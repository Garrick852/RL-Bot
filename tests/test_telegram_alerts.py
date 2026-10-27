"""No real network traffic; all credentials are synthetic test fixtures."""

import logging
import os
import unittest
from unittest.mock import Mock, patch

from governance.telegram_notifier import TelegramAlertEngine


class TelegramAlertTests(unittest.TestCase):
    def setUp(self):
        self.engine = TelegramAlertEngine()
        self.environment = patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "12345:SYNTHETIC_TEST_ONLY",
            "TELEGRAM_CHAT_ID": "synthetic-chat",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.response = Mock(status_code=200)
        self.response.json.return_value = {"ok": True}

    def test_all_reserved_markdown_characters(self):
        reserved = "\\_*[]()~`>#+-=|{}.!"
        self.assertEqual(
            self.engine.escape_markdown_v2(reserved),
            "".join("\\" + char for char in reserved),
        )
        self.assertEqual(self.engine.escape_markdown_v2("plain α 123"), "plain α 123")

    def test_success_payload_exact_changes_and_environment_only_credentials(self):
        with patch("requests.post", return_value=self.response) as post:
            self.assertTrue(self.engine.send_deployment_success(
                "US500.cash", {"profit_factor": 1.42, "sharpe": 1.18},
                {"Inp_Bars": {"old": 5, "new": 12}},
            ))
        url = post.call_args.args[0]
        options = post.call_args.kwargs
        self.assertEqual(url, "https://api.telegram.org/bot12345:SYNTHETIC_TEST_ONLY/sendMessage")
        self.assertEqual(options["timeout"], 5)
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(options["json"]["parse_mode"], "MarkdownV2")
        text = options["json"]["text"]
        self.assertIn("US500\\.cash", text)
        self.assertIn("1\\.42", text)
        self.assertIn('"old": 5', text)
        self.assertIn('"new": 12', text)
        self.assertIn("no live deployment authorization", text)
        self.response.close.assert_called_once()

    def test_fallback_stage_and_suspension_messages(self):
        with patch("requests.post", return_value=self.response) as post:
            for stage in (1, 2):
                self.assertTrue(self.engine.send_fallback_alert(stage, "OOS failed!", {
                    "status": "verified", "lot_size": .1, "secret": "NEVER_SEND",
                }))
                text = post.call_args.kwargs["json"]["text"]
                self.assertNotIn("NEVER_SEND", text)
                self.assertIn(f"Stage {stage}", text)
            self.assertIn("50%", text)
            self.assertTrue(self.engine.send_circuit_breaker("HK50.cash", "Three failures."))
            text = post.call_args.kwargs["json"]["text"]
            self.assertIn("SUSPENDED", text)
            self.assertIn("BLOCK ALL NEW ENTRIES", text)

    def test_transport_errors_never_leak_credentials_or_raise(self):
        for failure in (TimeoutError("12345:SYNTHETIC_TEST_ONLY synthetic-chat"),
                        RuntimeError("unexpected transport failure")):
            with self.subTest(failure=type(failure)), patch("requests.post", side_effect=failure):
                with self.assertLogs("governance.telegram_notifier", logging.ERROR) as logs:
                    self.assertFalse(self.engine.send_circuit_breaker("HK50", "failure"))
                output = "\n".join(logs.output)
                self.assertNotIn("SYNTHETIC_TEST_ONLY", output)
                self.assertNotIn("synthetic-chat", output)

    def test_bad_http_json_and_api_rejection_fail_closed(self):
        for code, payload in ((500, {"ok": True}), (302, {"ok": True}), (200, {"ok": False}),
                              (200, []), (200, {"ok": 1})):
            response = Mock(status_code=code)
            response.json.return_value = payload
            with self.subTest(code=code, payload=payload), patch("requests.post", return_value=response):
                with self.assertLogs("governance.telegram_notifier", logging.ERROR):
                    self.assertFalse(self.engine.send_circuit_breaker("A", "rejected"))
        self.response.json.side_effect = ValueError("bad JSON")
        with patch("requests.post", return_value=self.response):
            with self.assertLogs("governance.telegram_notifier", logging.ERROR):
                self.assertFalse(self.engine.send_circuit_breaker("A", "rejected"))

    def test_missing_credentials_oversized_and_invalid_inputs_make_no_request(self):
        with patch("requests.post") as post:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertLogs("governance.telegram_notifier", logging.ERROR):
                    self.assertFalse(self.engine.send_circuit_breaker("A", "failure"))
            with self.assertLogs("governance.telegram_notifier", logging.ERROR):
                self.assertFalse(self.engine.send_circuit_breaker("A", "x" * 4097))
                self.assertFalse(self.engine.send_fallback_alert(3, "wrong method", {}))
                self.assertFalse(self.engine.send_fallback_alert(True, "invalid stage", {}))
                self.assertFalse(self.engine.send_deployment_success("A", {"profit_factor": float("nan")}, {}))
                self.assertFalse(self.engine.send_circuit_breaker("", "failure"))
            post.assert_not_called()

    def test_response_cleanup_failure_does_not_raise(self):
        self.response.close.side_effect = RuntimeError("cleanup includes token")
        with patch("requests.post", return_value=self.response):
            with self.assertLogs("governance.telegram_notifier", logging.ERROR):
                self.assertTrue(self.engine.send_circuit_breaker("A", "failure"))


if __name__ == "__main__":
    unittest.main()
