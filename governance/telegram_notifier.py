"""Fail-safe Telegram alerts; orchestration requires explicit --deploy opt-in."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Mapping
from typing import Any

LOGGER = logging.getLogger(__name__)


class TelegramAlertEngine:
    """Format operational alerts and send only when explicitly called.

    Credentials are read using os.getenv at send time from TELEGRAM_BOT_TOKEN
    and TELEGRAM_CHAT_ID. No credentials are accepted as arguments or logged.
    Notification failure never implies success and is reported as False.
    This class does not verify governance results or authorize deployment.
    """

    @staticmethod
    def escape_markdown_v2(text: str) -> str:
        """Escape all Markdown V2 reserved characters, including backslashes."""
        return re.sub(r"([\\_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

    def _format_and_send(self, title: str, lines: list[str]) -> bool:
        text = f"*{self.escape_markdown_v2(title)}*\n"
        text += "\n".join(self.escape_markdown_v2(line) for line in lines)
        return self._send(text)

    def send_deployment_success(
        self, asset: str, metrics: Mapping[str, float], param_diff: Mapping[str, Any]
    ) -> bool:
        """Send a green validation/build summary, not proof of live deployment.

        Args:
            asset: Explicit instrument identifier.
            metrics: Finite profit_factor and sharpe values supplied by caller.
            param_diff: Exact parameter changes, ideally {name: {old, new}}.

        Returns:
            True only after Telegram confirms acceptance. Caller must establish
            actual L3/OOS/build success before invoking this method.
        """
        try:
            if not isinstance(asset, str) or not asset.strip():
                raise ValueError("asset must be nonempty")
            for name in ("profit_factor", "sharpe"):
                value = metrics[name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"invalid {name}")
            if not isinstance(param_diff, Mapping):
                raise ValueError("param_diff must be a mapping")
            return self._format_and_send("🟢 Validation & compilation success", [
                f"Asset: {asset}",
                "L3 validation: PASS (caller supplied)",
                "MQL5 compilation: 0 errors, 0 warnings (caller supplied)",
                f"OOS Profit Factor: {metrics['profit_factor']}",
                f"OOS Sharpe: {metrics['sharpe']}",
                f"Exact parameter changes: {self._json(dict(param_diff))}",
                "Scope: validation/build notification only; no live deployment authorization.",
            ])
        except Exception:
            LOGGER.error("Telegram success alert could not be formatted; no message sent")
            return False

    def send_fallback_alert(
        self, stage: int, reason: str, active_config: Mapping[str, Any]
    ) -> bool:
        """Send Stage 1/2 fallback instructions without transmitting full config.

        Args:
            stage: Exactly 1 or 2; Stage 3 uses send_circuit_breaker.
            reason: OOS rejection reason, without secrets or personal data.
            active_config: Config summary source. Only status, strategy_state,
                and lot_size fields are included; no full config dump.

        Returns:
            Whether Telegram accepted the message. No fallback is executed.
        """
        try:
            if type(stage) is not int or stage not in (1, 2):
                raise ValueError("fallback stage must be 1 or 2")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reason must be nonempty")
            if not isinstance(active_config, Mapping):
                raise ValueError("active_config must be a mapping")
            summary = {name: active_config[name] for name in ("status", "strategy_state", "lot_size")
                       if name in active_config}
            directive = (
                "Recalibrate: review expanded bounds and entropy 0.03; use a fresh OOS holdout."
                if stage == 1 else
                "Reject new parameters; retain verified last-known-good active_config.json; "
                "reduce reference risk/lot size to 50% (not compounded)."
            )
            return self._format_and_send(f"🟠 OOS fallback — Stage {stage}", [
                f"Reason: {reason}", directive,
                f"Active config summary: {self._json(summary)}",
                "Verify last-known-good evidence; unconfigured is not known-good.",
                "Directive only; no configuration or position changes performed.",
            ])
        except Exception:
            LOGGER.error("Telegram fallback alert could not be formatted; no message sent")
            return False

    def send_circuit_breaker(self, asset: str, reason: str) -> bool:
        """Send a red Stage 3 suspension alert; no trading action is performed."""
        try:
            if not isinstance(asset, str) or not asset.strip():
                raise ValueError("asset must be nonempty")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reason must be nonempty")
            return self._format_and_send("🔴 CRITICAL — Stage 3 circuit breaker", [
                f"Asset: {asset}", "Strategy state: SUSPENDED",
                "Trigger: consecutive OOS validation failures >= 3",
                f"Reason: {reason}", "BLOCK ALL NEW ENTRIES.",
                "Reject candidate parameters; retain verified last-known-good configuration.",
                "Suspension requires explicit recovery authorization.",
                "Delivery of this alert does not itself halt execution.",
            ])
        except Exception:
            LOGGER.error("Telegram circuit-breaker alert could not be formatted; no message sent")
            return False

    def _send(self, text: str) -> bool:
        """Send once without redirects/retries; redact all remote error content."""
        response = None
        try:
            token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            if not token or not chat_id:
                LOGGER.error("Telegram alert not sent: required environment credentials are missing")
                return False
            if re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token) is None:
                LOGGER.error("Telegram alert not sent: invalid bot-token format")
                return False
            # Telegram's limit is in text characters; UTF-16 units is conservative
            # for astral emoji. Refuse rather than silently truncate exact changes.
            if len(text.encode("utf-16-le")) // 2 > 4096:
                LOGGER.error("Telegram alert not sent: message exceeds 4096-character limit")
                return False
            import requests

            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id, "text": text,
                    "parse_mode": "MarkdownV2", "disable_web_page_preview": True,
                },
                timeout=5, allow_redirects=False,
            )
            if response.status_code != 200:
                LOGGER.error("Telegram transmission failed: HTTP status %s", response.status_code)
                return False
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                LOGGER.error("Telegram transmission failed: API did not confirm acceptance")
                return False
            return True
        except Exception:
            # Requests exceptions can embed the credential-bearing URL. Never log
            # exception text/tracebacks, response bodies, message text, or IDs.
            LOGGER.error("Telegram transmission failed: dependency, transport, timeout, or response error")
            return False
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    LOGGER.error("Telegram response cleanup failed")
