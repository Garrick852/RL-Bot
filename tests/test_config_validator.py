"""Tests for the configuration gatekeeper, independent of RL dependencies."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from governance.validator import ConfigValidator, ValidationState

ROOT = Path(__file__).resolve().parents[1]


class ConfigValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = self.directory / "bars.csv"
        self.data.write_bytes(b"timestamp,close\n2023-01-01T00:00:00Z,100\n")
        template = json.loads(
            (ROOT / "config" / "offline_experiment.template.json").read_text("utf-8")
        )
        self.template = template
        self.config = {
            key: copy.deepcopy(template[key])
            for key in ("dataset", "strategy", "costs", "execution", "splits", "frozen")
        }

        def fill_strings(value):
            for key, child in value.items():
                if isinstance(child, dict):
                    fill_strings(child)
                elif child is None:
                    value[key] = "explicit-test-fixture"

        fill_strings(self.config)
        self.config["frozen"] = True
        self.config["dataset"].update({
            "path": "bars.csv",
            "sha256": hashlib.sha256(self.data.read_bytes()).hexdigest(),
            "column_mapping": {"timestamp": "timestamp", "close": "close"},
        })
        self.config["strategy"]["shorting_allowed"] = False
        self.config["strategy"]["ema"].update(fast_period=12, slow_period=26)
        for name in ("spread", "commission", "slippage"):
            self.config["costs"][name]["value"] = 0
        self.config["costs"]["swap"].update(long_value=0, short_value=0)
        for name, start, end in (
            ("train", "2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
            ("validation", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
            ("test", "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        ):
            self.config["splits"][name] = {"start": start, "end": end}
        self.logger = logging.getLogger(f"{__name__}.{self.id()}")
        self.logger.addHandler(logging.NullHandler())

    def validator(self, config=None):
        return ConfigValidator(
            self.config if config is None else config,
            base_directory=self.directory,
            logger=self.logger,
        )

    def assert_invalid(self, validator, reason, **kwargs):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            self.assertEqual(validator.validate(**kwargs), ValidationState.INVALID)
        self.assertIn(reason, "\n".join(logs.output))
        self.assertTrue(validator.errors)

    def test_states_and_template_are_not_mutated(self):
        validator = self.validator(self.template)
        original = copy.deepcopy(self.template)
        self.assertEqual(validator.validate(), ValidationState.DRAFT_INCOMPLETE)
        self.assertTrue(validator.unresolved_fields)
        self.assertEqual(validator.evidence["dataset_sha256"], "UNCHECKED")
        self.assertEqual(self.template, original)
        validator = self.validator()
        self.assertEqual(
            validator.validate(check_evidence=False), ValidationState.EVIDENCE_UNCHECKED
        )
        self.assertEqual(validator.validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)
        self.assertEqual(validator.actual_dataset_sha256, self.config["dataset"]["sha256"])
        self.config["frozen"] = False
        self.assertEqual(validator.validate(), ValidationState.DRAFT_INCOMPLETE)
        self.assertEqual(validator.evidence["dataset_sha256"], "PASS")

    def test_missing_roots_and_frozen_nested_nulls_are_invalid(self):
        for key in ("dataset", "strategy", "costs", "execution", "splits", "frozen"):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                del config[key]
                self.assert_invalid(self.validator(config), key)
        self.config["execution"]["terminal_liquidation_cost_rule"] = None
        self.assert_invalid(self.validator(), "terminal_liquidation_cost_rule")

    def test_invalid_types_unknowns_and_nonfinite_values_fail(self):
        for value in (1, "true", [], {}):
            with self.subTest(value=value):
                config = copy.deepcopy(self.config)
                config["frozen"] = value
                self.assert_invalid(self.validator(config), "frozen")
        config = copy.deepcopy(self.config)
        config["costs"]["commission"]["value"] = float("nan")
        self.assert_invalid(self.validator(config), "must be finite")
        config = copy.deepcopy(self.config)
        config["dataset"]["sh256"] = "typo"
        self.assert_invalid(self.validator(config), "unknown keys")
        self.assert_invalid(self.validator([]), "config must be a mapping")
        self.assert_invalid(self.validator(), "check_evidence", check_evidence="false")

    def test_supplied_draft_errors_still_fail_closed(self):
        self.config["frozen"] = False
        for fast, slow in ((30, 10), (12, 12), (0, 26), (-1, 26), (True, 26), (12.5, 26)):
            with self.subTest(fast=fast, slow=slow):
                self.config["strategy"]["ema"].update(fast_period=fast, slow_period=slow)
                self.assert_invalid(self.validator(), "strategy.ema")

    def test_half_open_gaps_and_timezone_normalization(self):
        # Equal instants expressed with different offsets are legal boundaries.
        self.config["splits"]["validation"]["start"] = "2021-01-01T01:00:00+01:00"
        self.config["splits"]["test"]["start"] = "2022-02-01T00:00:00Z"
        self.assertEqual(self.validator().validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)
        for start, end in (
            ("2020-06-01T00:00:00Z", "2022-01-01T00:00:00Z"),
            ("2022-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
            ("2023-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        ):
            with self.subTest(start=start, end=end):
                self.config["splits"]["validation"] = {"start": start, "end": end}
                self.assert_invalid(self.validator(), "splits.")

    def test_bad_or_naive_timestamps_are_not_assigned_a_timezone(self):
        for value in ("2021-01-01", "2021-01-01T00:00:00", "not-a-date", "2021-02-30T00:00:00Z"):
            with self.subTest(value=value):
                self.config["splits"]["validation"]["start"] = value
                self.assert_invalid(self.validator(), "splits.validation.start")

    def test_hash_mismatch_missing_file_and_unreadable_file(self):
        validator = self.validator()
        self.data.write_bytes(b"tampered")
        self.assert_invalid(validator, "dataset.sha256 mismatch")
        self.assertEqual(validator.evidence["dataset_sha256"], "FAIL")
        self.assertEqual(validator.evidence["dataset_readability"], "PASS")
        self.data.unlink()
        self.assert_invalid(validator, "not a readable regular file")
        self.assertEqual(validator.evidence["dataset_readability"], "FAIL")
        self.assertIsNone(validator.actual_dataset_sha256)
        self.data.write_bytes(b"fixture")
        with patch.object(Path, "open", side_effect=PermissionError("test permission denied")):
            self.assert_invalid(validator, "test permission denied")

    def test_malformed_digest_and_remote_paths_fail(self):
        for value in ("A" * 64, "0" * 63, "", "not-a-hash"):
            with self.subTest(value=value):
                config = copy.deepcopy(self.config)
                config["dataset"]["sha256"] = value
                self.assert_invalid(self.validator(config), "dataset.sha256")
        for path in ("https://example.invalid/bars.csv", r"\\server\bars.csv"):
            with self.subTest(path=path):
                config = copy.deepcopy(self.config)
                config["dataset"]["path"] = path
                self.assert_invalid(self.validator(config), "must be local")

    def test_unexpected_exception_is_logged_and_clears_stale_success(self):
        validator = self.validator()
        self.assertEqual(validator.validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)
        with patch.object(validator, "validate_layer2", side_effect=RuntimeError("unexpected L2 error")):
            self.assert_invalid(validator, "RuntimeError: unexpected L2 error")
        self.assertIsNone(validator.actual_dataset_sha256)
        self.assertEqual(validator.evidence["dataset_sha256"], "UNCHECKED")
        self.assertEqual(validator.validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)

    def test_streaming_hash_uses_exact_bytes(self):
        content = b"\x00\xff\r\n" * 600_000
        self.data.write_bytes(content)
        self.config["dataset"]["sha256"] = hashlib.sha256(content).hexdigest()
        validator = self.validator()
        self.assertEqual(validator.validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)
        self.assertEqual(validator.actual_dataset_sha256, hashlib.sha256(content).hexdigest())

    def test_file_mutation_during_hash_is_rejected(self):
        stat = self.data.stat()
        changed = SimpleNamespace(
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            st_size=stat.st_size + 1,
            st_mtime_ns=stat.st_mtime_ns + 1,
        )
        validator = self.validator()
        with patch("governance.validator.os.fstat", side_effect=[stat, changed]):
            self.assert_invalid(validator, "dataset changed during hashing")
        self.assertNotEqual(validator.evidence["dataset_readability"], "PASS")

    def test_no_network_training_or_ledger_claim_from_hash_success(self):
        # A tiny fixture deliberately cannot establish OHLCV/gap/warm-up validity.
        self.data.write_bytes(b"not market data")
        self.config["dataset"]["sha256"] = hashlib.sha256(self.data.read_bytes()).hexdigest()
        self.assertEqual(self.validator().validate(), ValidationState.READY_FOR_OFFLINE_EXECUTION)


if __name__ == "__main__":
    unittest.main()
