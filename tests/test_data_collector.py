"""Quote, EMA, explicit-calendar gap, and warm-up isolation contracts."""

from __future__ import annotations

import tempfile
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.collector import (
    DataCollector,
    DataIntegrityError,
    PollutedDatasetError,
    quantize,
)


def quotes(times=None):
    if times is None:
        times = pd.date_range("2024-01-02T14:30:00Z", periods=12, freq="min")
    size = len(times)
    return pd.DataFrame({
        "timestamp": times,
        "open": np.full(size, 100.0),
        "high": np.full(size, 102.0),
        "low": np.full(size, 98.0),
        "close": np.full(size, 101.0),
        "volume": np.full(size, 10.0),
        "bid": np.full(size, 100.5),
        "ask": np.full(size, 101.5),
    })


class DataCollectorTests(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector()
        self.frame = quotes()

    def test_valid_quotes_are_not_mutated_and_equal_boundaries_pass(self):
        original = self.frame.copy(deep=True)
        self.collector.validate_ohlcv_invariants(self.frame)
        pd.testing.assert_frame_equal(self.frame, original)
        for column in ("open", "high", "low", "close", "bid", "ask"):
            self.frame[column] = 100.0
        self.frame["volume"] = 0
        self.collector.validate_ohlcv_invariants(self.frame)

    def test_missing_duplicate_empty_columns_and_wrong_frame(self):
        for column in self.frame.columns:
            with self.subTest(column=column), self.assertRaises(DataIntegrityError):
                self.collector.validate_ohlcv_invariants(self.frame.drop(columns=column))
        for frame in (
            self.frame.iloc[:0],
            pd.concat([self.frame, self.frame[["ask"]]], axis=1),
            [],
        ):
            with self.subTest(frame=type(frame)), self.assertRaises(DataIntegrityError):
                self.collector.validate_ohlcv_invariants(frame)

    def test_null_nonfinite_non_numeric_and_boolean_quotes_fail(self):
        for column in self.frame.columns:
            frame = self.frame.copy()
            frame.loc[0, column] = None
            with self.subTest(null=column), self.assertRaises(DataIntegrityError):
                self.collector.validate_ohlcv_invariants(frame)
        for value in (np.inf, -np.inf, np.nan, "101", True, complex(1, 2)):
            frame = self.frame.copy()
            frame["ask"] = value
            with self.subTest(value=value), self.assertRaises(DataIntegrityError):
                self.collector.validate_ohlcv_invariants(frame)

    def test_each_quote_invariant_is_enforced(self):
        for column, value, reason in (
            ("ask", 99, "ask >= bid"),
            ("high", 99, "high >= open"),
            ("high", 100.5, "high >= close"),
            ("low", 100.5, "low <= open"),
            ("close", 97, "low <= close"),
            ("volume", -1, "volume >= 0"),
        ):
            frame = self.frame.copy()
            frame.loc[3, column] = value
            with self.subTest(reason=reason), self.assertRaisesRegex(DataIntegrityError, reason):
                self.collector.validate_ohlcv_invariants(frame)

    def test_timestamps_require_explicit_offsets_and_strict_chronology(self):
        for times in (
            [pd.Timestamp("2024-01-01")] * 2,
            [1, 2],
            [pd.NaT, pd.Timestamp("2024-01-02T00:00:00Z")],
            ["2024-01-01T00:00:00Z"] * 2,
            ["2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z"],
        ):
            with self.subTest(times=times), self.assertRaises(DataIntegrityError):
                self.collector.validate_ohlcv_invariants(quotes(times))
        self.collector.validate_ohlcv_invariants(quotes([
            "2024-01-01T00:00:00Z", "2024-01-01T02:00:00+01:00",
        ]))

    def test_native_half_even_quantization_and_float_contract(self):
        for value, expected in ((.25, 0), (.75, 1), (1.25, 1), (-.25, 0), (-.75, -1)):
            self.assertEqual(quantize(value, .5), expected)
        for value in (1.234565, -1.234565, 0, 123.5):
            self.assertEqual(quantize(value, .00001), round(value / .00001) * .00001)
        for value, tick in ((np.inf, 1), (1, 0), (1, -1), (True, 1), (1, np.nan), (1e308, 1e-308)):
            with self.subTest(value=value, tick=tick), self.assertRaises(DataIntegrityError):
                quantize(value, tick)

    def test_ema_uses_exact_recursive_formula_and_quantized_feedback(self):
        frame = self.frame.iloc[:4].copy()
        frame["close"] = [100.0, 101.0, 99.0, 101.5]
        previous = quantize(100, .5)
        expected = [previous]
        for close in frame["close"].iloc[1:]:
            previous = round(((close - previous) * (2 / (3 + 1)) + previous) / .5) * .5
            expected.append(previous)
        series = self.collector.calculate_ema(frame, 3, .5)
        self.assertEqual(series.tolist(), expected)
        self.assertTrue(series.index.equals(frame.index))
        seeded = self.collector.calculate_ema(frame, 3, .5, initial_ema=99)
        self.assertEqual(seeded.iloc[0], quantize((100 - 99) * .5 + 99, .5))
        changed = frame.copy()
        changed.loc[3, "close"] = 99
        pd.testing.assert_series_equal(
            series.iloc[:3], self.collector.calculate_ema(changed, 3, .5).iloc[:3]
        )
        for period in (0, -1, True, 1.5):
            with self.subTest(period=period), self.assertRaises(DataIntegrityError):
                self.collector.calculate_ema(frame, period, .5)

    def test_gap_report_tolerance_and_consecutive_not_total(self):
        times = pd.DatetimeIndex(self.frame["timestamp"])
        collector = DataCollector(expected_rth_timestamps=times, bar_interval=pd.Timedelta("1min"))
        report = collector.check_gap_policy(self.frame.drop(index=[2, 3, 7, 8]), 2)
        self.assertEqual(report.max_consecutive_missing, 2)
        self.assertEqual(len(report.missing_rth_timestamps), 4)
        self.assertEqual(report.time_deltas.loc[4], pd.Timedelta("3min"))
        with self.assertRaises(PollutedDatasetError):
            collector.check_gap_policy(self.frame.drop(index=[2, 3, 4]), 2)
        with self.assertRaises(PollutedDatasetError):
            collector.check_gap_policy(self.frame.drop(index=[0]), 0)
        with self.assertRaises(PollutedDatasetError):
            collector.check_gap_policy(self.frame.drop(index=[11]), 0)
        self.assertEqual(collector.check_gap_policy(self.frame, 0).max_consecutive_missing, 0)
        for threshold in (-1, True, 1.5):
            with self.subTest(threshold=threshold), self.assertRaises(DataIntegrityError):
                collector.check_gap_policy(self.frame, threshold)

    def test_market_closures_reset_runs_and_offhours_do_not_replace_rth(self):
        # The supplied schedule already encodes a weekend and early close.
        times = pd.DatetimeIndex([
            "2024-01-05T20:58:00Z", "2024-01-05T20:59:00Z",
            "2024-01-08T14:30:00Z", "2024-01-08T14:31:00Z",
        ])
        collector = DataCollector(expected_rth_timestamps=times, bar_interval=pd.Timedelta("1min"))
        frame = quotes(times.delete([1, 2]))
        report = collector.check_gap_policy(frame, 1)
        self.assertEqual(report.max_consecutive_missing, 1)
        offhours = quotes(pd.DatetimeIndex(["2024-01-06T12:00:00Z"]))
        with self.assertRaises(PollutedDatasetError):
            collector.check_gap_policy(offhours, 1)

    def test_calendar_contract_must_be_explicit(self):
        with self.assertRaisesRegex(DataIntegrityError, "explicit RTH"):
            self.collector.check_gap_policy(self.frame, 0)
        times = pd.DatetimeIndex(self.frame["timestamp"])
        for schedule, interval in (
            (times, None), (None, pd.Timedelta("1min")), (times, pd.Timedelta(0)),
            (times[::-1], pd.Timedelta("1min")), (times.tz_localize(None), pd.Timedelta("1min")),
            (times, pd.Timedelta("2min")), (times[:0], pd.Timedelta("1min")),
        ):
            with self.subTest(schedule=schedule, interval=interval), self.assertRaises(DataIntegrityError):
                DataCollector(expected_rth_timestamps=schedule, bar_interval=interval)

    def test_warmup_is_exact_separate_and_half_open(self):
        start, end = self.frame["timestamp"].iloc[[6, 10]]
        original = self.frame.copy(deep=True)
        warmup, active = self.collector.get_split(self.frame, start, end, 2, 3)
        self.assertEqual(warmup.index.tolist(), list(range(6)))
        self.assertEqual(active.index.tolist(), [6, 7, 8, 9])
        self.assertTrue((warmup["timestamp"] < start).all())
        self.assertTrue((active["timestamp"] < end).all())
        warmup.loc[0, "close"] = -999
        active.loc[6, "close"] = -999
        pd.testing.assert_frame_equal(self.frame, original)

    def test_invalid_split_history_and_parameters_are_never_padded(self):
        start, end = self.frame["timestamp"].iloc[[6, 10]]
        for args in (
            (start, end, 3, 3), (start, start, 2, 3), (end, start, 2, 3),
            (start, end, 0, 3), (start, end, True, 3), (start, end, 1.5, 3),
            ("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", 1, 3),
        ):
            with self.subTest(args=args), self.assertRaises(DataIntegrityError):
                self.collector.get_split(self.frame, *args)

    def test_local_csv_loader_rejects_missing_quotes_duplicate_headers_and_bad_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.csv"
            self.frame.to_csv(path, index=False)
            loaded = self.collector.load_csv(path)
            self.assertEqual(str(loaded["timestamp"].dt.tz), "UTC")
            self.assertEqual(len(loaded), len(self.frame))
            self.collector.load_csv(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(DataIntegrityError, "expected_sha256"):
                self.collector.load_csv(path, expected_sha256="0" * 64)
            self.frame.drop(columns="bid").to_csv(path, index=False)
            with self.assertRaisesRegex(DataIntegrityError, "bid"):
                self.collector.load_csv(path)
            path.write_text("timestamp,ask,ask\n", encoding="utf-8")
            with self.assertRaisesRegex(DataIntegrityError, "duplicate"):
                self.collector.load_csv(path)
            self.frame.to_csv(path, index=False)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(DataIntegrityError, "null/NaN"):
                self.collector.load_csv(path)
            path.write_text("timestamp,open\n2024-01-01T00:00:00Z,1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(DataIntegrityError, "extra fields"):
                self.collector.load_csv(path)
            with self.assertRaisesRegex(DataIntegrityError, "local CSV"):
                self.collector.load_csv("https://example.invalid/quotes.csv")
            path.unlink()
            with self.assertRaisesRegex(DataIntegrityError, "cannot load CSV"):
                self.collector.load_csv(path)


if __name__ == "__main__":
    unittest.main()
