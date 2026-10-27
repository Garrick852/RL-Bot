"""MT5 conversion requires explicit quote semantics and timezone provenance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.collector import CORE_COLUMNS, DataCollector, DataIntegrityError
from data.parser_mt5 import MT5DataParser

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<SPREAD>"]


class MT5ParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bars.csv"
        self.mapping = Path(self.temp.name) / "symbols.json"
        self.mapping.write_bytes((ROOT / "config" / "broker_symbols.json").read_bytes())

    def parser(self, **kwargs):
        settings = {
            "symbol": "US500", "point_size": .01, "source_timezone": "UTC",
            "bid_ohlc_and_close_aligned_spread": True, "symbol_map_path": self.mapping,
        }
        settings.update(kwargs)
        return MT5DataParser(**settings)

    def write(self, delimiter=",", encoding="utf-8", rows=None, header=None):
        rows = rows if rows is not None else [
            ["2024.01.02", "09:30", "100", "102", "99", "101", "25", "2"],
            ["2024.01.02", "09:31:00.123", "101", "103", "100", "102", "30", "0"],
        ]
        lines = [delimiter.join(header or HEADER)] + [delimiter.join(row) for row in rows]
        self.path.write_text("\n".join(lines) + "\n", encoding=encoding)

    def test_formats_schema_and_reconstruction(self):
        for separator, encoding in ((",", "utf-8-sig"), ("\t", "utf-16"), (";", "utf-8")):
            with self.subTest(separator=separator, encoding=encoding):
                self.write(separator, encoding)
                frame = self.parser().parse_ohlcv(self.path)
                self.assertEqual(list(frame.columns), list(CORE_COLUMNS))
                self.assertEqual(frame["bid"].tolist(), [101, 102])
                self.assertEqual(frame["ask"].tolist(), [101 + 2*.01, 102])
                self.assertEqual(frame["volume"].tolist(), [25, 30])
                self.assertEqual(str(frame["timestamp"].dt.tz), "UTC")
                self.assertEqual(frame.attrs["symbol"], "US500.cash")
                self.assertEqual(frame.attrs["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
                DataCollector().validate_ohlcv_invariants(frame)

    def test_aliases_exact_broker_symbols_and_unknown_aliases(self):
        for alias in ("HK50", "US500", "US30"):
            self.assertEqual(self.parser(symbol=alias).symbol, alias + ".cash")
            self.assertEqual(self.parser(symbol=alias + ".cash").symbol, alias + ".cash")
        with self.assertRaisesRegex(DataIntegrityError, "unknown symbol"):
            self.parser(symbol="SP500")
        for content in ('[]', '{}', '{"US500": 1}', '{"US500":"a","US500":"b"}'):
            self.mapping.write_text(content)
            with self.subTest(content=content), self.assertRaises(DataIntegrityError):
                self.parser()

    def test_quote_contract_and_point_metadata_cannot_be_inferred(self):
        for patch in (
            {"bid_ohlc_and_close_aligned_spread": False},
            {"bid_ohlc_and_close_aligned_spread": 1},
            {"point_size": 0}, {"point_size": True}, {"point_size": float("nan")},
            {"source_timezone": "Not/AZone"}, {"source_timezone": ""},
        ):
            with self.subTest(patch=patch), self.assertRaises(DataIntegrityError):
                self.parser(**patch)

    def test_timezone_conversion_dst_ambiguity_and_nonexistent_time(self):
        self.write()
        frame = self.parser(source_timezone="America/New_York").parse_ohlcv(self.path)
        self.assertEqual(frame["timestamp"].iloc[0], pd.Timestamp("2024-01-02T14:30:00Z"))
        for date, time in (("2024.11.03", "01:30:00"), ("2024.03.10", "02:30:00")):
            self.write(rows=[[date, time, "100", "102", "99", "101", "25", "2"]])
            with self.subTest(date=date), self.assertRaisesRegex(DataIntegrityError, "timestamp"):
                self.parser(source_timezone="America/New_York").parse_ohlcv(self.path)

    def test_invalid_rows_never_filled_sorted_or_repaired(self):
        for index, value in (
            (0, "2024.02.30"), (1, "25:01"), (2, "999"), (3, "99"),
            (4, "102"), (5, "nan"), (6, "-1"), (7, "-1"), (7, ""),
            (7, "inf"), (5, "not-a-number"),
        ):
            row = ["2024.01.02", "09:30", "100", "102", "99", "101", "25", "2"]
            row[index] = value
            self.write(rows=[row])
            with self.subTest(index=index, value=value), self.assertRaises(DataIntegrityError):
                self.parser().parse_ohlcv(self.path)
        row = ["2024.01.02", "09:30", "100", "102", "99", "101", "25", "2"]
        self.write(rows=[row, row])
        with self.assertRaisesRegex(DataIntegrityError, "strictly increasing"):
            self.parser().parse_ohlcv(self.path)

    def test_unsupported_layout_and_malformed_csv_fail(self):
        for header, rows in (
            (HEADER[:-1], [["1"]*7]),
            (HEADER + ["<SPREAD>"], [["1"]*9]),
            (HEADER, [["1"]*9]),
            (HEADER, [[]]),
            (["<DATE>", "<TIME>", "<BID>", "<ASK>"], [["1"]*4]),
        ):
            self.write(rows=rows, header=header)
            with self.subTest(header=header), self.assertRaises(DataIntegrityError):
                self.parser().parse_ohlcv(self.path)
        with self.assertRaisesRegex(DataIntegrityError, "local file"):
            self.parser().parse_ohlcv("https://example.invalid/file.csv")
        self.path.unlink()
        with self.assertRaisesRegex(DataIntegrityError, "cannot parse"):
            self.parser().parse_ohlcv(self.path)

    def test_real_volume_does_not_override_explicit_tick_volume(self):
        self.write(header=HEADER + ["<VOL>"], rows=[
            ["2024.01.02", "09:30", "100", "102", "99", "101", "25", "2", "999"],
        ])
        self.assertEqual(self.parser().parse_ohlcv(self.path)["volume"].iloc[0], 25)


if __name__ == "__main__":
    unittest.main()
