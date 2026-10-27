"""Explicit-contract MT5 bar-export conversion to governed canonical quotes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from data.collector import CORE_COLUMNS, DataCollector, DataIntegrityError, _finite_real


def _local_path(value: str | Path) -> Path:
    text = str(value)
    if not text.strip() or text.startswith(("\\\\", "//")) or "://" in text:
        raise DataIntegrityError("parser inputs must be nonempty local file paths")
    return Path(value)


def _symbol_object(pairs: list[tuple[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for alias, symbol in pairs:
        if alias in result:
            raise DataIntegrityError(f"duplicate symbol alias: {alias}")
        if not alias.strip() or not isinstance(symbol, str) or not symbol.strip():
            raise DataIntegrityError("symbol aliases and broker symbols must be nonempty strings")
        result[alias] = symbol
    return result


class MT5DataParser:
    """Convert MT5-shaped bar CSV exports without inventing quote metadata.

    Args:
        symbol: Configured standard alias or an exact configured broker symbol.
        point_size: Explicit broker point size, not inferred from tick size,
            symbol spelling, decimal digits, or export prices.
        source_timezone: IANA timezone of naive export DATE/TIME values.
        bid_ohlc_and_close_aligned_spread: Must explicitly be True. The caller
            attests that OHLC are Bid prices and SPREAD is the close-time
            Ask-minus-Bid difference in broker points.
        symbol_map_path: Local alias map, relative to working directory when
            relative. Defaults to config/broker_symbols.json.

    Raises:
        DataIntegrityError: Metadata, symbol map, or quote contract is invalid.

    Notes:
        Only MT5's DATE/TIME/OHLC/TICKVOL/SPREAD bar layout is supported,
        including TDS files exported into that exact layout. Raw tick files,
        inherited/unknown spread semantics, and other TDS layouts are refused.
        A configured symbol is metadata, not proof of a file's instrument.
    """

    def __init__(
        self,
        *,
        symbol: str,
        point_size: float,
        source_timezone: str,
        bid_ohlc_and_close_aligned_spread: bool,
        symbol_map_path: str | Path = Path("config") / "broker_symbols.json",
    ) -> None:
        if bid_ohlc_and_close_aligned_spread is not True:
            raise DataIntegrityError("explicit Bid OHLC and close-aligned spread confirmation is required")
        self.point_size = _finite_real(point_size, "point_size")
        if self.point_size <= 0:
            raise DataIntegrityError("point_size must be strictly positive")
        if not isinstance(source_timezone, str) or not source_timezone.strip():
            raise DataIntegrityError("source_timezone must be an explicit IANA timezone")
        try:
            ZoneInfo(source_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise DataIntegrityError(f"unsupported source_timezone: {source_timezone!r}") from exc
        self.source_timezone = source_timezone
        try:
            raw = _local_path(symbol_map_path).read_bytes()
            mapping = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_symbol_object)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataIntegrityError(f"cannot load broker symbol map: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise DataIntegrityError("broker symbol map must be a nonempty JSON object")
        if not isinstance(symbol, str) or not symbol.strip():
            raise DataIntegrityError("symbol must be a nonempty configured alias or broker symbol")
        if symbol in mapping:
            self.symbol = mapping[symbol]
        elif symbol in mapping.values():
            self.symbol = symbol
        else:
            raise DataIntegrityError(f"unknown symbol alias or broker symbol: {symbol!r}")
        self.symbol_alias = symbol
        self.symbol_map_sha256 = hashlib.sha256(raw).hexdigest()

    def parse_ohlcv(self, filepath: str | Path) -> pd.DataFrame:
        """Parse UTF-8/BOM UTF-16 comma, tab, or semicolon MT5 bar CSV.

        Args:
            filepath: Local MT5-format bar file. DATE uses YYYY.MM.DD and TIME
                uses HH:MM or HH:MM:SS with optional fractional seconds.

        Returns:
            Chronological canonical CORE_COLUMNS with UTC-aware timestamp,
            volume from TICKVOL, Bid from CLOSE, and Ask = Bid + SPREAD *
            point_size. Source/contract identities are retained in df.attrs.
            No sorting, forward fills, missing-spread defaults, or aggregation
            are performed.

        Raises:
            DataIntegrityError: File, layout, numeric fields, timezone/DST, or
                canonical quote invariants fail.
        """
        try:
            raw = _local_path(filepath).read_bytes()
            encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = raw.decode(encoding)
            first_line = text.splitlines()[0] if text.splitlines() else ""
            dialect = csv.Sniffer().sniff(first_line, delimiters=",\t;")
            records = list(csv.reader(io.StringIO(text, newline=""), dialect))
            if not records:
                raise DataIntegrityError("MT5 export is empty")
            header = records[0]
            if len(header) != len(set(header)):
                raise DataIntegrityError("MT5 export has duplicate column headers")
            required = {
                "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
                "<TICKVOL>", "<SPREAD>",
            }
            if not required.issubset(header):
                raise DataIntegrityError(f"unsupported MT5/TDS layout; missing columns: {sorted(required - set(header))}")
            unknown = set(header) - required - {"<VOL>"}
            if unknown:
                raise DataIntegrityError(f"unsupported export columns: {sorted(unknown)}")
            for line, row in enumerate(records[1:], start=2):
                if len(row) != len(header):
                    raise DataIntegrityError(f"MT5 CSV row {line} has {len(row)} fields; expected {len(header)}")
            source = pd.read_csv(io.StringIO(text), sep=dialect.delimiter, dtype=str, skip_blank_lines=False)
        except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataIntegrityError(f"cannot parse MT5 export {filepath}: {exc}") from exc
        if source.empty or source[list(required)].isna().any().any():
            raise DataIntegrityError("MT5 export is empty or required fields contain nulls")
        dates, times = source["<DATE>"], source["<TIME>"]
        if not dates.str.fullmatch(r"\d{4}\.\d{2}\.\d{2}").all() or not times.str.fullmatch(
            r"\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
        ).all():
            raise DataIntegrityError("DATE/TIME must use YYYY.MM.DD and HH:MM[:SS[.fraction]]")
        try:
            # Localize rather than treating broker-local wall clock as UTC.
            naive = pd.DatetimeIndex([pd.Timestamp(f"{date.replace('.', '-')} {time}") for date, time in zip(dates, times)])
            timestamps = naive.tz_localize(
                self.source_timezone, ambiguous="raise", nonexistent="raise"
            ).tz_convert("UTC")
        except Exception as exc:
            # Timezone backends raise distinct DST exception types.
            raise DataIntegrityError(f"invalid or ambiguous broker timestamp: {exc}") from exc
        output = pd.DataFrame({"timestamp": timestamps})
        for source_name, target in (
            ("<OPEN>", "open"), ("<HIGH>", "high"), ("<LOW>", "low"),
            ("<CLOSE>", "close"), ("<TICKVOL>", "volume"), ("<SPREAD>", "_spread_points"),
        ):
            try:
                output[target] = pd.to_numeric(source[source_name], errors="raise")
            except (TypeError, ValueError) as exc:
                raise DataIntegrityError(f"{source_name} contains invalid numeric values: {exc}") from exc
            if not np.isfinite(output[target].to_numpy(dtype=np.float64)).all():
                raise DataIntegrityError(f"{source_name} contains non-finite values")
        if (output["_spread_points"] < 0).any():
            raise DataIntegrityError("SPREAD must be non-negative")
        output["bid"] = output["close"]
        with np.errstate(over="ignore", invalid="ignore"):
            output["ask"] = output["close"] + output["_spread_points"] * self.point_size
        output = output.loc[:, list(CORE_COLUMNS)]
        DataCollector().validate_ohlcv_invariants(output)
        output.attrs.update({
            "symbol": self.symbol,
            "symbol_alias": self.symbol_alias,
            "point_size": self.point_size,
            "source_timezone": self.source_timezone,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "symbol_map_sha256": self.symbol_map_sha256,
            "quote_contract": "bid_ohlc_close_aligned_spread_points",
            "timestamp_meaning": "export_bar_timestamp_not_certified_execution_time",
        })
        return output
