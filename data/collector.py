"""Strict local quote loading, tick-aligned EMA, and causal data partitions."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

CORE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "bid", "ask")
TimeInput = Union[str, datetime, pd.Timestamp]


class DataIntegrityError(ValueError):
    """Required quotes, timestamps, or feature inputs violate the contract."""


class PollutedDatasetError(DataIntegrityError):
    """Consecutive missing scheduled RTH bars exceed the declared tolerance."""


@dataclass(frozen=True)
class GapReport:
    """Evidence of observed deltas and missing expected RTH bar timestamps.

    Attributes:
        time_deltas: Consecutive observed row deltas; the first is NaT.
        missing_rth_timestamps: Missing bars within the caller's full schedule.
        max_consecutive_missing: Largest missing run within one contiguous
            scheduled session. Session closures reset the run.
    """

    time_deltas: pd.Series
    missing_rth_timestamps: pd.DatetimeIndex
    max_consecutive_missing: int


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise DataIntegrityError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise DataIntegrityError(f"{name} cannot be represented as a finite float") from exc
    if not math.isfinite(result):
        raise DataIntegrityError(f"{name} must be finite")
    return result


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise DataIntegrityError(f"{name} must be an integer >= {minimum}")
    return int(value)


def quantize(value: Real, tick_size: Real) -> float:
    """Quantize a finite value with Python's native round-half-to-even.

    Args:
        value: Value to place on the tick grid.
        tick_size: Strictly positive finite tick size.

    Returns:
        ``round(value / tick_size) * tick_size`` using binary floating point.
        Decimal-looking ties may not be exact binary ties; no decimal rounding
        or epsilon adjustment is substituted for the requested formula.

    Raises:
        DataIntegrityError: An input or the scaled/result value is non-finite,
            or tick_size is not strictly positive.
    """
    number = _finite_real(value, "value")
    tick = _finite_real(tick_size, "tick_size")
    if tick <= 0:
        raise DataIntegrityError("tick_size must be strictly positive")
    if not math.isfinite(number / tick):
        raise DataIntegrityError("value / tick_size is not finite")
    try:
        result = round(number / tick) * tick
    except OverflowError as exc:
        raise DataIntegrityError("quantized value overflowed") from exc
    if not math.isfinite(result):
        raise DataIntegrityError("quantized value is not finite")
    return result


def _timestamp(value: TimeInput, name: str) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise DataIntegrityError(f"{name} must be a timestamp, not an epoch number")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise DataIntegrityError(f"{name} must be non-null and timezone-aware")
        return stamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataIntegrityError(f"{name}: {exc}") from exc


def _timestamps(values: pd.Series, name: str) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex([_timestamp(value, name) for value in values])
    if not stamps.is_unique or not stamps.is_monotonic_increasing:
        raise DataIntegrityError(f"{name} must be unique and strictly increasing")
    return stamps


class DataCollector:
    """Load and transform historical quotes without repairs or execution.

    Args:
        expected_rth_timestamps: Complete timezone-aware expected bar timestamps
            for the exact window being audited, including leading/trailing
            expected bars. Session closures, holidays, and early closes MUST
            already be reflected here. None disables gap checking, not the gate.
        bar_interval: Positive pandas Timedelta specifying the nominal bar
            interval. Required together with expected_rth_timestamps.

    Raises:
        DataIntegrityError: Calendar inputs are partial, unordered, or invalid.
    """

    def __init__(
        self,
        *,
        expected_rth_timestamps: Optional[pd.DatetimeIndex] = None,
        bar_interval: Optional[pd.Timedelta] = None,
    ) -> None:
        self._expected_rth: Optional[pd.DatetimeIndex] = None
        self._bar_interval: Optional[pd.Timedelta] = None
        if (expected_rth_timestamps is None) != (bar_interval is None):
            raise DataIntegrityError("expected_rth_timestamps and bar_interval must be supplied together")
        if expected_rth_timestamps is not None:
            if not isinstance(expected_rth_timestamps, pd.DatetimeIndex):
                raise DataIntegrityError("expected_rth_timestamps must be a DatetimeIndex")
            if expected_rth_timestamps.empty:
                raise DataIntegrityError("expected_rth_timestamps must not be empty")
            if not isinstance(bar_interval, pd.Timedelta) or pd.isna(bar_interval) or bar_interval <= pd.Timedelta(0):
                raise DataIntegrityError("bar_interval must be a positive pandas Timedelta")
            expected = _timestamps(pd.Series(expected_rth_timestamps), "expected_rth_timestamps")
            if (expected.to_series().diff().dropna() < bar_interval).any():
                raise DataIntegrityError("expected RTH bars are closer than bar_interval")
            self._expected_rth = expected.copy()
            self._bar_interval = bar_interval

    def load_csv(
        self, path: Union[str, Path], *, expected_sha256: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a local CSV, validate quotes, and return UTC-aware timestamps.

        Args:
            path: Physical local CSV path; no URL or network source.
            expected_sha256: Optional expected digest. Parsing uses the same
                immutable bytes that are hashed, never a subsequent file read.

        Returns:
            Validated DataFrame. No missing quotes are filled or discarded.

        Raises:
            DataIntegrityError: File access, parsing, headers, or quotes fail.
        """
        if str(path).startswith(("\\\\", "//")) or "://" in str(path):
            raise DataIntegrityError("path must identify a local CSV file")
        try:
            raw = Path(path).read_bytes()
            if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise DataIntegrityError("loaded CSV bytes do not match expected_sha256")
            with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
                header = next(csv.reader(stream), None)
                if not header or len(header) != len(set(header)):
                    raise DataIntegrityError("CSV header is missing or contains duplicate columns")
                stream.seek(0)
                frame = pd.read_csv(stream, skip_blank_lines=False)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataIntegrityError(f"cannot load CSV {path}: {exc}") from exc
        if not isinstance(frame.index, pd.RangeIndex):
            raise DataIntegrityError("CSV rows contain extra fields or an implicit index")
        self.validate_ohlcv_invariants(frame)
        frame["timestamp"] = _timestamps(frame["timestamp"], "timestamp")
        return frame

    def validate_ohlcv_invariants(self, df: pd.DataFrame) -> None:
        """Reject null/non-finite quotes, crossed markets, and invalid OHLCV.

        Args:
            df: Nonempty chronological quote frame with all CORE_COLUMNS.

        Raises:
            DataIntegrityError: Columns, chronology, dtypes, or row invariants
                fail. Caller data is never sorted, filled, or mutated.
        """
        if not isinstance(df, pd.DataFrame):
            raise DataIntegrityError("df must be a pandas DataFrame")
        if not df.columns.is_unique:
            raise DataIntegrityError("df contains duplicate column names")
        missing = set(CORE_COLUMNS) - set(df.columns)
        if missing:
            raise DataIntegrityError(f"missing required columns: {sorted(missing)}")
        if df.empty:
            raise DataIntegrityError("df must contain at least one quote")
        if df.loc[:, list(CORE_COLUMNS)].isna().any().any():
            raise DataIntegrityError("core quotes contain null/NaN values")
        _timestamps(df["timestamp"], "timestamp")
        for column in CORE_COLUMNS[1:]:
            dtype = df[column].dtype
            if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                raise DataIntegrityError(f"{column} must contain real numeric quotes")
            if not np.isfinite(df[column].to_numpy(dtype=np.float64)).all():
                raise DataIntegrityError(f"{column} contains non-finite quotes")
        rules = {
            "ask >= bid": df["ask"] >= df["bid"],
            "high >= open": df["high"] >= df["open"],
            "high >= close": df["high"] >= df["close"],
            "low <= open": df["low"] <= df["open"],
            "low <= close": df["low"] <= df["close"],
            "high >= low": df["high"] >= df["low"],
            "volume >= 0": df["volume"] >= 0,
        }
        for rule, valid in rules.items():
            if not valid.all():
                positions = np.flatnonzero(~valid.to_numpy(dtype=bool))[:5].tolist()
                raise DataIntegrityError(f"{rule} violated at row positions {positions}")

    def calculate_ema(
        self,
        df: pd.DataFrame,
        period: int,
        tick_size: Real,
        *,
        initial_ema: Optional[Real] = None,
    ) -> pd.Series:
        """Calculate a causal EMA, quantizing the recursive state at every bar.

        Args:
            df: Validated quote frame, normally warm-up followed by active rows.
            period: Strictly positive integer N.
            tick_size: Strictly positive tick grid.
            initial_ema: Explicit EMA state BEFORE the first row. If omitted,
                seed the first row with its quantized close. No SMA or hidden
                historical observations are used.

        Returns:
            Series aligned to df.index, including warm-up rows. Callers MUST
            exclude warm-up values from scored active output.

        Raises:
            DataIntegrityError: Quotes, parameters, or arithmetic are invalid.
        """
        self.validate_ohlcv_invariants(df)
        length = _positive_integer(period, "period")
        quantize(0.0, tick_size)
        closes = df["close"].to_numpy(dtype=np.float64)
        output = np.empty(len(closes), dtype=np.float64)
        try:
            alpha = 2 / (length + 1)
        except OverflowError as exc:
            raise DataIntegrityError("period is too large for floating-point EMA arithmetic") from exc
        if alpha <= 0:
            raise DataIntegrityError("EMA alpha underflowed to zero")
        previous = None if initial_ema is None else quantize(initial_ema, tick_size)
        for index, close in enumerate(closes):
            raw = float(close) if previous is None else (float(close) - previous) * alpha + previous
            previous = quantize(raw, tick_size)
            output[index] = previous
        return pd.Series(output, index=df.index, name=f"ema_{length}")

    def check_gap_policy(self, df: pd.DataFrame, max_allowed_gaps: int) -> GapReport:
        """Count missing scheduled bars, resetting at declared session closures.

        Args:
            df: Observed quotes. Off-hours bars are allowed but cannot stand in
                for a missing expected RTH timestamp.
            max_allowed_gaps: Maximum consecutive missing bars, not elapsed
                minutes or total missing bars. Equality is allowed.

        Returns:
            Observed deltas and all missing RTH timestamps, even below tolerance.
            The caller-supplied schedule defines the entire audited window.

        Raises:
            DataIntegrityError: Quotes, threshold, or explicit calendar missing.
            PollutedDatasetError: A missing RTH run exceeds the threshold.
        """
        self.validate_ohlcv_invariants(df)
        maximum = _positive_integer(max_allowed_gaps, "max_allowed_gaps", allow_zero=True)
        if self._expected_rth is None or self._bar_interval is None:
            raise DataIntegrityError("gap checking requires an explicit RTH schedule and bar_interval")
        observed = _timestamps(df["timestamp"], "timestamp")
        expected = self._expected_rth
        absent = ~expected.isin(observed)
        run = longest = 0
        for index, missing in enumerate(absent):
            if index and expected[index] - expected[index - 1] != self._bar_interval:
                run = 0
            run = run + 1 if missing else 0
            longest = max(longest, run)
            if run > maximum:
                raise PollutedDatasetError(
                    f"{run} consecutive missing RTH bars exceed max_allowed_gaps={maximum}; "
                    f"run ends at {expected[index].isoformat()}"
                )
        deltas = pd.Series(observed, index=df.index, name="timestamp_delta").diff()
        return GapReport(deltas, expected[absent].copy(), longest)

    def get_split(
        self,
        df: pd.DataFrame,
        start_time: TimeInput,
        end_time: TimeInput,
        k_multiplier: int,
        slow_period: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Isolate exact preceding warm-up history from a half-open active split.

        Args:
            df: Chronological quote history containing sufficient prior bars.
            start_time: Inclusive timezone-aware active boundary.
            end_time: Exclusive timezone-aware active boundary.
            k_multiplier: Positive integer multiplier of the slow period.
            slow_period: Positive integer indicator period.

        Returns:
            Independent copies (warmup_df, active_df). Warm-up contains exactly
            k_multiplier * slow_period latest rows strictly before start_time;
            active rows satisfy start_time <= timestamp < end_time.

        Raises:
            DataIntegrityError: Invalid boundaries, insufficient history, or
                an empty active partition. Missing history is never padded.
        """
        self.validate_ohlcv_invariants(df)
        count = _positive_integer(k_multiplier, "k_multiplier") * _positive_integer(slow_period, "slow_period")
        start, end = _timestamp(start_time, "start_time"), _timestamp(end_time, "end_time")
        if start >= end:
            raise DataIntegrityError("start_time must be strictly before end_time")
        timestamps = _timestamps(df["timestamp"], "timestamp")
        prior = np.flatnonzero(timestamps < start)
        active = np.flatnonzero((timestamps >= start) & (timestamps < end))
        if len(prior) < count:
            raise DataIntegrityError(f"insufficient warm-up history: required {count}, available {len(prior)}")
        if not len(active):
            raise DataIntegrityError("active partition contains no observations")
        return df.iloc[prior[-count:]].copy(deep=True), df.iloc[active].copy(deep=True)
