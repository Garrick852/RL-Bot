"""Local CSV feature engineering and fail-closed distribution drift checks."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


CORE_FEATURES = ("ret1", "rsi", "bb_pos", "atr_pct", "macd_hist", "hr_sin", "hr_cos")


@dataclass(frozen=True)
class DriftResult:
    feature: str
    statistic: float
    p_value: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    results: tuple[DriftResult, ...]
    threshold: float

    @property
    def passed(self) -> bool:
        return not any(result.drifted for result in self.results)

    @property
    def drifted_features(self) -> tuple[str, ...]:
        return tuple(result.feature for result in self.results if result.drifted)


def load_symbol_map(path: str | Path) -> dict[str, str]:
    """Load standard-symbol to broker-symbol overrides from a JSON object."""
    with Path(path).open(encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and key and value
        for key, value in mapping.items()
    ):
        raise ValueError("symbol map must be a JSON object of non-empty strings")
    return mapping


def load_ohlcv_csv(path: str | Path) -> list[dict[str, float | datetime]]:
    """Read timestamp/open/high/low/close/volume CSV rows with strict validation."""
    rows: list[dict[str, float | datetime]] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
                bar = {
                    "timestamp": timestamp,
                    "open": float(raw["open"]),
                    "high": float(raw["high"]),
                    "low": float(raw["low"]),
                    "close": float(raw["close"]),
                    "volume": float(raw["volume"]),
                }
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid OHLCV row at line {line_number}") from error
            if not all(math.isfinite(float(bar[key])) for key in required - {"timestamp"}):
                raise ValueError(f"non-finite OHLCV value at line {line_number}")
            if float(bar["high"]) < max(float(bar["open"]), float(bar["close"]), float(bar["low"])):
                raise ValueError(f"invalid high value at line {line_number}")
            if float(bar["low"]) > min(float(bar["open"]), float(bar["close"]), float(bar["high"])):
                raise ValueError(f"invalid low value at line {line_number}")
            rows.append(bar)
    if not rows:
        raise ValueError("CSV contains no OHLCV rows")
    try:
        if any(
            current["timestamp"] <= previous["timestamp"]
            for previous, current in zip(rows, rows[1:])
        ):
            raise ValueError("CSV timestamps must be strictly increasing")
    except TypeError as error:
        raise ValueError("CSV timestamps must use a consistent timezone convention") from error
    return rows


def _rsi(changes: list[float], period: int = 14) -> list[float]:
    output = [50.0] * len(changes)
    for index in range(period, len(changes)):
        window = changes[index - period + 1 : index + 1]
        gains = sum(max(value, 0.0) for value in window) / period
        losses = sum(max(-value, 0.0) for value in window) / period
        output[index] = 100.0 if losses == 0.0 and gains else (
            50.0 if losses == 0.0 else 100.0 - 100.0 / (1.0 + gains / losses)
        )
    return output


def _ema(values: list[float], period: int) -> list[float]:
    alpha = 2.0 / (period + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def engineer_features(
    bars: Iterable[Mapping[str, float | datetime]], period: int = 14
) -> list[dict[str, float | datetime]]:
    """Create the documented core features from chronological OHLCV bars."""
    data = list(bars)
    if period < 2:
        raise ValueError("period must be at least 2")
    if not data:
        raise ValueError("at least one OHLCV bar is required")
    timestamps = [bar["timestamp"] for bar in data]
    if not all(isinstance(value, datetime) for value in timestamps):
        raise ValueError("each bar timestamp must be a datetime")
    try:
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("OHLCV bars must be strictly chronological")
    except TypeError as error:
        raise ValueError("OHLCV timestamps must use a consistent timezone convention") from error
    required = ("open", "high", "low", "close", "volume")
    for index, bar in enumerate(data):
        try:
            values = {name: float(bar[name]) for name in required}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid OHLCV values at row {index}") from error
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite OHLCV values at row {index}")
        if (
            values["close"] == 0
            or values["volume"] < 0
            or values["high"] < max(values["open"], values["close"], values["low"])
            or values["low"] > min(values["open"], values["close"], values["high"])
        ):
            raise ValueError(f"invalid OHLCV range at row {index}")
    closes = [float(bar["close"]) for bar in data]
    highs = [float(bar["high"]) for bar in data]
    lows = [float(bar["low"]) for bar in data]
    true_ranges = [
        highs[index] - lows[index]
        if index == 0
        else max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
        for index in range(len(data))
    ]
    changes = [0.0] + [
        (closes[index] - closes[index - 1]) / closes[index - 1]
        if closes[index - 1] else 0.0
        for index in range(1, len(closes))
    ]
    rsi = _rsi(changes, period)
    fast = _ema(closes, 12)
    slow = _ema(closes, 26)
    macd = [a - b for a, b in zip(fast, slow)]
    signal = _ema(macd, 9)
    features: list[dict[str, float | datetime]] = []
    for index, bar in enumerate(data):
        start = max(0, index - period + 1)
        window = closes[start : index + 1]
        average = statistics.fmean(window)
        deviation = statistics.pstdev(window) if len(window) > 1 else 0.0
        atr = statistics.fmean(true_ranges[start : index + 1])
        hour = timestamps[index].hour + timestamps[index].minute / 60.0
        features.append(
            {
                "timestamp": timestamps[index],
                "ret1": changes[index],
                "rsi": rsi[index],
                "bb_pos": 0.5 if deviation == 0 else (closes[index] - (average - 2 * deviation)) / (4 * deviation),
                "atr_pct": atr / closes[index] if closes[index] else 0.0,
                "macd_hist": macd[index] - signal[index],
                "hr_sin": math.sin(2 * math.pi * hour / 24),
                "hr_cos": math.cos(2 * math.pi * hour / 24),
            }
        )
    return features


def ks_two_sample(first: Iterable[float], second: Iterable[float]) -> tuple[float, float]:
    """Return two-sample KS statistic and exact (small samples) or asymptotic p-value."""
    left = sorted(float(value) for value in first)
    right = sorted(float(value) for value in second)
    if not left or not right:
        raise ValueError("both samples must contain at least one value")
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError("samples must contain only finite values")
    points = sorted(set(left + right))
    i = j = 0
    statistic = 0.0
    for point in points:
        while i < len(left) and left[i] <= point:
            i += 1
        while j < len(right) and right[j] <= point:
            j += 1
        statistic = max(statistic, abs(i / len(left) - j / len(right)))
    if statistic == 0.0:
        return statistic, 1.0
    if (
        len(left) <= 200
        and len(right) <= 200
        and len(set(left + right)) == len(left) + len(right)
    ):
        safe_paths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        safe_paths[0][0] = 1
        for left_count in range(len(left) + 1):
            for right_count in range(len(right) + 1):
                if left_count == right_count == 0:
                    continue
                difference = abs(left_count / len(left) - right_count / len(right))
                if difference >= statistic:
                    continue
                safe_paths[left_count][right_count] = (
                    (safe_paths[left_count - 1][right_count] if left_count else 0)
                    + (safe_paths[left_count][right_count - 1] if right_count else 0)
                )
        total_paths = math.comb(len(left) + len(right), len(left))
        return statistic, 1.0 - safe_paths[-1][-1] / total_paths
    effective_n = len(left) * len(right) / (len(left) + len(right))
    lam = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    p_value = min(1.0, max(0.0, 2.0 * sum(
        (-1.0) ** (term - 1) * math.exp(-2.0 * term * term * lam * lam)
        for term in range(1, 101)
    )))
    return statistic, p_value


def check_feature_drift(
    reference: Iterable[Mapping[str, float]],
    recent: Iterable[Mapping[str, float]],
    features: Iterable[str] = CORE_FEATURES,
    threshold: float = 0.01,
) -> DriftReport:
    """Compare distributions, marking p-values below threshold as a failed gate."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    baseline, current = list(reference), list(recent)
    results = []
    for feature in features:
        try:
            left = [float(row[feature]) for row in baseline]
            right = [float(row[feature]) for row in current]
        except KeyError as error:
            raise ValueError(f"missing drift feature: {error.args[0]}") from error
        statistic, p_value = ks_two_sample(left, right)
        results.append(DriftResult(feature, statistic, p_value, p_value < threshold))
    if not results:
        raise ValueError("at least one feature is required")
    return DriftReport(tuple(results), threshold)


def require_no_feature_drift(report: DriftReport) -> None:
    """Raise rather than allowing optimization to continue after a drift alert."""
    if not report.passed:
        raise RuntimeError(
            "feature drift gate failed for: " + ", ".join(report.drifted_features)
        )


def fit_scaler_contract(
    rows: Iterable[Mapping[str, float]],
    features: Iterable[str] = CORE_FEATURES,
) -> dict[str, object]:
    """Fit a deterministic standard-scaling contract for downstream policy inputs."""
    samples = list(rows)
    names = tuple(features)
    if not samples or not names:
        raise ValueError("scaler fitting requires rows and feature names")
    statistics_by_feature: dict[str, dict[str, float]] = {}
    for feature in names:
        try:
            values = [float(row[feature]) for row in samples]
        except KeyError as error:
            raise ValueError(f"missing scaler feature: {error.args[0]}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite scaler values for {feature}")
        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values)
        statistics_by_feature[feature] = {
            "mean": mean,
            "scale": deviation if deviation else 1.0,
        }
    return {
        "version": 1,
        "method": "standard",
        "sample_count": len(samples),
        "features": statistics_by_feature,
    }


def save_scaler_contract(contract: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(contract, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def fetch_mt5_bars(
    symbol: str,
    *,
    timeframe: str = "TIMEFRAME_M15",
    start_pos: int = 0,
    count: int = 10_000,
) -> list[dict[str, float | datetime]]:
    """Fetch a bounded bar window from an already-installed, initialized MT5 terminal."""
    if not symbol or start_pos < 0 or count < 1:
        raise ValueError("symbol, non-negative start_pos, and positive count are required")
    try:
        import MetaTrader5 as mt5
    except ImportError as error:
        raise RuntimeError("install the MT5 extra with `pip install .[mt5]`") from error
    mt5_timeframe = getattr(mt5, timeframe, None)
    if mt5_timeframe is None:
        raise ValueError(f"unknown MT5 timeframe: {timeframe}")
    rows = mt5.copy_rates_from_pos(symbol, mt5_timeframe, start_pos, count)
    if rows is None:
        raise RuntimeError(f"MT5 returned no rates for {symbol}: {mt5.last_error()}")
    result = []
    for row in rows:
        result.append(
            {
                "timestamp": datetime.fromtimestamp(int(row["time"]), tz=timezone.utc),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["tick_volume"]),
            }
        )
    if not result:
        raise RuntimeError(f"MT5 returned an empty rate window for {symbol}")
    return result


def fetch_ibkr_bars(
    symbol: str,
    *,
    host: str,
    port: int,
    client_id: int,
    duration: str = "30 D",
    bar_size: str = "15 mins",
    exchange: str = "SMART",
    currency: str = "USD",
    sec_type: str = "STK",
) -> list[dict[str, float | datetime]]:
    """Request historical bars through an explicitly addressed IBKR TWS/Gateway."""
    if not symbol or not host or port < 1 or client_id < 0 or not sec_type:
        raise ValueError("symbol, host, sec_type, positive port, and non-negative client_id are required")
    try:
        from ib_insync import IB, Contract, util
    except ImportError as error:
        raise RuntimeError("install the IBKR extra with `pip install .[ibkr]`") from error
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        contract = Contract(
            symbol=symbol, secType=sec_type, exchange=exchange, currency=currency
        )
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        )
        if not bars:
            raise RuntimeError(f"IBKR returned no historical bars for {symbol}")
        return [
            {
                "timestamp": util.parseIBDatetime(bar.date),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ]
    finally:
        ib.disconnect()


def write_feature_csv(
    features: Iterable[Mapping[str, float | datetime]], path: str | Path
) -> Path:
    """Persist engineered rows using a stable feature-column order."""
    rows = list(features)
    if not rows:
        raise ValueError("cannot write an empty feature set")
    columns = ["timestamp", *CORE_FEATURES]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if not set(columns).issubset(row):
                raise ValueError("feature row does not satisfy the core feature contract")
            if not isinstance(row["timestamp"], datetime) or any(
                not math.isfinite(float(row[key])) for key in CORE_FEATURES
            ):
                raise ValueError("feature row must contain a datetime and finite numeric features")
            writer.writerow(
                {
                    key: row[key].isoformat() if isinstance(row[key], datetime) else row[key]
                    for key in columns
                }
            )
    return destination
