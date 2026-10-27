"""Fail-closed out-of-sample candidate validation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class OOSMetrics:
    profit_factor: float
    sharpe: float
    max_drawdown_pct: float
    trade_count: int
    seed_expectancies: tuple[float, ...] = ()
    seed_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: OOSMetrics


def validate(
    metrics: OOSMetrics,
    *,
    min_profit_factor: float = 1.25,
    min_sharpe: float = 1.0,
    max_drawdown_pct: float = 10.0,
    min_trades: int = 30,
    require_seed_stability: bool = False,
    min_seeds: int = 3,
) -> ValidationResult:
    """Apply all OOS hard gates; non-finite or missing metric values fail closed."""
    if not isinstance(metrics, OOSMetrics):
        raise TypeError("metrics must be an OOSMetrics instance")
    for name, value in (
        ("min_profit_factor", min_profit_factor),
        ("min_sharpe", min_sharpe),
        ("max_drawdown_pct", max_drawdown_pct),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
    if (
        not math.isfinite(min_profit_factor)
        or min_profit_factor <= 0
        or not math.isfinite(min_sharpe)
        or not math.isfinite(max_drawdown_pct)
        or max_drawdown_pct < 0
        or isinstance(min_trades, bool)
        or not isinstance(min_trades, int)
        or min_trades < 1
        or isinstance(min_seeds, bool)
        or not isinstance(min_seeds, int)
        or min_seeds < 1
    ):
        raise ValueError("invalid OOS validation thresholds")
    reasons: list[str] = []
    numeric = (
        ("profit_factor", metrics.profit_factor),
        ("sharpe", metrics.sharpe),
        ("max_drawdown_pct", metrics.max_drawdown_pct),
    )
    for name, value in numeric:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f"{name} is not finite")
    if (
        isinstance(metrics.profit_factor, (int, float))
        and not isinstance(metrics.profit_factor, bool)
        and math.isfinite(metrics.profit_factor)
        and metrics.profit_factor < min_profit_factor
    ):
        reasons.append(f"profit_factor below {min_profit_factor}")
    if (
        isinstance(metrics.sharpe, (int, float))
        and not isinstance(metrics.sharpe, bool)
        and math.isfinite(metrics.sharpe)
        and metrics.sharpe < min_sharpe
    ):
        reasons.append(f"sharpe below {min_sharpe}")
    if (
        isinstance(metrics.max_drawdown_pct, (int, float))
        and not isinstance(metrics.max_drawdown_pct, bool)
        and math.isfinite(metrics.max_drawdown_pct)
        and metrics.max_drawdown_pct > max_drawdown_pct
    ):
        reasons.append(f"max_drawdown_pct above {max_drawdown_pct}")
    if (
        isinstance(metrics.max_drawdown_pct, (int, float))
        and not isinstance(metrics.max_drawdown_pct, bool)
        and math.isfinite(metrics.max_drawdown_pct)
        and metrics.max_drawdown_pct < 0
    ):
        reasons.append("max_drawdown_pct is negative")
    if isinstance(metrics.trade_count, bool) or not isinstance(metrics.trade_count, int):
        reasons.append("trade_count is not an integer")
    elif metrics.trade_count < min_trades:
        reasons.append(f"trade_count below {min_trades}")
    if require_seed_stability:
        if not isinstance(metrics.seed_ids, tuple) or not isinstance(
            metrics.seed_expectancies, tuple
        ):
            reasons.append("seed IDs and expectancies must be tuples")
        else:
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in metrics.seed_ids):
                reasons.append("seed IDs must be integers")
            elif len(metrics.seed_ids) < min_seeds or len(set(metrics.seed_ids)) < min_seeds:
                reasons.append(f"fewer than {min_seeds} distinct seed results")
            if len(metrics.seed_ids) != len(metrics.seed_expectancies):
                reasons.append("seed IDs and expectancy results do not match")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in metrics.seed_expectancies
        ):
            reasons.append("one or more seed expectancies are not positive")
    return ValidationResult(not reasons, tuple(reasons), metrics)


def split_oos(
    rows: Sequence[T],
    *,
    fraction: float = 0.2,
    min_train_rows: int = 1,
) -> tuple[Sequence[T], Sequence[T]]:
    """Reserve the latest fraction of chronological rows for an untouched OOS set."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("fraction must be numeric")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if isinstance(min_train_rows, bool) or not isinstance(min_train_rows, int) or min_train_rows < 1:
        raise ValueError("min_train_rows must be a positive integer")
    if not isinstance(rows, Sequence) or len(rows) <= min_train_rows:
        raise ValueError("rows must contain training data and at least one OOS row")
    split_at = int(len(rows) * (1.0 - fraction))
    split_at = max(min_train_rows, min(split_at, len(rows) - 1))
    return rows[:split_at], rows[split_at:]


def metrics_from_trades(
    trade_pnls: Iterable[float],
    *,
    starting_equity: float,
    seed_expectancies: Iterable[float] = (),
    seed_ids: Iterable[int] = (),
) -> OOSMetrics:
    """Derive PF, per-trade Sharpe, and peak-to-trough drawdown from OOS P&L."""
    pnls = [float(value) for value in trade_pnls]
    if not pnls:
        raise ValueError("at least one OOS trade is required")
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be finite and positive")
    if not all(math.isfinite(value) for value in pnls):
        raise ValueError("trade P&L values must be finite")
    gains = sum(value for value in pnls if value > 0)
    losses = -sum(value for value in pnls if value < 0)
    profit_factor = gains / losses if losses else (math.inf if gains else 0.0)
    deviation = statistics.stdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = (
        statistics.fmean(pnls) / deviation * math.sqrt(len(pnls))
        if deviation else 0.0
    )
    equity = peak = starting_equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)
    return OOSMetrics(
        profit_factor,
        sharpe,
        max_drawdown,
        len(pnls),
        tuple(float(value) for value in seed_expectancies),
        tuple(seed_ids),
    )
