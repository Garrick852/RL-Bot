"""Adversarial accounting, calendar, and provenance boundaries.

The Martingale inventory is an explicitly injected accounting fixture.
TradingEnv's production actions remain fixed-size short/flat/long targets;
these tests do not claim native grid-entry or per-tier execution support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.collector import CORE_COLUMNS, DataCollector, PollutedDatasetError
from engine_rl.envs.trading_env import DoubleEntryReconciliationError, TradingEnv
from governance.validator import ConfigValidator, ValidationState


def _quotes(times: pd.DatetimeIndex, bids: np.ndarray) -> pd.DataFrame:
    """Construct coherent synthetic quotes without missing core prices."""
    return pd.DataFrame({
        "timestamp": times,
        "open": bids,
        "high": bids + 2,
        "low": bids - 1,
        "close": bids,
        "volume": np.full(len(times), 100),
        "bid": bids,
        "ask": bids + 2,
        "signal": np.zeros(len(times)),
    })


@pytest.fixture
def martingale_inventory() -> Callable[..., tuple[TradingEnv, list[tuple[float, float]]]]:
    """Provide fresh eight-tier aggregate inventory for each tamper experiment."""
    def build(*, max_episode_steps: int | None = None):
        times = pd.date_range("2024-01-02T14:30:00Z", periods=9, freq="15min")
        bids = np.array([108, 107, 106, 105, 104, 103, 102, 101, 101], dtype=float)
        env = TradingEnv(
            _quotes(times, bids),
            feature_columns=["signal"],
            initial_cash=10000.0,
            quantity=1.0,
            point_value=2.0,
            slippage=0.25,
            commission=0.125,
            lambda_term=0.5,
            max_episode_steps=max_episode_steps,
        )
        env.reset(seed=7)
        tiers: list[tuple[float, float]] = []
        # Inject accounting-only entries; do not pretend repeated Long actions
        # can pyramid positions. Every tier includes its entry-side costs.
        for index in range(8):
            quantity = float(2**index)
            entry = float(bids[index] + 2 + env.slippage + env.commission)
            tiers.append((quantity, entry))
            env.fills.append({
                "row": index, "timestamp": times[index], "quantity": quantity,
                "price": entry, "kind": "ENTRY",
            })
        env.position = math.fsum(quantity for quantity, _ in tiers)
        env.entry_price = math.fsum(quantity * price for quantity, price in tiers) / env.position
        env._index = 8
        env._last_executed_index = 7
        env._steps = 8
        env._mark(7)
        return env, tiers

    return build


@pytest.mark.parametrize("exit_mode", ["data_end", "internal_limit", "explicit_truncate"])
def test_martingale_double_entry_reconciliation(
    martingale_inventory: Callable[..., tuple[TradingEnv, list[tuple[float, float]]]],
    exit_mode: str,
) -> None:
    """Liquidate all 255 units; reject a 1e-7 cash mutation at every exit path."""
    def build():
        env, tiers = martingale_inventory(
            max_episode_steps=9 if exit_mode == "internal_limit" else None
        )
        if exit_mode == "internal_limit":
            # Row 7 is nonfinal: the ninth step triggers the internal limit.
            env._index = 7
        return env, tiers

    def finish(env):
        return env.truncate() if exit_mode == "explicit_truncate" else env.step(2)

    env, tiers = build()
    assert len(tiers) == len(env.fills) == 8
    assert env.position == 255.0
    assert env.trades == []
    assert env.cash == env.initial_cash
    previous_equity = env.equity
    quantity = env.position
    exit_row = 8 if exit_mode == "data_end" else 7
    row = env._data.iloc[exit_row]
    mid = (float(row["bid"]) + float(row["ask"])) / 2
    half_spread = (float(row["ask"]) - float(row["bid"])) / 2
    expected_exit = mid - math.copysign(1.0, quantity) * (
        half_spread + env.slippage + env.commission + env.lambda_term
    )
    expected_pnl = math.fsum(
        (expected_exit - entry) * tier_quantity * env.point_value
        for tier_quantity, entry in tiers
    )
    ordinary_exit = mid - half_spread - env.slippage - env.commission
    lambda_cash_penalty = quantity * env.point_value * env.lambda_term

    _, reward, terminated, truncated, info = finish(env)

    assert terminated is (exit_mode == "data_end")
    assert truncated is (exit_mode != "data_end")
    assert env.trades[-1]["exit_price"] == expected_exit
    assert env.trades[-1]["quantity"] == quantity
    assert env.trades[-1]["exit_row"] == exit_row
    assert env.fills[-1]["quantity"] == -quantity
    assert env.fills[-1]["kind"] == "TERMINAL_EXIT"
    assert len(env.trades) == 1
    assert env.trades[-1]["terminal"] is True
    assert info["terminal_liquidation_pnl"] == pytest.approx(expected_pnl, rel=0, abs=1e-10)
    assert (ordinary_exit - expected_exit) * quantity * env.point_value == lambda_cash_penalty
    assert reward == pytest.approx(
        env.initial_cash + expected_pnl - previous_equity, rel=0, abs=1e-10
    )
    assert reward == pytest.approx(-lambda_cash_penalty, rel=0, abs=1e-10)
    assert info["unrealized_pnl"] == 0.0
    assert info["position"] == 0.0
    assert info["equity"] == info["cash"]
    assert math.isclose(
        info["cash"] - env.initial_cash,
        info["sum_of_realized_trades_pnl"] + info["terminal_liquidation_pnl"],
        rel_tol=0.0,
        abs_tol=1e-8,
    )

    within_tolerance, _ = build()
    within_tolerance.cash += 5e-9
    finish(within_tolerance)  # Small residual is deliberately within the contract.

    for tamper in (1e-7, -1e-7):
        attacked, _ = build()
        attacked.cash += tamper
        with pytest.raises(DoubleEntryReconciliationError, match="cash delta"):
            finish(attacked)
        with pytest.raises(RuntimeError, match="reset is required"):
            attacked.step(2)


@pytest.fixture
def ny_rth_quotes() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """An explicit winter NY session, 09:30–16:00 local, in UTC."""
    expected = pd.date_range(
        "2024-01-02T14:30:00Z", "2024-01-02T21:00:00Z",
        freq="15min", inclusive="left",
    )
    return _quotes(expected, np.full(len(expected), 100.0)), expected


def test_polluted_dataset_rth_gap(
    ny_rth_quotes: tuple[pd.DataFrame, pd.DatetimeIndex],
) -> None:
    """Five missing interior bars exceed four; five is permitted and disclosed."""
    frame, schedule = ny_rth_quotes
    collector = DataCollector(
        expected_rth_timestamps=schedule, bar_interval=pd.Timedelta("15min")
    )
    missing = schedule[6:11]
    polluted = frame.drop(index=range(6, 11)).copy()
    original = polluted.copy(deep=True)
    collector.validate_ohlcv_invariants(polluted)
    observed = pd.DatetimeIndex(polluted["timestamp"])
    assert observed[6] - observed[5] == pd.Timedelta("90min")
    with pytest.raises(
        PollutedDatasetError, match="5 consecutive missing RTH bars exceed max_allowed_gaps=4"
    ):
        collector.check_gap_policy(polluted, max_allowed_gaps=4)
    pd.testing.assert_frame_equal(polluted, original)
    report = collector.check_gap_policy(polluted, max_allowed_gaps=5)
    assert report.max_consecutive_missing == 5
    pd.testing.assert_index_equal(report.missing_rth_timestamps, missing)


@pytest.fixture
def frozen_active_config(
    tmp_path: Path, ny_rth_quotes: tuple[pd.DataFrame, pd.DatetimeIndex]
) -> tuple[Path, Path, str]:
    """Create an explicit validator-profile config, not a deployment-approved one."""
    frame, _ = ny_rth_quotes
    dataset = tmp_path / "dataset.csv"
    frame.loc[:, list(CORE_COLUMNS)].to_csv(dataset, index=False)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    fee = {"value": 0.0, "unit": "price", "application_rule": "per_unit_per_side"}
    config = {
        "frozen": True,
        "dataset": {
            "path": dataset.name, "sha256": digest, "format": "csv",
            "source_description": "synthetic adversarial fixture", "symbol": "TEST",
            "timeframe": "15min", "timezone": "UTC",
            "timestamp_meaning": "execution_quote_time", "price_basis": "bid_ask",
            "column_mapping": {name: name for name in CORE_COLUMNS},
            "missing_bar_policy": "explicit_rth_schedule",
        },
        "strategy": {
            "name": "fixed_ema_crossover",
            "ema": {"fast_period": 12, "slow_period": 26, "initialization_rule": "first_close_quantized"},
            "signal_rule": "cross", "position_between_crossovers": "retain_target",
            "shorting_allowed": True,
        },
        "costs": {
            "spread": {**fee, "application_rule": "embedded_bid_ask"},
            "slippage": dict(fee), "commission": dict(fee),
            "swap": {
                "long_value": 0.0, "short_value": 0.0, "unit": "price",
                "rollover_timezone": "UTC", "rollover_time": "00:00",
                "day_count_and_multiplier_rule": "explicit_zero",
            },
        },
        "execution": {
            "decision_timing": "before_current_quote", "fill_timing": "current_row",
            "fill_price_rule": "bid_ask", "final_bar_entry_policy": "forbid",
            "terminal_liquidation_price_rule": "bid_ask_adverse_lambda",
            "terminal_liquidation_cost_rule": "per_unit_price_costs_once",
        },
        "splits": {
            "boundary_convention": "start_inclusive_end_exclusive",
            "train": {"start": "2024-01-02T14:30:00Z", "end": "2024-01-02T16:30:00Z"},
            "validation": {"start": "2024-01-02T16:30:00Z", "end": "2024-01-02T18:30:00Z"},
            "test": {"start": "2024-01-02T18:30:00Z", "end": "2024-01-02T21:00:00Z"},
        },
    }
    active_config = tmp_path / "active_config.json"
    active_config.write_text(json.dumps(config), encoding="utf-8")
    return active_config, dataset, digest


def test_l3_cryptographic_rejection(
    frozen_active_config: tuple[Path, Path, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A valid control fails after a parseable quote change with a stale hash."""
    config_path, dataset, expected_hash = frozen_active_config
    config_bytes = config_path.read_bytes()
    validator = ConfigValidator(
        json.loads(config_bytes), base_directory=config_path.parent
    )
    assert validator.validate() is ValidationState.READY_FOR_OFFLINE_EXECUTION
    assert validator.evidence["dataset_sha256"] == "PASS"

    altered = pd.read_csv(dataset)
    altered.loc[0, "volume"] += 1
    altered.to_csv(dataset, index=False)
    DataCollector().load_csv(dataset)  # Quote semantics still pass; identity must not.
    actual_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert actual_hash != expected_hash
    exact_difference = f"dataset.sha256 mismatch: expected '{expected_hash}', actual {actual_hash}"

    with caplog.at_level(logging.ERROR, logger="governance.validator"):
        state = validator.validate()

    assert state is ValidationState.INVALID
    assert validator.evidence["dataset_readability"] == "PASS"
    assert validator.evidence["dataset_sha256"] == "FAIL"
    assert validator.actual_dataset_sha256 == actual_hash
    assert validator.errors == [f"ConfigValidationError: {exact_difference}"]
    assert exact_difference in caplog.text
    assert config_path.read_bytes() == config_bytes
