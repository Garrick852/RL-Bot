"""Current-quote single-position accounting with mandatory terminal liquidation."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from data.collector import DataCollector


class DoubleEntryReconciliationError(RuntimeError):
    """Cash and independently aggregated closed-trade P&L do not reconcile."""


STATE_FEATURES = ("position_side", "unrealized_pnl_fraction")


class TradingEnv(gym.Env):
    """Single-asset, fixed-size, cash-settled offline trading environment.

    Action indices 0/1/2 target short/flat/long. Cash is a realized-P&L
    settlement account, not a spot-asset purchase account. No leverage, margin,
    swap, or currency conversion is inferred.

    Args:
        active_df: Active-only quotes satisfying DataCollector invariants.
        feature_columns: Explicit finite, pre-scaled observation columns.
        initial_cash: Positive account starting balance.
        quantity: Positive absolute position size in instrument units.
        point_value: Positive cash value of one price point per unit.
        slippage: Non-negative adverse price units per unit per side.
        commission: Non-negative price units per unit per side, embedded once
            in effective fills, never charged again as a cash fee.
        lambda_term: Non-negative extra adverse price units on forced exits.
        max_episode_steps: Optional positive internal truncation limit.
        state_features: Optional account-state values appended, in the given
            order, after the feature columns: ``position_side`` (-1/0/+1) and
            ``unrealized_pnl_fraction`` (unrealized P&L / equity, both marked
            at the last executed row). Empty by default (market features only).

    Raises:
        ValueError: Parameters or observations violate the declared contract.

    Notes:
        Features for row t must be available before execution at row t quotes;
        callers must lag close-derived indicators if those are not yet known.
        Rewards are changes in conservative liquidation equity in cash units.
        An external TimeLimit cannot liquidate this environment automatically:
        use max_episode_steps here or call truncate() before stopping.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        active_df: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        initial_cash: float,
        quantity: float,
        point_value: float,
        slippage: float,
        commission: float,
        lambda_term: float,
        max_episode_steps: Optional[int] = None,
        state_features: Sequence[str] = (),
    ) -> None:
        super().__init__()
        DataCollector().validate_ohlcv_invariants(active_df)
        self._data = active_df.copy(deep=True)
        columns = list(feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("feature_columns must be nonempty and unique")
        if any(not isinstance(name, str) or name not in active_df for name in columns):
            raise ValueError("feature_columns must identify existing columns")
        for name in columns:
            if (
                not pd.api.types.is_numeric_dtype(active_df[name])
                or pd.api.types.is_bool_dtype(active_df[name])
                or pd.api.types.is_complex_dtype(active_df[name])
                or active_df[name].isna().any()
            ):
                raise ValueError(f"feature {name} must contain finite real numbers")
        raw_features = active_df[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(raw_features).all() or (
            np.abs(raw_features) > np.finfo(np.float32).max
        ).any():
            raise ValueError("features must be finite and representable as float32")
        self._features = raw_features.astype(np.float32)
        self.initial_cash = self._number(initial_cash, "initial_cash", positive=True)
        self.quantity = self._number(quantity, "quantity", positive=True)
        self.point_value = self._number(point_value, "point_value", positive=True)
        self.slippage = self._number(slippage, "slippage")
        self.commission = self._number(commission, "commission")
        self.lambda_term = self._number(lambda_term, "lambda_term")
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, (bool, np.bool_))
            or not isinstance(max_episode_steps, Integral)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer or None")
        self.max_episode_steps = max_episode_steps
        state = list(state_features)
        if (
            any(not isinstance(name, str) or name not in STATE_FEATURES for name in state)
            or len(set(state)) != len(state)
            or set(state) & set(columns)
        ):
            raise ValueError(f"state_features must be unique names from {STATE_FEATURES}")
        self.state_features = tuple(state)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(columns) + len(state),), dtype=np.float32
        )
        self._started = False
        self._finished = False
        self._index = 0
        self._last_executed_index: Optional[int] = None
        self._steps = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.entry_price: Optional[float] = None
        self.unrealized_pnl = 0.0
        self.equity = self.initial_cash
        self.realized_trades_pnl: list[float] = []
        self.terminal_liquidation_pnl = 0.0
        self.trades: list[dict[str, Any]] = []
        self.fills: list[dict[str, Any]] = []

    @staticmethod
    def _number(value: Real, name: str, *, positive: bool = False) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        result = float(value)
        if not math.isfinite(result) or (result <= 0 if positive else result < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
        return result

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh independent episode; reset is not a completed audit."""
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported; configure explicit constructor inputs")
        self._index = 0
        self._last_executed_index = None
        self._steps = 0
        self._started, self._finished = True, False
        self.cash = self.equity = self.initial_cash
        self.position = self.unrealized_pnl = self.terminal_liquidation_pnl = 0.0
        self.entry_price = None
        self.realized_trades_pnl = []
        self.trades = []
        self.fills = []
        return self._observation(0), self._info(None)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute only against current quotes, then return the next observation.

        Final or internally truncated rows ignore target entries/reversals and
        force-close existing exposure. Returned info names the executed row;
        equity is not marked at the next observation's quotes.
        """
        self._require_active()
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"invalid discrete action: {action!r}")
        index = self._index
        previous_equity = self.equity
        self._steps += 1
        terminated = index == len(self._data) - 1
        truncated = (
            not terminated
            and self.max_episode_steps is not None
            and self._steps >= self.max_episode_steps
        )
        if terminated or truncated:
            self._liquidate(index, terminal=True)
        else:
            target = (int(action) - 1) * self.quantity
            if target != self.position:
                self._liquidate(index, terminal=False)
                if target:
                    self._open(index, target)
            self._mark(index)
        self._last_executed_index = index
        if terminated or truncated:
            self._finished = True
            self._reconcile()
        reward = self._finite(self.equity - previous_equity, "reward")
        info = self._info(index)
        info["terminal_action_ignored"] = terminated or truncated
        if not self._finished:
            self._index += 1
        return self._observation(self._index), reward, terminated, truncated, info

    def truncate(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Force an early exit at the last consumed row, without peeking ahead.

        Before any step, uses row zero. Call this rather than dropping an open
        episode or relying on a wrapper to set truncated after step returns.
        """
        self._require_active()
        index = self._last_executed_index
        if index is None:
            index = 0
        previous_equity = self.equity
        self._liquidate(index, terminal=True)
        self._finished = True
        self._reconcile()
        return (
            self._observation(index),
            self._finite(self.equity - previous_equity, "reward"),
            False,
            True,
            self._info(index),
        )

    def _require_active(self) -> None:
        if not self._started or self._finished:
            raise RuntimeError("reset is required before stepping or after episode completion")

    def _observation(self, index: int) -> np.ndarray:
        features = self._features[index].copy()
        if not self.state_features:
            return features
        values = {
            "position_side": math.copysign(1.0, self.position) if self.position else 0.0,
            "unrealized_pnl_fraction": self.unrealized_pnl / self.equity if self.equity > 0 else 0.0,
        }
        state = np.array([values[name] for name in self.state_features], dtype=np.float32)
        return np.concatenate([features, state])

    @staticmethod
    def _finite(value: float, name: str) -> float:
        if not math.isfinite(value):
            raise DoubleEntryReconciliationError(f"non-finite accounting value: {name}")
        return value

    def _exit_price(self, index: int, *, terminal: bool) -> float:
        row = self._data.iloc[index]
        bid, ask = float(row["bid"]), float(row["ask"])
        sign = math.copysign(1.0, self.position)
        penalty = self.slippage + self.commission
        if terminal:
            # Algebraically mid - sign * (half_spread + costs + lambda).
            # Selecting bid/ask directly avoids cancellation in wide prices.
            penalty += self.lambda_term
        return self._finite((bid if sign > 0 else ask) - sign * penalty, "exit_price")

    def _open(self, index: int, target: float) -> None:
        row = self._data.iloc[index]
        sign = math.copysign(1.0, target)
        quote = float(row["ask"] if sign > 0 else row["bid"])
        self.entry_price = self._finite(
            quote + sign * (self.slippage + self.commission), "entry_price"
        )
        self.position = target
        self.fills.append({
            "row": index, "timestamp": row["timestamp"], "quantity": target,
            "price": self.entry_price, "kind": "ENTRY",
        })

    def _mark(self, index: int) -> None:
        if self.position:
            if self.entry_price is None:
                raise DoubleEntryReconciliationError("open position has no entry price")
            self.unrealized_pnl = self._finite(
                (self._exit_price(index, terminal=False) - self.entry_price)
                * self.position * self.point_value,
                "unrealized_pnl",
            )
        else:
            self.unrealized_pnl = 0.0
        self.equity = self._finite(self.cash + self.unrealized_pnl, "equity")

    def _liquidate(self, index: int, *, terminal: bool) -> None:
        if self.position:
            if self.entry_price is None:
                raise DoubleEntryReconciliationError("open position has no entry price")
            exit_price = self._exit_price(index, terminal=terminal)
            pnl = self._finite(
                (exit_price - self.entry_price) * self.position * self.point_value, "trade_pnl"
            )
            self.cash = self._finite(self.cash + pnl, "cash")
            trade = {
                "exit_row": index, "timestamp": self._data.iloc[index]["timestamp"],
                "quantity": self.position, "entry_price": self.entry_price,
                "exit_price": exit_price, "pnl": pnl, "terminal": terminal,
            }
            self.trades.append(trade)
            self.fills.append({
                "row": index, "timestamp": trade["timestamp"], "quantity": -self.position,
                "price": exit_price, "kind": "TERMINAL_EXIT" if terminal else "EXIT",
            })
            if terminal:
                self.terminal_liquidation_pnl = pnl
            else:
                self.realized_trades_pnl.append(pnl)
            self.position = 0.0
            self.entry_price = None
        self.unrealized_pnl = 0.0
        self.equity = self.cash

    def _reconcile(self) -> None:
        realized = math.fsum(self.realized_trades_pnl)
        total = realized + self.terminal_liquidation_pnl
        cash_change = self.cash - self.initial_cash
        if (
            not all(math.isfinite(value) for value in (total, cash_change, self.equity))
            or self.position != 0.0
            or self.unrealized_pnl != 0.0
            or not math.isclose(cash_change, total, rel_tol=0.0, abs_tol=1e-8)
            or not math.isclose(self.equity, self.cash, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise DoubleEntryReconciliationError(
                f"cash delta {cash_change!r} != realized {realized!r} + "
                f"terminal {self.terminal_liquidation_pnl!r}; "
                f"position={self.position!r}, unrealized={self.unrealized_pnl!r}"
            )

    def _info(self, executed_row: Optional[int]) -> dict[str, Any]:
        return {
            "executed_row": executed_row,
            "cash": self.cash,
            "equity": self.equity,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "sum_of_realized_trades_pnl": math.fsum(self.realized_trades_pnl),
            "terminal_liquidation_pnl": self.terminal_liquidation_pnl,
        }
