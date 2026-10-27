"""Gymnasium environment for discrete or continuous, single-asset simulations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Toy bar-by-bar environment; actions alter exposure, not broker positions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        returns: Sequence[float],
        *,
        continuous_actions: bool = False,
        transaction_cost: float = 0.0001,
    ) -> None:
        super().__init__()
        observations = np.asarray(features, dtype=np.float32)
        pnl_returns = np.asarray(returns, dtype=np.float32)
        if observations.ndim != 2 or len(observations) < 2:
            raise ValueError("features must be a 2D array containing at least two rows")
        if pnl_returns.ndim != 1 or len(pnl_returns) != len(observations):
            raise ValueError("returns must be a 1D array matching the feature row count")
        if not np.isfinite(observations).all() or not np.isfinite(pnl_returns).all():
            raise ValueError("features and returns must be finite")
        if transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative")
        self.features = observations
        self.returns = pnl_returns
        self.transaction_cost = float(transaction_cost)
        self.continuous_actions = continuous_actions
        self.action_space = (
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            if continuous_actions
            else spaces.Discrete(3)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observations.shape[1],), dtype=np.float32
        )
        self.index = 0
        self.position = 0.0
        self.equity = 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        self.position = 0.0
        self.equity = 1.0
        return self.features[self.index].copy(), {}

    def step(self, action):
        if self.continuous_actions:
            candidate = np.asarray(action, dtype=np.float32)
            if not self.action_space.contains(candidate):
                raise ValueError(f"invalid continuous action: {action!r}")
            exposure = float(candidate[0])
            exposure = float(np.clip(exposure, -1.0, 1.0))
        else:
            if not self.action_space.contains(action):
                raise ValueError(f"invalid action: {action!r}")
            exposure = float(action) - 1.0
        turnover = abs(exposure - self.position)
        reward = exposure * float(self.returns[self.index + 1]) - self.transaction_cost * turnover
        self.equity *= max(0.0, 1.0 + reward)
        self.position = exposure
        self.index += 1
        terminated = self.index >= len(self.features) - 1
        observation = self.features[self.index].copy()
        info = {"equity": self.equity, "position": self.position, "turnover": turnover}
        return observation, reward, terminated, False, info
