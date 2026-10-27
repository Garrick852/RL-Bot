"""Single source of truth for the 12-observation / 3-action linear policy contract.

The MQL5 ``EA_Template.mq5`` mirrors these constants exactly. A policy trained
with ``SB3Trainer(policy_architecture="linear_actor")`` on a ``TradingEnv`` built
from :func:`build_market_observations` columns plus
``state_features=STATE_OBSERVATION_FEATURES`` has a deterministic action equal to
``argmax(PPO_WEIGHTS @ obs + PPO_BIAS)``, which :func:`export_linear_policy`
extracts for ``MQL5Patcher`` (``OPTIMIZED_MATRIX`` anchors).

Timing: the decision at quote row ``t`` sees the alpha and Z-scored features of
the bar that closed before ``t`` (row ``t-1``), and the row-``t`` bid/ask spread.
This matches the EA, which evaluates closed bar shift 1 at the current tick.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.feature_engineer import TabularFeatureEngineer

ACTION_NAMES = ("SHORT", "FLAT", "LONG")  # TradingEnv targets -1/0/+1 units
ZSCORE_FEATURE_COLUMNS = tuple(f"z_{name}" for name in TabularFeatureEngineer.FEATURE_NAMES)
MARKET_OBSERVATION_COLUMNS = ("alpha_prediction", *ZSCORE_FEATURE_COLUMNS, "spread_fraction")
STATE_OBSERVATION_FEATURES = ("position_side", "unrealized_pnl_fraction")
OBSERVATION_NAMES = (*MARKET_OBSERVATION_COLUMNS, *STATE_OBSERVATION_FEATURES)
OBS_DIM = len(OBSERVATION_NAMES)
ACTION_DIM = len(ACTION_NAMES)
if (OBS_DIM, ACTION_DIM) != (12, 3):
    raise RuntimeError("linear policy contract must remain 12 observations x 3 actions")


def build_market_observations(
    quotes: pd.DataFrame, scaled_features: pd.DataFrame, alpha: pd.Series
) -> pd.DataFrame:
    """Append the ten market observation columns with closed-bar causality.

    Args:
        quotes: Quote rows with ``bid``/``ask`` (any extra columns are kept).
        scaled_features: Training-partition Z-scored features, columns exactly
            ``TabularFeatureEngineer.FEATURE_NAMES``, indexed like ``quotes``.
        alpha: Alpha prediction for each bar computed from that bar's features.

    Returns:
        Copy of ``quotes`` with MARKET_OBSERVATION_COLUMNS, alpha/features lagged
        one row; rows lacking a complete finite observation are dropped.
    """
    names = list(TabularFeatureEngineer.FEATURE_NAMES)
    if list(scaled_features.columns) != names:
        raise ValueError(f"scaled_features columns must be exactly {names}")
    if not (quotes.index.equals(scaled_features.index) and quotes.index.equals(alpha.index)):
        raise ValueError("quotes, scaled_features, and alpha must share one index")
    bid = quotes["bid"].astype(float)
    ask = quotes["ask"].astype(float)
    if (bid <= 0).any() or (ask < bid).any():
        raise ValueError("quotes require positive bid and ask >= bid")
    frame = quotes.copy()
    frame["alpha_prediction"] = alpha.astype(float).shift(1)
    for name, column in zip(names, ZSCORE_FEATURE_COLUMNS):
        frame[column] = scaled_features[name].astype(float).shift(1)
    frame["spread_fraction"] = (ask - bid) / bid
    values = frame[list(MARKET_OBSERVATION_COLUMNS)].to_numpy(dtype=float)
    return frame.loc[np.isfinite(values).all(axis=1)]


def export_linear_policy(model: Any) -> dict[str, list]:
    """Extract ``PPO_WEIGHTS`` (3x12) and ``PPO_BIAS`` (3) from a linear-actor PPO.

    Refuses any policy whose deterministic action is not exactly
    ``argmax(W @ obs + b)`` on the 12-dim contract observation.
    """
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import FlattenExtractor

    policy = model.policy
    if not isinstance(policy.action_space, spaces.Discrete) or policy.action_space.n != ACTION_DIM:
        raise ValueError(f"policy must have Discrete({ACTION_DIM}) actions")
    if tuple(policy.observation_space.shape) != (OBS_DIM,):
        raise ValueError(f"policy observation must be ({OBS_DIM},)")
    if not isinstance(policy.pi_features_extractor, FlattenExtractor):
        raise ValueError("policy feature extractor must be the identity FlattenExtractor")
    if any(True for _ in policy.mlp_extractor.policy_net.parameters()):
        raise ValueError("actor has hidden layers; train with policy_architecture='linear_actor'")
    weights = policy.action_net.weight.detach().cpu().double().numpy()
    bias = policy.action_net.bias.detach().cpu().double().numpy()
    if weights.shape != (ACTION_DIM, OBS_DIM) or bias.shape != (ACTION_DIM,):
        raise ValueError("unexpected action head shape")
    if not (np.isfinite(weights).all() and np.isfinite(bias).all()):
        raise ValueError("policy parameters must be finite")
    return {"PPO_WEIGHTS": weights.tolist(), "PPO_BIAS": bias.tolist()}


def linear_policy_action(weights: Any, bias: Any, observation: Any) -> int:
    """Reference for the EA: argmax(W @ obs + b), ties to the lowest index."""
    logits = np.asarray(weights, dtype=float) @ np.asarray(observation, dtype=float) + np.asarray(bias, dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    return int(np.argmax(logits))
