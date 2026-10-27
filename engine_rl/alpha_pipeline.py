"""Hash-verified XGBoost/ONNX alpha enrichment for the 12-input linear-policy contract.

``attach_alpha`` turns raw quote bars into the ten market observation columns of
:mod:`engine_rl.linear_policy` using the *exact* artifacts the EA consumes: the
manifest's feature windows, its training-partition Z-score (never refitted on RL
data), and the ONNX graph itself (float32 inputs, as ``OnnxRun`` in MQL5).
``TradingEnvAdapter`` fixes the environment to those columns plus the two
position-state features, so every trained policy is 12 x 3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.feature_engineer import TabularFeatureEngineer
from engine_rl.envs.trading_env import TradingEnv
from engine_rl.linear_policy import (
    MARKET_OBSERVATION_COLUMNS, STATE_OBSERVATION_FEATURES, build_market_observations,
)
from models.onnx_bridge import MANIFEST_CONTRACT, OnnxAlphaRuntime, sha256_file
from models.zscore import ZScoreParameters

OBSERVATION_CONTRACT = "alpha_linear_policy_12x3_v1"
# Python feature-engineer window -> EA_Template input carrying the same window.
EA_WINDOW_INPUTS: Mapping[str, str] = {
    "rsi_window": "InpRsiWindow", "macd_fast": "InpMacdFast", "macd_slow": "InpMacdSlow",
    "macd_signal": "InpMacdSignal", "atr_window": "InpAtrWindow",
    "bollinger_window": "InpBBWindow", "price_action_window": "InpPAWindow",
}

__all__ = [
    "AlphaArtifactError", "AlphaModel", "EA_WINDOW_INPUTS", "OBSERVATION_CONTRACT",
    "TradingEnvAdapter", "alpha_label_end", "attach_alpha", "load_alpha_model",
]


class AlphaArtifactError(ValueError):
    """The alpha manifest/ONNX pair is missing, altered, or contract-incompatible."""


def _reject_constant(value: str) -> None:
    raise AlphaArtifactError(f"non-finite JSON constant in alpha manifest: {value}")


@dataclass(frozen=True)
class AlphaModel:
    """Verified alpha artifacts; picklable so spawned workers never re-resolve paths."""

    manifest: Mapping[str, Any]
    manifest_sha256: str
    onnx_path: Path
    onnx_sha256: str
    zscore: ZScoreParameters
    engineer: TabularFeatureEngineer

    def runtime(self) -> OnnxAlphaRuntime:
        """Open the ONNX session, re-checking the hash at time of use."""
        return OnnxAlphaRuntime(self.onnx_path, expected_sha256=self.onnx_sha256)

    @property
    def training_window(self) -> Mapping[str, Any]:
        return self.manifest["training_window"]


def load_alpha_model(manifest_path: str | Path, expected_sha256: str) -> AlphaModel:
    """Load an exported alpha manifest and its ONNX graph under strict identity checks.

    Args:
        manifest_path: ``*.manifest.json`` written by ``export_xgb_alpha_onnx``.
        expected_sha256: Frozen config's SHA-256 of the manifest bytes.

    Returns:
        Verified :class:`AlphaModel`.

    Raises:
        AlphaArtifactError: Hash mismatch, unsupported contract, missing leakage
            evidence, or an ONNX file that differs from the manifest.
    """
    path = Path(manifest_path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise AlphaArtifactError(f"alpha manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    manifest = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(manifest, dict) or manifest.get("contract") != MANIFEST_CONTRACT:
        raise AlphaArtifactError("unsupported alpha manifest contract")
    window = manifest.get("training_window")
    if not isinstance(window, dict) or set(window) != {
        "first_feature_timestamp", "last_feature_timestamp", "label_horizon_bars",
    } or type(window["label_horizon_bars"]) is not int or window["label_horizon_bars"] < 1:
        raise AlphaArtifactError("alpha manifest lacks training_window leakage evidence; re-export it")
    params = manifest.get("feature_engineer")
    if not isinstance(params, dict) or set(params) != set(EA_WINDOW_INPUTS) or any(
        type(value) is not int for value in params.values()
    ):
        raise AlphaArtifactError(f"alpha feature_engineer must define exactly {sorted(EA_WINDOW_INPUTS)}")
    zscore = manifest.get("zscore") or {}
    try:
        parameters = ZScoreParameters(
            feature_names=tuple(zscore.get("feature_names", ())),
            mean=tuple(zscore.get("mean", ())), std=tuple(zscore.get("std", ())),
        )
        engineer = TabularFeatureEngineer(**params)
    except (ValueError, TypeError) as exc:
        raise AlphaArtifactError(f"invalid alpha manifest parameters: {exc}") from exc
    if parameters.feature_names != tuple(TabularFeatureEngineer.FEATURE_NAMES):
        raise AlphaArtifactError("alpha Z-score feature order differs from the EA observation contract")
    name = manifest.get("onnx_file")
    if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".onnx"):
        raise AlphaArtifactError("onnx_file must be a bare .onnx file name beside the manifest")
    onnx_path = path.parent / name
    onnx_sha = sha256_file(onnx_path)
    if onnx_sha != manifest.get("onnx_sha256"):
        raise AlphaArtifactError("ONNX artifact does not match the manifest SHA-256")
    return AlphaModel(manifest, actual, onnx_path.absolute(), onnx_sha, parameters, engineer)


def alpha_label_end(timestamps: pd.Series, model: AlphaModel) -> pd.Timestamp:
    """Return the last bar any alpha training label observed (last row + horizon bars).

    Positional, so RTH gaps are honored. The alpha's last training feature row
    must be present in ``timestamps``; otherwise the evidence cannot be proven.
    """
    window = model.training_window
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    last = pd.Timestamp(window["last_feature_timestamp"]).tz_convert("UTC")
    matches = np.flatnonzero(stamps == last)
    if len(matches) != 1:
        raise AlphaArtifactError("alpha training window is not provably inside this dataset")
    position = int(matches[0]) + window["label_horizon_bars"]
    if position >= len(stamps):
        raise AlphaArtifactError("alpha training labels extend beyond this dataset")
    return stamps[position]


def attach_alpha(df: pd.DataFrame, model: AlphaModel) -> pd.DataFrame:
    """Pre-compute the ten market observation columns for chronological bars.

    Features use the manifest windows, are Z-scored with the manifest
    training-partition statistics, and scored by the ONNX graph in float32.
    Alpha and features are lagged one closed bar; spread is row ``t``.

    Args:
        df: Contiguous chronological quote bars (warm-up followed by active rows)
            with timestamp, high, low, close, bid, and ask.
        model: Verified :class:`AlphaModel`.

    Returns:
        Copy of ``df`` (original index) plus ``MARKET_OBSERVATION_COLUMNS``;
        rows without a complete finite observation (warm-up) are dropped.
    """
    features = model.engineer.generate_features(df)
    if len(features) != len(df):
        raise AlphaArtifactError("feature rows do not align with input bars")
    features.index = df.index
    scaled = model.zscore.transform(features)
    values = scaled.to_numpy(dtype=np.float64)
    complete = np.isfinite(values).all(axis=1)
    alpha = pd.Series(np.nan, index=df.index, dtype=np.float64)
    if complete.any():
        prediction = model.runtime().predict(values[complete].astype(np.float32))
        alpha.iloc[np.flatnonzero(complete)] = prediction.astype(np.float64)
    return build_market_observations(df, scaled, alpha)


class TradingEnvAdapter(TradingEnv):
    """``TradingEnv`` locked to the 12-input contract (10 market + 2 position-state)."""

    def __init__(self, active_df: pd.DataFrame, **settings: Any) -> None:
        for name in ("feature_columns", "state_features"):
            if name in settings:
                raise ValueError(f"{name} is fixed by the {OBSERVATION_CONTRACT} contract")
        super().__init__(
            active_df, feature_columns=list(MARKET_OBSERVATION_COLUMNS),
            state_features=STATE_OBSERVATION_FEATURES, **settings,
        )
