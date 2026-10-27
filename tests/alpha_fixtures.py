"""Shared synthetic bars plus a tiny exported XGBoost/ONNX alpha for pipeline tests."""

from __future__ import annotations

import atexit
import functools
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA_WINDOWS = {
    "rsi_window": 3, "macd_fast": 3, "macd_slow": 6, "macd_signal": 3,
    "atr_window": 3, "bollinger_window": 5, "price_action_window": 5,
}
ALPHA_HORIZON = 2
ALPHA_TRAIN_ROWS = 150
BARS = 400


def make_bars(count: int = BARS) -> pd.DataFrame:
    """Sine-wave bid/ask bars with noisy ranges, so no alpha feature is constant."""
    rng = np.random.default_rng(852)
    times = pd.date_range("2024-01-01", periods=count, freq="min", tz="UTC")
    midpoint = np.round(100 + np.sin(np.arange(count) / 3) + rng.normal(0, .05, count), 2)
    return pd.DataFrame({
        "timestamp": times, "open": midpoint,
        "high": np.round(midpoint + .5 + rng.uniform(0, 1, count), 2),
        "low": np.round(midpoint - .5 - rng.uniform(0, 1, count), 2),
        "close": midpoint, "volume": np.ones(count),
        "bid": midpoint - .01, "ask": midpoint + .01,
    })


@functools.cache
def _exported_alpha() -> Path:
    from data.feature_engineer import TabularFeatureEngineer
    from models.onnx_bridge import export_xgb_alpha_onnx
    from models.xgb_alpha import XGBAlphaModel

    directory = Path(tempfile.mkdtemp(prefix="alpha_fixture_"))
    atexit.register(shutil.rmtree, directory, True)
    bars = make_bars().iloc[:ALPHA_TRAIN_ROWS]
    engineer = TabularFeatureEngineer(**ALPHA_WINDOWS)
    X, y = engineer.align_and_clean(
        engineer.generate_features(bars), engineer.generate_labels(bars, ALPHA_HORIZON),
    )
    model = XGBAlphaModel(
        horizon=ALPHA_HORIZON, max_moving_average_window=engineer.max_moving_average_window,
        n_splits=2, model_params={"n_estimators": 10, "max_depth": 2, "learning_rate": .1},
        standardize=True,
    )
    model.fit_walk_forward(X, y)
    export_xgb_alpha_onnx(model, X, onnx_path=directory / "xgb_alpha.onnx", feature_engineer=engineer)
    return directory


def install_alpha(destination: Path) -> tuple[str, str]:
    """Copy the cached manifest/ONNX pair into ``destination``; return (relative path, SHA-256)."""
    source = _exported_alpha()
    for name in ("xgb_alpha.onnx", "xgb_alpha.manifest.json"):
        shutil.copyfile(source / name, destination / name)
    manifest = destination / "xgb_alpha.manifest.json"
    return manifest.name, hashlib.sha256(manifest.read_bytes()).hexdigest()
