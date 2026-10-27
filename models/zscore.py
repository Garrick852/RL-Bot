"""Training-partition Z-score standardization shared by XGBoost, ONNX and MQL5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from data.collector import DataIntegrityError
from data.feature_engineer import TabularFeatureEngineer, assert_causal_feature_frame

ZSCORE_DDOF = 0


@dataclass(frozen=True)
class ZScoreParameters:
    """Population (ddof=0) mean/std vectors in exact ``FEATURE_NAMES`` order."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        if count == 0 or len(self.mean) != count or len(self.std) != count:
            raise DataIntegrityError("Z-score vectors must be nonempty and match feature count")
        if tuple(self.feature_names) != TabularFeatureEngineer.FEATURE_NAMES:
            raise DataIntegrityError("Z-score feature order must equal FEATURE_NAMES")
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise DataIntegrityError("Z-score mean must be finite and std strictly positive")

    @classmethod
    def from_values(cls, feature_names: Sequence[str], values: np.ndarray) -> "ZScoreParameters":
        """Fit from a finite 2-D matrix; constant columns are rejected."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise DataIntegrityError("Z-score inputs must be a nonempty finite matrix")
        std = matrix.std(axis=0, ddof=ZSCORE_DDOF)
        constant = [str(name) for name, value in zip(feature_names, std) if not value > 0]
        if constant:
            raise DataIntegrityError(f"constant training features cannot be Z-scored: {constant}")
        return cls(
            feature_names=tuple(str(name) for name in feature_names),
            mean=tuple(float(v) for v in matrix.mean(axis=0)),
            std=tuple(float(v) for v in std),
        )

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        """Return float64 ``(values - mean) / std``."""
        return (np.asarray(values, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(self.std)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale ``X`` after an exact column-order check, preserving its index."""
        if list(X.columns) != list(self.feature_names):
            raise DataIntegrityError("feature columns do not match Z-score feature order")
        return pd.DataFrame(
            self.transform_values(X.to_numpy(dtype=np.float64)), index=X.index, columns=X.columns,
        )


def fit_zscore(X_train: pd.DataFrame) -> ZScoreParameters:
    """Fit Z-score vectors on a certified causal training partition."""
    assert_causal_feature_frame(X_train)
    return ZScoreParameters.from_values(X_train.columns, X_train.to_numpy(dtype=np.float64))
