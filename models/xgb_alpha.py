"""Purged walk-forward XGBoost alpha modeling for offline research."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

from data.collector import DataIntegrityError
from data.feature_engineer import assert_causal_feature_frame
from models.zscore import ZScoreParameters

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldMetrics:
    """Out-of-sample diagnostics for one strictly later validation fold."""

    fold: int
    train_rows: int
    validation_rows: int
    mse: float
    directional_accuracy: float
    spearman_rank_correlation: Optional[float]


class XGBAlphaModel:
    """Train and validate an XGBoost return regressor with a purged time split.

    Args:
        horizon: Forward-label horizon in bars.
        max_moving_average_window: Largest lookback used by the feature set.
            The TimeSeriesSplit gap is at least their sum.
        n_splits: Number of expanding walk-forward validation folds.
        model_params: Additional XGBRegressor parameters. CPU/thread and model
            determinism parameters are controlled by this class.
        random_state: Explicit model seed.
        standardize: Train on Z-scored features. Each fold fits mean/std on
            its own training rows only; the final model uses the full supplied
            training partition. Required for ONNX/MQL5 export, whose runtime
            input is the Z-scored vector.
        logger: Metric and validation logger.

    Notes:
        ``fit_walk_forward`` expects only the caller's training partition.
        Fold validation scores are OOS relative to each fold; after evaluation,
        the final deployable-to-research artifact is refit on the supplied
        training partition. This class has no deployment or execution path.
    """

    def __init__(
        self,
        *,
        horizon: int,
        max_moving_average_window: int,
        n_splits: int = 5,
        model_params: Optional[dict[str, Any]] = None,
        random_state: int = 0,
        standardize: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.horizon = self._positive_int(horizon, "horizon")
        self.max_moving_average_window = self._positive_int(
            max_moving_average_window, "max_moving_average_window",
        )
        self.n_splits = self._positive_int(n_splits, "n_splits")
        self.random_state = self._positive_int(random_state, "random_state", allow_zero=True)
        self.gap = self.horizon + self.max_moving_average_window
        self.model_params = dict(model_params or {})
        protected = {"n_jobs", "random_state", "objective", "tree_method"}
        conflict = protected.intersection(self.model_params)
        if conflict:
            raise ValueError(f"model_params cannot override governed fields: {sorted(conflict)}")
        self.logger = logger if logger is not None else LOGGER
        self.model: Any = None
        self.fold_metrics: list[FoldMetrics] = []
        self.feature_names: tuple[str, ...] = ()
        self.standardize = bool(standardize)
        self.zscore: Optional[ZScoreParameters] = None

    @staticmethod
    def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
        minimum = 0 if allow_zero else 1
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    def _validate_xy(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert_causal_feature_frame(X)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise DataIntegrityError("X and y must be Series/DataFrame with identical indices")
        if y.name != f"forward_return_{self.horizon}":
            raise DataIntegrityError(
                f"target must be generated for the configured horizon {self.horizon}"
            )
        if not isinstance(X.index, pd.DatetimeIndex) or X.index.tz is None:
            raise DataIntegrityError("X/y require a timezone-aware DatetimeIndex")
        if X.empty or not X.index.is_unique or not X.index.is_monotonic_increasing:
            raise DataIntegrityError("X/y must be nonempty, unique, and chronological")
        x_values = X.to_numpy(dtype=np.float64)
        y_values = y.to_numpy(dtype=np.float64)
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise DataIntegrityError("X/y must be fully cleaned and finite before fitting")
        if X.attrs.get("max_moving_average_window") != self.max_moving_average_window:
            raise DataIntegrityError(
                "model embargo window must equal the feature engineer's certified maximum lookback"
            )

    def _new_regressor(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=self.random_state,
            **self.model_params,
        )

    @staticmethod
    def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float, Optional[float]]:
        mse = float(np.mean(np.square(prediction - y_true)))
        directional_accuracy = float(np.mean(np.sign(prediction) == np.sign(y_true)))
        correlation = float(spearmanr(y_true, prediction).statistic)
        spearman = correlation if math.isfinite(correlation) else None
        return mse, directional_accuracy, spearman

    def fit_walk_forward(self, X: pd.DataFrame, y: pd.Series) -> list[FoldMetrics]:
        """Run expanding purged folds, log OOS metrics, then refit the final model.

        Args:
            X: Cleaned, certified causal feature frame from the train partition.
            y: Aligned forward-return labels for the same train partition.

        Returns:
            Immutable OOS metrics for each fold.

        Raises:
            DataIntegrityError: Inputs are misaligned, noncausal, or non-finite.
            ValueError: The requested fold/gap cannot be formed.
        """
        self._validate_xy(X, y)
        splitter = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        self.fold_metrics = []
        y_values = y.to_numpy(dtype=np.float64)
        for fold, (train_idx, validation_idx) in enumerate(splitter.split(X), start=1):
            if train_idx[-1] + self.gap >= validation_idx[0]:
                raise AssertionError("purge gap is smaller than the configured horizon/lookback embargo")
            X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
            if self.standardize:
                fold_scaler = ZScoreParameters.from_values(X.columns, X_train.to_numpy(dtype=np.float64))
                X_train, X_validation = fold_scaler.transform(X_train), fold_scaler.transform(X_validation)
            model = self._new_regressor()
            model.fit(X_train, y.iloc[train_idx], verbose=False)
            prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
            if prediction.shape != y_values[validation_idx].shape or not np.isfinite(prediction).all():
                raise DataIntegrityError(f"fold {fold} produced malformed/non-finite predictions")
            mse, accuracy, spearman = self._metrics(y_values[validation_idx], prediction)
            metrics = FoldMetrics(
                fold=fold,
                train_rows=len(train_idx),
                validation_rows=len(validation_idx),
                mse=mse,
                directional_accuracy=accuracy,
                spearman_rank_correlation=spearman,
            )
            self.fold_metrics.append(metrics)
            self.logger.info(
                "XGB OOS fold=%d train=%d validation=%d gap=%d MSE=%.10g "
                "directional_accuracy=%.6f Spearman=%s",
                fold, len(train_idx), len(validation_idx), self.gap,
                mse, accuracy, spearman,
            )
        self.zscore = (
            ZScoreParameters.from_values(X.columns, X.to_numpy(dtype=np.float64))
            if self.standardize else None
        )
        self.model = self._new_regressor()
        self.model.fit(self._model_input(X), y, verbose=False)
        self.feature_names = tuple(str(name) for name in X.columns)
        return list(self.fold_metrics)

    def _model_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.zscore.transform(X) if self.zscore is not None else X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict from raw certified features, applying the fitted Z-score if any."""
        if self.model is None:
            raise RuntimeError("fit_walk_forward must complete before predict")
        assert_causal_feature_frame(X)
        return np.asarray(self.model.predict(self._model_input(X)), dtype=np.float64)

    def export(self, path: str | Path = "reports/xgb_alpha.json") -> Path:
        """Atomically save the fitted XGBoost model in JSON format.

        Args:
            path: Local destination, defaulting to ``reports/xgb_alpha.json``.

        Returns:
            Resolved artifact path.

        Raises:
            RuntimeError: No final model has been fitted.
            ValueError: Destination is not a JSON model path.
            OSError: Artifact cannot be persisted.
        """
        if self.model is None:
            raise RuntimeError("fit_walk_forward must complete before export")
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            raise ValueError("XGBoost export destination must use .json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".json", dir=destination.parent,
        )
        os.close(descriptor)
        try:
            self.model.save_model(name)
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.logger.info(
            "Exported XGBoost JSON model to %s; feature_order=%s",
            destination, self.feature_names,
        )
        return destination

    def load(self, path: str | Path) -> Any:
        """Load a persisted JSON model for deterministic offline inference."""
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(path))
        self.model = model
        self.zscore = None  # JSON holds trees only; Z-score vectors live in the ONNX manifest.
        self.feature_names = tuple(model.get_booster().feature_names or ())
        return model
