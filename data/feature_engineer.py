"""Causal tabular feature and forward-label generation for market bars."""

from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np
import pandas as pd

from data.collector import DataIntegrityError

CAUSAL_FEATURE_CONTRACT = "backward_rolling_including_current_v1"


def _feature_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(str(column) for column in frame.columns).encode("utf-8"))
    if isinstance(frame.index, pd.DatetimeIndex):
        digest.update(frame.index.asi8.tobytes())
        digest.update(str(frame.index.tz).encode("utf-8"))
    else:
        digest.update(pd.util.hash_pandas_object(frame.index, index=True).to_numpy().tobytes())
    digest.update(frame.to_numpy(dtype=np.float64).tobytes())
    return digest.hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class TabularFeatureEngineer:
    """Build deterministic OHLC features using no observations after each row.

    All rolling calculations use right-aligned windows (the current row and
    earlier rows only); exponential calculations use ``adjust=False``. The
    returned frame carries a contract marker consumed by ``XGBAlphaModel``.

    Args:
        rsi_window: Wilder RSI lookback.
        macd_fast: Fast MACD EMA span.
        macd_slow: Slow MACD EMA span; must exceed ``macd_fast``.
        macd_signal: MACD signal EMA span.
        atr_window: Wilder ATR lookback.
        bollinger_window: Bollinger bandwidth lookback.
        price_action_window: High/low distance lookback.
    """

    FEATURE_NAMES = (
        "rsi", "macd", "macd_signal", "macd_histogram", "atr",
        "bollinger_width", "distance_from_rolling_high", "distance_from_rolling_low",
    )

    def __init__(
        self,
        *,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_window: int = 14,
        bollinger_window: int = 20,
        price_action_window: int = 20,
    ) -> None:
        self.rsi_window = _positive_integer(rsi_window, "rsi_window")
        self.macd_fast = _positive_integer(macd_fast, "macd_fast")
        self.macd_slow = _positive_integer(macd_slow, "macd_slow")
        self.macd_signal = _positive_integer(macd_signal, "macd_signal")
        self.atr_window = _positive_integer(atr_window, "atr_window")
        self.bollinger_window = _positive_integer(bollinger_window, "bollinger_window")
        self.price_action_window = _positive_integer(price_action_window, "price_action_window")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be less than macd_slow")
        self.max_moving_average_window = max(
            self.rsi_window, self.macd_fast, self.macd_slow, self.macd_signal,
            self.atr_window, self.bollinger_window, self.price_action_window,
        )

    def _validate_bars(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        if not isinstance(df, pd.DataFrame):
            raise DataIntegrityError("df must be a pandas DataFrame")
        required = {"timestamp", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise DataIntegrityError(f"feature input missing columns: {sorted(missing)}")
        if not df.columns.is_unique:
            raise DataIntegrityError("feature input contains duplicate columns")
        if df.empty or not df.index.is_unique:
            raise DataIntegrityError("feature input must be nonempty with a unique row index")
        try:
            parsed = [pd.Timestamp(value) for value in df["timestamp"]]
            if any(stamp.tzinfo is None or stamp.utcoffset() is None for stamp in parsed):
                raise DataIntegrityError("feature timestamps must be timezone-aware")
            timestamps = pd.DatetimeIndex([stamp.tz_convert("UTC") for stamp in parsed])
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataIntegrityError(f"invalid feature timestamps: {exc}") from exc
        if timestamps.hasnans or not timestamps.is_unique or not timestamps.is_monotonic_increasing:
            raise DataIntegrityError("feature timestamps must be non-null, unique, and increasing")
        if df.loc[:, ["high", "low", "close"]].isna().any().any():
            raise DataIntegrityError("feature price inputs contain null values")
        for column in ("high", "low", "close"):
            values = df[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise DataIntegrityError(f"{column} contains non-finite prices")
        if (df["high"] < df["close"]).any() or (df["low"] > df["close"]).any():
            raise DataIntegrityError("OHLC invariant failed: high >= close >= low")
        if (df.loc[:, ["high", "low", "close"]] <= 0).any().any():
            raise DataIntegrityError("OHLC feature prices must be strictly positive")
        return timestamps

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum, volatility, and price-action features causally.

        For every output row ``t``, all operands are drawn from indices ``<=t``.
        No centered windows, backward label shifts, or future data are used.

        Args:
            df: Chronological bars with timestamp, high, low, and close.

        Returns:
            Feature-only DataFrame indexed by UTC timestamps. Initial warm-up
            rows remain NaN for joint cleaning with the target.

        Raises:
            DataIntegrityError: Required prices or chronology are invalid.
        """
        timestamps = self._validate_bars(df)
        work = df.copy()
        work.index = timestamps
        high, low, close = (work[name].astype(np.float64) for name in ("high", "low", "close"))

        change = close.diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)
        average_gain = gain.ewm(
            alpha=1 / self.rsi_window, min_periods=self.rsi_window, adjust=False,
        ).mean()
        average_loss = loss.ewm(
            alpha=1 / self.rsi_window, min_periods=self.rsi_window, adjust=False,
        ).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + relative_strength)
        rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
        rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)

        fast_ema = close.ewm(span=self.macd_fast, min_periods=self.macd_fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.macd_slow, min_periods=self.macd_slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(
            span=self.macd_signal, min_periods=self.macd_signal, adjust=False,
        ).mean()

        previous_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(
            alpha=1 / self.atr_window, min_periods=self.atr_window, adjust=False,
        ).mean()

        rolling_close = close.rolling(self.bollinger_window, min_periods=self.bollinger_window)
        middle = rolling_close.mean()
        deviation = rolling_close.std(ddof=0)
        bollinger_width = (4 * deviation / middle).where(middle != 0)

        rolling_high = high.rolling(self.price_action_window, min_periods=self.price_action_window).max()
        rolling_low = low.rolling(self.price_action_window, min_periods=self.price_action_window).min()
        distance_high = (close - rolling_high) / rolling_high
        distance_low = (close - rolling_low) / rolling_low

        features = pd.DataFrame({
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd - macd_signal,
            "atr": atr,
            "bollinger_width": bollinger_width,
            "distance_from_rolling_high": distance_high,
            "distance_from_rolling_low": distance_low,
        }, index=timestamps)
        if not features.columns.equals(pd.Index(self.FEATURE_NAMES)):
            raise AssertionError("feature set diverged from the reviewed causal contract")
        finite_or_nan = np.isfinite(features.to_numpy()) | features.isna().to_numpy()
        if not finite_or_nan.all():
            raise DataIntegrityError("generated feature values contain infinity")
        features.attrs["causal_contract"] = CAUSAL_FEATURE_CONTRACT
        features.attrs["feature_names"] = list(self.FEATURE_NAMES)
        features.attrs["max_moving_average_window"] = self.max_moving_average_window
        features.attrs["uses_future_rows"] = False
        features.attrs["causal_fingerprint"] = _feature_fingerprint(features)
        return features

    def generate_labels(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Create forward simple returns, aligned to the decision-time row.

        The negative shift is intentional and isolated to labels:
        ``label[t] = close[t+horizon] / close[t] - 1``. Feature columns must
        never be created with this alignment.

        Args:
            df: Validated chronological bars with positive close prices.
            horizon: Number of future bars represented by each label.

        Returns:
            Forward-return Series. Its final ``horizon`` entries are NaN.
        """
        horizon = _positive_integer(horizon, "horizon")
        timestamps = self._validate_bars(df)
        close = pd.Series(df["close"].to_numpy(dtype=np.float64), index=timestamps)
        labels = close.shift(-horizon).div(close).sub(1.0)
        labels.name = f"forward_return_{horizon}"
        return labels

    @staticmethod
    def align_and_clean(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Drop warm-up and horizon-edge rows while preserving exact alignment.

        Args:
            X: Causally certified feature frame.
            y: Forward-return target on the same row index.

        Returns:
            Feature/target pair with identical indices and only finite rows.

        Raises:
            DataIntegrityError: Inputs do not have matching unique indices,
                or contain infinities.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise DataIntegrityError("X must be a DataFrame and y must be a Series")
        assert_causal_feature_frame(X)
        if not X.index.equals(y.index) or not X.index.is_unique:
            raise DataIntegrityError("X and y must have identical unique indices before cleaning")
        if X.empty or X.columns.empty:
            raise DataIntegrityError("X must contain rows and feature columns")
        if not X.columns.is_unique or any(
            not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
            for dtype in X.dtypes
        ):
            raise DataIntegrityError("X must have unique, numeric feature columns")
        x_values = X.to_numpy(dtype=np.float64)
        y_values = y.to_numpy(dtype=np.float64)
        if np.isinf(x_values).any() or np.isinf(y_values).any():
            raise DataIntegrityError("X and y cannot contain infinite values")
        valid = np.isfinite(x_values).all(axis=1) & np.isfinite(y_values)
        clean_x = X.loc[valid].copy()
        clean_y = y.loc[valid].copy()
        if not clean_x.index.equals(clean_y.index):
            raise AssertionError("joint cleaning failed to preserve exact X/y alignment")
        clean_x.attrs.update(X.attrs)
        clean_x.attrs["causal_fingerprint"] = _feature_fingerprint(clean_x)
        return clean_x, clean_y


def assert_causal_feature_frame(X: pd.DataFrame) -> None:
    """Reject feature matrices without the engineer's backward-only contract.

    A value-only DataFrame cannot reveal its own provenance. The alpha model
    therefore accepts only frames produced by ``generate_features`` and
    carrying its certificate; future-shifted labels are deliberately not
    accepted as features.
    """
    if not isinstance(X, pd.DataFrame):
        raise DataIntegrityError("X must be a DataFrame")
    try:
        fingerprint = _feature_fingerprint(X)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataIntegrityError(f"causal feature frame is not numeric: {exc}") from exc
    if (
        X.attrs.get("causal_contract") != CAUSAL_FEATURE_CONTRACT
        or X.attrs.get("uses_future_rows") is not False
        or X.attrs.get("feature_names") != list(X.columns)
        or not X.columns.isin(TabularFeatureEngineer.FEATURE_NAMES).all()
        or X.attrs.get("causal_fingerprint") != fingerprint
    ):
        raise DataIntegrityError(
            "X lacks a valid backward-only feature certificate, was modified, "
            "or includes an unapproved feature"
        )
