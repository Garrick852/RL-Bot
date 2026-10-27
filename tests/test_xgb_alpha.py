"""Causality, label alignment, purged splits, metrics, and artifact tests."""

import numpy as np
import pandas as pd
import pytest

from data.collector import DataIntegrityError
from data.feature_engineer import (
    CAUSAL_FEATURE_CONTRACT, TabularFeatureEngineer, assert_causal_feature_frame,
)
from models.xgb_alpha import XGBAlphaModel


def bars(count=220):
    timestamps = pd.date_range("2024-01-02", periods=count, freq="15min", tz="UTC")
    close = 100 + np.cumsum(np.sin(np.arange(count) / 4) * .15 + .02)
    return pd.DataFrame({
        "timestamp": timestamps, "open": close - .02, "high": close + .5,
        "low": close - .5, "close": close,
    })


def test_features_are_invariant_to_changes_strictly_after_each_cutoff():
    frame = bars()
    engineer = TabularFeatureEngineer()
    original = engineer.generate_features(frame)
    for cutoff in (40, 100, 160):
        changed = frame.copy()
        changed.loc[cutoff:, ["high", "low", "close", "open"]] *= 1.75
        recomputed = engineer.generate_features(changed)
        pd.testing.assert_frame_equal(
            original.iloc[:cutoff], recomputed.iloc[:cutoff], check_exact=True,
        )
    assert original.attrs["causal_contract"] == CAUSAL_FEATURE_CONTRACT
    assert original.attrs["uses_future_rows"] is False
    assert original.index.equals(pd.DatetimeIndex(frame["timestamp"]))


def test_forward_labels_point_exactly_horizon_bars_ahead():
    frame = bars(60)
    engineer = TabularFeatureEngineer()
    horizon = 5
    labels = engineer.generate_labels(frame, horizon)
    close = frame["close"].to_numpy()
    np.testing.assert_allclose(
        labels.iloc[:-horizon].to_numpy(),
        close[horizon:] / close[:-horizon] - 1,
        rtol=0, atol=1e-14,
    )
    assert labels.iloc[-horizon:].isna().all()
    assert labels.index.equals(pd.DatetimeIndex(frame["timestamp"]))


def test_alignment_removes_warmup_and_forward_label_tail_together():
    frame = bars(80)
    engineer = TabularFeatureEngineer()
    X = engineer.generate_features(frame)
    y = engineer.generate_labels(frame, 3)
    clean_x, clean_y = engineer.align_and_clean(X, y)
    assert clean_x.index.equals(clean_y.index)
    assert clean_x.index.is_monotonic_increasing
    assert clean_x.attrs["causal_contract"] == CAUSAL_FEATURE_CONTRACT
    assert clean_x.index[0] > X.index[0]
    assert clean_x.index[-1] < X.index[-1]
    with pytest.raises(DataIntegrityError, match="identical unique indices"):
        engineer.align_and_clean(X, y.iloc[1:])


def test_feature_input_rejects_naive_timestamps_and_bad_ohlc():
    engineer = TabularFeatureEngineer()
    frame = bars(40)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="timezone-aware"):
        engineer.generate_features(frame)
    frame = bars(40)
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1
    with pytest.raises(DataIntegrityError, match="OHLC invariant"):
        engineer.generate_features(frame)


def test_alpha_model_rejects_uncertified_or_future_added_feature():
    frame = bars()
    engineer = TabularFeatureEngineer()
    X, y = engineer.align_and_clean(
        engineer.generate_features(frame), engineer.generate_labels(frame, 2),
    )
    uncertified = pd.DataFrame(X.to_numpy(), columns=X.columns, index=X.index)
    with pytest.raises(DataIntegrityError, match="backward-only"):
        XGBAlphaModel(horizon=2, max_moving_average_window=26).fit_walk_forward(uncertified, y)
    leaked = X.copy()
    leaked["future_close"] = frame.set_index("timestamp")["close"].shift(-1).reindex(leaked.index)
    with pytest.raises(DataIntegrityError, match="backward-only"):
        assert_causal_feature_frame(leaked)
    modified = X.copy()
    modified.iloc[-1, 0] = 123456.0
    with pytest.raises(DataIntegrityError, match="modified"):
        assert_causal_feature_frame(modified)


def test_purged_walk_forward_metrics_and_json_reload_are_deterministic(tmp_path):
    frame = bars(260)
    engineer = TabularFeatureEngineer()
    X, y = engineer.align_and_clean(
        engineer.generate_features(frame), engineer.generate_labels(frame, 3),
    )
    model = XGBAlphaModel(
        horizon=3, max_moving_average_window=engineer.max_moving_average_window,
        n_splits=3, model_params={"n_estimators": 12, "max_depth": 2, "learning_rate": .1},
        random_state=7,
    )
    folds = model.fit_walk_forward(X, y)
    assert model.gap == 3 + engineer.max_moving_average_window
    assert len(folds) == 3
    assert all(fold.train_rows > 0 and fold.validation_rows > 0 for fold in folds)
    assert all(np.isfinite(fold.mse) and 0 <= fold.directional_accuracy <= 1 for fold in folds)
    assert all(fold.spearman_rank_correlation is None or np.isfinite(
        fold.spearman_rank_correlation
    ) for fold in folds)
    expected = model.model.predict(X.iloc[-10:])
    artifact = model.export(tmp_path / "reports" / "xgb_alpha.json")
    assert artifact.is_file()
    loaded = XGBAlphaModel(
        horizon=3, max_moving_average_window=engineer.max_moving_average_window,
        n_splits=3, random_state=7,
    )
    restored = loaded.load(artifact)
    np.testing.assert_array_equal(expected, restored.predict(X.iloc[-10:]))


def test_invalid_embargo_and_nonfinite_data_fail_closed():
    frame = bars(150)
    engineer = TabularFeatureEngineer()
    X, y = engineer.align_and_clean(
        engineer.generate_features(frame), engineer.generate_labels(frame, 4),
    )
    with pytest.raises(DataIntegrityError, match="embargo window"):
        XGBAlphaModel(horizon=4, max_moving_average_window=2).fit_walk_forward(X, y)
    with pytest.raises(DataIntegrityError, match="configured horizon"):
        XGBAlphaModel(
            horizon=5, max_moving_average_window=engineer.max_moving_average_window,
        ).fit_walk_forward(X, y)
    dirty_y = y.copy()
    dirty_y.iloc[0] = np.inf
    with pytest.raises(DataIntegrityError, match="finite"):
        XGBAlphaModel(
            horizon=4, max_moving_average_window=engineer.max_moving_average_window,
        ).fit_walk_forward(X, dirty_y)
