"""Zero-drift gate between Python features/ONNX and the MQL5 parity dumper.

Synthetic tests always run and prove (a) the line-by-line algorithm used by
``engine_mt5/include/FeatureBuilder.mqh`` reproduces pandas, (b) the ONNX
export is hash-locked and equivalent to XGBoost, and (c) the comparator fails
closed on drift. The real MT5 gate runs only when ``RLBOT_MT5_PARITY_DIR``
points at a directory containing the exported manifest/ONNX plus the
``mql5_bars.csv`` and ``mql5_features.csv`` written by
``engine_mt5/scripts/validate_features.mq5``. A skip is NOT parity evidence.
"""

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxmltools")

from data.collector import DataIntegrityError
from data.feature_engineer import TabularFeatureEngineer
from models.feature_parity import (
    MT5_TIME_FORMAT, PARITY_COLUMNS, FeatureParityError, build_python_reference,
    compare_feature_parity, load_mql5_features, load_mt5_bars, run_parity_gate,
)
from models.onnx_bridge import (
    OnnxAlphaRuntime, OnnxExportError, ZScoreParameters, export_xgb_alpha_onnx, fit_zscore,
    load_manifest,
)
from models.xgb_alpha import XGBAlphaModel

ROOT = Path(__file__).resolve().parents[1]
FEATURE_BUILDER = ROOT / "engine_mt5" / "include" / "FeatureBuilder.mqh"
ATOL = 1e-4


def make_bars(count=700, seed=7):
    rng = np.random.default_rng(seed)
    close = 4500 + np.cumsum(rng.normal(0, 3, count))
    close[200:215] = close[199]  # flat run: zero gains/losses and zero-width segments
    spread_up = np.abs(rng.normal(0, 2, count))
    spread_down = np.abs(rng.normal(0, 2, count))
    spread_up[200:215] = spread_down[200:215] = 0.0
    timestamps = pd.date_range("2024-03-01 00:00", periods=count, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "open": close, "high": close + spread_up,
        "low": close - spread_down, "close": close,
    })


def mql5_mirror_features(high, low, close, windows):
    """Pure-Python transcription of CFeatureBuilder::Build (keep in lock-step)."""
    rsi_w, fast_w, slow_w = windows["rsi_window"], windows["macd_fast"], windows["macd_slow"]
    sig_w, atr_w = windows["macd_signal"], windows["atr_window"]
    bb_w, pa_w = windows["bollinger_window"], windows["price_action_window"]

    class Ewm:
        def __init__(self, alpha, min_periods):
            self.alpha, self.min_periods, self.nobs, self.value = alpha, min_periods, 0, 0.0

        def valid(self):
            return self.nobs >= self.min_periods

        def update(self, x):
            if self.nobs == 0:
                self.value = x
            else:
                old_wt = 1.0 - self.alpha
                if self.value != x:
                    self.value = (old_wt * self.value + self.alpha * x) / (old_wt + self.alpha)
            self.nobs += 1

    gain, loss = Ewm(1 / rsi_w, rsi_w), Ewm(1 / rsi_w, rsi_w)
    fast, slow = Ewm(2 / (fast_w + 1), fast_w), Ewm(2 / (slow_w + 1), slow_w)
    signal, atr = Ewm(2 / (sig_w + 1), sig_w), Ewm(1 / atr_w, atr_w)
    first_valid = max(rsi_w, (slow_w - 1) + (sig_w - 1), atr_w - 1, bb_w - 1, pa_w - 1)
    rows = np.full((len(close), 8), np.nan)
    for i, (h, l, c) in enumerate(zip(high, low, close)):
        if i > 0:
            change = c - close[i - 1]
            gain.update(change if change > 0 else 0.0)
            loss.update(-change if change < 0 else 0.0)
        fast.update(c)
        slow.update(c)
        macd_ok = fast.valid() and slow.valid()
        macd = fast.value - slow.value if macd_ok else 0.0
        if macd_ok:
            signal.update(macd)
        tr = h - l
        if i > 0:
            pc = close[i - 1]
            tr = max(tr, max(abs(h - pc), abs(l - pc)))
        atr.update(tr)
        if i < first_valid or not (gain.valid() and signal.valid() and atr.valid()):
            continue
        if loss.value == 0.0:
            rsi = 100.0 if gain.value > 0.0 else 50.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + gain.value / loss.value)
        window = close[i - bb_w + 1:i + 1]
        mean = sum(window) / bb_w
        ss = sum((value - mean) * (value - mean) for value in window)
        hh = max(high[i - pa_w + 1:i + 1])
        ll = min(low[i - pa_w + 1:i + 1])
        rows[i] = (rsi, macd, signal.value, macd - signal.value, atr.value,
                   4.0 * math.sqrt(ss / bb_w) / mean, (c - hh) / hh, (c - ll) / ll)
    return rows


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("onnx_parity")
    engineer = TabularFeatureEngineer()
    bars = make_bars()
    X, y = engineer.align_and_clean(engineer.generate_features(bars), engineer.generate_labels(bars, 4))
    model = XGBAlphaModel(
        horizon=4, max_moving_average_window=engineer.max_moving_average_window, n_splits=3,
        model_params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
        standardize=True,
    )
    model.fit_walk_forward(X, y)
    manifest = export_xgb_alpha_onnx(
        model, X, onnx_path=directory / "xgb_alpha.onnx", feature_engineer=engineer,
    )
    return directory, manifest, bars, model, X


def write_simulated_dump(directory, bars, manifest, onnx_path, export_rows=400):
    """Emulate validate_features.mq5 output using the MQL5-mirror algorithm."""
    high, low, close = (bars[c].to_list() for c in ("high", "low", "close"))
    raw = mql5_mirror_features(high, low, close, manifest["feature_engineer"])
    zs = manifest["zscore"]
    scaled = (raw - np.asarray(zs["mean"])) / np.asarray(zs["std"])
    window = slice(len(bars) - export_rows, len(bars))
    runtime = OnnxAlphaRuntime(onnx_path, expected_sha256=manifest["onnx_sha256"])
    prediction = runtime.predict(scaled[window].astype(np.float32))
    stamps = bars["timestamp"].dt.strftime(MT5_TIME_FORMAT)
    bar_lines = ["timestamp,open,high,low,close"] + [
        f"{t},{o!r},{h!r},{l!r},{c!r}"
        for t, o, h, l, c in zip(stamps, bars["open"], high, low, close)
    ]
    (directory / "mql5_bars.csv").write_text("\n".join(bar_lines) + "\n", encoding="ascii")
    feature_lines = [",".join(["timestamp", *PARITY_COLUMNS])]
    for row, stamp in enumerate(stamps.iloc[window]):
        values = [*raw[window][row], *scaled[window][row], float(prediction[row])]
        feature_lines.append(",".join([stamp, *(repr(float(v)) for v in values)]))
    (directory / "mql5_features.csv").write_text("\n".join(feature_lines) + "\n", encoding="ascii")


def test_mql5_algorithm_mirror_matches_pandas_exactly_enough():
    bars = make_bars()
    engineer = TabularFeatureEngineer()
    expected = engineer.generate_features(bars).to_numpy()
    windows = {name: getattr(engineer, name) for name in (
        "rsi_window", "macd_fast", "macd_slow", "macd_signal", "atr_window",
        "bollinger_window", "price_action_window")}
    mirror = mql5_mirror_features(*(bars[c].to_list() for c in ("high", "low", "close")), windows)
    first_pandas = int(np.flatnonzero(~np.isnan(expected).any(axis=1))[0])
    first_mirror = int(np.flatnonzero(~np.isnan(mirror).any(axis=1))[0])
    assert first_pandas == first_mirror == 33
    np.testing.assert_allclose(mirror[first_mirror:], expected[first_pandas:], rtol=0, atol=1e-9)
    assert np.isnan(mirror[:first_mirror]).all()
    flat_rsi = expected[214, 0]
    assert flat_rsi == mirror[214, 0]


def test_feature_builder_source_keeps_reviewed_order_and_formulas():
    source = FEATURE_BUILDER.read_text(encoding="utf-8")
    names = re.search(r"names\[ALPHA_FEATURE_COUNT\]\s*=\s*\{(.*?)\};", source, re.S).group(1)
    assert tuple(re.findall(r'"([a-z_]+)"', names)) == TabularFeatureEngineer.FEATURE_NAMES
    for fragment in (
        "(old_wt * value + alpha * x) / (old_wt + alpha)",
        "MathSqrt(ss / m_bb_window)",
        "2.0 / (m_macd_fast + 1.0)",
        "1.0 / m_rsi_window",
        "(m_macd_slow - 1) + (m_macd_signal - 1)",
    ):
        assert fragment in source
    for builtin in ("iRSI(", "iMACD(", "iATR(", "iBands(", "iMA("):
        assert builtin not in source


def test_onnx_export_is_hash_locked_shaped_and_equivalent(exported):
    import onnx

    directory, manifest, _, model, X = exported
    graph = onnx.load(directory / "xgb_alpha.onnx").graph
    assert [i.name for i in graph.input] == ["features"]
    assert [o.name for o in graph.output] == ["prediction"]
    assert graph.input[0].type.tensor_type.shape.dim[1].dim_value == 8
    assert graph.output[0].type.tensor_type.shape.dim[1].dim_value == 1
    assert manifest["zscore"]["feature_names"] == list(TabularFeatureEngineer.FEATURE_NAMES)
    assert manifest["xgb_onnx_max_abs_difference"] <= 1e-6
    assert manifest["deployment"] == "OFFLINE_PARITY_ONLY_NOT_WIRED_TO_EA"
    assert load_manifest(directory / "xgb_alpha.manifest.json")["onnx_sha256"] == manifest["onnx_sha256"]
    zscore_bytes = (directory / "xgb_alpha_zscore.csv").read_bytes()
    assert zscore_bytes.startswith(b"feature,mean,std\r\nrsi,")
    np.testing.assert_allclose(
        np.std(X.to_numpy(), axis=0, ddof=0), manifest["zscore"]["std"], rtol=1e-12,
    )
    runtime = OnnxAlphaRuntime(directory / "xgb_alpha.onnx", expected_sha256=manifest["onnx_sha256"])
    scaled = ZScoreParameters(
        tuple(manifest["zscore"]["feature_names"]), tuple(manifest["zscore"]["mean"]),
        tuple(manifest["zscore"]["std"]),
    ).transform(X)
    np.testing.assert_allclose(runtime.predict(scaled.to_numpy()), model.predict(X), rtol=0, atol=1e-6)
    with pytest.raises(OnnxExportError, match="hash mismatch"):
        OnnxAlphaRuntime(directory / "xgb_alpha.onnx", expected_sha256="0" * 64)


def test_export_requires_standardized_model_and_matching_partition(exported, tmp_path):
    _, _, bars, model, X = exported
    engineer = TabularFeatureEngineer()
    X_raw, y = engineer.align_and_clean(engineer.generate_features(bars), engineer.generate_labels(bars, 4))
    raw_model = XGBAlphaModel(horizon=4, max_moving_average_window=26, n_splits=2,
                              model_params={"n_estimators": 5})
    raw_model.fit_walk_forward(X_raw, y)
    with pytest.raises(OnnxExportError, match="standardize=True"):
        export_xgb_alpha_onnx(raw_model, X_raw, onnx_path=tmp_path / "a.onnx")
    subset, _ = engineer.align_and_clean(
        engineer.generate_features(bars.iloc[:500]), engineer.generate_labels(bars.iloc[:500], 4),
    )
    with pytest.raises(OnnxExportError, match="partition"):
        export_xgb_alpha_onnx(model, subset, onnx_path=tmp_path / "b.onnx")
    assert not (tmp_path / "a.onnx").exists() and not (tmp_path / "b.onnx").exists()


def test_fold_standardization_uses_only_fold_training_rows(exported):
    _, _, _, model, X = exported
    assert model.zscore == fit_zscore(X)
    assert len(model.fold_metrics) == 3


def test_zscore_rejects_constant_and_uncertified_features():
    engineer = TabularFeatureEngineer()
    bars = make_bars()
    X, _ = engineer.align_and_clean(engineer.generate_features(bars), engineer.generate_labels(bars, 4))
    with pytest.raises(DataIntegrityError):
        fit_zscore(X.copy().assign(rsi=1.0))
    with pytest.raises(DataIntegrityError):
        ZScoreParameters(tuple(X.columns), (0.0,) * 8, (1.0,) * 7 + (0.0,))


def test_simulated_mql5_dump_passes_zero_drift_gate(exported, tmp_path):
    directory, manifest, bars, _, _ = exported
    for name in ("xgb_alpha.onnx", "xgb_alpha.manifest.json"):
        (tmp_path / name).write_bytes((directory / name).read_bytes())
    write_simulated_dump(tmp_path, bars, manifest, tmp_path / "xgb_alpha.onnx")
    report = run_parity_gate(tmp_path, atol=ATOL)
    assert report.rows == 400
    assert max(report.max_abs_difference.values()) < 1e-8


def test_parity_gate_fails_closed_on_drift_missing_rows_and_header(exported, tmp_path):
    directory, manifest, bars, _, _ = exported
    onnx_path = directory / "xgb_alpha.onnx"
    write_simulated_dump(tmp_path, bars, manifest, onnx_path)
    reference = build_python_reference(load_mt5_bars(tmp_path / "mql5_bars.csv"), manifest, onnx_path)
    mql5 = load_mql5_features(tmp_path / "mql5_features.csv")

    drifted = mql5.copy()
    drifted.iloc[10, drifted.columns.get_loc("raw_atr")] += 2 * ATOL
    with pytest.raises(FeatureParityError, match="raw_atr"):
        compare_feature_parity(reference, drifted, atol=ATOL)

    drifted = mql5.copy()
    drifted.iloc[-1, drifted.columns.get_loc("prediction")] += 2 * ATOL
    with pytest.raises(FeatureParityError, match="prediction"):
        compare_feature_parity(reference, drifted, atol=ATOL)

    with pytest.raises(FeatureParityError, match="absent"):
        compare_feature_parity(reference.iloc[:-1], mql5, atol=ATOL)

    text = (tmp_path / "mql5_features.csv").read_text(encoding="ascii")
    (tmp_path / "mql5_features.csv").write_text(text.replace("z_rsi", "z_RSI", 1), encoding="ascii")
    with pytest.raises(FeatureParityError, match="header"):
        load_mql5_features(tmp_path / "mql5_features.csv")


def test_reference_rejects_tampered_onnx(exported, tmp_path):
    directory, manifest, bars, _, _ = exported
    tampered = tmp_path / "xgb_alpha.onnx"
    tampered.write_bytes((directory / "xgb_alpha.onnx").read_bytes() + b"\0")
    with pytest.raises(OnnxExportError, match="hash mismatch"):
        build_python_reference(bars, manifest, tampered)


@pytest.mark.skipif(
    not os.getenv("RLBOT_MT5_PARITY_DIR"),
    reason="RLBOT_MT5_PARITY_DIR unset: real MT5 parity NOT verified (skip is not evidence)",
)
def test_real_mt5_dump_matches_python_within_1e_4():
    directory = Path(os.environ["RLBOT_MT5_PARITY_DIR"])
    report = run_parity_gate(directory, atol=ATOL)
    minimum = int(os.getenv("RLBOT_MT5_PARITY_MIN_ROWS", "1000"))
    assert report.rows >= minimum, f"only {report.rows} MQL5 rows compared; need >= {minimum}"
    print(json.dumps({k: v for k, v in report.max_abs_difference.items()}, indent=2))
