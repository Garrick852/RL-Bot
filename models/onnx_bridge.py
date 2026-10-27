"""Offline ONNX export of the XGBoost alpha model for MQL5 parity research.

The exported graph is the unchanged tree ensemble of an ``XGBAlphaModel``
trained with ``standardize=True``: its native input is the float32 Z-scored
feature vector in exact ``TabularFeatureEngineer.FEATURE_NAMES`` order, and its
output is one float32 forward-return prediction. Standardization is NOT folded
into the graph because float32 de-scaling flips tree splits near thresholds.
The training-partition mean/std ship in a manifest and an MQL5-readable CSV.
Nothing here deploys or wires a live EA.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.collector import DataIntegrityError
from data.feature_engineer import TabularFeatureEngineer
from models.zscore import ZSCORE_DDOF, ZScoreParameters, fit_zscore

ONNX_INPUT_NAME = "features"
ONNX_OUTPUT_NAME = "prediction"
ONNX_DEFAULT_OPSET = 13
ONNX_ML_OPSET = 1
ONNX_IR_VERSION = 8
MANIFEST_CONTRACT = "xgb_alpha_onnx_zscore_v1"

__all__ = [
    "OnnxAlphaRuntime", "OnnxExportError", "ZScoreParameters", "export_xgb_alpha_onnx",
    "feature_engineer_params", "fit_zscore", "load_manifest", "mql5_zscore_inputs", "sha256_file",
]


class OnnxExportError(RuntimeError):
    """The ONNX artifact cannot be produced or proven equivalent."""


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _build_graph(booster: Any, feature_count: int) -> Any:
    import onnx
    from onnx import TensorProto, helper
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    anonymous = copy.copy(booster)
    # onnxmltools only accepts the anonymous f0..fN naming; order is verified by the caller.
    anonymous.feature_names = None
    trees = convert_xgboost(
        anonymous,
        initial_types=[(ONNX_INPUT_NAME, FloatTensorType([None, feature_count]))],
        target_opset=ONNX_DEFAULT_OPSET,
    )
    if len(trees.graph.input) != 1 or len(trees.graph.output) != 1:
        raise OnnxExportError("converted XGBoost graph must have exactly one input and output")
    graph = helper.make_graph(
        [*trees.graph.node,
         helper.make_node("Identity", [trees.graph.output[0].name], [ONNX_OUTPUT_NAME],
                          name="prediction_identity")],
        "xgb_alpha_zscore_input",
        [helper.make_tensor_value_info(ONNX_INPUT_NAME, TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info(ONNX_OUTPUT_NAME, TensorProto.FLOAT, [None, 1])],
        initializer=list(trees.graph.initializer),
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", ONNX_DEFAULT_OPSET),
            helper.make_opsetid("ai.onnx.ml", ONNX_ML_OPSET),
        ],
        producer_name="rl-bot-offline-xgb-alpha",
    )
    model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


class OnnxAlphaRuntime:
    """Hash-verified ONNX Runtime session for offline reference predictions."""

    def __init__(self, onnx_path: str | Path, *, expected_sha256: str | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(onnx_path)
        actual = sha256_file(self.path)
        if expected_sha256 is not None and actual != expected_sha256:
            raise OnnxExportError(
                f"ONNX artifact hash mismatch: expected {expected_sha256}, got {actual}"
            )
        self.sha256 = actual
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"],
        )

    def predict(self, scaled: np.ndarray) -> np.ndarray:
        """Run float32 scaled rows and return one float32 prediction per row."""
        matrix = np.asarray(scaled, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise DataIntegrityError("ONNX inputs must be a finite 2-D matrix")
        output = self.session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: matrix})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)


def export_xgb_alpha_onnx(
    model: Any,
    X_train: pd.DataFrame,
    *,
    onnx_path: str | Path = "reports/xgb_alpha.onnx",
    manifest_path: str | Path | None = None,
    zscore_csv_path: str | Path | None = None,
    feature_engineer: TabularFeatureEngineer | None = None,
    equivalence_atol: float = 1e-6,
) -> dict[str, Any]:
    """Export a fitted ``XGBAlphaModel`` to Z-score-input ONNX plus sidecars.

    Args:
        model: ``XGBAlphaModel(standardize=True)`` after ``fit_walk_forward``.
        X_train: The exact certified training partition passed to
            ``fit_walk_forward``; its Z-score must equal ``model.zscore``.
        onnx_path: ONNX destination.
        manifest_path: JSON manifest destination; defaults beside the ONNX file.
        zscore_csv_path: MQL5-readable ``feature,mean,std`` CSV; defaults beside
            the ONNX file.
        feature_engineer: Engineer whose windows produced ``X_train``; recorded
            so parity references are rebuilt with identical windows.
        equivalence_atol: Max allowed |XGBoost - ONNX| on the scaled X_train.

    Returns:
        The persisted manifest dictionary.

    Raises:
        OnnxExportError: Unfitted model, feature-order drift, or non-equivalence.
    """
    booster_model = getattr(model, "model", None)
    zscore = getattr(model, "zscore", None)
    if booster_model is None:
        raise OnnxExportError("XGBAlphaModel must be fitted before ONNX export")
    if not isinstance(zscore, ZScoreParameters):
        raise OnnxExportError("ONNX export requires XGBAlphaModel(standardize=True)")
    engineer = feature_engineer or TabularFeatureEngineer()
    if X_train.attrs.get("max_moving_average_window") != engineer.max_moving_average_window:
        raise OnnxExportError("feature_engineer windows do not match the certified X_train")
    if fit_zscore(X_train) != zscore:
        raise OnnxExportError("X_train is not the partition the model's Z-score was fitted on")
    horizon = getattr(model, "horizon", None)
    if not isinstance(X_train.index, pd.DatetimeIndex) or X_train.index.tz is None:
        raise OnnxExportError("X_train must keep its tz-aware timestamp index for leakage evidence")
    if type(horizon) is not int or horizon < 1:
        raise OnnxExportError("model must declare its positive integer label horizon")
    booster = booster_model.get_booster()
    if tuple(booster.feature_names or ()) != zscore.feature_names:
        raise OnnxExportError(
            f"booster feature order {booster.feature_names} != {zscore.feature_names}"
        )
    onnx_model = _build_graph(booster, len(zscore.feature_names))

    destination = Path(onnx_path)
    if destination.suffix.lower() != ".onnx":
        raise ValueError("ONNX export destination must use .onnx")
    manifest_destination = Path(manifest_path) if manifest_path else destination.with_suffix(".manifest.json")
    zscore_destination = Path(zscore_csv_path) if zscore_csv_path else destination.with_name(
        f"{destination.stem}_zscore.csv"
    )
    payload = onnx_model.SerializeToString()
    _atomic_write_bytes(destination, payload)
    digest = hashlib.sha256(payload).hexdigest()

    runtime = OnnxAlphaRuntime(destination, expected_sha256=digest)
    scaled = zscore.transform(X_train)
    onnx_prediction = runtime.predict(scaled.to_numpy()).astype(np.float64)
    xgb_prediction = np.asarray(booster_model.predict(scaled), dtype=np.float64)
    max_difference = float(np.max(np.abs(onnx_prediction - xgb_prediction)))
    if not np.isfinite(max_difference) or max_difference > equivalence_atol:
        destination.unlink(missing_ok=True)
        raise OnnxExportError(
            f"ONNX/XGBoost equivalence failed: max |diff|={max_difference:.3e} > {equivalence_atol:.1e}"
        )

    import onnxruntime
    import xgboost

    lines = ["feature,mean,std"] + [
        f"{name},{mean!r},{std!r}"
        for name, mean, std in zip(zscore.feature_names, zscore.mean, zscore.std)
    ]
    # CRLF: MQL5 FILE_TXT FileReadString splits lines on "\r\n".
    _atomic_write_bytes(zscore_destination, ("\r\n".join(lines) + "\r\n").encode("ascii"))
    manifest = {
        "contract": MANIFEST_CONTRACT,
        "onnx_file": destination.name,
        "onnx_sha256": digest,
        "zscore_csv_file": zscore_destination.name,
        "zscore_csv_sha256": sha256_file(zscore_destination),
        "input": {"name": ONNX_INPUT_NAME, "dtype": "float32", "shape": [None, len(zscore.feature_names)]},
        "output": {"name": ONNX_OUTPUT_NAME, "dtype": "float32", "shape": [None, 1]},
        "opset": {"ai.onnx": ONNX_DEFAULT_OPSET, "ai.onnx.ml": ONNX_ML_OPSET},
        "ir_version": ONNX_IR_VERSION,
        "zscore": {**asdict(zscore), "ddof": ZSCORE_DDOF, "source": "training_partition"},
        "feature_engineer": feature_engineer_params(engineer),
        "training_rows": int(len(X_train)),
        # Consumers must keep RL partitions after the last label bar (last + horizon bars).
        "training_window": {
            "first_feature_timestamp": X_train.index[0].tz_convert("UTC").isoformat(),
            "last_feature_timestamp": X_train.index[-1].tz_convert("UTC").isoformat(),
            "label_horizon_bars": horizon,
        },
        "xgb_onnx_max_abs_difference": max_difference,
        "xgb_onnx_equivalence_atol": equivalence_atol,
        "versions": {"xgboost": xgboost.__version__, "onnxruntime": onnxruntime.__version__},
        "deployment": "OFFLINE_PARITY_ONLY_NOT_WIRED_TO_EA",
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(manifest_destination, serialized.encode("utf-8"))
    return json.loads(serialized)


def feature_engineer_params(engineer: TabularFeatureEngineer) -> dict[str, int]:
    """Return the constructor windows needed to rebuild an identical engineer."""
    names: Sequence[str] = (
        "rsi_window", "macd_fast", "macd_slow", "macd_signal",
        "atr_window", "bollinger_window", "price_action_window",
    )
    return {name: int(getattr(engineer, name)) for name in names}


def mql5_zscore_inputs(manifest: Mapping[str, Any]) -> dict[str, float]:
    """Map manifest Z-score vectors to EA_Template ``InpAlphaMean<i>``/``InpAlphaStd<i>`` inputs.

    The result is a scalar parameter mapping accepted by ``MQL5Patcher.patch_source``.
    """
    zscore = manifest["zscore"]
    parameters = ZScoreParameters(
        feature_names=tuple(zscore["feature_names"]),
        mean=tuple(zscore["mean"]),
        std=tuple(zscore["std"]),
    )
    inputs: dict[str, float] = {}
    for index, (mean, std) in enumerate(zip(parameters.mean, parameters.std)):
        inputs[f"InpAlphaMean{index}"] = float(mean)
        inputs[f"InpAlphaStd{index}"] = float(std)
    return inputs


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate an export manifest."""
    manifest: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("contract") != MANIFEST_CONTRACT:
        raise OnnxExportError("unsupported ONNX manifest contract")
    zscore = manifest.get("zscore") or {}
    ZScoreParameters(
        feature_names=tuple(zscore.get("feature_names", ())),
        mean=tuple(zscore.get("mean", ())),
        std=tuple(zscore.get("std", ())),
    )
    return dict(manifest)
