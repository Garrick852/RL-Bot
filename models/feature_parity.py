"""Zero-drift comparison between Python features/ONNX and the MQL5 dumper.

Both sides consume the *same* closed-bar history: ``validate_features.mq5``
writes every bar it used (warm-up included) to ``mql5_bars.csv`` and its
computed rows to ``mql5_features.csv``. Python rebuilds features from that bar
file, so EMA seeds and rolling windows start on identical observations.
MT5 bar times are broker-server clock values; both files are labeled UTC
purely for exact alignment, not as a claim about the true exchange timezone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.collector import DataIntegrityError
from data.feature_engineer import TabularFeatureEngineer
from models.onnx_bridge import OnnxAlphaRuntime, ZScoreParameters, load_manifest

MT5_TIME_FORMAT = "%Y.%m.%d %H:%M:%S"
FEATURE_NAMES = TabularFeatureEngineer.FEATURE_NAMES
RAW_COLUMNS = tuple(f"raw_{name}" for name in FEATURE_NAMES)
SCALED_COLUMNS = tuple(f"z_{name}" for name in FEATURE_NAMES)
PARITY_COLUMNS = RAW_COLUMNS + SCALED_COLUMNS + ("prediction",)


class FeatureParityError(AssertionError):
    """Python and MQL5 feature/prediction streams diverged or cannot align."""


@dataclass(frozen=True)
class ParityReport:
    """Successful comparison summary."""

    rows: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    max_abs_difference: dict[str, float]


def _parse_mt5_timestamps(values: pd.Series) -> pd.DatetimeIndex:
    try:
        parsed = pd.to_datetime(values, format=MT5_TIME_FORMAT, utc=True)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"MT5 timestamps must use {MT5_TIME_FORMAT!r}: {exc}") from exc
    index = pd.DatetimeIndex(parsed, name="timestamp")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise DataIntegrityError("MT5 timestamps must be non-null, unique, and increasing")
    return index


def load_mt5_bars(path: str | Path) -> pd.DataFrame:
    """Load ``mql5_bars.csv`` into a frame accepted by ``TabularFeatureEngineer``."""
    frame = pd.read_csv(path, dtype={"timestamp": str})
    required = ["timestamp", "open", "high", "low", "close"]
    if list(frame.columns) != required:
        raise DataIntegrityError(f"bar file columns must be exactly {required}")
    timestamps = _parse_mt5_timestamps(frame["timestamp"])
    bars = frame.loc[:, ["open", "high", "low", "close"]].astype(np.float64)
    bars.insert(0, "timestamp", timestamps)
    return bars.reset_index(drop=True)


def load_mql5_features(path: str | Path) -> pd.DataFrame:
    """Load ``mql5_features.csv`` with an exact header and finite values."""
    frame = pd.read_csv(path, dtype={"timestamp": str})
    expected = ["timestamp", *PARITY_COLUMNS]
    if list(frame.columns) != expected:
        raise FeatureParityError(f"MQL5 feature header mismatch; expected {expected}")
    index = _parse_mt5_timestamps(frame["timestamp"])
    values = frame.loc[:, list(PARITY_COLUMNS)].astype(np.float64)
    values.index = index
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise FeatureParityError("MQL5 feature file must be nonempty and fully finite")
    return values


def build_python_reference(
    bars: pd.DataFrame,
    manifest: Mapping[str, Any],
    onnx_path: str | Path,
) -> pd.DataFrame:
    """Compute raw features, Z-scores, and ONNX predictions from bars in Python.

    Rows still in indicator warm-up are dropped. The ONNX file must match the
    manifest hash before inference.
    """
    engineer = TabularFeatureEngineer(**manifest["feature_engineer"])
    raw = engineer.generate_features(bars).dropna()
    zscore_spec = manifest["zscore"]
    zscore = ZScoreParameters(
        feature_names=tuple(zscore_spec["feature_names"]),
        mean=tuple(zscore_spec["mean"]),
        std=tuple(zscore_spec["std"]),
    )
    scaled = zscore.transform(raw).to_numpy(dtype=np.float64)
    runtime = OnnxAlphaRuntime(onnx_path, expected_sha256=manifest["onnx_sha256"])
    prediction = runtime.predict(scaled.astype(np.float32)).astype(np.float64)
    reference = pd.DataFrame(raw.to_numpy(dtype=np.float64), index=raw.index, columns=list(RAW_COLUMNS))
    reference[list(SCALED_COLUMNS)] = scaled
    reference["prediction"] = prediction
    reference.index.name = "timestamp"
    return reference


def compare_feature_parity(
    python_frame: pd.DataFrame,
    mql5_frame: pd.DataFrame,
    *,
    atol: float = 1e-4,
) -> ParityReport:
    """Require every MQL5 row to exist in Python and match within ``atol``.

    Raises:
        FeatureParityError: Missing timestamps, column drift, or any value
            outside ``numpy.allclose(..., rtol=0, atol=atol)``.
    """
    for label, frame in (("python", python_frame), ("mql5", mql5_frame)):
        if list(frame.columns) != list(PARITY_COLUMNS):
            raise FeatureParityError(f"{label} columns must be exactly {list(PARITY_COLUMNS)}")
        if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise FeatureParityError(f"{label} frame must be nonempty, unique, and chronological")
    missing = mql5_frame.index.difference(python_frame.index)
    if len(missing):
        raise FeatureParityError(
            f"{len(missing)} MQL5 timestamps absent from valid Python rows; first={missing[0]}"
        )
    aligned = python_frame.loc[mql5_frame.index]
    differences: dict[str, float] = {}
    failures: list[str] = []
    for column in PARITY_COLUMNS:
        left = aligned[column].to_numpy(dtype=np.float64)
        right = mql5_frame[column].to_numpy(dtype=np.float64)
        delta = np.abs(left - right)
        differences[column] = float(np.nanmax(delta)) if delta.size else 0.0
        if not np.allclose(left, right, rtol=0.0, atol=atol):
            worst = int(np.nanargmax(delta)) if np.isfinite(delta).any() else 0
            failures.append(
                f"{column}: max|diff|={differences[column]:.3e} at {mql5_frame.index[worst]} "
                f"(python={left[worst]!r}, mql5={right[worst]!r})"
            )
    if failures:
        raise FeatureParityError("feature parity drift > atol=%g: %s" % (atol, "; ".join(failures)))
    return ParityReport(
        rows=len(mql5_frame),
        first_timestamp=mql5_frame.index[0],
        last_timestamp=mql5_frame.index[-1],
        max_abs_difference=differences,
    )


def run_parity_gate(artifact_dir: str | Path, *, atol: float = 1e-4) -> ParityReport:
    """Validate a directory holding manifest, ONNX, bar dump, and feature dump."""
    directory = Path(artifact_dir)
    manifest_files = sorted(directory.glob("*.manifest.json"))
    if len(manifest_files) != 1:
        raise FeatureParityError(f"expected exactly one *.manifest.json in {directory}")
    manifest = load_manifest(manifest_files[0])
    bars = load_mt5_bars(directory / "mql5_bars.csv")
    reference = build_python_reference(bars, manifest, directory / manifest["onnx_file"])
    return compare_feature_parity(reference, load_mql5_features(directory / "mql5_features.csv"), atol=atol)
