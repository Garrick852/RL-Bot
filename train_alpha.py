"""Train and export the causal XGBoost alpha consumed by the offline RL pipeline.

Offline research step 4A: fit ``XGBAlphaModel`` with purged walk-forward folds on
the rows strictly before the RL train split of the *same* hash-locked dataset,
export the Z-score-input ONNX graph plus manifest, and prove the export loads
through the orchestrator's own verifier. No deployment, network, or MT5 access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from data.collector import DataCollector
from data.feature_engineer import TabularFeatureEngineer
from engine_rl.alpha_pipeline import alpha_label_end, load_alpha_model
from models.onnx_bridge import export_xgb_alpha_onnx
from models.xgb_alpha import XGBAlphaModel
from orchestrator import _atomic_json

LOGGER = logging.getLogger(__name__)
REPORT_CONTRACT = "xgb_alpha_training_report_v1"
ONNX_NAME = "xgb_alpha.onnx"
MANIFEST_NAME = "xgb_alpha.manifest.json"
ZSCORE_NAME = "xgb_alpha_zscore.csv"
REPORT_NAME = "train_alpha.report.json"
ARTIFACTS = (ONNX_NAME, MANIFEST_NAME, ZSCORE_NAME, REPORT_NAME)
DEFAULT_MODEL_PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}


class AlphaTrainingRefused(RuntimeError):
    """The alpha cannot be trained or exported without violating a guardrail."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlphaTrainingRefused(message)


def _utc(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AlphaTrainingRefused(f"{name} is not a timestamp: {value!r}") from exc
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def train_alpha(
    dataset_path: str | Path,
    dataset_sha256: str,
    rl_train_start: str | pd.Timestamp,
    output_dir: str | Path,
    *,
    horizon: int,
    alpha_start: Optional[str | pd.Timestamp] = None,
    windows: Optional[dict[str, int]] = None,
    n_splits: int = 5,
    random_state: int = 0,
    model_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Fit, export, and self-verify the alpha; return the persisted training report.

    Args:
        dataset_path: Local MT5-derived CSV that the RL experiment config also uses.
        dataset_sha256: Exact lowercase SHA-256 of ``dataset_path``.
        rl_train_start: Timezone-aware RL ``splits.train.start``. Alpha rows and
            every label bar they consume must lie strictly before it.
        output_dir: Destination directory; existing alpha artifacts are refused.
        horizon: Forward-return label horizon in bars.
        alpha_start: Optional tz-aware inclusive start of the alpha partition.
        windows: ``TabularFeatureEngineer`` window overrides.
        n_splits: Purged ``TimeSeriesSplit`` folds.
        random_state: XGBoost seed.
        model_params: XGBRegressor parameters (governed fields are refused).

    Returns:
        The training report, including the config snippet for ``features``.

    Raises:
        AlphaTrainingRefused: Any integrity, leakage, or export guardrail fails.
    """
    _require(isinstance(dataset_sha256, str) and len(dataset_sha256) == 64
             and all(char in "0123456789abcdef" for char in dataset_sha256),
             "dataset_sha256 must be a lowercase SHA-256")
    destination = Path(output_dir)
    existing = [name for name in ARTIFACTS if (destination / name).exists()]
    _require(not existing, f"refusing to overwrite existing alpha artifacts: {existing}")
    barrier = _utc(rl_train_start, "rl_train_start")
    lower = _utc(alpha_start, "alpha_start") if alpha_start is not None else None

    try:
        frame = DataCollector().load_csv(dataset_path, expected_sha256=dataset_sha256)
        stamps = pd.DatetimeIndex(frame["timestamp"])
        _require(stamps.is_unique and stamps.is_monotonic_increasing,
                 "dataset timestamps must be unique and chronological")
        _require(bool((stamps >= barrier).any()), "dataset has no rows at or after rl_train_start")
        mask = stamps < barrier
        if lower is not None:
            _require(lower < barrier, "alpha_start must precede rl_train_start")
            mask &= stamps >= lower
        partition = frame.loc[mask].reset_index(drop=True)
        _require(not partition.empty, "no dataset rows fall inside the alpha partition")

        engineer = TabularFeatureEngineer(**(windows or {}))
        # Labels come from the partition only, so no label bar can reach the RL split.
        X, y = engineer.align_and_clean(
            engineer.generate_features(partition), engineer.generate_labels(partition, horizon),
        )
        _require(len(X) > 0, "no complete feature/label rows; extend the alpha partition")
        model = XGBAlphaModel(
            horizon=horizon, max_moving_average_window=engineer.max_moving_average_window,
            n_splits=n_splits, random_state=random_state,
            model_params=dict(DEFAULT_MODEL_PARAMS if model_params is None else model_params),
            standardize=True,
        )
        folds = model.fit_walk_forward(X, y)
    except AlphaTrainingRefused:
        raise
    except (ValueError, RuntimeError, AssertionError) as exc:
        raise AlphaTrainingRefused(f"alpha training failed: {exc}") from exc

    destination.mkdir(parents=True, exist_ok=True)
    created = [destination / name for name in ARTIFACTS]
    try:
        manifest = export_xgb_alpha_onnx(
            model, X, onnx_path=destination / ONNX_NAME, manifest_path=destination / MANIFEST_NAME,
            zscore_csv_path=destination / ZSCORE_NAME, feature_engineer=engineer,
        )
        manifest_sha = hashlib.sha256((destination / MANIFEST_NAME).read_bytes()).hexdigest()
        # Round-trip through the orchestrator's verifier: same checks the RL run applies.
        alpha = load_alpha_model(destination / MANIFEST_NAME, manifest_sha)
        label_end = alpha_label_end(frame["timestamp"], alpha)
        _require(label_end < barrier,
                 f"alpha label end {label_end.isoformat()} is not before rl_train_start")
        report = {
            "contract": REPORT_CONTRACT,
            "dataset": {"path": str(dataset_path), "sha256": dataset_sha256, "rows": int(len(frame))},
            "rl_train_start": barrier.isoformat(),
            "alpha_partition": {
                "start": partition["timestamp"].iloc[0].isoformat(),
                "end_exclusive": barrier.isoformat(),
                "rows": int(len(partition)),
                "training_rows": int(len(X)),
                "label_end": label_end.isoformat(),
            },
            "horizon": horizon,
            "feature_engineer": manifest["feature_engineer"],
            "model": {"random_state": random_state, "n_splits": n_splits, "purge_gap": model.gap,
                      "params": model.model_params},
            "walk_forward_oos": [asdict(fold) for fold in folds],
            "artifacts": {"manifest_sha256": manifest_sha, "onnx_sha256": alpha.onnx_sha256,
                          "zscore_csv_sha256": manifest["zscore_csv_sha256"]},
            "features_config": {
                "alpha_manifest_path": str((destination / MANIFEST_NAME).resolve()),
                "alpha_manifest_sha256": manifest_sha,
            },
            "scope": "OFFLINE_RESEARCH_ONLY",
        }
        _atomic_json(destination / REPORT_NAME, report)
    except Exception as exc:
        for path in created:
            path.unlink(missing_ok=True)
        if isinstance(exc, AlphaTrainingRefused):
            raise
        raise AlphaTrainingRefused(f"alpha export failed verification: {exc}") from exc
    LOGGER.info("Alpha exported: manifest=%s sha256=%s", destination / MANIFEST_NAME, manifest_sha)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; exit 0 only when artifacts are exported and verified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Hash-locked dataset CSV shared with the RL config")
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--rl-train-start", required=True, help="tz-aware RL splits.train.start")
    parser.add_argument("--alpha-start", default=None, help="Optional tz-aware alpha partition start")
    parser.add_argument("--horizon", type=int, required=True, help="Forward-return label horizon (bars)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/alpha"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_MODEL_PARAMS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MODEL_PARAMS["max_depth"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_MODEL_PARAMS["learning_rate"])
    defaults = TabularFeatureEngineer()
    window_names = ("rsi_window", "macd_fast", "macd_slow", "macd_signal",
                    "atr_window", "bollinger_window", "price_action_window")
    for name in window_names:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        report = train_alpha(
            args.dataset, args.dataset_sha256, args.rl_train_start, args.output_dir,
            horizon=args.horizon, alpha_start=args.alpha_start,
            windows={name: getattr(args, name) for name in window_names},
            n_splits=args.n_splits, random_state=args.seed,
            model_params={"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                          "learning_rate": args.learning_rate},
        )
    except (AlphaTrainingRefused, OSError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"status": "EXPORTED", "artifacts": report["artifacts"],
                      "features_config": report["features_config"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
