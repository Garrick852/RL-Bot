"""Write an accepted single-shot holdout audit + burn flag for ``--deploy`` gate tests."""

import hashlib
import json
from pathlib import Path


def write_accepted_holdout(root: Path, seed_models: dict[int, Path], alpha: dict) -> Path:
    """Persist ``reports/true_holdout_eval.json`` and its burn flag under ``root``.

    Args:
        root: Repository root every recorded path is relative to.
        seed_models: Seed -> frozen model (must live under ``root``).
        alpha: ``{"manifest_sha256", "onnx_sha256"}`` accepted with the policy.
    """
    root = Path(root)
    dataset_sha, candidate_sha = "d" * 64, "c" * 64
    flag = root / "data" / "holdout_used.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"dataset_sha256": dataset_sha, "candidate_sha256": candidate_sha}))
    report = {
        "contract": "true_holdout_eval_v1", "median_oos": 1.4, "reconciliation_ok": True, "gap_ok": True,
        "evaluated_once": True, "accepted": True,
        "theta_accept": {"metric": "median_profit_factor", "value": 1.25},
        "holdout": {"dataset_sha256": dataset_sha, "burn_flag": "data/holdout_used.flag"},
        "candidate": {"path": "config/best_candidate_config.json", "sha256": candidate_sha,
                      "alpha_manifest_sha256": alpha["manifest_sha256"], "alpha_onnx_sha256": alpha["onnx_sha256"]},
        "seeds": [{"seed": seed, "status": "COMPLETE",
                   "model_path": Path(path).resolve().relative_to(root.resolve()).as_posix(),
                   "model_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                  for seed, path in seed_models.items()],
    }
    path = root / "reports" / "true_holdout_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return path
