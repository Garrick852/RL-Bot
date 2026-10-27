"""Persistent, CPU-bounded Optuna search over the offline multi-seed pipeline.

The reused test partition is optimization data, not an untouched acceptance
holdout. Exported parameters require a separately frozen fresh-holdout run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import statistics
import uuid
from pathlib import Path
from typing import Any, Sequence

import optuna
from optuna.trial import FrozenTrial, TrialState

from governance.validator import ConfigValidator, ValidationState
from orchestrator import (
    OfflineOrchestrator, _atomic_json, _local_path, _reject_constant, _unique_object,
    _worker_initializer,
)

LOGGER = logging.getLogger(__name__)
STORAGE_URL = "sqlite:///reports/optuna_vps_study.db"
STUDY_NAME = "offline_ppo_vps"
SAMPLER_SEED = 852
SEARCH_SPACE = {
    "version": 2, "metric": "median_profit_factor", "sampler_seed": SAMPLER_SEED,
    "learning_rate": [1e-5, 1e-3], "ent_coef": [1e-4, 3e-2],
    "batch_size": [64, 128, 256, 512],
}


class OfflineOptimizer:
    """Wrap the real offline pipeline without external operational side effects.

    Args:
        config_path: Complete frozen experiment config, not the repository's
            unconfigured active-config placeholder. Relative data paths are
            resolved against this config before making trial copies.
        max_workers: Upper bound on parallel seed workers; trials themselves
            are always sequential. Each seed retains one numerical thread.
        study_name: Persistent study identity within the fixed local database.
        candidate_path: Best sampled parameter dictionary, not an active config.
    """

    def __init__(
        self,
        config_path: str | Path = "config/active_config.json",
        *,
        max_workers: int = 1,
        study_name: str = STUDY_NAME,
        candidate_path: str | Path = "config/best_candidate_config.json",
    ) -> None:
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if not isinstance(study_name, str) or not study_name.strip():
            raise ValueError("study_name must be nonempty")
        source = _local_path(str(config_path), Path.cwd()).resolve()
        raw = source.read_bytes()
        base: Any = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        if not isinstance(base, dict):
            raise ValueError("base experiment must be a JSON object")
        validator = ConfigValidator(base, base_directory=source.parent)
        if validator.validate() != ValidationState.READY_FOR_OFFLINE_EXECUTION:
            raise ValueError(
                f"base experiment refused: errors={validator.errors}; "
                f"unresolved={validator.unresolved_fields}"
            )
        preflight = OfflineOrchestrator()
        preflight.config = base
        preflight._validate_profile()
        self.base_config = copy.deepcopy(base)
        self.base_config["dataset"]["path"] = str(
            _local_path(base["dataset"]["path"], source.parent).resolve()
        )
        self.base_config["features"]["alpha_manifest_path"] = str(
            _local_path(base["features"]["alpha_manifest_path"], source.parent).resolve()
        )
        # Operational configuration is irrelevant to a search candidate.
        self.base_config.pop("deployment", None)
        self.base_config["orchestration"]["max_workers"] = min(
            max_workers, base["orchestration"]["max_workers"],
        )
        self.base_config["optimization"] = {"role": "SEARCH_ONLY_REUSED_HOLDOUT"}
        self.study_name = study_name
        self.candidate_path = _local_path(str(candidate_path), Path.cwd()).resolve()
        if self.candidate_path == source:
            raise ValueError("candidate export must not overwrite the base experiment")
        self.reports = Path("reports").resolve()
        self.contract = {
            "base_config_sha256": hashlib.sha256(raw).hexdigest(),
            "search_space": SEARCH_SPACE,
            "seed_workers": self.base_config["orchestration"]["max_workers"],
        }
        self._exported_best: int | None = None

    def objective(self, trial: optuna.Trial) -> float:
        """Run every seed with sampled settings; prune invalid/undefined evidence.

        Threshold failure alone is not pruning: low but defined performance is
        useful to TPE. No failed seed is discarded or substituted.
        """
        try:
            config = copy.deepcopy(self.base_config)
            parameters = config["training"]["ppo_hyperparameters"]
            parameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            parameters["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 3e-2, log=True)
            parameters["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            # Alpha windows are frozen in the hash-locked manifest; EMAs only drive the baseline.
            directory = self.reports / "optuna_trials" / f"trial-{trial.number}-{uuid.uuid4().hex}"
            config_path = directory / "trial_config.json"
            _atomic_json(config_path, config)
            trial.set_user_attr("trial_config_path", str(config_path))
            trial.set_user_attr("requires_fresh_holdout", True)
            report = OfflineOrchestrator(directory).run_pipeline(config_path, deploy=False)
            trial.set_user_attr("audit_report_path", str(directory / "offline_audit_report.json"))
            if (
                report["run_status"] != "COMPLETE"
                or report["validation_state"] != ValidationState.READY_FOR_OFFLINE_EXECUTION.value
                or report["rl_training_performed"] is not True
                or report["scope"] != "HYPERPARAMETER_SEARCH_ONLY"
            ):
                raise ValueError(f"trial pipeline refused/failed: {report.get('errors', [])}")
            seeds = config["training"]["seeds"]
            rows = report["results"]
            if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
                raise ValueError("incomplete or duplicate multi-seed evidence")
            values = [row["metrics"]["profit_factor"] for row in rows]
            if any(
                row["status"] != "COMPLETE"
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for row, value in zip(rows, values)
            ):
                raise ValueError("every seed requires a complete, finite Profit Factor")
            median = statistics.median(values)
            reported = report["median_oos_metrics"]["profit_factor"]
            if type(reported) not in (int, float) or not math.isfinite(reported) or not math.isclose(
                median, reported, rel_tol=0, abs_tol=1e-12,
            ):
                raise ValueError("reported median does not match every configured seed")
            trial.set_user_attr("eligible_objective", True)
            return float(median)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Optuna trial %s pruned: %s", trial.number, reason)
            trial.set_user_attr("failure_reason", reason)
            raise optuna.TrialPruned(reason) from exc

    def export_best(self, study: optuna.Study, _trial: FrozenTrial | None = None) -> None:
        """Atomically export only a new, eligible best parameter dictionary.

        Called once at resume as well, so an interrupted/missing export is
        repaired from committed study evidence before new trials begin.
        """
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not complete:
            return
        best = study.best_trial
        if best.user_attrs.get("eligible_objective") is not True:
            raise ValueError("best trial lacks verified multi-seed search evidence")
        if best.number == self._exported_best:
            return
        _atomic_json(self.candidate_path, best.params)
        self._exported_best = best.number
        LOGGER.info("Exported search high-water mark trial=%s median_PF=%s", best.number, best.value)

    def optimize(self, n_trials: int) -> optuna.Study:
        """Resume a contract-bound SQLite study, running one trial at a time.

        An exclusive process lock prevents competing VPS invocations. Stale
        locks require operator inspection; they are never silently removed.
        Sampler seeding stabilizes fresh serial searches, not the exact RNG
        continuation across process restarts.
        """
        if type(n_trials) is not int or n_trials < 1:
            raise ValueError("n_trials must be a positive integer")
        self.reports.mkdir(parents=True, exist_ok=True)
        # Spawn inherits these limits before importing this module/Optuna.
        _worker_initializer()
        lock = self.reports / "optuna_vps_study.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.close(descriptor)
            study = optuna.create_study(
                study_name=self.study_name, storage=STORAGE_URL,
                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
                direction="maximize", load_if_exists=True,
            )
            prior = study.user_attrs.get("search_contract")
            if prior is None:
                if study.get_trials(deepcopy=False):
                    raise ValueError("cannot attach an unbound existing study with trials")
                study.set_user_attr("search_contract", self.contract)
            elif prior != self.contract:
                raise ValueError("study config/search-space/worker contract changed; use a new study name")
            if study.direction != optuna.study.StudyDirection.MAXIMIZE:
                raise ValueError("study must maximize median Profit Factor")
            self._exported_best = None
            self.export_best(study)
            study.optimize(self.objective, n_trials=n_trials, n_jobs=1, callbacks=[self.export_best])
            return study
        finally:
            lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Optimize an explicitly frozen local experiment; no operational mode exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", nargs="?", type=Path, default=Path("config/active_config.json"))
    parser.add_argument("--n-trials", type=int, required=True, help="Additional trials on this invocation")
    parser.add_argument("--max-workers", type=int, default=1, help="Seed-worker cap; trials are serial")
    parser.add_argument("--study-name", default=STUDY_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        study = OfflineOptimizer(
            args.config_path, max_workers=args.max_workers, study_name=args.study_name,
        ).optimize(args.n_trials)
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not complete:
            LOGGER.error("No eligible completed trials; no best candidate exported")
            return 1
        print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))
        return 0
    except Exception:
        LOGGER.exception("Offline optimization halted")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
