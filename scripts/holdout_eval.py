"""Single-shot true-holdout evaluation of the winning PPO candidate.

Order of operations (fail-closed, anti p-hacking):

1. Verify the holdout CSV SHA-256. A mismatch raises ``ValueError``; nothing is burned.
2. Refuse with ``HoldoutAlreadyUsedError`` if ``data/holdout_used.flag`` (or a prior audit) exists.
3. Verify the candidate offline: trial config, evidence, frozen linear-actor
   models, and alpha artifacts must all match their recorded SHA-256 hashes.
   Holdout contents are not read here, so a broken candidate cannot waste the holdout.
4. Atomically create the flag (``O_EXCL``). The holdout is now permanently burned.
5. Only then load the data, check the gap policy, attach the alpha, build
   ``TradingEnvAdapter`` episodes, and evaluate every predeclared seed.
6. Persist ``reports/true_holdout_eval.json``, even when evaluation fails.

No deployment, network, or MT5 access exists in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine_rl.alpha_pipeline import AlphaModel, load_alpha_model  # noqa: E402
from orchestrator import (  # noqa: E402
    OfflineOrchestrator, _atomic_json, _json_ready, _local_path, _reject_constant, _unique_object,
)

LOGGER = logging.getLogger("holdout_eval")
FLAG_PATH = Path("data/holdout_used.flag")
REPORT_PATH = Path("reports/true_holdout_eval.json")
DEFAULT_CANDIDATE = Path("config/best_candidate_config.json")
CANDIDATE_CONTRACT = "holdout_candidate_v1"
REPORT_CONTRACT = "true_holdout_eval_v1"
ACCEPT_METRIC = "median_profit_factor"


class HoldoutAlreadyUsedError(RuntimeError):
    """The single-shot holdout has already been consumed."""


class HoldoutPreflightError(ValueError):
    """The candidate or its artifacts failed integrity checks; nothing was burned."""


@dataclass(frozen=True)
class Candidate:
    """Hash-verified candidate evidence required to evaluate the holdout."""

    path: Path
    sha256: str
    trial_number: Any
    config: dict[str, Any]
    config_sha256: str
    evidence_sha256: str
    theta_accept: float
    frozen_models: tuple[dict[str, Any], ...]
    alpha: AlphaModel


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutPreflightError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_bytes().decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )


def _utc(value: str, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HoldoutPreflightError(f"{name} is not a timestamp: {value!r}") from exc
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def verify_dataset_sha256(data_path: str | Path, expected_sha256: str) -> str:
    """Hash the holdout bytes; raise ``ValueError`` on any mismatch.

    Returns:
        The verified lowercase digest.
    """
    if not _is_sha256(expected_sha256):
        raise ValueError("--sha256 must be a lowercase 64-character SHA-256")
    actual = _sha256(Path(data_path))
    if actual != expected_sha256:
        raise ValueError(f"holdout SHA-256 mismatch: expected {expected_sha256}, actual {actual}")
    return actual


def ensure_unused(flag_path: Path, report_path: Path) -> None:
    """Refuse if the burn flag or a prior holdout audit exists."""
    if flag_path.exists():
        raise HoldoutAlreadyUsedError(f"holdout already burned: {flag_path} exists")
    if report_path.exists():
        raise HoldoutAlreadyUsedError(f"holdout audit {report_path} exists; flag was removed after use")


def burn(flag_path: Path, payload: dict[str, Any]) -> None:
    """Atomically and exclusively create the burn flag before any evaluation."""
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(flag_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as exc:
        raise HoldoutAlreadyUsedError(f"holdout already burned: {flag_path} exists") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_candidate(candidate_path: str | Path, holdout_start: pd.Timestamp) -> Candidate:
    """Verify every candidate artifact by hash; read no holdout contents.

    Raises:
        HoldoutPreflightError: Any binding, hash, architecture, or chronology check fails.
    """
    path = Path(candidate_path)
    try:
        candidate = _load_json(path)
    except (OSError, ValueError) as exc:
        raise HoldoutPreflightError(f"cannot read candidate {path}: {exc}") from exc
    keys = {"contract", "trial_number", "objective", "params", "trial_config_path", "trial_config_sha256",
            "evidence_path", "evidence_sha256", "theta_accept", "requires_fresh_holdout"}
    _require(isinstance(candidate, dict) and set(candidate) == keys
             and candidate["contract"] == CANDIDATE_CONTRACT,
             f"candidate must be an optimizer-exported {CANDIDATE_CONTRACT} with exactly {sorted(keys)}")
    theta = candidate["theta_accept"]
    _require(isinstance(theta, dict) and set(theta) == {"metric", "value"} and theta["metric"] == ACCEPT_METRIC
             and type(theta["value"]) in (int, float) and math.isfinite(theta["value"]) and theta["value"] > 0,
             f"theta_accept must be a positive finite {ACCEPT_METRIC} threshold")

    config_path = Path(candidate["trial_config_path"])
    _require(_is_sha256(candidate["trial_config_sha256"]) and config_path.is_file()
             and _sha256(config_path) == candidate["trial_config_sha256"], "trial config SHA-256 mismatch")
    config = _load_json(config_path)
    profile = OfflineOrchestrator()
    profile.config = config
    try:
        profile._validate_profile()
    except Exception as exc:
        raise HoldoutPreflightError(f"trial config profile refused: {exc}") from exc
    minimum = config["evaluation"]["research_gate_thresholds"]["min_profit_factor"]
    _require(theta["value"] >= minimum,
             f"theta_accept {theta['value']} weakens predeclared min_profit_factor {minimum}")

    evidence_path = Path(candidate["evidence_path"])
    _require(_is_sha256(candidate["evidence_sha256"]) and evidence_path.is_file()
             and _sha256(evidence_path) == candidate["evidence_sha256"], "candidate evidence SHA-256 mismatch")
    evidence = _load_json(evidence_path)
    _require(isinstance(evidence, dict) and evidence.get("run_status") == "COMPLETE"
             and evidence.get("validation_state") == "READY_FOR_OFFLINE_EXECUTION"
             and evidence.get("models_frozen_before_oos") is True
             and evidence.get("config_hash") == candidate["trial_config_sha256"],
             "evidence is not a complete frozen run of this trial config")
    seeds = config["training"]["seeds"]
    rows = evidence.get("results")
    _require(isinstance(rows, list) and sorted(row.get("seed") for row in rows) == sorted(seeds)
             and len(rows) == len(seeds), "evidence must hold exactly one result per predeclared seed")
    by_seed = {row["seed"]: row for row in rows}
    frozen = []
    for seed in seeds:
        training = by_seed[seed].get("training") or {}
        _require(training.get("seed") == seed and training.get("policy_architecture") == "linear_actor",
                 f"seed {seed} is not a linear_actor policy")
        model = Path(training.get("model_path", ""))
        _require(model.is_file() and _sha256(model) == training.get("model_sha256"),
                 f"seed {seed} frozen model SHA-256 mismatch")
        frozen.append({"seed": seed, "model_path": str(model), "model_sha256": training["model_sha256"]})

    features = config["features"]
    try:
        alpha = load_alpha_model(_local_path(features["alpha_manifest_path"], config_path.parent),
                                 features["alpha_manifest_sha256"])
    except Exception as exc:
        raise HoldoutPreflightError(f"alpha artifacts refused: {exc}") from exc
    recorded = evidence.get("alpha") or {}
    _require(recorded.get("manifest_sha256") == alpha.manifest_sha256
             and recorded.get("onnx_sha256") == alpha.onnx_sha256, "alpha differs from the evaluated candidate")

    search_end = max(pd.Timestamp(bounds["end"]) for bounds in config["splits"].values()
                     if isinstance(bounds, dict) and "end" in bounds)
    _require(holdout_start >= search_end,
             f"holdout start {holdout_start.isoformat()} overlaps search data ending {search_end.isoformat()}")
    return Candidate(path, _sha256(path), candidate["trial_number"], config, candidate["trial_config_sha256"],
                     candidate["evidence_sha256"], float(theta["value"]), tuple(frozen), alpha)


def _verify_linear(candidate: Candidate) -> None:
    """Load each frozen model on CPU and prove the actor has no hidden layers."""
    from stable_baselines3 import PPO

    from engine_rl.linear_policy import export_linear_policy
    from engine_rl.sb3_trainer import SB3Trainer

    for frozen in candidate.frozen_models:
        SB3Trainer(seed=frozen["seed"], num_threads=1)
        try:
            export_linear_policy(PPO.load(frozen["model_path"], device="cpu"))
        except Exception as exc:
            raise HoldoutPreflightError(f"seed {frozen['seed']} is not an exportable linear actor: {exc}") from exc


def _load_schedule(path: str | Path) -> pd.DatetimeIndex:
    try:
        values = _load_json(Path(path))
    except (OSError, ValueError) as exc:
        raise HoldoutPreflightError(f"cannot read RTH schedule {path}: {exc}") from exc
    _require(isinstance(values, list) and bool(values) and all(isinstance(v, str) for v in values),
             "RTH schedule must be a nonempty JSON list of ISO timestamps")
    stamps = pd.DatetimeIndex([_utc(value, "rth timestamp") for value in values])
    _require(stamps.is_unique and stamps.is_monotonic_increasing, "RTH schedule must be unique and chronological")
    return stamps


def _evaluate(
    candidate: Candidate, data_path: Path, sha256: str, start: pd.Timestamp, end: pd.Timestamp,
    schedule: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Post-burn evaluation; every failure is recorded rather than retried."""
    import numpy as np

    from data.collector import DataCollector, PollutedDatasetError
    from engine_rl.alpha_pipeline import attach_alpha
    from engine_rl.envs.trading_env import DoubleEntryReconciliationError
    from engine_rl.linear_policy import MARKET_OBSERVATION_COLUMNS
    from orchestrator import _evaluate_seed

    config = candidate.config
    outcome: dict[str, Any] = {"gap_ok": False, "reconciliation_ok": False, "median_oos": None,
                               "seeds": [], "errors": [], "partition": None, "max_consecutive_missing": None}
    try:
        frame = DataCollector().load_csv(data_path, expected_sha256=sha256)
        interval = pd.Timedelta(config["dataset"]["timeframe"])
        collector = DataCollector(expected_rth_timestamps=schedule, bar_interval=interval)
        if not (schedule[0] <= frame["timestamp"].iloc[0] and schedule[-1] >= frame["timestamp"].iloc[-1]):
            raise ValueError("RTH schedule must cover the full holdout dataset window")
        try:
            gap = collector.check_gap_policy(frame, config["orchestration"]["max_allowed_gaps"])
        except PollutedDatasetError as exc:
            outcome["errors"].append(f"PollutedDatasetError: {exc}")
            return outcome
        outcome["gap_ok"] = True
        outcome["max_consecutive_missing"] = gap.max_consecutive_missing

        run = config["orchestration"]
        history = max(config["features"]["warmup_bars"],
                      run["k_multiplier"] * config["strategy"]["ema"]["slow_period"])
        warmup, active = collector.get_split(frame, start, end, 1, history)
        combined = pd.concat([warmup, active], ignore_index=True)
        enriched = attach_alpha(combined, candidate.alpha)
        enriched = enriched.loc[enriched.index >= len(warmup)]
        if len(enriched) != len(active):
            raise ValueError("warm-up cannot produce a complete alpha observation for every holdout row")
        active = enriched.reset_index(drop=True)
        observation = active[list(MARKET_OBSERVATION_COLUMNS)].to_numpy()
        if not (np.isfinite(observation).all() and (np.abs(observation) <= np.finfo(np.float32).max).all()):
            raise ValueError("holdout observations must be finite and float32 representable")
        outcome["partition"] = {
            "rows": len(active), "warmup_rows": len(warmup),
            "first_scored": active["timestamp"].iloc[0].isoformat(),
            "last_scored": active["timestamp"].iloc[-1].isoformat(),
        }

        profile = OfflineOrchestrator()
        profile.config = config
        settings = profile._env_settings()
        annualization = config["evaluation"]["annualization_periods"]
        reconciled = True
        for frozen in candidate.frozen_models:
            try:
                result = _evaluate_seed(frozen, active, settings, annualization)
            except DoubleEntryReconciliationError as exc:
                reconciled = False
                outcome["errors"].append(f"seed {frozen['seed']} DoubleEntryReconciliationError: {exc}")
                outcome["seeds"].append({"seed": frozen["seed"], "status": "RECONCILIATION_FAILED"})
                continue
            outcome["seeds"].append({
                "seed": frozen["seed"], "status": result["status"], "model_sha256": frozen["model_sha256"],
                "metrics": result["metrics"], "undefined_reasons": result["undefined_reasons"],
                "initial_cash": result["initial_cash"], "final_cash": result["final_cash"],
                "terminal_liquidation_pnl": result["terminal_liquidation_pnl"],
            })
        outcome["reconciliation_ok"] = reconciled
        values = [row.get("metrics", {}).get("profit_factor") for row in outcome["seeds"]]
        if reconciled and all(type(value) in (int, float) and math.isfinite(value) for value in values):
            outcome["median_oos"] = float(statistics.median(values))
        elif reconciled:
            outcome["errors"].append("undefined Profit Factor on at least one seed; median_oos is null")
    except Exception as exc:
        LOGGER.exception("Holdout evaluation failed after burn")
        outcome["errors"].append(f"{type(exc).__name__}: {exc}")
    return outcome


def run_holdout(
    data_path: str | Path,
    sha256: str,
    holdout_start: str,
    holdout_end: str,
    rth_schedule_path: str | Path,
    *,
    candidate_path: str | Path = DEFAULT_CANDIDATE,
    flag_path: str | Path = FLAG_PATH,
    report_path: str | Path = REPORT_PATH,
) -> dict[str, Any]:
    """Run the single-shot holdout evaluation and persist its audit.

    Raises:
        ValueError: Holdout SHA-256 mismatch or candidate preflight refusal (no burn).
        HoldoutAlreadyUsedError: The holdout was already burned.
    """
    data = Path(data_path)
    flag, report_file = Path(flag_path), Path(report_path)
    digest = verify_dataset_sha256(data, sha256)
    ensure_unused(flag, report_file)
    start, end = _utc(holdout_start, "holdout_start"), _utc(holdout_end, "holdout_end")
    _require(start < end, "holdout_start must precede holdout_end")
    schedule = _load_schedule(rth_schedule_path)
    schedule_sha = _sha256(Path(rth_schedule_path))
    candidate = load_candidate(candidate_path, start)
    _verify_linear(candidate)

    burned_at = datetime.now(timezone.utc).isoformat()
    burn(flag, {
        "burned_at": burned_at, "dataset_sha256": digest, "candidate_sha256": candidate.sha256,
        "holdout_start": start.isoformat(), "holdout_end": end.isoformat(),
    })
    LOGGER.warning("Holdout %s burned at %s; evaluating once", digest, burned_at)

    outcome = _evaluate(candidate, data, digest, start, end, schedule)
    evaluated_once = True
    median = outcome["median_oos"]
    accepted = bool(
        median is not None and median >= candidate.theta_accept
        and outcome["reconciliation_ok"] and outcome["gap_ok"] and evaluated_once
    )
    report = {
        "contract": REPORT_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "median_oos": median,
        "reconciliation_ok": bool(outcome["reconciliation_ok"]),
        "gap_ok": bool(outcome["gap_ok"]),
        "evaluated_once": evaluated_once,
        "accepted": accepted,
        "theta_accept": {"metric": ACCEPT_METRIC, "value": candidate.theta_accept},
        "status": "COMPLETE" if median is not None else "FAILED",
        "errors": outcome["errors"],
        "holdout": {
            "dataset_path": str(data), "dataset_sha256": digest,
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "rth_schedule_sha256": schedule_sha,
            "max_consecutive_missing": outcome["max_consecutive_missing"],
            "partition": outcome["partition"], "burn_flag": str(flag), "burned_at": burned_at,
        },
        "candidate": {
            "path": str(candidate.path), "sha256": candidate.sha256, "trial_number": candidate.trial_number,
            "trial_config_sha256": candidate.config_sha256, "evidence_sha256": candidate.evidence_sha256,
            "alpha_manifest_sha256": candidate.alpha.manifest_sha256,
            "alpha_onnx_sha256": candidate.alpha.onnx_sha256,
        },
        "seeds": outcome["seeds"],
        "scope": "OFFLINE_RESEARCH_ONLY",
    }
    report = _json_ready(report)
    _atomic_json(report_file, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Exit 0 when accepted, 1 when rejected/failed after burn, 2 when refused before burn."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="True-holdout CSV (with prior warm-up bars)")
    parser.add_argument("--sha256", required=True, help="Expected SHA-256 of --data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CANDIDATE, help="Optimizer-exported candidate")
    parser.add_argument("--holdout-start", required=True, help="Inclusive tz-aware first scored bar")
    parser.add_argument("--holdout-end", required=True, help="Exclusive tz-aware end")
    parser.add_argument("--rth-schedule", required=True, type=Path,
                        help="JSON list of expected RTH timestamps covering the full --data window")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        report = run_holdout(args.data, args.sha256, args.holdout_start, args.holdout_end, args.rth_schedule,
                             candidate_path=args.config)
    except (ValueError, HoldoutAlreadyUsedError, OSError) as exc:
        print(json.dumps({"status": "REFUSED", "burned": FLAG_PATH.exists(),
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps({key: report[key] for key in
                      ("median_oos", "reconciliation_ok", "gap_ok", "evaluated_once", "accepted")}, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
