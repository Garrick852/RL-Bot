"""CPU-governed multi-seed PPO research with an explicit opt-in local build gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import statistics
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from governance.validator import ConfigValidator, ValidationState

LOGGER = logging.getLogger(__name__)


class PipelineError(ValueError):
    """An explicit offline contract or pipeline prerequisite was violated."""


class PipelineRefused(PipelineError):
    """Validation returned a non-executable state without a new constraint error."""


def _reject_constant(value: str) -> None:
    raise PipelineError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise PipelineError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PipelineError(reason)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(type(value) in (int, float), f"{name} must be numeric, not boolean")
    number = float(value)
    _require(math.isfinite(number) and (number > 0 if positive else number >= 0),
             f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _local_path(value: str, base: Path) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), "local path must be nonempty")
    _require(not value.startswith(("\\\\", "//")) and "://" not in value, "network paths are forbidden")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Persist strict JSON with atomic replacement and no success-shaped fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _worker_initializer() -> None:
    # Spawn workers import numerical libraries only after these limits are set.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"


def _json_ready(value: Any) -> Any:
    """Normalize nested evidence timestamps without hiding unsupported objects."""
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _evaluate(
    seed: int, active: Any, env_settings: dict[str, Any], annualization: float,
    model: Any = None, baseline: Optional[str] = None,
) -> dict[str, Any]:
    """Measure a frozen deterministic policy or fixed baseline on one partition."""
    import numpy as np
    from engine_rl.envs.trading_env import TradingEnv

    env = TradingEnv(active, **env_settings)
    observation, _ = env.reset(seed=seed)
    equity = [env.initial_cash]
    for index in range(len(active)):
        if baseline == "buy_and_hold":
            action = 2
        elif baseline == "fixed_ema_crossover":
            action = int(active["baseline_target"].iloc[index]) + 1
        else:
            _require(model is not None, "evaluation requires a frozen model")
            predicted, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(predicted).item())
        observation, _, terminated, truncated, info = env.step(action)
        equity.append(info["equity"])
        if terminated or truncated:
            break
    _require(terminated and not truncated, f"seed {seed} failed to complete the full OOS episode")
    values = np.asarray(equity, dtype=np.float64)
    pnls = [float(trade["pnl"]) for trade in env.trades]
    gains = math.fsum(pnl for pnl in pnls if pnl > 0)
    losses = -math.fsum(pnl for pnl in pnls if pnl < 0)
    reasons: dict[str, str] = {}
    pf = gains / losses if losses > 0 else None
    if pf is None:
        reasons["profit_factor"] = "NO_LOSSES_OR_NO_TRADES"
    sharpe = None
    if (values <= 0).any():
        reasons["sharpe"] = "NON_POSITIVE_EQUITY"
    else:
        returns = np.diff(values) / values[:-1]
        deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        if deviation > 0:
            sharpe = float(np.mean(returns) / deviation * math.sqrt(annualization))
        else:
            reasons["sharpe"] = "INSUFFICIENT_OR_ZERO_VARIANCE_RETURNS"
    drawdown = float(np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values)))
    metrics = {
        "profit_factor": pf, "sharpe": sharpe, "max_drawdown": drawdown,
        "trade_count": len(pnls),
        "expectancy": statistics.fmean(pnls) if pnls else None,
    }
    for key, value in metrics.items():
        if value is not None and not math.isfinite(value):
            metrics[key] = None
            reasons[key] = "NON_FINITE_METRIC"
    return {
        "seed": seed, "status": "COMPLETE", "metrics": metrics, "undefined_reasons": reasons,
        "final_cash": env.cash, "initial_cash": env.initial_cash,
        "normal_realized_pnl": math.fsum(env.realized_trades_pnl),
        "terminal_liquidation_pnl": env.terminal_liquidation_pnl,
        "unrealized_pnl": env.unrealized_pnl,
        "trades": env.trades, "fills": env.fills, "equity": equity,
    }


def _train_seed(
    seed: int, train: Any, validation: Any, settings: dict[str, Any],
    annualization: float, budget: int, model_path: str,
    hyperparameters: dict[str, Any], policy_architecture: str,
) -> dict[str, Any]:
    """Train without receiving test data; freeze the final predeclared checkpoint."""
    from engine_rl.sb3_trainer import SB3Trainer
    from engine_rl.envs.trading_env import TradingEnv

    trainer = SB3Trainer(seed=seed, num_threads=1, hyperparameters=hyperparameters,
                         policy_architecture=policy_architecture)
    model = trainer.train(TradingEnv(train, **settings), budget)
    validation_result = _evaluate(seed, validation, settings, annualization, model=model)
    model.save(model_path)
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    return {
        "seed": seed, "model_path": model_path, "model_sha256": digest,
        "actual_timesteps": model.num_timesteps, "telemetry": trainer.telemetry,
        "validation": validation_result,
        "ppo_hyperparameters": trainer.hyperparameters,
        "policy_architecture": trainer.policy_architecture,
    }


def _evaluate_seed(
    frozen: dict[str, Any], test: Any, settings: dict[str, Any], annualization: float,
) -> dict[str, Any]:
    """Load exactly the frozen model after the all-seed training barrier."""
    from engine_rl.sb3_trainer import SB3Trainer

    SB3Trainer(seed=frozen["seed"], num_threads=1)
    from stable_baselines3 import PPO
    path = Path(frozen["model_path"])
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == frozen["model_sha256"],
             "frozen model identity mismatch before OOS")
    model = PPO.load(path, device="cpu")
    result = _evaluate(frozen["seed"], test, settings, annualization, model=model)
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == frozen["model_sha256"],
             "model changed during OOS evaluation")
    return {**result, "training": frozen}


class OfflineOrchestrator:
    """Coordinate strict preflight, frozen PPO policies, OOS, and optional builds.

    Args:
        report_directory: Destination for offline_audit_report.json and retained
            per-run snapshots. Defaults to the explicitly requested reports path.
    """

    def __init__(self, report_directory: str | Path = "reports") -> None:
        self.report_directory = Path(report_directory)
        self.state = ValidationState.DRAFT_INCOMPLETE
        self.config: dict[str, Any] = {}
        self.context: dict[str, Any] = {}

    def run_pipeline(self, config_path: str | Path, *, deploy: bool = False) -> dict[str, Any]:
        """Validate input, execute every declared seed, and persist a final audit.

        Args:
            config_path: Local UTF-8 JSON configuration path. Dataset paths are
                resolved relative to this file, never a broker or remote source.
            deploy: Explicit permission for staged compilation and Telegram.

        Returns:
            Persisted research report; build status is independent of research
            PASS. No patching, compiler process, or notification without deploy.

        Raises:
            OSError: A failure report cannot be persisted.
        """
        self.state = ValidationState.DRAFT_INCOMPLETE
        self.config = {}
        config_hash: Optional[str] = None
        data_hash: Optional[str] = None
        results: list[dict[str, Any]] = []
        validator: Optional[ConfigValidator] = None
        run_id = uuid.uuid4().hex
        self.context = {"run_id": run_id, "run_status": "NOT_STARTED", "errors": []}
        self.context["deployment"] = {"requested": deploy, "status": "NOT_REQUESTED"}
        self.context["state_transitions"] = [self.state.value]
        try:
            _require(type(deploy) is bool, "deploy must be a boolean")
            path = _local_path(str(config_path), Path.cwd())
            raw = path.read_bytes()
            config_hash = hashlib.sha256(raw).hexdigest()
            config = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            _require(isinstance(config, dict), "config must be a JSON object")
            self.config = config
            self._config_base = path.parent
            validator = ConfigValidator(config, base_directory=path.parent)
            self.state = validator.validate(check_evidence=False)
            self.context["state_transitions"].append(self.state.value)
            self.context["validation_errors"] = validator.errors
            if self.state != ValidationState.EVIDENCE_UNCHECKED:
                raise PipelineRefused(
                    f"L1/L2 refused execution: {self.state.value}; "
                    f"errors={validator.errors}; unresolved={validator.unresolved_fields}"
                )
            self._validate_profile()
            import pandas as pd
            import numpy as np
            from data.collector import DataCollector

            run_directory = self.report_directory / "runs" / run_id
            run_directory.mkdir(parents=True, exist_ok=False)
            (run_directory / "config.snapshot.json").write_bytes(raw)
            source = _local_path(config["dataset"]["path"], path.parent)
            snapshot = run_directory / "dataset.snapshot.csv"
            # Load and hash one retained byte snapshot, not a later re-read for workers.
            snapshot.write_bytes(source.read_bytes())
            collector = DataCollector()
            frame = collector.load_csv(snapshot, expected_sha256=config["dataset"]["sha256"])
            data_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            validator.validate_layer3(config["dataset"]["path"])
            _require(data_hash == config["dataset"]["sha256"], "loaded dataset snapshot SHA-256 mismatch")
            _require(data_hash == validator.actual_dataset_sha256, "source changed relative to loaded snapshot")
            self.context["evidence"] = dict(validator.evidence)
            run = config["orchestration"]
            expected = pd.DatetimeIndex([pd.Timestamp(value) for value in run["expected_rth_timestamps"]])
            collector = DataCollector(
                expected_rth_timestamps=expected, bar_interval=pd.Timedelta(config["dataset"]["timeframe"])
            )
            _require(expected[0] <= frame["timestamp"].iloc[0]
                     and expected[-1] >= frame["timestamp"].iloc[-1],
                     "explicit RTH schedule must cover the full dataset window")
            gap_report = collector.check_gap_policy(frame, run["max_allowed_gaps"])
            self.context["gap_evidence"] = {
                "missing_rth_timestamps": [stamp.isoformat() for stamp in gap_report.missing_rth_timestamps],
                "max_consecutive_missing": gap_report.max_consecutive_missing,
            }
            partitions = {}
            partition_evidence = {}
            fast = config["strategy"]["ema"]["fast_period"]
            slow = config["strategy"]["ema"]["slow_period"]
            for name in ("train", "validation", "test"):
                bounds = config["splits"][name]
                _require(pd.Timestamp(bounds["start"]) >= frame["timestamp"].iloc[0]
                         and pd.Timestamp(bounds["end"]) <= frame["timestamp"].iloc[-1] + pd.Timedelta(config["dataset"]["timeframe"]),
                         f"dataset does not cover {name} boundaries")
                warmup, active = collector.get_split(
                    frame, bounds["start"], bounds["end"], run["k_multiplier"], slow
                )
                combined = pd.concat([warmup, active], ignore_index=True)
                for period, feature in ((fast, "ema_fast"), (slow, "ema_slow")):
                    # Lag close-based indicators so row-t fills cannot use row-t close.
                    values = collector.calculate_ema(
                        combined, period, config["instrument"]["tick_size"]
                    ).shift(1)
                    active[feature] = values.iloc[len(warmup):].to_numpy()
                active["baseline_target"] = (
                    (active["ema_fast"] > active["ema_slow"]).astype(int)
                    - (active["ema_fast"] < active["ema_slow"]).astype(int)
                )
                partitions[name] = active
                partition_evidence[name] = {
                    "rows": len(active), "warmup_rows": len(warmup),
                    "first_scored": active["timestamp"].iloc[0].isoformat(),
                    "last_scored": active["timestamp"].iloc[-1].isoformat(),
                }
            features = config["features"]["ordered_names"]
            train = partitions["train"][features]
            mean = train.mean()
            scale = train.std(ddof=0)
            # This is a declared standard-scaler zero-variance rule, not missing data padding.
            scale = scale.mask(scale == 0, 1.0)
            _require(np.isfinite(mean.to_numpy()).all() and np.isfinite(scale.to_numpy()).all(),
                     "training scaler statistics must be finite")
            for active in partitions.values():
                active[features] = (active[features] - mean) / scale
                values = active[features].to_numpy()
                _require(np.isfinite(values).all() and (np.abs(values) <= np.finfo(np.float32).max).all(),
                         "scaled features must be finite and float32 representable")
            self.context["scaler"] = {
                "fit_partition": "train_only", "mean": mean.to_dict(), "scale": scale.to_dict(),
                "zero_variance_rule": "unit_scale", "feature_order": features,
            }
            self.context["partition_evidence"] = partition_evidence
            self.context["splits"] = config["splits"]
            self.state = ValidationState.READY_FOR_OFFLINE_EXECUTION
            self.context["state_transitions"].append(self.state.value)
            self.context["run_status"] = "RUNNING"
            self.context["artifacts"] = {
                "config_snapshot": str(run_directory / "config.snapshot.json"),
                "dataset_snapshot": str(snapshot),
            }
            cpu_count = os.cpu_count()
            _require(cpu_count is not None and cpu_count > 0, "CPU capacity could not be established")
            seeds = config["training"]["seeds"]
            workers = min(run["max_workers"], cpu_count, len(seeds))
            self.context["cpu_governance"] = {"max_workers": workers, "threads_per_worker": 1}
            settings = self._env_settings()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_initializer,
            ) as pool:
                training_futures = {
                    pool.submit(
                        _train_seed, seed, partitions["train"], partitions["validation"],
                        settings, config["evaluation"]["annualization_periods"],
                        config["training"]["total_timesteps_per_seed"],
                        str((run_directory / f"seed-{seed}.zip").absolute()),
                        config["training"]["ppo_hyperparameters"],
                        config["training"].get("policy_architecture", "mlp_tanh_128x128"),
                    ): seed for seed in seeds
                }
                frozen_models = []
                futures = {}
                try:
                    for future in as_completed(training_futures):
                        frozen_models.append(future.result())
                    _require(len(frozen_models) == len(seeds), "not every model reached the OOS freeze barrier")
                    self.context["models_frozen_before_oos"] = True
                    futures = {
                        pool.submit(
                            _evaluate_seed, frozen, partitions["test"], settings,
                            config["evaluation"]["annualization_periods"],
                        ): frozen["seed"] for frozen in frozen_models
                    }
                    for future in as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            raise PipelineError(f"seed {futures[future]} failed: {exc}") from exc
                except Exception:
                    for future in list(training_futures) + list(futures):
                        future.cancel()
                    raise
            results.sort(key=lambda result: seeds.index(result["seed"]))
            self.context["baselines"] = {
                name: _evaluate(
                    0, partitions["test"], settings,
                    config["evaluation"]["annualization_periods"], baseline=name,
                ) for name in ("buy_and_hold", "fixed_ema_crossover")
            }
            self.context["run_status"] = "COMPLETE"
        except Exception as exc:
            LOGGER.exception("Offline pipeline halted: %s: %s", type(exc).__name__, exc)
            if self.context["run_status"] == "RUNNING":
                self.context["run_status"] = "FAILED"
            elif not isinstance(exc, PipelineRefused):
                self.state = ValidationState.INVALID
                self.context["state_transitions"].append(self.state.value)
            if validator is not None:
                self.context["evidence"] = dict(validator.evidence)
                self.context["validation_errors"] = list(validator.errors)
            self.context["errors"].append(f"{type(exc).__name__}: {exc}")
        report = self.export_audit_report(results, config_hash, data_hash)
        if deploy:
            if report["run_status"] != "COMPLETE":
                self.context["deployment"] = {"requested": True, "status": "BLOCKED_INCOMPLETE"}
            else:
                self.context["deployment"] = self.execute_deployment_gate(report)
            report = self.export_audit_report(results, config_hash, data_hash)
        return report

    def _env_settings(self) -> dict[str, Any]:
        config = self.config
        return {
            "feature_columns": config["features"]["ordered_names"],
            "initial_cash": config["instrument"]["starting_cash"],
            "quantity": config["orchestration"]["quantity"],
            "point_value": config["instrument"]["point_value_per_unit"],
            "slippage": config["costs"]["slippage"]["value"],
            "commission": config["costs"]["commission"]["value"],
            "lambda_term": config["orchestration"]["lambda_term"],
        }

    def _validate_profile(self) -> None:
        """Refuse unsupported economics, feature contracts, or implicit settings."""
        config = self.config
        _require(config["template_version"] == 1 and type(config["template_version"]) is int
                 and config["specification_status"] == "FROZEN_PPO_RESEARCH",
                 "template_version=1 and specification_status=FROZEN_PPO_RESEARCH required")
        run = config["orchestration"]
        required = {
            "expected_rth_timestamps", "max_allowed_gaps", "k_multiplier",
            "max_workers", "quantity", "lambda_term",
        }
        _require(isinstance(run, dict) and set(run) == required, f"orchestration requires exactly {sorted(required)}")
        for name in ("k_multiplier", "max_workers"):
            _integer(run[name], name)
        _integer(run["max_allowed_gaps"], "max_allowed_gaps", minimum=0)
        _number(run["quantity"], "quantity", positive=True)
        _number(run["lambda_term"], "lambda_term")
        _require(isinstance(run["expected_rth_timestamps"], list) and bool(run["expected_rth_timestamps"]),
                 "expected_rth_timestamps must be an explicit nonempty schedule")
        seeds = config["training"]["seeds"]
        _require(isinstance(seeds, list) and len(seeds) >= 3
                 and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
                 and len(set(seeds)) == len(seeds), "training.seeds requires >=3 distinct uint32 integers")
        _require(config["training"]["algorithm"] == "PPO", "only PPO training is implemented")
        _integer(config["training"]["total_timesteps_per_seed"], "total_timesteps_per_seed")
        _require(config["training"]["checkpoint_selection_rule"] == "final_checkpoint",
                 "only predeclared final_checkpoint selection is supported")
        from engine_rl.sb3_trainer import validate_policy_architecture, validate_ppo_hyperparameters
        validate_ppo_hyperparameters(config["training"]["ppo_hyperparameters"])
        if "policy_architecture" in config["training"]:
            validate_policy_architecture(config["training"]["policy_architecture"])
        if "optimization" in config:
            _require(config["optimization"] == {"role": "SEARCH_ONLY_REUSED_HOLDOUT"},
                     "optimization must explicitly mark reused holdout search")
        _require(config["training"]["device"] == "cpu" and type(config["training"]["threads_per_worker"]) is int
                 and config["training"]["threads_per_worker"] == 1, "CPU workers require exactly one numerical thread")
        _require(config["dataset"]["format"] == "csv" and config["dataset"]["timezone"] == "UTC",
                 "only local UTC csv input is supported")
        _require(config["dataset"]["missing_bar_policy"] == "explicit_rth_schedule", "explicit_rth_schedule policy required")
        _require(config["dataset"]["price_basis"] == "bid_ask"
                 and config["dataset"]["timestamp_meaning"] == "execution_quote_time",
                 "bid_ask prices at execution_quote_time are required")
        from data.collector import CORE_COLUMNS
        _require(config["dataset"]["column_mapping"] == {name: name for name in CORE_COLUMNS},
                 "column_mapping must explicitly identify the canonical quote columns")
        _require(config["strategy"]["name"] == "fixed_ema_crossover"
                 and config["strategy"]["ema"]["initialization_rule"] == "first_close_quantized",
                 "only first_close_quantized EMA features are supported")
        _require(config["features"]["ordered_names"] == ["ema_fast", "ema_slow"]
                 and config["features"]["scaler_method"] == "standard"
                 and config["features"]["scaler_fit_partition"] == "train_only"
                 and config["features"]["refit_on_validation_or_test"] is False,
                 "features require ordered lagged EMA fast/slow with train-only standard scaling")
        _require(config["features"]["warmup_bars"] == run["k_multiplier"] * config["strategy"]["ema"]["slow_period"],
                 "features.warmup_bars must equal k_multiplier * slow_period")
        _require(type(config["features"]["warmup_bars"]) is int, "warmup_bars must be an integer")
        mapping = config["policy"]["discrete_action_index_to_target"]
        _require(mapping == {"0": -1, "1": 0, "2": 1}
                 and all(type(value) is int for value in mapping.values())
                 and config["policy"]["action_space_type"] == "discrete"
                 and config["policy"]["controls"] == "target_position"
                 and config["policy"]["reward_definition"] == "liquidation_equity_delta",
                 "policy must explicitly adopt discrete short/flat/long targets")
        instrument = config["instrument"]
        for name in ("starting_cash", "point_value_per_unit", "tick_size", "minimum_position_increment"):
            _number(instrument[name], f"instrument.{name}", positive=True)
        _require(instrument["asset_type"] == "cash_settled_derivative"
                 and instrument["position_sizing_rule"] == "fixed_quantity"
                 and instrument["account_currency"] == instrument["quote_currency"]
                 and instrument["currency_conversion_rule"] == "identity"
                 and instrument["margin_rule"] == "none"
                 and instrument["leverage"] == 1
                 and instrument["bankruptcy_policy"] == "report_fail",
                 "only fixed-size unlevered same-currency cash-settlement profile is supported")
        increment = run["quantity"] / instrument["minimum_position_increment"]
        _require(math.isclose(increment, round(increment), rel_tol=0, abs_tol=1e-8), "quantity violates minimum_position_increment")
        costs = config["costs"]
        _require(costs["spread"]["value"] == 0 and costs["spread"]["unit"] == "price"
                 and costs["spread"]["application_rule"] == "embedded_bid_ask",
                 "spread must be sourced from bid/ask, never added twice")
        for name in ("commission", "slippage"):
            _number(costs[name]["value"], f"costs.{name}.value")
            _require(costs[name]["unit"] == "price"
                     and costs[name]["application_rule"] == "per_unit_per_side", f"unsupported {name} units/application")
        _require(costs["swap"]["long_value"] == 0 and costs["swap"]["short_value"] == 0
                 and costs["swap"]["day_count_and_multiplier_rule"] == "explicit_zero",
                 "nonzero swap is not implemented")
        execution = {
            "decision_timing": "before_current_quote",
            "fill_timing": "current_row", "fill_price_rule": "bid_ask",
            "final_bar_entry_policy": "forbid",
            "terminal_liquidation_price_rule": "bid_ask_adverse_lambda",
            "terminal_liquidation_cost_rule": "per_unit_price_costs_once",
        }
        _require(config["execution"] == execution, "unsupported execution contract")
        safety = config["safety"]
        _require(safety == {
            "offline_only": True, "network_access_allowed": False, "broker_execution_allowed": False,
            "active_config_writes_allowed": False, "automatic_deployment_allowed": False,
        } and all(type(value) is bool for value in safety.values()), "offline safety flags must be explicit booleans")
        evaluation = config["evaluation"]
        _require(evaluation["deterministic_policy"] is True, "OOS policy must be deterministic")
        _require(evaluation["baseline_rules"] == ["buy_and_hold", "fixed_ema_crossover"],
                 "both fixed baselines must be explicitly declared")
        _number(evaluation["annualization_periods"], "annualization_periods", positive=True)
        _require(evaluation["sharpe_definition"] == "bar_simple_returns_sample_std_zero_rf"
                 and evaluation["return_sampling_frequency"] == config["dataset"]["timeframe"]
                 and evaluation["undefined_metric_policy"] == "null_and_fail", "unsupported metric definition")
        thresholds = evaluation["research_gate_thresholds"]
        expected = {"min_profit_factor", "min_sharpe", "max_drawdown", "min_trades", "min_expectancy"}
        _require(isinstance(thresholds, dict) and set(thresholds) == expected, f"thresholds require exactly {sorted(expected)}")
        for name in expected:
            _number(thresholds[name], f"thresholds.{name}")
        _integer(thresholds["min_trades"], "min_trades")
        _require(thresholds["min_profit_factor"] >= 1.25 and thresholds["min_sharpe"] >= 1
                 and 0 < thresholds["max_drawdown"] <= .1 and thresholds["min_trades"] >= 30,
                 "thresholds cannot weaken the specified research profile")

    def export_audit_report(
        self,
        results: Sequence[dict[str, Any]],
        config_hash: Optional[str],
        data_hash: Optional[str],
    ) -> dict[str, Any]:
        """Persist strict all-seed research metrics independently of build status."""
        training = self.config.get("training")
        seeds = training.get("seeds", []) if isinstance(training, dict) else []
        if not isinstance(seeds, list) or not all(type(seed) is int for seed in seeds):
            seeds = []
        valid = (
            self.context.get("run_status") == "COMPLETE"
            and len(results) == len(seeds) and len(seeds) >= 3
            and len(set(seeds)) == len(seeds)
            and len({row["seed"] for row in results}) == len(results)
            and {row["seed"] for row in results} == set(seeds)
            and all(row["status"] == "COMPLETE" for row in results)
            and self.state == ValidationState.READY_FOR_OFFLINE_EXECUTION
            and config_hash is not None and data_hash is not None
            and self.context.get("models_frozen_before_oos") is True
            and all("training" in row for row in results)
            and set(self.context.get("baselines", {})) == {"buy_and_hold", "fixed_ema_crossover"}
        )
        medians: dict[str, Optional[float]] = {}
        for name in ("profit_factor", "sharpe", "max_drawdown"):
            values = [row["metrics"][name] for row in results]
            medians[name] = (
                statistics.median(values)
                if valid and all(type(value) in (int, float) and math.isfinite(value) for value in values)
                else None
            )
        evaluation = self.config.get("evaluation")
        thresholds = evaluation.get("research_gate_thresholds", {}) if isinstance(evaluation, dict) else {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        threshold_pass = bool(
            valid and all(value is not None for value in medians.values())
            and medians["profit_factor"] >= thresholds["min_profit_factor"]
            and medians["sharpe"] >= thresholds["min_sharpe"]
            and all(
                row["metrics"]["max_drawdown"] <= thresholds["max_drawdown"]
                and row["metrics"]["trade_count"] >= thresholds["min_trades"]
                and row["metrics"]["expectancy"] is not None
                and row["metrics"]["expectancy"] > thresholds["min_expectancy"]
                and row["final_cash"] > 0
                for row in results
            )
        )
        dataset = self.config.get("dataset")
        search_only = "optimization" in self.config
        report = {
            "audit_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            **self.context, "validation_state": self.state.value,
            "config_hash": config_hash, "dataset_hash": data_hash,
            "configured_dataset_hash": dataset.get("sha256") if isinstance(dataset, dict) else None,
            "run_kind": "SB3_PPO_MULTI_SEED", "rl_training_performed": bool(results),
            "seeds_used": [row["seed"] for row in results], "configured_seeds": seeds,
            "median_oos_metrics": medians, "thresholds": thresholds,
            "threshold_pass": threshold_pass, "pass": threshold_pass and not search_only,
            "decision": "PASS" if threshold_pass and not search_only else "FAIL",
            "missing_seeds": [seed for seed in seeds if seed not in {row["seed"] for row in results}],
            "metric_definitions": {
                "profit_factor": "sum positive net closed-trade P&L / abs(sum negative net closed-trade P&L)",
                "sharpe": "mean simple marked-equity bar return / sample std * sqrt(annualization_periods); zero risk-free",
                "max_drawdown": "fractional marked-equity peak-to-trough; includes initial and terminal equity",
                "aggregation": "componentwise median of every configured seed; even N averages middle two",
            },
            "reasons": (
                ([] if threshold_pass else ["INCOMPLETE_OR_RESEARCH_THRESHOLDS_NOT_MET"])
                + (["SEARCH_REQUIRES_FRESH_HOLDOUT"] if search_only else [])
            ),
            "scope": "HYPERPARAMETER_SEARCH_ONLY" if search_only else "OFFLINE_RESEARCH_ONLY",
            "results": list(results),
        }
        report = _json_ready(report)
        destination = self.report_directory / "offline_audit_report.json"
        run_report = self.report_directory / "runs" / self.context.get("run_id", uuid.uuid4().hex) / "audit.json"
        _atomic_json(run_report, report)
        _atomic_json(destination, report)
        return report

    def execute_deployment_gate(self, report: dict[str, Any]) -> dict[str, Any]:
        """Opt-in staged build only; never load a terminal or install a policy.

        Research readiness, research PASS, and local build status are separate.
        A PPO action policy is not equivalent to tagged EMA parameters: this
        gate explicitly requires the fixed-parameter-only binding contract.
        The original source and its binary remain unchanged.
        """
        outcome: dict[str, Any] = {"requested": False, "status": "BLOCKED_NOT_AUTHORIZED"}
        if self.context.get("deployment", {}).get("requested") is not True:
            LOGGER.error("Build gate refused: explicit deploy authorization is missing")
            return outcome

        from governance.fallback import AdaptiveFallbackStateMachine
        from governance.telegram_notifier import TelegramAlertEngine
        from engine_mt5.patcher import MQL5Patcher

        outcome = {"requested": True, "status": "BLOCKED"}
        notifier = TelegramAlertEngine()
        asset = self.config["dataset"]["symbol"]
        lock_path: Optional[Path] = None
        lock_acquired = False
        staged: Optional[Path] = None
        try:
            _require(report["run_kind"] == "SB3_PPO_MULTI_SEED"
                     and report["rl_training_performed"] is True
                     and report["run_status"] == "COMPLETE"
                     and report["validation_state"] == ValidationState.READY_FOR_OFFLINE_EXECUTION.value,
                     "only completed real PPO OOS results can enter the build gate")
            _require("optimization" not in self.config, "search candidates require fresh holdout evaluation")
            deployment = self.config["deployment"]
            required = {"source_path", "source_sha256", "metaeditor_path", "parameters", "parameter_binding"}
            _require(isinstance(deployment, dict) and set(deployment) == required,
                     f"deployment requires exactly {sorted(required)}")
            _require(deployment["parameter_binding"] == "fixed_strategy_parameters_not_ppo_policy",
                     "MQL5 policy export is unimplemented; explicit fixed-parameter-only binding required")
            source = _local_path(deployment["source_path"], self._config_base).absolute()
            compiler = _local_path(deployment["metaeditor_path"], self._config_base).absolute()
            # Persistent state is keyed to the asset/source, not the current run.
            key = hashlib.sha256(f"{asset}|{source}".encode()).hexdigest()
            state_path = self.report_directory / "fallback" / f"{key}.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = state_path.with_suffix(".lock")
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            lock_acquired = True
            persisted = {"consecutive_failures": 0, "suspended": False, "processed_runs": []}
            if state_path.exists():
                persisted = json.loads(state_path.read_text("utf-8"), object_pairs_hook=_unique_object,
                                       parse_constant=_reject_constant)
            count = _integer(persisted["consecutive_failures"], "consecutive_failures", minimum=0)
            _require(type(persisted["suspended"]) is bool and isinstance(persisted["processed_runs"], list),
                     "invalid persistent fallback journal")
            _require(persisted["suspended"] == (count >= 3), "inconsistent fallback suspension journal")
            if persisted["suspended"]:
                outcome["status"] = "SUSPENDED"
                outcome["notification_delivered"] = self._notify(
                    notifier.send_circuit_breaker, asset, "Suspension latch requires explicit external recovery"
                )
                return outcome
            run_id = self.context["run_id"]
            _require(run_id not in persisted["processed_runs"], "this run already entered the build gate")
            persisted["processed_runs"].append(run_id)
            if not report["pass"]:
                persisted["consecutive_failures"] = count + 1
                directive = AdaptiveFallbackStateMachine().evaluate(count + 1)
                persisted["suspended"] = directive["state"].value == "SUSPENDED"
                _atomic_json(state_path, persisted)
                outcome.update(status=directive["state"].value, directive=directive)
                if persisted["suspended"]:
                    outcome["notification_delivered"] = self._notify(
                        notifier.send_circuit_breaker, asset, "OOS research gates failed for three consecutive candidates"
                    )
                else:
                    outcome["notification_delivered"] = self._notify(
                        notifier.send_fallback_alert, directive["stage"],
                        "OOS research gates failed; no retry on the same test holdout",
                        {"status": "retained_not_verified_by_gate"},
                    )
                return outcome
            source_bytes = source.read_bytes()
            _require(hashlib.sha256(source_bytes).hexdigest() == deployment["source_sha256"],
                     "MQL5 source identity mismatch")
            parameters = deployment["parameters"]
            _require(isinstance(parameters, dict) and bool(parameters), "explicit MQL5 parameter mapping required")
            _require(all(not isinstance(value, (list, tuple, dict)) for value in parameters.values()),
                     "matrix/policy parameters are outside the fixed-parameter-only build gate")
            staged = source.with_name(f"{source.stem}.{run_id}.staged.mq5")
            _require(not staged.exists(), "staged source already exists")
            with staged.open("xb") as destination:
                destination.write(source_bytes)
            patcher = MQL5Patcher()
            patcher.patch_source(staged, parameters)
            log_path = patcher.compile_ea(compiler, staged)
            # Never infer a deployed PPO policy from a clean fixed-parameter build.
            outcome.update(
                status="STAGED_BUILD_SUCCEEDED", staged_source=str(staged),
                staged_binary=str(staged.with_suffix(".ex5")), compile_log=str(log_path),
                original_source_unchanged=True, live_deployment_performed=False,
                policy_exported=False,
            )
            persisted["consecutive_failures"] = 0
            _atomic_json(state_path, persisted)
            outcome["notification_delivered"] = self._notify(
                notifier.send_deployment_success, asset, report["median_oos_metrics"], parameters,
            )
            return outcome
        except Exception as exc:
            LOGGER.error("Opt-in build blocked: %s: %s", type(exc).__name__, exc)
            outcome.update(status="INVALID", reason=f"{type(exc).__name__}: {exc}")
            if staged is not None:
                try:
                    staged.with_suffix(".ex5").unlink(missing_ok=True)
                except OSError as cleanup_error:
                    outcome["cleanup_error"] = str(cleanup_error)
                    LOGGER.error("Unsafe staged binary could not be removed: %s", cleanup_error)
            outcome["notification_delivered"] = self._notify(
                notifier.send_circuit_breaker, asset, "Opt-in local build failed; inspect local audit"
            )
            return outcome
        finally:
            if lock_acquired and lock_path is not None:
                try:
                    lock_path.unlink()
                except OSError as exc:
                    outcome.update(status="INVALID", lock_cleanup_error=str(exc))
                    LOGGER.error("Build lock could not be released: %s", exc)

    @staticmethod
    def _notify(method: Any, *args: Any) -> bool:
        """Isolate notification failures even if a custom notifier violates its contract."""
        try:
            return method(*args) is True
        except Exception:
            LOGGER.error("Operational notification failed; build decision is unchanged")
            return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run offline by default; --deploy explicitly permits local builds/alerts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="Explicit local experiment JSON")
    parser.add_argument("--report-directory", type=Path, default=Path("reports"))
    parser.add_argument("--deploy", action="store_true",
                        help="Opt in to staged MQL5 patch/compile and Telegram; no terminal deployment")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = OfflineOrchestrator(args.report_directory).run_pipeline(args.config_path, deploy=args.deploy)
    except OSError:
        LOGGER.exception("Audit persistence failed")
        return 1
    print(json.dumps({
        "run_status": report["run_status"], "decision": report["decision"],
        "median_oos_metrics": report["median_oos_metrics"],
        "deployment": report["deployment"],
    }, indent=2, allow_nan=False))
    return 0 if report["run_status"] == "COMPLETE" and (
        not args.deploy or report["deployment"]["status"] == "STAGED_BUILD_SUCCEEDED"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
