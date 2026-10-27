"""Real spawned-process PPO/OOS tests and fail-closed pipeline refusals."""

import copy
import hashlib
import json
import logging
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import Future

import numpy as np
import pandas as pd

from orchestrator import OfflineOrchestrator, _evaluate, main
from governance.validator import ValidationState

ROOT = Path(__file__).resolve().parents[1]


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "experiment.json"
        self.reports = self.root / "reports"
        times = pd.date_range("2024-01-01", periods=230, freq="min", tz="UTC")
        midpoint = 100 + np.sin(np.arange(230) / 3)
        self.frame = pd.DataFrame({
            "timestamp": times, "open": midpoint, "high": midpoint + 1,
            "low": midpoint - 1, "close": midpoint, "volume": np.ones(230),
            "bid": midpoint - .01, "ask": midpoint + .01,
        })
        self.frame.to_csv(self.root / "bars.csv", index=False)
        config = json.loads((ROOT / "config" / "offline_experiment.template.json").read_text("utf-8"))
        # Fill every informational field explicitly for this synthetic test only.
        def fill(value):
            if isinstance(value, dict):
                return {key: fill(child) for key, child in value.items()}
            return "test_metadata_only" if value is None else value
        self.config = fill(config)
        self.config.update(frozen=True, specification_status="FROZEN_PPO_RESEARCH")
        self.config["dataset"].update(
            path="bars.csv", sha256=hashlib.sha256((self.root / "bars.csv").read_bytes()).hexdigest(),
            format="csv", timezone="UTC", timeframe="1min", price_basis="bid_ask",
            timestamp_meaning="execution_quote_time", missing_bar_policy="explicit_rth_schedule",
            column_mapping={name: name for name in self.frame.columns},
        )
        self.config["strategy"].update(name="fixed_ema_crossover", shorting_allowed=True)
        self.config["strategy"]["ema"] = {
            "fast_period": 2, "slow_period": 3, "initialization_rule": "first_close_quantized",
        }
        for name, start, end in (("train", 10, 70), ("validation", 70, 110), ("test", 110, 230)):
            self.config["splits"][name] = {
                "start": (times[0] + pd.Timedelta(minutes=start)).isoformat(),
                "end": (times[0] + pd.Timedelta(minutes=end)).isoformat(),
            }
        self.config["instrument"].update(
            asset_type="cash_settled_derivative", starting_cash=10000,
            point_value_per_unit=1, tick_size=.01, position_sizing_rule="fixed_quantity",
            minimum_position_increment=1, margin_rule="none", leverage=1,
            quote_currency="USD", account_currency="USD", currency_conversion_rule="identity",
            bankruptcy_policy="report_fail",
        )
        self.config["training"].update(
            algorithm="PPO", seeds=[7, 42, 99], threads_per_worker=1,
            total_timesteps_per_seed=2048, checkpoint_selection_rule="final_checkpoint",
            ppo_hyperparameters={
                "n_steps": 2048, "batch_size": 256, "learning_rate": 3e-4,
                "clip_range": .2, "gae_lambda": .95, "gamma": .99,
            },
        )
        self.config["features"].update(
            ordered_names=["ema_fast", "ema_slow"], warmup_bars=6, scaler_method="standard",
        )
        self.config["policy"].update(
            discrete_action_index_to_target={"0": -1, "1": 0, "2": 1}, action_space_type="discrete",
            controls="target_position", reward_definition="liquidation_equity_delta",
        )
        self.config["execution"] = {
            "decision_timing": "before_current_quote", "fill_timing": "current_row",
            "fill_price_rule": "bid_ask", "final_bar_entry_policy": "forbid",
            "terminal_liquidation_price_rule": "bid_ask_adverse_lambda",
            "terminal_liquidation_cost_rule": "per_unit_price_costs_once",
        }
        for name in ("spread", "commission", "slippage"):
            self.config["costs"][name].update(
                value=0, unit="price",
                application_rule="embedded_bid_ask" if name == "spread" else "per_unit_per_side",
            )
        self.config["costs"]["swap"].update(
            long_value=0, short_value=0, day_count_and_multiplier_rule="explicit_zero",
        )
        self.config["evaluation"].update(
            deterministic_policy=True, baseline_rules=["buy_and_hold", "fixed_ema_crossover"],
            annualization_periods=252*390, return_sampling_frequency="1min",
            sharpe_definition="bar_simple_returns_sample_std_zero_rf",
            undefined_metric_policy="null_and_fail",
            research_gate_thresholds={
                "min_profit_factor": 1.25, "min_sharpe": 1, "max_drawdown": .1,
                "min_trades": 30, "min_expectancy": 0,
            },
        )
        self.config["orchestration"] = {
            "expected_rth_timestamps": [stamp.isoformat() for stamp in times],
            "max_allowed_gaps": 0, "k_multiplier": 2, "max_workers": 2,
            "quantity": 1, "lambda_term": .01,
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")

    def run_failure(self, state="INVALID"):
        with self.assertLogs("orchestrator", logging.ERROR):
            report = OfflineOrchestrator(self.reports).run_pipeline(self.path)
        self.assertFalse(report["pass"])
        self.assertEqual(report["validation_state"], state)
        self.assertEqual(report["run_status"], "NOT_STARTED")
        self.assertTrue(report["errors"])
        self.assertTrue((self.reports / "offline_audit_report.json").is_file())
        return report

    def test_real_spawned_workers_train_freeze_and_evaluate_without_deployment(self):
        with patch.object(OfflineOrchestrator, "execute_deployment_gate",
                          side_effect=AssertionError("offline mode entered deployment")) as gate:
            report = OfflineOrchestrator(self.reports).run_pipeline(self.path)
            gate.assert_not_called()
        self.assertEqual(report["run_status"], "COMPLETE", report["errors"])
        self.assertEqual(report["validation_state"], "READY_FOR_OFFLINE_EXECUTION")
        self.assertEqual(report["seeds_used"], [7, 42, 99])
        self.assertEqual(report["config_hash"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(report["dataset_hash"], self.config["dataset"]["sha256"])
        self.assertEqual(len(report["results"]), 3)
        self.assertEqual(report["state_transitions"], [
            "DRAFT_INCOMPLETE", "EVIDENCE_UNCHECKED", "READY_FOR_OFFLINE_EXECUTION",
        ])
        self.assertLessEqual(report["cpu_governance"]["max_workers"], 2)
        self.assertFalse(report["pass"])
        self.assertTrue(report["rl_training_performed"])
        self.assertTrue(report["models_frozen_before_oos"])
        self.assertEqual(report["deployment"]["status"], "NOT_REQUESTED")
        self.assertEqual(set(report["baselines"]), {"buy_and_hold", "fixed_ema_crossover"})
        self.assertEqual(report["splits"], self.config["splits"])
        for row in report["results"]:
            self.assertEqual(row["training"]["actual_timesteps"], 2048)
            self.assertEqual(
                hashlib.sha256(Path(row["training"]["model_path"]).read_bytes()).hexdigest(),
                row["training"]["model_sha256"],
            )
            self.assertTrue(row["training"]["telemetry"])
            self.assertEqual(row["unrealized_pnl"], 0)
            self.assertAlmostEqual(
                row["final_cash"] - row["initial_cash"],
                row["normal_realized_pnl"] + row["terminal_liquidation_pnl"], places=8,
            )
        saved = json.loads((self.reports / "offline_audit_report.json").read_text())
        self.assertEqual(saved, report)

    def test_invalid_configs_never_start_workers(self):
        for change in ("seeds", "frozen", "network", "gap", "cost", "hash"):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                if change == "seeds":
                    config["training"]["seeds"] = [7, 7, 99]
                elif change == "frozen":
                    config["frozen"] = False
                elif change == "network":
                    config["safety"]["broker_execution_allowed"] = True
                elif change == "gap":
                    config["orchestration"]["expected_rth_timestamps"].insert(1, "2024-01-01T00:00:30Z")
                elif change == "cost":
                    config["costs"]["commission"]["unit"] = "cash"
                else:
                    config["dataset"]["sha256"] = "0"*64
                self.path.write_text(json.dumps(config))
                with patch("orchestrator.ProcessPoolExecutor") as pool:
                    self.run_failure("DRAFT_INCOMPLETE" if change == "frozen" else "INVALID")
                    pool.assert_not_called()

    def test_duplicate_json_and_bad_config_shapes_get_failure_reports(self):
        for content in ('{"frozen": true, "frozen": false}', '{"dataset":2,"training":2}', '[]', 'NaN'):
            with self.subTest(content=content):
                self.path.write_text(content)
                self.run_failure()

    def test_worker_exception_keeps_failure_record(self):
        with patch("orchestrator.ProcessPoolExecutor", side_effect=RuntimeError("worker startup failed")):
            with self.assertLogs("orchestrator", logging.ERROR):
                report = OfflineOrchestrator(self.reports).run_pipeline(self.path)
        self.assertEqual(report["run_status"], "FAILED")
        self.assertFalse(report["pass"])
        self.assertIn("worker startup failed", report["errors"][0])

    def test_deterministic_evaluation_reports_undefined_no_trade_metrics(self):
        frame = self.frame.iloc[:1].copy()
        frame["ema_fast"], frame["ema_slow"] = 0., 0.
        settings = dict(feature_columns=["ema_fast", "ema_slow"], initial_cash=10000,
                        quantity=1, point_value=1, slippage=0, commission=0, lambda_term=0)
        first = _evaluate(7, frame, settings, 252, baseline="buy_and_hold")
        self.assertEqual(first, _evaluate(7, frame, settings, 252, baseline="buy_and_hold"))
        self.assertIsNone(first["metrics"]["profit_factor"])
        self.assertIsNone(first["metrics"]["sharpe"])
        frame = self.frame.iloc[:40].copy()
        frame["ema_fast"], frame["ema_slow"] = 0., 0.
        measured = _evaluate(42, frame, settings, 252, baseline="buy_and_hold")
        self.assertEqual(measured, _evaluate(42, frame, settings, 252, baseline="buy_and_hold"))
        self.assertGreater(measured["metrics"]["trade_count"], 0)

    def test_real_cli_persists_report_and_spawn_safe_main_guard(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "orchestrator.py"), str(self.path),
             "--report-directory", str(self.reports)],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.reports / "offline_audit_report.json").read_text())
        self.assertEqual(report["run_status"], "COMPLETE")
        self.assertFalse(report["pass"])
        self.assertEqual(report["deployment"]["status"], "NOT_REQUESTED")
        self.assertIn('"median_oos_metrics"', result.stdout)

    def test_cli_refusal_returns_nonzero(self):
        self.path.write_text("{}")
        with self.assertLogs(level=logging.ERROR):
            self.assertEqual(main([str(self.path), "--report-directory", str(self.reports)]), 1)

    def test_componentwise_median_and_per_seed_research_guards(self):
        orchestrator = OfflineOrchestrator(self.reports)
        orchestrator.config = self.config
        orchestrator.state = ValidationState.READY_FOR_OFFLINE_EXECUTION
        orchestrator.context = {
            "run_status": "COMPLETE", "run_id": "synthetic-aggregate-test",
            "models_frozen_before_oos": True,
            "baselines": {"buy_and_hold": {}, "fixed_ema_crossover": {}},
        }
        results = []
        for seed, pf, sharpe in ((7, 1.1, 1.3), (42, 1.5, 1.1), (99, 10, 1.5)):
            results.append({
                "seed": seed, "status": "COMPLETE", "final_cash": 10001,
                "metrics": {"profit_factor": pf, "sharpe": sharpe, "max_drawdown": .05,
                            "trade_count": 40, "expectancy": 1},
                "trades": [], "fills": [],
                "training": {"synthetic_unit_test_evidence": True},
            })
        report = orchestrator.export_audit_report(results, "0"*64, "1"*64)
        self.assertEqual(report["median_oos_metrics"]["profit_factor"], 1.5)
        self.assertEqual(report["median_oos_metrics"]["sharpe"], 1.3)
        self.assertTrue(report["threshold_pass"])
        self.assertTrue(report["pass"])
        self.config["optimization"] = {"role": "SEARCH_ONLY_REUSED_HOLDOUT"}
        search_report = orchestrator.export_audit_report(results, "0"*64, "1"*64)
        self.assertTrue(search_report["threshold_pass"])
        self.assertFalse(search_report["pass"])
        self.assertEqual(search_report["scope"], "HYPERPARAMETER_SEARCH_ONLY")
        self.config.pop("optimization")
        results[1]["metrics"]["profit_factor"] = None
        report = orchestrator.export_audit_report(results, "0"*64, "1"*64)
        self.assertIsNone(report["median_oos_metrics"]["profit_factor"])
        self.assertFalse(report["threshold_pass"])
        results[1]["metrics"]["profit_factor"] = 1.5
        results[0]["metrics"]["max_drawdown"] = .11
        report = orchestrator.export_audit_report(results, "0"*64, "1"*64)
        self.assertFalse(report["threshold_pass"])

    def test_training_barrier_and_deploy_opt_in_after_persisted_research_report(self):
        completed_training = []
        evaluated = []

        class ImmediatePool:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args):
                future = Future()
                try:
                    future.set_result(function(*args))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        def train(seed, train_frame, validation, _settings, _annualization, _budget, model_path, hyperparameters,
                  policy_architecture):
            self.assertEqual(policy_architecture, self.config["training"]["policy_architecture"])
            self.assertEqual(hyperparameters, self.config["training"]["ppo_hyperparameters"])
            self.assertLess(train_frame["timestamp"].max(), validation["timestamp"].min())
            self.assertLess(validation["timestamp"].max(), self.frame["timestamp"].iloc[110])
            completed_training.append(seed)
            return {"seed": seed, "model_path": model_path, "actual_timesteps": 2048}

        def evaluate(frozen, test, _settings, _annualization):
            self.assertEqual(set(completed_training), {7, 42, 99})
            self.assertEqual(test["timestamp"].min(), self.frame["timestamp"].iloc[110])
            evaluated.append(frozen["seed"])
            return {
                "seed": frozen["seed"], "status": "COMPLETE", "training": frozen,
                "final_cash": 10001, "metrics": {
                    "profit_factor": 1.5, "sharpe": 1.2, "max_drawdown": .05,
                    "trade_count": 40, "expectancy": 1,
                },
            }

        def gate(orchestrator, report):
            self.assertTrue(report["pass"])
            self.assertEqual(set(evaluated), {7, 42, 99})
            saved = json.loads((self.reports / "offline_audit_report.json").read_text())
            self.assertEqual(saved, report)
            return {"requested": True, "status": "STAGED_BUILD_SUCCEEDED"}

        with patch("orchestrator.ProcessPoolExecutor", ImmediatePool), \
             patch("orchestrator._train_seed", side_effect=train), \
             patch("orchestrator._evaluate_seed", side_effect=evaluate):
            with patch.object(OfflineOrchestrator, "execute_deployment_gate",
                              side_effect=AssertionError("offline build side effect")):
                report = OfflineOrchestrator(self.reports).run_pipeline(self.path)
                self.assertTrue(report["pass"])
            completed_training.clear()
            evaluated.clear()
            with patch.object(OfflineOrchestrator, "execute_deployment_gate", gate):
                report = OfflineOrchestrator(self.reports).run_pipeline(self.path, deploy=True)
                self.assertEqual(report["deployment"]["status"], "STAGED_BUILD_SUCCEEDED")

    def test_deploy_refusal_never_enters_build_gate(self):
        self.config["frozen"] = False
        self.save()
        with patch.object(OfflineOrchestrator, "execute_deployment_gate") as gate:
            with self.assertLogs("orchestrator", logging.ERROR):
                report = OfflineOrchestrator(self.reports).run_pipeline(self.path, deploy=True)
        gate.assert_not_called()
        self.assertEqual(report["deployment"]["status"], "BLOCKED_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
