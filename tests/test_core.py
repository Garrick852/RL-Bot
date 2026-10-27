import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import subprocess
import numpy as np

from quant_optimization_bot.adapters.config_watcher import ActiveConfigWatcher
from quant_optimization_bot.data.collector import (
    check_feature_drift,
    engineer_features,
    fit_scaler_contract,
    ks_two_sample,
)
from quant_optimization_bot.engine_mt5.patcher import patch_and_compile, patch_source_text
from quant_optimization_bot.engine_mt5.mt5_runner import run_tester
from quant_optimization_bot.engine_rl.result_analysis import analyze_policy, write_rl_analysis
from quant_optimization_bot.governance.fallback import (
    FallbackStage,
    StrategyState,
    handle_failure,
)
from quant_optimization_bot.governance.persistence import persist_validated_config
from quant_optimization_bot.governance.report_analyzer import analyze_result_report, write_analysis_report
from quant_optimization_bot.governance.validator import OOSMetrics, split_oos, validate
from quant_optimization_bot.orchestrator import MatrixJob, run_matrix, worker_limit


class GovernanceTests(unittest.TestCase):
    def test_report_analysis_separates_tester_metrics_from_deployment_evidence(self):
        raw = {"profit_factor": 0.8, "sharpe": -0.2, "max_drawdown_pct": 14.0, "trade_count": 12}
        analysis = analyze_result_report(raw)
        self.assertEqual(analysis.status, "UNVERIFIED")
        self.assertFalse(analysis.deployment_eligible)
        self.assertTrue(any("held-out OOS" in item for item in analysis.recommendations))
        self.assertTrue(any("below break-even" in item for item in analysis.recommendations))
        self.assertTrue(any("negative" in item for item in analysis.recommendations))
        self.assertTrue(any("below 30" in item for item in analysis.recommendations))

        passed_metrics = OOSMetrics(1.4, 1.2, 5.0, 40)
        drift = check_feature_drift(
            [{"feature": float(index)} for index in range(20)],
            [{"feature": float(index)} for index in range(20)],
            features=("feature",),
        )
        approved = analyze_result_report(
            raw,
            oos_metrics=passed_metrics,
            drift_report=drift,
        )
        self.assertEqual(approved.status, "VALIDATED")
        self.assertTrue(approved.deployment_eligible)
        with tempfile.TemporaryDirectory() as temporary:
            report_path = write_analysis_report(approved, Path(temporary) / "analysis.md")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Deployment eligible:** yes", report)
            self.assertIn("## Recommendations", report)

    def test_rl_report_evaluates_actions_and_ranks_indicator_sensitivity(self):
        class Environment:
            features = np.asarray(
                [[-2.0, 0.0], [-1.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                dtype=np.float32,
            )

            def reset(self):
                self.index = 0
                self.position = 0.0
                return self.features[0], {}

            def step(self, action):
                exposure = float(action) - 1.0
                reward = exposure * 0.01 - abs(exposure - self.position) * 0.0001
                self.position = exposure
                self.index += 1
                terminated = self.index == len(self.features) - 1
                return (
                    self.features[self.index],
                    reward,
                    terminated,
                    False,
                    {"position": self.position},
                )

        class Policy:
            def predict(self, observation, deterministic):
                self.assert_deterministic = deterministic
                return (2 if observation[0] > 0 else 0), None

        analysis = analyze_policy(
            Policy(),
            Environment(),
            feature_names=("rsi", "atr_pct"),
            training_health={
                "approx_kl": 0.03,
                "clip_fraction": 0.5,
                "explained_variance": 0.4,
            },
        )
        self.assertEqual(analysis.status, "UNVERIFIED")
        self.assertFalse(analysis.deployment_eligible)
        self.assertEqual(analysis.action_summary, {"0": 2, "2": 1})
        self.assertGreater(analysis.feature_sensitivity["rsi"], 0)
        self.assertEqual(analysis.feature_sensitivity["atr_pct"], 0)
        self.assertTrue(any("approx_kl" in item for item in analysis.recommendations))
        self.assertTrue(any("clip_fraction" in item for item in analysis.recommendations))
        self.assertTrue(any("explained_variance" in item for item in analysis.recommendations))
        with tempfile.TemporaryDirectory() as temporary:
            path = write_rl_analysis(analysis, Path(temporary) / "rl.md")
            report = path.read_text(encoding="utf-8")
            self.assertIn("RL strategy and indicator analysis", report)
            self.assertIn("`rsi`", report)
            self.assertIn("not causal indicator importance", report)

    def test_failed_oos_or_drift_remains_blocked(self):
        failed_oos = analyze_result_report(
            {},
            oos_metrics=OOSMetrics(0.8, 0.2, 20.0, 8),
        )
        self.assertEqual(failed_oos.status, "REJECTED")
        self.assertFalse(failed_oos.deployment_eligible)
        shifted = check_feature_drift(
            [{"feature": float(index)} for index in range(20)],
            [{"feature": float(index + 100)} for index in range(20)],
            features=("feature",),
        )
        blocked = analyze_result_report(
            {},
            oos_metrics=OOSMetrics(1.5, 1.2, 5.0, 40),
            drift_report=shifted,
        )
        self.assertEqual(blocked.status, "DRIFT_BLOCKED")
        self.assertFalse(blocked.deployment_eligible)
        self.assertTrue(any("shifted feature" in item for item in blocked.recommendations))

    def test_validation_enforces_all_hard_gates(self):
        metrics = OOSMetrics(1.3, 1.2, 9.0, 30, (0.1, 0.2, 0.3), (7, 42, 99))
        self.assertTrue(validate(metrics, require_seed_stability=True).passed)
        duplicated = OOSMetrics(1.3, 1.2, 9.0, 30, (0.1, 0.2, 0.3), (7, 7, 7))
        self.assertFalse(validate(duplicated, require_seed_stability=True).passed)
        rejected = validate(
            OOSMetrics(1.1, 0.9, 11.0, 12, (0.1, -0.2), (7, 42)),
            require_seed_stability=True,
        )
        self.assertFalse(rejected.passed)
        self.assertEqual(len(rejected.reasons), 6)
        self.assertFalse(validate(OOSMetrics("1.4", 1.1, 8.0, 40)).passed)

    def test_oos_split_reserves_latest_rows(self):
        train, test = split_oos(list(range(10)))
        self.assertEqual(train, list(range(8)))
        self.assertEqual(test, [8, 9])
        with self.assertRaises(ValueError):
            split_oos([1], fraction=0.2)

    def test_fallback_retries_then_degrades_and_suspends(self):
        rejected = OOSMetrics(0.8, 0.2, 20.0, 8)
        accepted = OOSMetrics(1.4, 1.2, 6.0, 40)
        retried = handle_failure(
            rejected,
            {"lot_size": 1.0},
            consecutive_failures=2,
            recalibrate=lambda: ({"lot_size": 0.7}, accepted),
        )
        self.assertTrue(retried.accepted)
        self.assertEqual(retried.consecutive_failures, 0)
        passing = handle_failure(
            accepted,
            {"lot_size": 1.0},
            candidate_parameters={"lot_size": 0.7},
            consecutive_failures=0,
        )
        self.assertEqual(passing.parameters, {"lot_size": 0.7})
        degraded = handle_failure(rejected, {"lot_size": 1.0}, consecutive_failures=0)
        self.assertEqual(degraded.stage, FallbackStage.DEGRADE)
        self.assertEqual(degraded.lot_size_scale, 0.5)
        self.assertFalse(degraded.close_only)
        halted = handle_failure(rejected, {}, consecutive_failures=2)
        self.assertEqual(halted.stage, FallbackStage.CIRCUIT_BREAKER)
        self.assertEqual(halted.state, StrategyState.SUSPENDED)
        self.assertTrue(halted.close_only)

    def test_only_validated_config_is_persisted_and_hash_verified(self):
        metrics = OOSMetrics(1.4, 1.1, 8.0, 40)
        result = validate(metrics)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "active.json"
            config = persist_validated_config(path, {"InpFastEMA": 12}, result)
            self.assertEqual(ActiveConfigWatcher(path).read_validated(), config)
            path.write_text(path.read_text(encoding="utf-8").replace("12", "13"))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ActiveConfigWatcher(path).read_validated()
        with self.assertRaisesRegex(ValueError, "failed OOS"):
            persist_validated_config("unused.json", {}, validate(OOSMetrics(0, 0, 50, 1)))
        forged = type(result)(True, (), OOSMetrics(0.5, 0.2, 30.0, 3))
        with self.assertRaisesRegex(ValueError, "failed OOS"):
            persist_validated_config("unused.json", {}, forged)


class DataTests(unittest.TestCase):
    def test_feature_contract_and_ks_drift_gate(self):
        bars = []
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for index in range(40):
            close = 100.0 + index * 0.2
            bars.append(
                {
                    "timestamp": start + timedelta(minutes=15 * index),
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        features = engineer_features(bars)
        self.assertEqual(len(features), len(bars))
        self.assertEqual(
            {"ret1", "rsi", "bb_pos", "atr_pct", "macd_hist", "hr_sin", "hr_cos"},
            set(features[0]) - {"timestamp"},
        )
        statistic, p_value = ks_two_sample(list(range(20)), list(range(100, 120)))
        self.assertEqual(statistic, 1.0)
        self.assertLess(p_value, 0.01)
        self.assertAlmostEqual(
            ks_two_sample([0, 1, 2, 3], [10, 11, 12, 13])[1],
            2.0 / 70.0,
        )
        report = check_feature_drift(
            [{"core": float(value)} for value in range(20)],
            [{"core": float(value + 100)} for value in range(20)],
            features=("core",),
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.drifted_features, ("core",))
        contract = fit_scaler_contract(
            [{"feature": 1.0}, {"feature": 3.0}],
            features=("feature",),
        )
        self.assertEqual(contract["features"]["feature"], {"mean": 2.0, "scale": 1.0})


class PatcherTests(unittest.TestCase):
    def test_tester_run_writes_companion_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            terminal = folder / "terminal64.exe"
            ini = folder / "tester.ini"
            report = folder / "run.xml"
            terminal.write_text("")
            ini.write_text(f"[Tester]\nReport={report}\n")

            def successful_test(command, **_kwargs):
                config = Path(next(arg for arg in command if arg.startswith("/config:")).removeprefix("/config:"))
                report_path = next(
                    line.split("=", 1)[1]
                    for line in config.read_text().splitlines()
                    if line.startswith("Report=")
                )
                Path(report_path).write_text(
                    "<TesterReport><ProfitFactor>0.9</ProfitFactor>"
                    "<Sharpe>0.2</Sharpe><TotalTrades>10</TotalTrades></TesterReport>"
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "quant_optimization_bot.engine_mt5.mt5_runner.subprocess.run",
                side_effect=successful_test,
            ):
                result = run_tester(terminal, ini, report)
            self.assertEqual(result.analysis.status, "UNVERIFIED")
            self.assertEqual(result.metrics["profit_factor"], 0.9)
            self.assertTrue(result.analysis_path.is_file())
            markdown = result.analysis_path.read_text(encoding="utf-8")
            self.assertIn("Deployment eligible:** no", markdown)
            self.assertIn("profit factor is below break-even", markdown)

    def test_patches_only_exact_optimized_declaration(self):
        source = (
            "input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars\n"
            "input int InpOther = 7;\n"
        )
        result = patch_source_text(source, {"InpLeftBars": 12})
        self.assertIn("InpLeftBars = 12;", result)
        self.assertIn("InpOther = 7;", result)
        with self.assertRaisesRegex(ValueError, "found 0"):
            patch_source_text(source, {"InpOther": 9})
        with self.assertRaises(ValueError):
            patch_source_text(source, {"InpLeftBars": "4; input int Bad = 1"})

    def test_source_is_replaced_only_after_clean_compile_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "Expert.mq5"
            compiler = folder / "metaeditor64.exe"
            source.write_text("input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars\n")
            compiler.write_text("")

            def successful_compile(command, **_kwargs):
                staged_source = Path(next(arg for arg in command if arg.startswith("/compile:")).removeprefix("/compile:"))
                log_arg = next(arg for arg in command if arg.startswith("/log:"))
                Path(log_arg.removeprefix("/log:")).write_text("0 errors, 0 warnings")
                staged_source.with_suffix(".ex5").write_bytes(b"compiled artifact")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "quant_optimization_bot.engine_mt5.patcher.subprocess.run",
                side_effect=successful_compile,
            ):
                patch_and_compile(source, {"InpLeftBars": 12}, metaeditor=compiler)
            self.assertIn("InpLeftBars = 12", source.read_text())
            self.assertEqual(source.with_suffix(".ex5").read_bytes(), b"compiled artifact")

    def test_failed_compile_preserves_source_and_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "Expert.mq5"
            compiler = folder / "metaeditor64.exe"
            source.write_bytes(
                b"\xff\xfe" + "input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars\r\n".encode("utf-16-le")
            )
            compiler.write_text("")
            binary = source.with_suffix(".ex5")
            binary.write_bytes(b"old binary")

            def failed_compile(command, **_kwargs):
                log_arg = next(arg for arg in command if arg.startswith("/log:"))
                Path(log_arg.removeprefix("/log:")).write_text("1 errors, 0 warnings")
                return subprocess.CompletedProcess(command, 1, "", "")

            with patch(
                "quant_optimization_bot.engine_mt5.patcher.subprocess.run",
                side_effect=failed_compile,
            ):
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    patch_and_compile(source, {"InpLeftBars": 12}, metaeditor=compiler)
            self.assertIn("InpLeftBars = 5;".encode("utf-16-le"), source.read_bytes())
            self.assertEqual(binary.read_bytes(), b"old binary")

    def test_successful_compile_preserves_utf16_source_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "Expert.mq5"
            compiler = folder / "metaeditor64.exe"
            source.write_bytes(
                b"\xff\xfe" + "input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars\r\n".encode("utf-16-le")
            )
            compiler.write_text("")

            def successful_compile(command, **_kwargs):
                staged_source = Path(next(arg for arg in command if arg.startswith("/compile:")).removeprefix("/compile:"))
                Path(next(arg for arg in command if arg.startswith("/log:")).removeprefix("/log:")).write_text(
                    "0 errors, 0 warnings"
                )
                staged_source.with_suffix(".ex5").write_bytes(b"new binary")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "quant_optimization_bot.engine_mt5.patcher.subprocess.run",
                side_effect=successful_compile,
            ):
                patch_and_compile(source, {"InpLeftBars": 12}, metaeditor=compiler)
            content = source.read_bytes()
            self.assertTrue(content.startswith(b"\xff\xfe"))
            self.assertIn("InpLeftBars = 12;".encode("utf-16-le"), content)


class OrchestratorTests(unittest.TestCase):
    def test_cpu_limit_and_rl_jobs_execute_sequentially(self):
        self.assertEqual(worker_limit(cpu_count=8), 4)
        self.assertEqual(worker_limit(configured=5, cpu_count=4), 2)
        jobs = [
            MatrixJob("EURUSD", "PPO", kind="sb3"),
            MatrixJob("US500.cash", "PPO2", kind="torchrl"),
        ]
        result = run_matrix(jobs, lambda job: job.strategy)
        self.assertEqual([value for _, value in result], ["PPO", "PPO2"])


if __name__ == "__main__":
    unittest.main()
