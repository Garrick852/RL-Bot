"""Opt-in build, policy/alpha injection, persistent fallback, and notifier-isolation tests."""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from alpha_fixtures import install_alpha
from engine_mt5.patcher import CompilationError, SourcePatchError
from engine_rl import linear_policy as contract
from governance.validator import ValidationState
from models.onnx_bridge import load_manifest, mql5_zscore_inputs
from orchestrator import POLICY_PARAMETER_BINDING, OfflineOrchestrator
from test_ea_template import TEMPLATE, contract_env, parse_matrix, parse_vector

SEEDS = (7, 42, 99)


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    """Untrained but structurally exact linear-actor and MLP PPO checkpoints."""
    torch = pytest.importorskip("torch")
    from stable_baselines3 import PPO

    directory = tmp_path_factory.mktemp("policies")
    previous = torch.get_num_threads()
    try:
        models = {}
        for name, arch in (("linear", {"pi": [], "vf": [128, 128]}), ("mlp", {"pi": [16], "vf": [16]})):
            model = PPO("MlpPolicy", contract_env(60), policy_kwargs={"net_arch": arch}, seed=3, device="cpu")
            path = directory / f"{name}.zip"
            model.save(path)
            models[name] = path
        expected = contract.export_linear_policy(PPO.load(models["linear"], device="cpu"))
    finally:
        torch.set_num_threads(previous)
    return models, expected


@pytest.fixture
def gate(tmp_path, policies):
    models, _ = policies
    source = tmp_path / "strategy.mq5"
    source.write_bytes(TEMPLATE.read_bytes())
    source.with_suffix(".ex5").write_bytes(b"original binary")
    manifest_name, manifest_sha = install_alpha(tmp_path)
    manifest = load_manifest(tmp_path / manifest_name)
    orchestrator = OfflineOrchestrator(tmp_path / "reports")
    orchestrator._config_base = tmp_path
    orchestrator.config = {
        "dataset": {"symbol": "TEST"},
        "training": {"seeds": list(SEEDS)},
        "features": {"alpha_manifest_path": manifest_name, "alpha_manifest_sha256": manifest_sha},
        "deployment": {
            "source_path": str(source), "metaeditor_path": str(tmp_path / "metaeditor64.exe"),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameters": {},
            "parameter_binding": POLICY_PARAMETER_BINDING, "policy_seed": 42,
        },
    }
    orchestrator.context = {
        "run_id": "candidate-1", "deployment": {"requested": True},
        "alpha": {"manifest_sha256": manifest_sha, "onnx_sha256": manifest["onnx_sha256"]},
    }
    model_sha = hashlib.sha256(models["linear"].read_bytes()).hexdigest()
    report = {
        "run_kind": "SB3_PPO_MULTI_SEED", "rl_training_performed": True,
        "run_status": "COMPLETE",
        "validation_state": ValidationState.READY_FOR_OFFLINE_EXECUTION.value,
        "pass": True, "median_oos_metrics": {
            "profit_factor": 1.5, "sharpe": 1.2, "max_drawdown": .05,
        },
        "results": [{"seed": seed, "training": {
            "model_path": str(models["linear"]), "model_sha256": model_sha,
            "policy_architecture": "linear_actor",
        }} for seed in SEEDS],
    }
    notifier = Mock()
    notifier.send_deployment_success.return_value = True
    notifier.send_fallback_alert.return_value = True
    notifier.send_circuit_breaker.return_value = True
    with patch("governance.telegram_notifier.TelegramAlertEngine", return_value=notifier):
        yield orchestrator, report, source, notifier


def input_value(text, name):
    return re.search(rf"^input\s+\w+\s+{name}\s*=\s*([^;]+);", text, re.M).group(1).strip()


def compile_fixture(_compiler, source):
    """Represent compiler output without invoking MetaEditor."""
    source = Path(source)
    source.with_suffix(".ex5").write_bytes(b"new staged binary")
    log = source.with_suffix(".log")
    log.write_text("0 error(s), 0 warning(s)")
    return log


def journal(orchestrator):
    paths = list((orchestrator.report_directory / "fallback").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_success_injects_verified_policy_and_alpha_into_staged_source_only(gate, policies):
    orchestrator, report, source, notifier = gate
    _, expected = policies
    original = source.read_bytes()
    with patch("engine_mt5.patcher.MQL5Patcher.compile_ea", side_effect=compile_fixture):
        result = orchestrator.execute_deployment_gate(report)
    assert result["status"] == "STAGED_BUILD_SUCCEEDED", result.get("reason")
    staged = Path(result["staged_source"]).read_text()
    weights = parse_matrix(staged, "PPO_WEIGHTS")
    bias = parse_vector(staged, "PPO_BIAS")
    assert weights.shape == (contract.ACTION_DIM, contract.OBS_DIM)
    np.testing.assert_array_equal(weights, np.asarray(expected["PPO_WEIGHTS"]))
    np.testing.assert_array_equal(bias, np.asarray(expected["PPO_BIAS"]))
    manifest = load_manifest(source.parent / "xgb_alpha.manifest.json")
    for name, value in mql5_zscore_inputs(manifest).items():
        assert float(input_value(staged, name)) == value
    windows = manifest["feature_engineer"]
    assert int(input_value(staged, "InpRsiWindow")) == windows["rsi_window"]
    assert int(input_value(staged, "InpMacdSlow")) == windows["macd_slow"]
    assert int(input_value(staged, "InpPAWindow")) == windows["price_action_window"]
    assert source.read_bytes() == original
    assert source.with_suffix(".ex5").read_bytes() == b"original binary"
    assert result["policy_exported"] is True
    assert result["policy_seed"] == 42
    assert result["policy_model_sha256"] == report["results"][1]["training"]["model_sha256"]
    assert result["alpha_onnx_sha256"] == manifest["onnx_sha256"]
    assert result["required_onnx_file"] == "xgb_alpha.onnx"
    assert result["live_deployment_performed"] is False
    assert journal(orchestrator)["consecutive_failures"] == 0
    notifier.send_deployment_success.assert_called_once()
    summary = notifier.send_deployment_success.call_args.args[2]
    assert "PPO_WEIGHTS" not in summary and summary["policy_seed"] == 42
    assert not list((orchestrator.report_directory / "fallback").glob("*.lock"))


def test_fallback_counts_are_persisted_and_suspension_cannot_auto_reset(gate):
    orchestrator, report, _, notifier = gate
    report["pass"] = False
    with patch("engine_mt5.patcher.MQL5Patcher") as patcher:
        for index, state in enumerate(("RECALIBRATE_AND_RETRY", "GRACEFUL_DEGRADATION", "SUSPENDED"), 1):
            orchestrator.context["run_id"] = f"candidate-{index}"
            result = orchestrator.execute_deployment_gate(report)
            assert result["status"] == state
            assert journal(orchestrator)["consecutive_failures"] == index
            if index == 2:
                assert result["directive"]["risk_size_scale"] == .5
        # A fresh coordinator with passing metrics must honor the on-disk latch.
        resumed = OfflineOrchestrator(orchestrator.report_directory)
        resumed.config = orchestrator.config
        resumed._config_base = orchestrator._config_base
        resumed.context = {"run_id": "passing-candidate", "deployment": {"requested": True}}
        report["pass"] = True
        assert resumed.execute_deployment_gate(report)["status"] == "SUSPENDED"
        assert journal(resumed)["consecutive_failures"] == 3
        patcher.assert_not_called()
    assert notifier.send_fallback_alert.call_count == 2
    assert notifier.send_circuit_breaker.call_count == 2


@pytest.mark.parametrize("failure", [SourcePatchError("missing anchor"), CompilationError("one warning"),
                                    RuntimeError("unexpected build exception")])
def test_failed_build_invalidates_and_deletes_only_staged_binary(gate, failure):
    orchestrator, report, source, notifier = gate

    def fail(_compiler, staged):
        Path(staged).with_suffix(".ex5").write_bytes(b"must be removed")
        raise failure

    with patch("engine_mt5.patcher.MQL5Patcher.compile_ea", side_effect=fail):
        result = orchestrator.execute_deployment_gate(report)
    assert result["status"] == "INVALID"
    assert not source.with_name("strategy.candidate-1.staged.ex5").exists()
    assert source.with_suffix(".ex5").read_bytes() == b"original binary"
    notifier.send_circuit_breaker.assert_called_once()
    notifier.send_deployment_success.assert_not_called()


def test_real_missing_anchor_never_invokes_compiler(gate):
    orchestrator, report, _, notifier = gate
    orchestrator.config["deployment"]["parameters"] = {"Missing": 9}
    with patch("engine_mt5.patcher.MQL5Patcher.compile_ea") as compiler:
        assert orchestrator.execute_deployment_gate(report)["status"] == "INVALID"
        compiler.assert_not_called()
    notifier.send_circuit_breaker.assert_called_once()


def test_config_supplied_policy_matrix_or_alpha_override_is_refused(gate):
    orchestrator, report, source, _ = gate
    for parameters in ({"PPO_WEIGHTS": [[0.0] * 12] * 3}, {"InpAlphaMean0": 0.0}, {"InpRsiWindow": 99}):
        orchestrator.config["deployment"]["parameters"] = parameters
        with patch("engine_mt5.patcher.MQL5Patcher") as patcher:
            result = orchestrator.execute_deployment_gate(report)
        assert result["status"] == "INVALID"
        patcher.assert_not_called()
        orchestrator.context["run_id"] += "-next"
    assert not list(source.parent.glob("*.staged.mq5"))


@pytest.mark.parametrize("tamper", ["mlp", "model_sha", "seed", "binding", "alpha", "undeclared_seed"])
def test_unverifiable_policy_or_alpha_never_reaches_patcher(gate, policies, tamper):
    orchestrator, report, source, _ = gate
    models, _ = policies
    selected = report["results"][1]["training"]
    if tamper == "mlp":
        selected.update(model_path=str(models["mlp"]),
                        model_sha256=hashlib.sha256(models["mlp"].read_bytes()).hexdigest(),
                        policy_architecture="mlp_tanh_128x128")
    elif tamper == "model_sha":
        selected["model_sha256"] = "0" * 64
    elif tamper == "seed":
        report["results"] = [row for row in report["results"] if row["seed"] != 42]
    elif tamper == "binding":
        orchestrator.config["deployment"]["parameter_binding"] = "fixed_strategy_parameters_not_ppo_policy"
    elif tamper == "alpha":
        orchestrator.context["alpha"]["onnx_sha256"] = "0" * 64
    else:
        orchestrator.config["deployment"]["policy_seed"] = 5
    with patch("engine_mt5.patcher.MQL5Patcher") as patcher:
        result = orchestrator.execute_deployment_gate(report)
    assert result["status"] == "INVALID"
    patcher.assert_not_called()
    assert not list(source.parent.glob("*.staged.mq5"))


@pytest.mark.parametrize("passing", [True, False])
def test_notifier_exception_never_changes_build_or_fallback_decision(gate, passing):
    orchestrator, report, _, notifier = gate
    report["pass"] = passing
    notifier.send_deployment_success.side_effect = RuntimeError("network")
    notifier.send_fallback_alert.side_effect = RuntimeError("network")
    with patch("engine_mt5.patcher.MQL5Patcher.compile_ea", side_effect=compile_fixture):
        result = orchestrator.execute_deployment_gate(report)
    assert result["status"] == ("STAGED_BUILD_SUCCEEDED" if passing else "RECALIBRATE_AND_RETRY")
    assert result["notification_delivered"] is False


def test_unauthorized_gate_has_no_source_journal_or_notification_effects(gate):
    orchestrator, report, source, notifier = gate
    orchestrator.context["deployment"]["requested"] = False
    assert orchestrator.execute_deployment_gate(report)["status"] == "BLOCKED_NOT_AUTHORIZED"
    assert orchestrator.report_directory.exists() is False
    assert source.read_bytes() == TEMPLATE.read_bytes()
    notifier.send_circuit_breaker.assert_not_called()
    notifier.send_deployment_success.assert_not_called()


def test_locked_journal_fails_closed_without_removing_other_runs_lock(gate):
    orchestrator, report, source, _ = gate
    key = hashlib.sha256(f"TEST|{source}".encode()).hexdigest()
    directory = orchestrator.report_directory / "fallback"
    directory.mkdir(parents=True)
    lock = directory / f"{key}.lock"
    lock.write_text("another run")
    assert orchestrator.execute_deployment_gate(report)["status"] == "INVALID"
    assert lock.read_text() == "another run"


def test_corrupt_journal_and_changed_source_fail_closed(gate):
    orchestrator, report, source, _ = gate
    source.write_text("changed after freeze")
    assert orchestrator.execute_deployment_gate(report)["status"] == "INVALID"
    key = hashlib.sha256(f"TEST|{source}".encode()).hexdigest()
    (orchestrator.report_directory / "fallback" / f"{key}.json").write_text(
        '{"consecutive_failures": 3, "suspended": false, "processed_runs": []}'
    )
    assert orchestrator.execute_deployment_gate(report)["status"] == "INVALID"


def test_search_evidence_cannot_enter_build_gate(gate):
    orchestrator, report, source, _ = gate
    orchestrator.config["optimization"] = {"role": "SEARCH_ONLY_REUSED_HOLDOUT"}
    with patch("engine_mt5.patcher.MQL5Patcher") as patcher:
        result = orchestrator.execute_deployment_gate(report)
        assert result["status"] == "INVALID"
        assert "fresh holdout" in result["reason"]
        patcher.assert_not_called()
    assert not orchestrator.report_directory.exists()
    assert source.read_bytes() == TEMPLATE.read_bytes()
