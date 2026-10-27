"""Opt-in build, persistent fallback, and notifier-isolation integration tests."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from engine_mt5.patcher import CompilationError, SourcePatchError
from governance.validator import ValidationState
from orchestrator import OfflineOrchestrator


@pytest.fixture
def gate(tmp_path):
    source = tmp_path / "strategy.mq5"
    source.write_text("input int InpFast = 5; // OPTIMIZED:InpFast\n")
    source.with_suffix(".ex5").write_bytes(b"original binary")
    orchestrator = OfflineOrchestrator(tmp_path / "reports")
    orchestrator._config_base = tmp_path
    orchestrator.config = {
        "dataset": {"symbol": "TEST"},
        "deployment": {
            "source_path": str(source), "metaeditor_path": str(tmp_path / "metaeditor64.exe"),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameters": {"InpFast": 8},
            "parameter_binding": "fixed_strategy_parameters_not_ppo_policy",
        },
    }
    orchestrator.context = {"run_id": "candidate-1", "deployment": {"requested": True}}
    report = {
        "run_kind": "SB3_PPO_MULTI_SEED", "rl_training_performed": True,
        "run_status": "COMPLETE",
        "validation_state": ValidationState.READY_FOR_OFFLINE_EXECUTION.value,
        "pass": True, "median_oos_metrics": {
            "profit_factor": 1.5, "sharpe": 1.2, "max_drawdown": .05,
        },
    }
    notifier = Mock()
    notifier.send_deployment_success.return_value = True
    notifier.send_fallback_alert.return_value = True
    notifier.send_circuit_breaker.return_value = True
    with patch("governance.telegram_notifier.TelegramAlertEngine", return_value=notifier):
        yield orchestrator, report, source, notifier


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


def test_success_patches_and_compiles_staged_source_only(gate):
    orchestrator, report, source, notifier = gate
    with patch("engine_mt5.patcher.MQL5Patcher.compile_ea", side_effect=compile_fixture):
        result = orchestrator.execute_deployment_gate(report)
    assert result["status"] == "STAGED_BUILD_SUCCEEDED"
    assert "InpFast = 8;" in Path(result["staged_source"]).read_text()
    assert "InpFast = 5;" in source.read_text()
    assert source.with_suffix(".ex5").read_bytes() == b"original binary"
    assert result["policy_exported"] is False
    assert result["live_deployment_performed"] is False
    assert journal(orchestrator)["consecutive_failures"] == 0
    notifier.send_deployment_success.assert_called_once()
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


def test_policy_matrix_parameters_are_refused_by_fixed_parameter_gate(gate):
    orchestrator, report, source, _ = gate
    orchestrator.config["deployment"]["parameters"] = {"InpFast": 8, "PPO_WEIGHTS": [[0.0] * 12] * 4}
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
    assert not orchestrator.report_directory.exists()
    assert "InpFast = 5;" in source.read_text()
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
    assert "InpFast = 5;" in source.read_text()
