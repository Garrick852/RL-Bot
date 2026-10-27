"""Single-shot holdout: SHA lock, burn-before-evaluate, integrity preflight, and audit fields."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import test_orchestrator as experiment_fixtures
from engine_rl.alpha_pipeline import load_alpha_model
from engine_rl.envs.trading_env import DoubleEntryReconciliationError
from scripts import holdout_eval
from scripts.holdout_eval import HoldoutAlreadyUsedError, HoldoutPreflightError, main, run_holdout
from test_ea_template import contract_env


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def linear_models(tmp_path_factory):
    """Untrained but structurally exact linear-actor and MLP PPO checkpoints."""
    torch = pytest.importorskip("torch")
    from stable_baselines3 import PPO

    directory = tmp_path_factory.mktemp("holdout_policies")
    previous = torch.get_num_threads()
    try:
        paths = {}
        for name, arch in (("linear", {"pi": [], "vf": [128, 128]}), ("mlp", {"pi": [16], "vf": [16]})):
            model = PPO("MlpPolicy", contract_env(60), policy_kwargs={"net_arch": arch}, seed=3, device="cpu")
            paths[name] = directory / f"{name}.zip"
            model.save(paths[name])
    finally:
        torch.set_num_threads(previous)
    return paths


class Setup:
    """A complete synthetic search result plus an isolated holdout window (rows 320-400)."""

    def __init__(self, root: Path, models: dict[str, Path]):
        fixture = experiment_fixtures.OrchestratorTests()
        fixture.setUp()
        self.cleanup = fixture.doCleanups
        times = fixture.frame["timestamp"]
        fixture.config["splits"]["test"] = {
            "start": (times[0] + pd.Timedelta(minutes=270)).isoformat(),
            "end": (times[0] + pd.Timedelta(minutes=300)).isoformat(),
        }
        fixture.save()
        self.fixture, self.root, self.times = fixture, root, times
        self.trial_config = fixture.path
        seeds = fixture.config["training"]["seeds"]
        model_dir = root / "models"
        model_dir.mkdir()
        rows = []
        for seed in seeds:
            path = model_dir / f"seed-{seed}.zip"
            shutil.copyfile(models["linear"], path)
            rows.append({"seed": seed, "status": "COMPLETE", "metrics": {"profit_factor": 1.5}, "training": {
                "seed": seed, "model_path": str(path), "model_sha256": sha(path),
                "policy_architecture": "linear_actor",
            }})
        features = fixture.config["features"]
        alpha = load_alpha_model(fixture.root / features["alpha_manifest_path"], features["alpha_manifest_sha256"])
        self.evidence = root / "candidate_evidence.json"
        self.write_evidence({
            "run_status": "COMPLETE", "validation_state": "READY_FOR_OFFLINE_EXECUTION",
            "models_frozen_before_oos": True, "config_hash": sha(self.trial_config),
            "alpha": {"manifest_sha256": alpha.manifest_sha256, "onnx_sha256": alpha.onnx_sha256},
            "results": rows,
        })
        self.data = fixture.root / "bars.csv"
        self.schedule = root / "rth.json"
        self.schedule.write_text(json.dumps([stamp.isoformat() for stamp in times]))
        self.flag = root / "data" / "holdout_used.flag"
        self.report = root / "reports" / "true_holdout_eval.json"
        self.candidate = root / "best_candidate_config.json"
        self.write_candidate()

    def write_evidence(self, evidence):
        self.evidence.write_text(json.dumps(evidence))

    def write_candidate(self, **overrides):
        candidate = {
            "contract": "holdout_candidate_v1", "trial_number": 0,
            "objective": {"metric": "median_profit_factor", "value": 1.5}, "params": {"learning_rate": 3e-4},
            "trial_config_path": str(self.trial_config), "trial_config_sha256": sha(self.trial_config),
            "evidence_path": str(self.evidence), "evidence_sha256": sha(self.evidence),
            "theta_accept": {"metric": "median_profit_factor", "value": 1.25},
            "requires_fresh_holdout": True, **overrides,
        }
        self.candidate.write_text(json.dumps(candidate))

    def run(self, data=None, digest=None, start=320, end=400):
        data = data or self.data
        return run_holdout(
            data, digest or sha(data),
            (self.times[0] + pd.Timedelta(minutes=start)).isoformat(),
            (self.times[0] + pd.Timedelta(minutes=end)).isoformat(),
            self.schedule, candidate_path=self.candidate, flag_path=self.flag, report_path=self.report,
        )


@pytest.fixture
def setup(tmp_path, linear_models):
    value = Setup(tmp_path, linear_models)
    try:
        yield value
    finally:
        value.cleanup()


def test_sha_mismatch_raises_value_error_and_never_burns(setup):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup.run(digest="0" * 64)
    assert not setup.flag.exists() and not setup.report.exists()


def test_existing_flag_or_prior_audit_refuses(setup):
    setup.flag.parent.mkdir()
    setup.flag.write_text("burned")
    with pytest.raises(HoldoutAlreadyUsedError):
        setup.run()
    assert setup.flag.read_text() == "burned"
    setup.flag.unlink()
    setup.report.parent.mkdir()
    setup.report.write_text("{}")
    with pytest.raises(HoldoutAlreadyUsedError, match="flag was removed"):
        setup.run()
    assert not setup.flag.exists()


@pytest.mark.parametrize("tamper", ["candidate_contract", "trial_config", "evidence", "model", "mlp_actor",
                                    "weak_theta", "overlap", "missing_seed", "alpha"])
def test_integrity_failures_refuse_before_burning(setup, linear_models, tamper):
    start = 320
    if tamper == "candidate_contract":
        setup.write_candidate(contract="params_only")
    elif tamper == "trial_config":
        setup.trial_config.write_bytes(setup.trial_config.read_bytes() + b" ")
    elif tamper == "evidence":
        setup.evidence.write_bytes(setup.evidence.read_bytes() + b" ")
    elif tamper == "weak_theta":
        setup.write_candidate(theta_accept={"metric": "median_profit_factor", "value": 1.0})
    elif tamper == "overlap":
        start = 290
    else:
        evidence = json.loads(setup.evidence.read_text())
        row = evidence["results"][0]["training"]
        if tamper == "model":
            Path(row["model_path"]).write_bytes(b"tampered")
        elif tamper == "mlp_actor":
            shutil.copyfile(linear_models["mlp"], row["model_path"])
            row["model_sha256"] = sha(row["model_path"])
        elif tamper == "missing_seed":
            evidence["results"].pop()
        else:
            evidence["alpha"]["onnx_sha256"] = "0" * 64
        setup.write_evidence(evidence)
        setup.write_candidate()
    with pytest.raises(HoldoutPreflightError):
        setup.run(start=start)
    assert not setup.flag.exists() and not setup.report.exists()


def test_flag_is_burned_before_any_evaluation_and_report_has_required_fields(setup):
    import orchestrator

    real = orchestrator._evaluate_seed
    calls = []

    def guarded(frozen, active, settings, annualization):
        assert setup.flag.is_file(), "evaluation started before the holdout was burned"
        calls.append(frozen["seed"])
        assert active["timestamp"].iloc[0] == setup.times[320]
        return real(frozen, active, settings, annualization)

    with patch("orchestrator._evaluate_seed", side_effect=guarded):
        report = setup.run()
    assert calls == setup.fixture.config["training"]["seeds"]
    saved = json.loads(setup.report.read_text())
    assert saved == report
    for field in ("median_oos", "reconciliation_ok", "gap_ok", "evaluated_once", "accepted"):
        assert field in saved
    assert saved["evaluated_once"] is True
    assert saved["gap_ok"] is True and saved["reconciliation_ok"] is True
    assert saved["holdout"]["partition"]["rows"] == 80
    burned = json.loads(setup.flag.read_text())
    assert burned["dataset_sha256"] == sha(setup.data) and burned["candidate_sha256"] == sha(setup.candidate)
    medians = [row["metrics"]["profit_factor"] for row in saved["seeds"]]
    if saved["median_oos"] is None:
        assert None in medians and saved["accepted"] is False
    else:
        assert saved["accepted"] is (saved["median_oos"] >= 1.25)
    with pytest.raises(HoldoutAlreadyUsedError):
        setup.run()


def test_acceptance_requires_theta_and_every_check(setup):
    def scored(value):
        def fake(frozen, active, settings, annualization):
            return {"status": "COMPLETE", "metrics": {"profit_factor": value}, "undefined_reasons": [],
                    "initial_cash": 10000, "final_cash": 10100, "terminal_liquidation_pnl": 0}
        return fake

    with patch("orchestrator._evaluate_seed", side_effect=scored(1.3)):
        report = setup.run()
    assert report["median_oos"] == 1.3 and report["accepted"] is True

    setup.flag.chmod(0o666)  # the burn flag is created read-only
    setup.flag.unlink()
    setup.report.unlink()
    with patch("orchestrator._evaluate_seed", side_effect=scored(1.2)):
        report = setup.run()
    assert report["median_oos"] == 1.2 and report["accepted"] is False


def test_reconciliation_failure_is_recorded_after_burn(setup):
    with patch("orchestrator._evaluate_seed", side_effect=DoubleEntryReconciliationError("cash drift")):
        report = setup.run()
    assert setup.flag.is_file()
    assert report["reconciliation_ok"] is False and report["accepted"] is False
    assert report["median_oos"] is None and report["evaluated_once"] is True
    assert any("DoubleEntryReconciliationError" in error for error in report["errors"])


def test_gap_failure_is_recorded_without_evaluation(setup):
    frame = setup.fixture.frame.drop(index=350)
    polluted = setup.root / "holdout_gap.csv"
    frame.to_csv(polluted, index=False)
    with patch("orchestrator._evaluate_seed", side_effect=AssertionError("must not evaluate")):
        report = setup.run(data=polluted)
    assert setup.flag.is_file()
    assert report["gap_ok"] is False and report["accepted"] is False and report["median_oos"] is None
    assert report["evaluated_once"] is True


def test_cli_exit_codes(setup, monkeypatch, capsys):
    monkeypatch.chdir(setup.root)
    args = ["--data", str(setup.data), "--config", str(setup.candidate),
            "--holdout-start", (setup.times[0] + pd.Timedelta(minutes=320)).isoformat(),
            "--holdout-end", (setup.times[0] + pd.Timedelta(minutes=400)).isoformat(),
            "--rth-schedule", str(setup.schedule)]
    assert main([*args, "--sha256", "0" * 64]) == 2
    assert not Path("data/holdout_used.flag").exists()
    with patch("orchestrator._evaluate_seed", side_effect=DoubleEntryReconciliationError("drift")):
        assert main([*args, "--sha256", sha(setup.data)]) == 1
    assert Path("data/holdout_used.flag").is_file() and Path("reports/true_holdout_eval.json").is_file()
    assert main([*args, "--sha256", sha(setup.data)]) == 2
    assert '"burned": true' in capsys.readouterr().out


def test_script_runs_directly_from_repository_root():
    root = Path(holdout_eval.__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts" / "holdout_eval.py"), "--help"],
                               capture_output=True, text=True, cwd=root.parent)
    assert completed.returncode == 0, completed.stderr
    assert "--rth-schedule" in completed.stdout
