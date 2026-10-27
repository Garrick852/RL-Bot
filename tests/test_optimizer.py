"""Persistent search, exact medians, air-gap enforcement, and real PPO wiring."""

import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState

import test_orchestrator as experiment_fixtures
import optimizer as optimizer_module
from optimizer import OfflineOptimizer, SAMPLER_SEED, STORAGE_URL, main


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture = experiment_fixtures.OrchestratorTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def evidence(config, values=(1.1, 1.5, 20.0)):
    return {
        "run_status": "COMPLETE", "validation_state": "READY_FOR_OFFLINE_EXECUTION",
        "rl_training_performed": True, "scope": "HYPERPARAMETER_SEARCH_ONLY",
        "median_oos_metrics": {"profit_factor": sorted(values)[1]},
        "results": [
            {"seed": seed, "status": "COMPLETE", "metrics": {"profit_factor": value}}
            for seed, value in zip(config["training"]["seeds"], values)
        ],
    }


def successful_pipeline(path, *, deploy):
    assert deploy is False
    config = json.loads(Path(path).read_text())
    assert "deployment" not in config
    return evidence(config)


def test_trials_copy_config_inject_parameters_and_export_exact_best_dictionary(experiment):
    before = experiment.path.read_bytes()
    optimizer = OfflineOptimizer(experiment.path, max_workers=1)
    base = copy.deepcopy(optimizer.base_config)
    seen = []

    def pipeline(path, *, deploy):
        config = json.loads(Path(path).read_text())
        seen.append(config)
        assert deploy is False
        assert Path(config["dataset"]["path"]).is_absolute()
        assert config["orchestration"]["max_workers"] == 1
        assert config["training"]["threads_per_worker"] == 1
        assert config["features"]["warmup_bars"] == (
            config["orchestration"]["k_multiplier"] * config["strategy"]["ema"]["slow_period"]
        )
        return evidence(config)

    with patch("optimizer.OfflineOrchestrator.run_pipeline", side_effect=pipeline), \
         patch("orchestrator.OfflineOrchestrator.execute_deployment_gate",
               side_effect=AssertionError("search attempted operational side effects")):
        study = optimizer.optimize(2)
    assert len(seen) == 2
    assert study.best_value == 1.5
    assert optimizer.base_config == base
    assert experiment.path.read_bytes() == before
    best = study.best_params
    assert seen[0]["training"]["ppo_hyperparameters"]["learning_rate"] == study.trials[0].params["learning_rate"]
    assert seen[0]["training"]["ppo_hyperparameters"]["ent_coef"] == study.trials[0].params["ent_coef"]
    assert seen[0]["strategy"]["ema"]["slow_period"] == study.trials[0].params["ema_slow_period"]
    assert json.loads(Path("config/best_candidate_config.json").read_text()) == best
    assert Path("reports/optuna_vps_study.db").is_file()
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert study.trials[0].user_attrs["eligible_objective"] is True
    assert study.trials[0].user_attrs["requires_fresh_holdout"] is True


def test_sqlite_resume_adds_trials_and_repairs_missing_export(experiment):
    with patch("optimizer.OfflineOrchestrator.run_pipeline", side_effect=successful_pipeline):
        first = OfflineOptimizer(experiment.path).optimize(1)
        Path("config/best_candidate_config.json").unlink()
        second = OfflineOptimizer(experiment.path).optimize(1)
    assert first.study_name == second.study_name
    assert len(second.trials) == 2
    assert second.trials[1].number == 1
    assert json.loads(Path("config/best_candidate_config.json").read_text()) == second.best_params
    assert not Path("reports/optuna_vps_study.lock").exists()


def test_changed_config_or_worker_contract_cannot_resume_same_study(experiment):
    with patch("optimizer.OfflineOrchestrator.run_pipeline", side_effect=successful_pipeline):
        OfflineOptimizer(experiment.path).optimize(1)
    with pytest.raises(ValueError, match="contract changed"):
        OfflineOptimizer(experiment.path, max_workers=2).optimize(1)
    experiment.config["training"]["total_timesteps_per_seed"] = 4096
    experiment.save()
    with pytest.raises(ValueError, match="contract changed"):
        OfflineOptimizer(experiment.path).optimize(1)
    assert not Path("reports/optuna_vps_study.lock").exists()


@pytest.mark.parametrize("failure", ["crash", "invalid", "missing_seed", "duplicate_seed",
                                   "undefined_pf", "wrong_median", "failed_seed"])
def test_broken_multiseed_evidence_is_pruned_never_exported(experiment, failure):
    def fail(path, *, deploy):
        assert deploy is False
        if failure == "crash":
            raise RuntimeError("training crashed")
        config = json.loads(Path(path).read_text())
        report = evidence(config)
        if failure == "invalid":
            report["validation_state"] = "INVALID"
        elif failure == "missing_seed":
            report["results"].pop()
        elif failure == "duplicate_seed":
            report["results"][1]["seed"] = 7
        elif failure == "undefined_pf":
            report["results"][0]["metrics"]["profit_factor"] = None
        elif failure == "wrong_median":
            report["median_oos_metrics"]["profit_factor"] = 20.0
        else:
            report["results"][0]["status"] = "FAILED"
        return report

    with patch("optimizer.OfflineOrchestrator.run_pipeline", side_effect=fail):
        study = OfflineOptimizer(experiment.path).optimize(1)
    assert study.trials[0].state == TrialState.PRUNED
    assert study.trials[0].user_attrs["failure_reason"]
    assert not Path("config/best_candidate_config.json").exists()


def test_export_only_changes_on_strict_high_water_mark(experiment):
    optimizer = OfflineOptimizer(experiment.path)
    study = optuna.create_study(direction="maximize")
    for value, param in ((1.5, 1e-4), (1.0, 2e-4), (1.5, 3e-4), (2.0, 4e-4)):
        trial = optuna.trial.create_trial(
            params={"learning_rate": param},
            distributions={"learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True)},
            value=value, user_attrs={"eligible_objective": True},
        )
        study.add_trial(trial)
        with patch("optimizer._atomic_json", wraps=optimizer_module._atomic_json) as write:
            optimizer.export_best(study, study.trials[-1])
            assert write.call_count == (1 if value == 2.0 or param == 1e-4 else 0)
    assert json.loads(optimizer.candidate_path.read_text()) == {"learning_rate": 4e-4}


def test_unconfigured_placeholder_is_refused_and_base_cannot_be_overwritten(experiment):
    Path("config").mkdir()
    placeholder = Path("config/active_config.json")
    placeholder.write_text('{"status":"unconfigured","strategy_state":"SUSPENDED","parameters":{}}')
    assert main(["--n-trials", "1"]) == 1
    assert not Path("reports/optuna_vps_study.db").exists()
    with pytest.raises(ValueError, match="overwrite"):
        OfflineOptimizer(experiment.path, candidate_path=experiment.path)


def test_exclusive_search_lock_is_not_stolen(experiment):
    Path("reports").mkdir()
    lock = Path("reports/optuna_vps_study.lock")
    lock.write_text("other process")
    with pytest.raises(FileExistsError):
        OfflineOptimizer(experiment.path).optimize(1)
    assert lock.read_text() == "other process"


def test_dataset_tampered_after_base_validation_is_pruned_before_workers(experiment):
    optimizer = OfflineOptimizer(experiment.path)
    dataset = experiment.root / "bars.csv"
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    with patch("orchestrator.ProcessPoolExecutor") as pool:
        study = optimizer.optimize(1)
        pool.assert_not_called()
    assert study.trials[0].state == TrialState.PRUNED
    report = json.loads(Path(study.trials[0].user_attrs["audit_report_path"]).read_text())
    assert report["validation_state"] == "INVALID"
    assert report["deployment"]["status"] == "NOT_REQUESTED"
    assert not optimizer.candidate_path.exists()


def test_fresh_serial_studies_have_seeded_matching_first_suggestions(experiment):
    with patch("optimizer.OfflineOrchestrator.run_pipeline", side_effect=successful_pipeline):
        a = OfflineOptimizer(experiment.path, study_name="a").optimize(1)
        b = OfflineOptimizer(experiment.path, study_name="b").optimize(1)
    assert a.trials[0].params == b.trials[0].params
    assert a.user_attrs["search_contract"]["search_space"]["sampler_seed"] == SAMPLER_SEED
    assert STORAGE_URL == "sqlite:///reports/optuna_vps_study.db"


def test_real_optuna_trial_trains_all_seeds_with_sampled_profile(experiment):
    times = experiment.frame["timestamp"]
    for name, start, end in (("train", 130, 170), ("validation", 170, 190), ("test", 190, 230)):
        experiment.config["splits"][name] = {
            "start": (times.iloc[0] + pd.Timedelta(minutes=start)).isoformat(),
            "end": (times.iloc[0] + pd.Timedelta(minutes=end)).isoformat(),
        }
    experiment.save()
    with patch("orchestrator.OfflineOrchestrator.execute_deployment_gate",
               side_effect=AssertionError("search must stay offline")):
        study = OfflineOptimizer(experiment.path).optimize(1)
    trial = study.trials[0]
    report = json.loads(Path(trial.user_attrs["audit_report_path"]).read_text())
    assert report["run_status"] == "COMPLETE", report["errors"]
    assert report["models_frozen_before_oos"] is True
    assert report["scope"] == "HYPERPARAMETER_SEARCH_ONLY"
    assert report["pass"] is False
    assert "SEARCH_REQUIRES_FRESH_HOLDOUT" in report["reasons"]
    assert report["deployment"]["status"] == "NOT_REQUESTED"
    assert report["seeds_used"] == [7, 42, 99]
    for row in report["results"]:
        parameters = row["training"]["ppo_hyperparameters"]
        for name in ("learning_rate", "ent_coef", "batch_size"):
            assert parameters[name] == trial.params[name]
        assert row["training"]["actual_timesteps"] == 2048
    median = report["median_oos_metrics"]["profit_factor"]
    if median is None:
        assert trial.state == TrialState.PRUNED
    else:
        assert math.isclose(trial.value, median)
