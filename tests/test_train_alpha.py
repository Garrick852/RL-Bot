"""Alpha training CLI: leakage barrier, artifact round-trip, and fail-closed refusals."""

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_fixtures import ALPHA_WINDOWS, make_bars
from engine_rl.alpha_pipeline import alpha_label_end, attach_alpha, load_alpha_model
from train_alpha import ARTIFACTS, MANIFEST_NAME, AlphaTrainingRefused, main, train_alpha

HORIZON = 2
FAST_MODEL = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}


@pytest.fixture()
def dataset(tmp_path: Path) -> tuple[Path, str, pd.DataFrame]:
    frame = make_bars()
    path = tmp_path / "bars.csv"
    frame.to_csv(path, index=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), frame


def _train(path: Path, sha: str, start: str, out: Path, **overrides) -> dict:
    options = {"horizon": HORIZON, "windows": ALPHA_WINDOWS, "n_splits": 2, "model_params": FAST_MODEL}
    options.update(overrides)
    return train_alpha(path, sha, start, out, **options)


def test_exports_verified_alpha_strictly_before_rl_split(dataset, tmp_path):
    path, sha, frame = dataset
    start = frame["timestamp"].iloc[200]
    out = tmp_path / "alpha"
    report = _train(path, sha, start.isoformat(), out)

    assert all((out / name).is_file() for name in ARTIFACTS)
    manifest_sha = hashlib.sha256((out / MANIFEST_NAME).read_bytes()).hexdigest()
    assert report["features_config"] == {
        "alpha_manifest_path": str((out / MANIFEST_NAME).resolve()), "alpha_manifest_sha256": manifest_sha,
    }
    alpha = load_alpha_model(out / MANIFEST_NAME, manifest_sha)
    assert alpha.onnx_sha256 == report["artifacts"]["onnx_sha256"]
    label_end = alpha_label_end(frame["timestamp"], alpha)
    assert label_end < start and report["alpha_partition"]["label_end"] == label_end.isoformat()
    assert pd.Timestamp(alpha.training_window["last_feature_timestamp"]) < start
    assert report["model"]["purge_gap"] == HORIZON + max(ALPHA_WINDOWS.values())
    assert len(report["walk_forward_oos"]) == 2
    assert json.loads((out / "train_alpha.report.json").read_text("utf-8")) == report
    enriched = attach_alpha(frame, alpha)
    assert np.isfinite(enriched["alpha_prediction"].iloc[200:]).all()


def test_rl_bars_never_change_the_alpha(dataset, tmp_path):
    path, sha, frame = dataset
    start = frame["timestamp"].iloc[200]
    first = _train(path, sha, start.isoformat(), tmp_path / "a")
    tampered = frame.copy()
    tampered.loc[200:, ["open", "close"]] += 5.0
    tampered.loc[200:, ["high", "bid", "ask"]] += 5.0
    tampered.loc[200:, "low"] += 5.0
    other = tmp_path / "tampered.csv"
    tampered.to_csv(other, index=False)
    second = _train(other, hashlib.sha256(other.read_bytes()).hexdigest(), start.isoformat(), tmp_path / "b")
    assert first["artifacts"]["onnx_sha256"] == second["artifacts"]["onnx_sha256"]
    assert first["walk_forward_oos"] == second["walk_forward_oos"]


@pytest.mark.parametrize("case", ["sha", "naive_start", "no_rl_rows", "start_after_barrier", "too_few_rows"])
def test_refusals_leave_no_artifacts(dataset, tmp_path, case):
    path, sha, frame = dataset
    start = frame["timestamp"].iloc[200].isoformat()
    kwargs = {}
    if case == "sha":
        sha = "0" * 64
    elif case == "naive_start":
        start = "2024-01-01 03:00:00"
    elif case == "no_rl_rows":
        start = (frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)).isoformat()
    elif case == "start_after_barrier":
        kwargs["alpha_start"] = start
    else:
        start = frame["timestamp"].iloc[12].isoformat()
    out = tmp_path / "alpha"
    with pytest.raises(AlphaTrainingRefused):
        _train(path, sha, start, out, **kwargs)
    assert not any((out / name).exists() for name in ARTIFACTS)


def test_refuses_to_overwrite_hash_locked_artifacts(dataset, tmp_path):
    path, sha, frame = dataset
    out = tmp_path / "alpha"
    out.mkdir()
    (out / MANIFEST_NAME).write_bytes(b"locked")
    with pytest.raises(AlphaTrainingRefused, match="overwrite"):
        _train(path, sha, frame["timestamp"].iloc[200].isoformat(), out)
    assert (out / MANIFEST_NAME).read_bytes() == b"locked"


def test_cli_exit_codes(dataset, tmp_path):
    path, sha, frame = dataset
    windows = [arg for key, value in ALPHA_WINDOWS.items() for arg in (f"--{key.replace('_', '-')}", str(value))]
    base = [str(path), "--dataset-sha256", sha, "--rl-train-start", frame["timestamp"].iloc[200].isoformat(),
            "--horizon", str(HORIZON), "--n-splits", "2", "--n-estimators", "10", "--max-depth", "2", *windows]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert main([*base, "--output-dir", str(tmp_path / "ok")]) == 0
    assert json.loads(buffer.getvalue())["status"] == "EXPORTED"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert main([*base, "--output-dir", str(tmp_path / "ok")]) == 1
    assert json.loads(buffer.getvalue())["status"] == "REFUSED"
