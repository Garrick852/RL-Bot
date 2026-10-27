"""Optional Stable-Baselines3 PPO production-track helpers."""

from __future__ import annotations

import math
from pathlib import Path


def train(
    environment,
    *,
    dataset_size: int,
    output_path: str | Path,
    seed: int = 42,
    total_timesteps: int | None = None,
    evaluation_environment=None,
    evaluation_is_out_of_sample: bool = False,
    feature_names: tuple[str, ...] | None = None,
    report_path: str | Path | None = None,
):
    """Train the documented CPU PPO profile; optional packages load on demand."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if total_timesteps is None:
        total_timesteps = 125 * dataset_size
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError("install the SB3 extra with `pip install .[sb3]`") from error
    model = PPO(
        "MlpPolicy",
        environment,
        policy_kwargs={
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.Tanh,
        },
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        clip_range=0.2,
        gae_lambda=0.95,
        gamma=0.99,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(output_path))
    if evaluation_environment is not None:
        if not feature_names:
            raise ValueError("feature_names are required to analyze an RL evaluation")
        from .result_analysis import analyze_policy, write_rl_analysis

        analysis = analyze_policy(
            model,
            evaluation_environment,
            feature_names=feature_names,
            evaluated_out_of_sample=evaluation_is_out_of_sample,
            training_health=health_telemetry(model),
        )
        destination = (
            Path(report_path)
            if report_path is not None
            else Path(output_path).with_suffix(".rl-analysis.md")
        )
        write_rl_analysis(analysis, destination)
    return model


def health_telemetry(model) -> dict[str, float | None]:
    """Return the latest PPO diagnostics where SB3 exposes them."""
    logger = getattr(model, "logger", None)
    values = getattr(logger, "name_to_value", {}) if logger is not None else {}
    telemetry: dict[str, float | None] = {}
    for name in ("train/approx_kl", "train/clip_fraction", "train/explained_variance"):
        value = values.get(name)
        telemetry[name.rsplit("/", 1)[-1]] = (
            float(value) if value is not None and math.isfinite(float(value)) else None
        )
    return telemetry
