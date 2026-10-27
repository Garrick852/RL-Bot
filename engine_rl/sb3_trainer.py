"""CPU-governed bounded Track A PPO training; no OOS or deployment authorization."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from engine_rl.envs.trading_env import TradingEnv

LOGGER = logging.getLogger(__name__)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
METRICS = ("approx_kl", "clip_fraction", "explained_variance")
DEFAULT_PPO_HYPERPARAMETERS: dict[str, int | float] = {
    "n_steps": 2048, "batch_size": 256, "learning_rate": 3e-4,
    "clip_range": .2, "gae_lambda": .95, "gamma": .99,
}
# "linear_actor" removes every hidden actor layer, so the deterministic action is
# argmax(W @ obs + b) and can be exported exactly (engine_rl.linear_policy). The
# critic keeps hidden layers; it is a training aid and is never exported.
POLICY_ARCHITECTURES: dict[str, dict[str, list[int]]] = {
    "mlp_tanh_128x128": {"pi": [128, 128], "vf": [128, 128]},
    "linear_actor": {"pi": [], "vf": [128, 128]},
}
DEFAULT_POLICY_ARCHITECTURE = "mlp_tanh_128x128"


def validate_policy_architecture(name: Any) -> str:
    """Return a supported named actor/critic layout or raise ValueError."""
    if not isinstance(name, str) or name not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {sorted(POLICY_ARCHITECTURES)}")
    return name


def validate_ppo_hyperparameters(parameters: Mapping[str, Any]) -> dict[str, int | float]:
    """Validate the bounded CPU profile shared by preflight and training.

    Only learning rate, entropy, and divisible minibatch size are tunable.
    Legacy six-key profiles explicitly retain SB3's zero entropy coefficient.
    """
    required = set(DEFAULT_PPO_HYPERPARAMETERS)
    if not isinstance(parameters, Mapping) or set(parameters) not in (required, required | {"ent_coef"}):
        raise ValueError("PPO hyperparameters require the six profile keys and optional ent_coef")
    for name, expected in DEFAULT_PPO_HYPERPARAMETERS.items():
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"PPO {name} must be finite numeric")
        if name not in ("learning_rate", "batch_size") and value != expected:
            raise ValueError(f"PPO {name} must remain {expected}")
    if type(parameters["n_steps"]) is not int:
        raise ValueError("PPO n_steps must be an integer")
    if type(parameters["batch_size"]) is not int or parameters["batch_size"] not in (64, 128, 256, 512):
        raise ValueError("PPO batch_size must be one of 64, 128, 256, 512")
    if not 0 < parameters["learning_rate"] <= 1:
        raise ValueError("PPO learning_rate must be in (0, 1]")
    entropy = parameters.get("ent_coef", 0.0)
    if type(entropy) not in (int, float) or not math.isfinite(entropy) or not 0 <= entropy <= 1:
        raise ValueError("PPO ent_coef must be finite and in [0, 1]")
    return {**parameters, "ent_coef": entropy}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _telemetry_callback(
    records: list[dict[str, Any]], logger: logging.Logger
) -> BaseCallback:
    """Build a real SB3 callback only after CPU governance has been applied."""
    from stable_baselines3.common.callbacks import BaseCallback

    class TelemetryCallback(BaseCallback):
        """Capture post-update diagnostics without fabricating missing values."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._last_update: int | None = None

        def _capture(self) -> None:
            update = getattr(self.model, "_n_updates", 0)
            if not update or update == self._last_update:
                return
            values = self.logger.name_to_value
            record: dict[str, Any] = {
                "timesteps": self.num_timesteps,
                "optimizer_updates": update,
            }
            for name in METRICS:
                value = values.get(f"train/{name}")
                valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
                record[name] = float(value) if valid else None
                if not valid:
                    logger.warning("PPO telemetry unavailable/non-finite: %s at update=%s", name, update)
            records.append(record)
            logger.info(
                "PPO health steps=%s updates=%s approx_kl=%s clip_fraction=%s explained_variance=%s",
                self.num_timesteps, update, *(record[name] for name in METRICS),
            )
            self._last_update = update

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            # SB3 trains after rollout_end; next rollout_start sees that update.
            self._capture()

        def _on_training_end(self) -> None:
            self._capture()
            if not records:
                logger.warning("No optimizer-update telemetry was produced")

    return TelemetryCallback()


class SB3Trainer:
    """Train the bounded lightweight CPU PPO profile on an instantiated environment.

    Args:
        seed: Explicit uint32 training seed; no implicit experiment seed.
        num_threads: Positive PyTorch intra-op thread budget. Defaults to the
            conservative single-thread CPU profile.
        logger: Logger for thread governance and optimizer telemetry.
        hyperparameters: Explicit bounded PPO overrides; None retains the
            original fixed profile. Rollout size stays fixed.
        policy_architecture: Named layout from POLICY_ARCHITECTURES. The
            default keeps the original 2x128 tanh actor and critic.

    Attributes:
        telemetry: One diagnostic record per observed PPO optimizer update.
        model: Most recently initialized PPO instance, or None before training.

    Notes:
        Thread limits are process-wide and retained, not restored after train.
        For strongest OpenMP/BLAS control instantiate in a fresh process before
        importing numerical libraries. Already initialized non-PyTorch BLAS
        runtimes may not honor later environment-variable changes. This is a
        thread budget, not a temperature monitor or OS CPU-affinity controller.
    """

    def __init__(
        self,
        *,
        seed: int,
        num_threads: int = 1,
        logger: logging.Logger | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
        policy_architecture: str = DEFAULT_POLICY_ARCHITECTURE,
    ) -> None:
        self.hyperparameters = validate_ppo_hyperparameters(
            DEFAULT_PPO_HYPERPARAMETERS if hyperparameters is None else hyperparameters
        )
        self.policy_architecture = validate_policy_architecture(policy_architecture)
        self.num_threads = _positive_integer(num_threads, "num_threads")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        self.seed = int(seed)
        self.logger = logger if logger is not None else LOGGER
        self.telemetry: list[dict[str, Any]] = []
        self.model: PPO | None = None
        for name in THREAD_VARIABLES:
            os.environ[name] = str(self.num_threads)
        os.environ["OMP_DYNAMIC"] = "FALSE"
        os.environ["MKL_DYNAMIC"] = "FALSE"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install the training extra: python -m pip install ".[sb3]"') from exc
        torch.set_num_threads(self.num_threads)
        self.logger.info(
            "CPU governance: intra-op threads=%s; device=cpu; process-wide limits retained",
            torch.get_num_threads(),
        )

    def train(self, env: TradingEnv, total_timesteps: int) -> PPO:
        """Train a new PPO instance using only explicitly provided observations.

        Args:
            env: Instantiated root TradingEnv containing training data only.
                Dataset isolation and training-only scaling are caller duties.
            total_timesteps: Positive requested budget. PPO collects complete
                2048-step rollouts, so actual timesteps round up to that size.

        Returns:
            CPU PPO model. Telemetry is retained on this trainer. No evaluation,
            model persistence, research acceptance, or deployment is performed.

        Raises:
            ValueError: Environment type or training budget is invalid.
            RuntimeError: Optional SB3 dependencies are missing.
        """
        budget = _positive_integer(total_timesteps, "total_timesteps")
        try:
            import torch
            from stable_baselines3 import PPO
            from engine_rl.envs.trading_env import TradingEnv
        except ImportError as exc:
            raise RuntimeError('Install the training extra: python -m pip install ".[sb3]"') from exc
        if not isinstance(env, TradingEnv):
            raise ValueError("env must be an instantiated engine_rl.envs.trading_env.TradingEnv")
        torch.set_num_threads(self.num_threads)
        self.telemetry = []
        self.model = PPO(
            "MlpPolicy", env,
            **self.hyperparameters,
            policy_kwargs={
                "net_arch": {
                    key: list(value) for key, value in POLICY_ARCHITECTURES[self.policy_architecture].items()
                },
                "activation_fn": torch.nn.Tanh,
            },
            seed=self.seed, device="cpu", verbose=0,
        )
        self.model.learn(
            total_timesteps=budget,
            callback=_telemetry_callback(self.telemetry, self.logger),
        )
        return self.model
