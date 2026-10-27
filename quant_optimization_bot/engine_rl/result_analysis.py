"""Out-of-sample RL policy evaluation and indicator-sensitivity reporting."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..governance.validator import OOSMetrics, ValidationResult, validate


@dataclass(frozen=True)
class RLResultAnalysis:
    """Evaluation metrics plus descriptive policy sensitivity by input feature."""

    status: str
    metrics: OOSMetrics
    total_return_pct: float
    mean_step_reward: float
    action_summary: Mapping[str, int]
    feature_sensitivity: Mapping[str, float]
    training_health: Mapping[str, float | None]
    validation: ValidationResult
    recommendations: tuple[str, ...]
    evaluated_out_of_sample: bool

    @property
    def deployment_eligible(self) -> bool:
        return False

    def to_markdown(self) -> str:
        lines = [
            "# RL strategy and indicator analysis",
            "",
            f"**Status:** {self.status}",
            "**Deployment eligible:** no",
            f"**Evaluation data marked OOS:** {'yes' if self.evaluated_out_of_sample else 'no'}",
            "",
            "## Policy performance",
            "",
            f"- Profit factor: {self.metrics.profit_factor:.6g}",
            f"- Per-trade Sharpe (not annualized): {self.metrics.sharpe:.6g}",
            f"- Maximum drawdown: {self.metrics.max_drawdown_pct:.6g}%",
            f"- Closed trade count: {self.metrics.trade_count}",
            f"- Compounded return: {self.total_return_pct:.6g}%",
            f"- Mean per-step reward: {self.mean_step_reward:.6g}",
            "",
            "## Learned actions",
            "",
        ]
        lines.extend(
            f"- `{action}`: {count} step(s)"
            for action, count in sorted(self.action_summary.items())
        )
        if not self.action_summary:
            lines.append("- No actions were recorded.")
        lines.extend(["", "## Indicator / feature sensitivity", ""])
        if self.feature_sensitivity:
            lines.append(
                "Mean absolute action change after perturbing one feature by one "
                "evaluation-set standard deviation:"
            )
            lines.extend(
                f"- `{feature}`: {sensitivity:.6g}"
                for feature, sensitivity in sorted(
                    self.feature_sensitivity.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
        else:
            lines.append("- No feature sensitivity was computed.")
        lines.extend(
            [
                "",
                "Sensitivity is descriptive policy dependence, not causal indicator "
                "importance or evidence that changing an indicator improves returns.",
                "",
                "## PPO training health",
                "",
            ]
        )
        if self.training_health:
            lines.extend(
                f"- `{name}`: {'unavailable' if value is None else f'{value:.6g}'}"
                for name, value in sorted(self.training_health.items())
            )
        else:
            lines.append("- No PPO training telemetry was supplied.")
        lines.extend(["", "## OOS hard gates", ""])
        lines.append(
            "- Passed the configured single-run metric gates."
            if self.validation.passed
            else "- Failed: " + "; ".join(self.validation.reasons) + "."
        )
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"{index}. {item}" for index, item in enumerate(self.recommendations, 1))
        return "\n".join(lines) + "\n"


def _action_value(action: object) -> float:
    value = np.asarray(action, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("policy returned an empty or non-finite action")
    return float(value.reshape(-1)[0])


def _predict(model, observation: np.ndarray) -> object:
    result = model.predict(observation, deterministic=True)
    return result[0] if isinstance(result, tuple) else result


def _closed_trade_pnls(rewards: Sequence[float], positions: Sequence[float]) -> list[float]:
    trades: list[float] = []
    current_pnl = 0.0
    current_direction = 0
    for reward, position in zip(rewards, positions):
        direction = 1 if position > 0 else -1 if position < 0 else 0
        if direction == 0 and current_direction:
            current_pnl += reward
            trades.append(current_pnl)
            current_pnl = 0.0
            current_direction = 0
            continue
        if direction != current_direction:
            if current_direction:
                trades.append(current_pnl)
            current_pnl = 0.0
            current_direction = direction
        if current_direction:
            current_pnl += reward
    if current_direction:
        trades.append(current_pnl)
    return trades


def analyze_policy(
    model,
    environment,
    *,
    feature_names: Sequence[str],
    evaluated_out_of_sample: bool = False,
    training_health: Mapping[str, float | None] | None = None,
    sensitivity_sample_limit: int = 512,
) -> RLResultAnalysis:
    """Roll out a policy and rank inputs by action sensitivity on an evaluation set.

    The caller must supply a separate chronological OOS environment and explicitly
    mark it as OOS. The function does not infer data provenance or apply drift gates.
    """
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("feature_names must be unique non-empty strings")
    if sensitivity_sample_limit < 1:
        raise ValueError("sensitivity_sample_limit must be positive")
    features = np.asarray(getattr(environment, "features", ()), dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(names) or len(features) < 2:
        raise ValueError("environment must expose features matching feature_names")
    if not np.isfinite(features).all():
        raise ValueError("evaluation features must be finite")

    observation, _ = environment.reset()
    rewards: list[float] = []
    positions: list[float] = []
    actions: dict[str, int] = {}
    terminated = truncated = False
    max_steps = len(features) - 1
    while not (terminated or truncated):
        action = _predict(model, observation)
        action_value = _action_value(action)
        action_label = str(int(action_value)) if action_value.is_integer() else f"{action_value:.6g}"
        actions[action_label] = actions.get(action_label, 0) + 1
        observation, reward, terminated, truncated, info = environment.step(action)
        rewards.append(float(reward))
        positions.append(float(info.get("position", action_value)))
        if len(rewards) > max_steps:
            raise RuntimeError("evaluation environment exceeded its expected episode length")
    if not rewards:
        raise ValueError("evaluation environment produced no rewards")
    if not all(math.isfinite(value) for value in rewards + positions):
        raise ValueError("evaluation produced non-finite rewards or positions")

    trade_pnls = _closed_trade_pnls(rewards, positions)
    if trade_pnls:
        positive = sum(value for value in trade_pnls if value > 0)
        negative = -sum(value for value in trade_pnls if value < 0)
        profit_factor = positive / negative if negative else (math.inf if positive else 0.0)
        deviation = statistics.stdev(trade_pnls) if len(trade_pnls) > 1 else 0.0
        sharpe = (
            statistics.fmean(trade_pnls) / deviation * math.sqrt(len(trade_pnls))
            if deviation
            else 0.0
        )
        equity = peak = 1.0
        drawdown = 0.0
        for reward in rewards:
            equity *= max(0.0, 1.0 + reward)
            peak = max(peak, equity)
            if peak > 0:
                drawdown = max(drawdown, (peak - equity) / peak * 100.0)
    else:
        profit_factor = sharpe = drawdown = 0.0
        equity = 1.0
    metrics = OOSMetrics(profit_factor, sharpe, drawdown, len(trade_pnls))
    validation = validate(metrics)

    sample = features[:sensitivity_sample_limit].astype(np.float32)
    baseline = np.asarray([_action_value(_predict(model, row)) for row in sample])
    sensitivity: dict[str, float] = {}
    for index, name in enumerate(names):
        scale = float(np.std(sample[:, index]))
        if scale == 0.0:
            sensitivity[name] = 0.0
            continue
        perturbed = sample.copy()
        perturbed[:, index] += scale
        changed = np.asarray(
            [_action_value(_predict(model, row)) for row in perturbed]
        )
        sensitivity[name] = float(np.mean(np.abs(changed - baseline)))

    health = dict(training_health or {})
    recommendations: list[str] = []
    if not evaluated_out_of_sample:
        status = "UNVERIFIED"
        recommendations.append(
            "Evaluate on a separately reserved chronological OOS dataset; training-episode "
            "results do not establish generalization."
        )
    elif not validation.passed:
        status = "REJECTED"
        for reason in validation.reasons:
            recommendations.append(f"Resolve OOS gate failure: {reason}.")
    else:
        status = "OOS_METRICS_PASS"
        recommendations.append(
            "Single-run OOS metric gates passed. Repeat across at least three distinct seeds "
            "(for example 7, 42, 99) and run the feature-drift gate before promotion."
        )
    if metrics.trade_count < 30:
        recommendations.append(
            "Increase the independent OOS trade sample; fewer than 30 closed trades is inconclusive."
        )
    if metrics.profit_factor < 1.25:
        recommendations.append(
            "Review entry/exit rules and transaction-cost assumptions; OOS profit factor is below 1.25."
        )
    if metrics.sharpe < 1.0:
        recommendations.append(
            "Review risk sizing and regime sensitivity; per-trade OOS Sharpe is below 1.0."
        )
    if metrics.max_drawdown_pct > 10.0:
        recommendations.append(
            "Reduce exposure or add risk controls; OOS drawdown exceeds the 10% default ceiling."
        )
    if actions and max(actions.values()) / len(rewards) > 0.95:
        recommendations.append(
            "The policy selected one action on over 95% of steps; inspect for action collapse "
            "and verify reward scaling and exploration."
        )
    if any(value is None for value in health.values()):
        recommendations.append("Some PPO health telemetry is unavailable; inspect the training log.")
    kl = health.get("approx_kl")
    if kl is not None and (kl < 0.001 or kl > 0.02):
        recommendations.append(
            "PPO approx_kl is outside the expected 0.001-0.02 range; review update size and learning rate."
        )
    clip = health.get("clip_fraction")
    if clip is not None and clip >= 0.4:
        recommendations.append(
            "PPO clip_fraction is at least 0.4; updates may be too aggressive."
        )
    explained = health.get("explained_variance")
    if explained is not None and explained < 0.70:
        recommendations.append(
            "PPO explained_variance is below 0.70; review value fit, reward scaling, and features."
        )
    ranked = sorted(sensitivity.items(), key=lambda item: (-item[1], item[0]))
    if ranked:
        names_to_review = [name for name, value in ranked if value > 0]
        if names_to_review:
            recommendations.append(
                "Prioritize ablation and robustness checks for the most policy-sensitive inputs: "
                + ", ".join(names_to_review[:3])
                + ". Sensitivity is not causal importance."
            )
        else:
            recommendations.append(
                "Actions were insensitive to the supplied indicators under one-standard-deviation "
                "perturbations; verify feature scaling and policy behavior."
            )
    recommendations.append(
        "Do not deploy from this report alone: multi-seed stability, drift checks, and independent review remain required."
    )
    return RLResultAnalysis(
        status=status,
        metrics=metrics,
        total_return_pct=(equity - 1.0) * 100.0,
        mean_step_reward=statistics.fmean(rewards),
        action_summary=actions,
        feature_sensitivity=sensitivity,
        training_health=health,
        validation=validation,
        recommendations=tuple(recommendations),
        evaluated_out_of_sample=evaluated_out_of_sample,
    )


def write_rl_analysis(analysis: RLResultAnalysis, path: str | Path) -> Path:
    """Write a Markdown RL result report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(analysis.to_markdown(), encoding="utf-8")
    return destination
