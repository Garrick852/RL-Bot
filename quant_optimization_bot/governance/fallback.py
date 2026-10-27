"""Three-stage fail-safe response to a rejected optimization candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .validator import OOSMetrics, ValidationResult, validate


class StrategyState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"


class FallbackStage(str, Enum):
    RETRY = "RECALIBRATE_AND_RETRY"
    DEGRADE = "GRACEFUL_DEGRADATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


@dataclass(frozen=True)
class FallbackResult:
    stage: FallbackStage
    state: StrategyState
    accepted: bool
    parameters: Mapping[str, object]
    lot_size_scale: float
    consecutive_failures: int
    close_only: bool
    validation: ValidationResult


def expand_parameter_bounds(
    bounds: Mapping[str, tuple[float, float] | list[float]],
    *,
    factor: float = 1.25,
) -> dict[str, tuple[float, float]]:
    """Expand each finite search interval about its midpoint for stage-1 retries."""
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("factor must be finite and greater than 1")
    expanded: dict[str, tuple[float, float]] = {}
    for name, limits in bounds.items():
        if len(limits) != 2:
            raise ValueError(f"bounds for {name} must have two values")
        low, high = float(limits[0]), float(limits[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"invalid bounds for {name}")
        center = (low + high) / 2.0
        half_width = (high - low) * factor / 2.0
        expanded[name] = (center - half_width, center + half_width)
    return expanded


def recalibration_settings() -> dict[str, float]:
    """Return the documented stage-1 entropy and lookback adjustments."""
    return {"ppo_entropy_coefficient": 0.03, "lookback_scale": 0.8}


def handle_failure(
    candidate_metrics: OOSMetrics,
    last_known_good: Mapping[str, object],
    *,
    candidate_parameters: Mapping[str, object] | None = None,
    consecutive_failures: int,
    recalibrate: Callable[[], tuple[Mapping[str, object], OOSMetrics]] | None = None,
    retry_validator: Callable[[OOSMetrics], ValidationResult] = validate,
    circuit_breaker_after: int = 3,
) -> FallbackResult:
    """Retry once, otherwise retain known-good parameters with reduced sizing."""
    if consecutive_failures < 0:
        raise ValueError("consecutive_failures cannot be negative")
    if circuit_breaker_after < 1:
        raise ValueError("circuit_breaker_after must be positive")
    initial = validate(candidate_metrics)
    if initial.passed:
        if candidate_parameters is None:
            raise ValueError("candidate_parameters are required for a passing candidate")
        return FallbackResult(
            FallbackStage.RETRY, StrategyState.ACTIVE, True,
            dict(candidate_parameters), 1.0, 0, False, initial,
        )
    if recalibrate is not None:
        retry_parameters, retry_metrics = recalibrate()
        retry_result = retry_validator(retry_metrics)
        if retry_result.passed:
            return FallbackResult(
                FallbackStage.RETRY, StrategyState.ACTIVE, True,
                dict(retry_parameters), 1.0, 0, False, retry_result,
            )
        initial = retry_result
    failures = consecutive_failures + 1
    if failures >= circuit_breaker_after:
        return FallbackResult(
            FallbackStage.CIRCUIT_BREAKER, StrategyState.SUSPENDED, False,
            dict(last_known_good), 0.5, failures, True, initial,
        )
    return FallbackResult(
        FallbackStage.DEGRADE, StrategyState.ACTIVE, False,
        dict(last_known_good), 0.5, failures, False, initial,
    )
