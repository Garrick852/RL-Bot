"""Pure three-stage adaptive fallback directives for rejected OOS candidates."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, TypedDict


class FallbackState(str, Enum):
    """Directive states, not authorization to trade or deploy parameters."""

    NO_ACTION = "NO_ACTION"
    RECALIBRATE_AND_RETRY = "RECALIBRATE_AND_RETRY"
    GRACEFUL_DEGRADATION = "GRACEFUL_DEGRADATION"
    SUSPENDED = "SUSPENDED"


class CriticalAlert(TypedDict):
    """Transport-independent alert; no notification is sent by this module."""

    severity: Literal["CRITICAL"]
    code: Literal["OOS_CIRCUIT_BREAKER"]
    message: str
    consecutive_failures: int


class FallbackDirective(TypedDict):
    """Actions for a separately authorized coordinator to interpret."""

    state: FallbackState
    stage: int
    consecutive_failures: int
    reject_candidate: bool
    expand_optimization_bounds: bool
    ppo_ent_coef: Optional[float]
    retain_last_known_good: bool
    active_config_action: Literal["UNCHANGED", "RETAIN_LAST_KNOWN_GOOD"]
    risk_size_scale: Optional[float]
    risk_size_reference: Literal["LAST_KNOWN_GOOD", "NOT_APPLICABLE"]
    block_new_entries: bool
    require_verified_last_known_good: bool
    require_fresh_oos_holdout: bool
    require_explicit_recovery_authorization: bool
    alert: Optional[CriticalAlert]


class AdaptiveFallbackStateMachine:
    """Map an already-updated consecutive OOS failure count to strict directives.

    The caller owns persistent failure counts, verified last-known-good identity,
    suspension latching, and all execution. Evaluation does not increment/reset
    a counter, write files, mutate optimization bounds, resize positions, place
    orders, train a model, or send alerts.

    A zero count returns NO_ACTION, never ACTIVE or permission to resume.
    Once SUSPENDED has been acted on, a coordinator MUST retain that latch until
    separately authorized recovery; passing a smaller count is not recovery.
    """

    def evaluate(self, consecutive_failures: int) -> FallbackDirective:
        """Evaluate failure count without performing the returned actions.

        Args:
            consecutive_failures: Number of already-recorded consecutive
                rejected OOS candidates, not the count before this failure.

        Returns:
            A fresh independent directive. First failure requests reviewed
            bound expansion and entropy 0.03 (the protocol's retry profile).
            Second failure rejects the candidate and requests half the verified
            last-known-good risk size, never half an already degraded size.
            Three or more failures request suspension and a critical alert.

        Raises:
            ValueError: Count is negative, boolean, or not a Python integer.
        """
        if type(consecutive_failures) is not int or consecutive_failures < 0:
            raise ValueError("consecutive_failures must be a non-negative integer")

        directive: FallbackDirective = {
            "state": FallbackState.NO_ACTION,
            "stage": 0,
            "consecutive_failures": consecutive_failures,
            "reject_candidate": consecutive_failures > 0,
            "expand_optimization_bounds": False,
            "ppo_ent_coef": None,
            "retain_last_known_good": False,
            "active_config_action": "UNCHANGED",
            "risk_size_scale": None,
            "risk_size_reference": "NOT_APPLICABLE",
            "block_new_entries": False,
            "require_verified_last_known_good": False,
            "require_fresh_oos_holdout": False,
            "require_explicit_recovery_authorization": False,
            "alert": None,
        }
        if consecutive_failures == 0:
            return directive

        directive.update(
            retain_last_known_good=True,
            active_config_action="RETAIN_LAST_KNOWN_GOOD",
            require_verified_last_known_good=True,
        )
        if consecutive_failures == 1:
            directive.update(
                state=FallbackState.RECALIBRATE_AND_RETRY,
                stage=1,
                expand_optimization_bounds=True,
                ppo_ent_coef=0.03,
                require_fresh_oos_holdout=True,
            )
        elif consecutive_failures == 2:
            directive.update(
                state=FallbackState.GRACEFUL_DEGRADATION,
                stage=2,
                risk_size_scale=0.5,
                risk_size_reference="LAST_KNOWN_GOOD",
            )
        else:
            directive.update(
                state=FallbackState.SUSPENDED,
                stage=3,
                risk_size_scale=0.0,
                risk_size_reference="LAST_KNOWN_GOOD",
                block_new_entries=True,
                require_explicit_recovery_authorization=True,
                alert={
                    "severity": "CRITICAL",
                    "code": "OOS_CIRCUIT_BREAKER",
                    "message": (
                        f"CRITICAL: OOS circuit breaker SUSPENDED after "
                        f"{consecutive_failures} consecutive validation failures. "
                        "Reject candidate parameters; block all new entries; "
                        "retain verified last-known-good configuration. "
                        "Explicit recovery authorization is required."
                    ),
                    "consecutive_failures": consecutive_failures,
                },
            )
        return directive
