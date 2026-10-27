"""Validated active-config reader used by read-only broker adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..governance.validator import OOSMetrics, validate


class ActiveConfigWatcher:
    """Read an active config on demand and verify its hash before exposing parameters."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_validated(self) -> dict:
        config = json.loads(self.path.read_text(encoding="utf-8"))
        if config.get("status") != "validated" or config.get("strategy_state") != "ACTIVE":
            raise RuntimeError("active configuration is not validated and ACTIVE")
        canonical_body = {
            key: value for key, value in config.items() if key != "sha256"
        }
        canonical = json.dumps(canonical_body, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if config.get("sha256") != expected:
            raise ValueError("active configuration SHA-256 verification failed")
        if not isinstance(config.get("parameters"), dict):
            raise ValueError("active configuration parameters must be an object")
        metrics = config.get("oos_metrics")
        if not isinstance(metrics, dict):
            raise ValueError("active configuration is missing OOS metrics")
        try:
            oos = OOSMetrics(
                profit_factor=metrics["profit_factor"],
                sharpe=metrics["sharpe"],
                max_drawdown_pct=metrics["max_drawdown_pct"],
                trade_count=metrics["trade_count"],
                seed_expectancies=tuple(metrics.get("seed_expectancies", ())),
                seed_ids=tuple(metrics.get("seed_ids", ())),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("active configuration has malformed OOS metrics") from error
        try:
            result = validate(
                oos,
                require_seed_stability=bool(oos.seed_expectancies or oos.seed_ids),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("active configuration has invalid OOS metrics") from error
        if not result.passed:
            raise ValueError(
                "active configuration failed OOS validation: " + "; ".join(result.reasons)
            )
        return config
