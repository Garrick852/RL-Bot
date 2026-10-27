"""Atomic persistence of validated active configuration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .validator import ValidationResult, validate


def persist_validated_config(
    path: str | Path,
    parameters: Mapping[str, object],
    validation: ValidationResult,
) -> dict[str, object]:
    """Write a deterministic, hashed active config only after a passed OOS gate."""
    rechecked = validate(
        validation.metrics,
        require_seed_stability=bool(
            validation.metrics.seed_expectancies or validation.metrics.seed_ids
        ),
    )
    if not validation.passed or not rechecked.passed:
        raise ValueError("refusing to persist a candidate that failed OOS validation")
    body: dict[str, object] = {
        "status": "validated",
        "strategy_state": "ACTIVE",
        "parameters": dict(parameters),
        "oos_metrics": {
            "profit_factor": validation.metrics.profit_factor,
            "sharpe": validation.metrics.sharpe,
            "max_drawdown_pct": validation.metrics.max_drawdown_pct,
            "trade_count": validation.metrics.trade_count,
            "seed_expectancies": list(validation.metrics.seed_expectancies),
            "seed_ids": list(validation.metrics.seed_ids),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    body["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temp_path = stream.name
            json.dump(body, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    except OSError:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
        raise
    return body
