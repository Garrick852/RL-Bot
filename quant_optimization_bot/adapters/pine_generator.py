"""Translate validated simple thresholds into a TradingView Pine Script stub."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping


def generate_pine(
    parameters: Mapping[str, object],
    output_path: str | Path,
    *,
    source_hash: str,
) -> Path:
    """Generate a strategy-rule export from explicitly named numeric thresholds."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source_hash must be a lowercase SHA-256 hex digest")
    supported = ("InpRSILongThreshold", "InpBBPosThreshold")
    values: dict[str, float] = {}
    for key in supported:
        if key not in parameters:
            raise ValueError(f"missing Pine rule parameter: {key}")
        value = float(parameters[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite Pine rule parameter: {key}")
        values[key] = value
    body = f"""//@version=5
strategy("RL-Bot validated rules", overlay=false)
// Validated config SHA-256: {source_hash}
rsiThreshold = {values["InpRSILongThreshold"]:.8g}
bbPosThreshold = {values["InpBBPosThreshold"]:.8g}
rsiValue = ta.rsi(close, 14)
basis = ta.sma(close, 20)
deviation = 2.0 * ta.stdev(close, 20)
bbPos = deviation == 0.0 ? 0.5 : (close - (basis - deviation)) / (2.0 * deviation)
longSignal = rsiValue >= rsiThreshold and bbPos >= bbPosThreshold
plot(rsiValue, "RSI")
plot(bbPos, "BB position")
plotshape(longSignal, title="Long rule", style=shape.triangleup, location=location.bottom)
if longSignal
    strategy.entry("Long", strategy.long)
"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination
