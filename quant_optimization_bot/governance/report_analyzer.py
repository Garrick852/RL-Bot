"""Decision-support analysis for backtest and out-of-sample result reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..data.collector import DriftReport
from .fallback import FallbackResult, FallbackStage
from .validator import OOSMetrics, ValidationResult, validate


@dataclass(frozen=True)
class ReportAnalysis:
    """A report's evaluation status, hard-gate results, and ordered next steps."""

    status: str
    raw_metrics: Mapping[str, float]
    validation: ValidationResult | None
    drift_report: DriftReport | None
    fallback: FallbackResult | None
    recommendations: tuple[str, ...]

    @property
    def deployment_eligible(self) -> bool:
        return (
            self.status == "VALIDATED"
            and self.validation is not None
            and self.validation.passed
            and self.drift_report is not None
            and self.drift_report.passed
        )

    def to_markdown(self) -> str:
        lines = [
            "# Result report analysis",
            "",
            f"**Status:** {self.status}",
            f"**Deployment eligible:** {'yes' if self.deployment_eligible else 'no'}",
            "",
            "## Backtest metrics",
            "",
        ]
        if self.raw_metrics:
            lines.extend(
                f"- `{name}`: {value:.6g}"
                for name, value in sorted(self.raw_metrics.items())
            )
        else:
            lines.append("- No parseable tester metrics were provided.")
        lines.extend(["", "## OOS validation", ""])
        if self.validation is None:
            lines.append("- Not supplied; in-sample/tester metrics are not treated as OOS evidence.")
        elif self.validation.passed:
            lines.append("- Passed all configured OOS hard gates.")
        else:
            lines.extend(f"- Failed: {reason}." for reason in self.validation.reasons)
        lines.extend(["", "## Feature drift", ""])
        if self.drift_report is None:
            lines.append("- Not checked; deployment remains blocked.")
        elif self.drift_report.passed:
            lines.append(
                f"- Passed KS checks for {len(self.drift_report.results)} feature(s) "
                f"at p-value threshold {self.drift_report.threshold:g}."
            )
        else:
            lines.append("- Failed for: " + ", ".join(self.drift_report.drifted_features) + ".")
        if self.fallback is not None:
            lines.extend(
                [
                    "",
                    "## Fallback state",
                    "",
                    f"- Stage: `{self.fallback.stage.value}`",
                    f"- Strategy state: `{self.fallback.state.value}`",
                    f"- Close-only: {'yes' if self.fallback.close_only else 'no'}",
                ]
            )
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"{index}. {text}" for index, text in enumerate(self.recommendations, 1))
        return "\n".join(lines) + "\n"


def analyze_result_report(
    raw_metrics: Mapping[str, float],
    *,
    oos_metrics: OOSMetrics | None = None,
    drift_report: DriftReport | None = None,
    fallback: FallbackResult | None = None,
    require_seed_stability: bool = False,
) -> ReportAnalysis:
    """Analyze tester observations without mistaking in-sample results for OOS proof."""
    normalized: dict[str, float] = {}
    recommendations: list[str] = []
    for name, value in raw_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            recommendations.append(f"Review `{name}`: the report contains a non-finite or non-numeric value.")
        else:
            normalized[name] = float(value)

    validation = (
        validate(oos_metrics, require_seed_stability=require_seed_stability)
        if oos_metrics is not None
        else None
    )
    if validation is not None and not validation.passed:
        status = "REJECTED"
    elif drift_report is not None and not drift_report.passed:
        status = "DRIFT_BLOCKED"
    elif validation is None:
        status = "UNVERIFIED"
    elif drift_report is None:
        status = "DRIFT_UNCHECKED"
    else:
        status = "VALIDATED"

    if validation is None:
        recommendations.append(
            "Run a chronological held-out OOS evaluation; tester/optimization metrics alone "
            "cannot authorize deployment."
        )
    else:
        metric = validation.metrics
        if (
            isinstance(metric.profit_factor, (int, float))
            and not isinstance(metric.profit_factor, bool)
            and math.isfinite(metric.profit_factor)
            and metric.profit_factor < 1.25
        ):
            recommendations.append(
                "Reject this candidate and revisit entry/exit assumptions; OOS profit factor "
                "is below 1.25."
            )
        if (
            isinstance(metric.sharpe, (int, float))
            and not isinstance(metric.sharpe, bool)
            and math.isfinite(metric.sharpe)
            and metric.sharpe < 1.0
        ):
            recommendations.append(
                "Improve risk-adjusted OOS returns; Sharpe is below 1.0. Check costs, "
                "position sizing, and regime sensitivity."
            )
        if (
            isinstance(metric.max_drawdown_pct, (int, float))
            and not isinstance(metric.max_drawdown_pct, bool)
            and math.isfinite(metric.max_drawdown_pct)
            and metric.max_drawdown_pct > 10.0
        ):
            recommendations.append(
                "Reduce downside exposure and retest; OOS drawdown exceeds the 10% default ceiling."
            )
        if (
            isinstance(metric.trade_count, int)
            and not isinstance(metric.trade_count, bool)
            and metric.trade_count < 30
        ):
            recommendations.append(
                "Gather more independent OOS trades before drawing conclusions; fewer than 30 "
                "trades is not a sufficient sample."
            )
        for reason in validation.reasons:
            if not any(reason in item for item in recommendations):
                recommendations.append(f"Resolve OOS validation failure: {reason}.")

    if drift_report is None:
        recommendations.append(
            "Run the KS drift gate against the historical feature baseline before considering deployment."
        )
    elif not drift_report.passed:
        recommendations.append(
            "Pause optimization/deployment and investigate the shifted feature distributions: "
            + ", ".join(drift_report.drifted_features)
            + "."
        )

    profit_factor = normalized.get("profit_factor")
    if profit_factor is not None and profit_factor < 1.0:
        recommendations.append(
            "The reported tester profit factor is below break-even; inspect trade costs and "
            "strategy logic before further parameter tuning."
        )
    sharpe = normalized.get("sharpe")
    if sharpe is not None and sharpe < 0.0:
        recommendations.append("The reported tester Sharpe is negative; do not promote this run.")
    drawdown = normalized.get("max_drawdown_pct")
    if drawdown is not None and drawdown > 10.0:
        recommendations.append(
            "Tester drawdown exceeds the default 10% review threshold; reduce risk and validate again."
        )
    trades = normalized.get("trade_count")
    if trades is not None and trades < 30:
        recommendations.append("The tester trade sample is below 30; extend the test period or collect more data.")

    if fallback is not None:
        if fallback.stage is FallbackStage.CIRCUIT_BREAKER:
            recommendations.append(
                "Circuit breaker is active: keep the strategy suspended and close-only until "
                "a fresh candidate passes all gates."
            )
        elif fallback.stage is FallbackStage.DEGRADE:
            recommendations.append(
                "Keep the last known-good parameters and reduced sizing; do not activate the rejected candidate."
            )
        elif not fallback.accepted:
            recommendations.append("The retry did not pass; retain fallback settings and investigate before re-queuing.")

    if status == "VALIDATED":
        recommendations.append(
            "OOS and drift gates passed. Persist the validated configuration, then review the "
            "change and compile result before any deployment."
        )
    elif not any("deployment" in item.lower() or "deploy" in item.lower() for item in recommendations):
        recommendations.append("Do not deploy: at least one required governance gate is unresolved or failed.")

    return ReportAnalysis(
        status,
        normalized,
        validation,
        drift_report,
        fallback,
        tuple(dict.fromkeys(recommendations)),
    )


def write_analysis_report(analysis: ReportAnalysis, path: str | Path) -> Path:
    """Write a Markdown analysis report without changing the source tester report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(analysis.to_markdown(), encoding="utf-8")
    return destination
