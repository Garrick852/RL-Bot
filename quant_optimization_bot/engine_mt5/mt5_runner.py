"""Isolated MT5 Strategy Tester CLI invocation and XML report parsing."""

from __future__ import annotations

import subprocess
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..governance.report_analyzer import ReportAnalysis, analyze_result_report, write_analysis_report


@dataclass(frozen=True)
class TesterResult:
    returncode: int
    report_path: Path
    metrics: Mapping[str, float]
    log_path: Path
    analysis: ReportAnalysis
    analysis_path: Path


def write_tester_ini(
    path: str | Path,
    *,
    expert: str,
    symbol: str,
    timeframe: str,
    from_date: str,
    to_date: str,
    report_path: str | Path,
    parameters: Mapping[str, object],
) -> Path:
    """Write a minimal non-visual Strategy Tester configuration."""
    values = {
        "Expert": expert,
        "Symbol": symbol,
        "Period": timeframe,
        "FromDate": from_date,
        "ToDate": to_date,
        "Report": str(Path(report_path).resolve()),
        "ShutdownTerminal": "1",
        "Visual": "0",
        "Optimization": "0",
    }
    lines = ["[Tester]"] + [f"{key}={value}" for key, value in values.items()]
    if parameters:
        lines.append("[TesterInputs]")
        for key, value in parameters.items():
            if not key.replace("_", "").isalnum():
                raise ValueError(f"invalid tester input name: {key!r}")
            if isinstance(value, bool):
                encoded = "true" if value else "false"
            elif isinstance(value, (int, float, str)):
                encoded = str(value)
            else:
                raise TypeError(f"unsupported tester input: {key}")
            if any(char in encoded for char in "\r\n"):
                raise ValueError(f"invalid tester input value: {key}")
            lines.append(f"{key}={encoded}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def parse_xml_report(path: str | Path) -> dict[str, float]:
    """Extract common Strategy Tester metrics by normalized XML element name."""
    root = ET.parse(path).getroot()
    aliases = {
        "profitfactor": "profit_factor",
        "sharpe": "sharpe",
        "sharperatio": "sharpe",
        "maximaldrawdownpercent": "max_drawdown_pct",
        "trades": "trade_count",
        "totaltrades": "trade_count",
    }
    metrics: dict[str, float] = {}
    for element in root.iter():
        name = "".join(char for char in element.tag.lower() if char.isalnum())
        if name in aliases and element.text:
            try:
                metrics[aliases[name]] = float(element.text.strip().replace("%", ""))
            except ValueError as error:
                raise ValueError(f"invalid numeric tester metric in {element.tag}") from error
    return metrics


def run_tester(
    terminal: str | Path,
    ini_path: str | Path,
    report_path: str | Path,
    *,
    timeout_seconds: int = 3600,
    portable_directory: str | Path | None = None,
) -> TesterResult:
    """Run one isolated terminal process; does not place or route live orders."""
    executable = Path(terminal).resolve(strict=True)
    ini = Path(ini_path).resolve(strict=True)
    report = Path(report_path).resolve()
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if report.exists():
        raise FileExistsError(f"report path already exists; use a fresh run-specific path: {report}")
    portable = (
        Path(portable_directory).resolve()
        if portable_directory is not None
        else report.parent / f"mt5-portable-{uuid.uuid4().hex}"
    )
    portable.mkdir(parents=True, exist_ok=True)
    log = portable / "tester.log"
    process = subprocess.run(
        [str(executable), "/portable", f"/config:{ini}"],
        cwd=portable,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    log.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"MT5 Strategy Tester exited with {process.returncode}; see {log}")
    if not report.is_file():
        raise FileNotFoundError(f"MT5 report was not created: {report}")
    metrics = parse_xml_report(report)
    analysis = analyze_result_report(metrics)
    analysis_path = write_analysis_report(
        analysis, report.with_suffix(report.suffix + ".analysis.md")
    )
    return TesterResult(process.returncode, report, metrics, log, analysis, analysis_path)
