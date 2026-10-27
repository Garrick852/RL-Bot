"""CPU-governed, regime-prioritized matrix scheduler."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar

R = TypeVar("R")


@dataclass(frozen=True)
class MatrixJob:
    asset: str
    strategy: str
    kind: str = "mt5"
    regime_priority: int = 0
    parameters: Mapping[str, object] | None = None


def worker_limit(configured: int | None = None, cpu_count: int | None = None) -> int:
    """Resolve CPU worker count to at most half of the available cores."""
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("cpu_count must be positive")
    maximum = max(1, cores // 2)
    if configured is not None and (
        isinstance(configured, bool) or not isinstance(configured, int) or configured < 1
    ):
        raise ValueError("configured worker count must be positive")
    return min(configured, maximum) if configured is not None else maximum


def load_matrix(path: str | Path) -> tuple[list[MatrixJob], int]:
    """Expand a validated asset/strategy matrix into FIFO MT5 jobs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("matrix configuration must be a JSON object")
    assets, strategies = raw.get("assets"), raw.get("strategies")
    if (
        not isinstance(assets, list) or not assets
        or not all(isinstance(value, str) and value for value in assets)
        or not isinstance(strategies, list) or not strategies
        or not all(isinstance(value, str) and value for value in strategies)
    ):
        raise ValueError("matrix assets and strategies must be non-empty string arrays")
    bounds = raw.get("parameter_bounds", {})
    if not isinstance(bounds, dict):
        raise ValueError("parameter_bounds must be an object")
    for name, limits in bounds.items():
        if (
            not isinstance(name, str)
            or not isinstance(limits, list)
            or len(limits) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in limits)
            or not all(math.isfinite(value) for value in limits)
            or limits[0] > limits[1]
        ):
            raise ValueError(f"invalid parameter bounds for {name!r}")
    jobs = [
        MatrixJob(asset, strategy, "mt5", parameters={})
        for asset in assets
        for strategy in strategies
    ]
    return jobs, worker_limit(raw.get("max_parallel_tasks"))


def run_matrix(
    jobs: Iterable[MatrixJob],
    runner: Callable[[MatrixJob], R],
    *,
    max_workers: int | None = None,
) -> list[tuple[MatrixJob, R]]:
    """Run MT5 jobs with bounded process concurrency, then RL jobs sequentially."""
    ordered = list(jobs)
    if any(job.kind not in {"mt5", "sb3", "torchrl"} for job in ordered):
        raise ValueError("job kind must be mt5, sb3, or torchrl")
    limit = worker_limit(max_workers)
    prioritized = sorted(
        enumerate(ordered),
        key=lambda item: (-item[1].regime_priority, item[0]),
    )
    mt5_jobs = [job for _, job in prioritized if job.kind == "mt5"]
    rl_jobs = [job for _, job in prioritized if job.kind != "mt5"]
    results: list[tuple[MatrixJob, R]] = []
    if mt5_jobs:
        with ProcessPoolExecutor(max_workers=limit) as pool:
            futures = [(job, pool.submit(runner, job)) for job in mt5_jobs]
            results.extend((job, future.result()) for job, future in futures)
    results.extend((job, runner(job)) for job in rl_jobs)
    return results


def build_matrix_jobs(
    matrix_path: str | Path,
    *,
    kind_by_strategy: Mapping[str, str] | None = None,
) -> tuple[list[MatrixJob], int]:
    """Expand matrix entries and route named strategies to optional RL tracks."""
    jobs, workers = load_matrix(matrix_path)
    routes = kind_by_strategy or {}
    routed: list[MatrixJob] = []
    for job in jobs:
        kind = routes.get(job.strategy, job.kind)
        if kind not in {"mt5", "sb3", "torchrl"}:
            raise ValueError(f"unsupported task kind for {job.strategy}: {kind}")
        routed.append(MatrixJob(job.asset, job.strategy, kind, job.regime_priority, job.parameters))
    return routed, workers
