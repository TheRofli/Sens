"""Adaptive CPU resource policy shared by local ASR engines."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CpuPlan:
    logical: int
    physical: int
    inference_threads: int
    reserved_logical: int
    source: str


def choose_cpu_plan(
    logical: int | None = None,
    physical: int | None = None,
    override: str | None = None,
) -> CpuPlan:
    logical = max(1, int(logical or os.cpu_count() or 1))
    if physical is None:
        try:
            import psutil

            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    physical = max(1, min(logical, int(physical or max(1, logical // 2))))
    configured = override if override is not None else os.environ.get("SENS_ASR_THREADS")
    if configured:
        try:
            threads = max(1, min(logical, int(configured)))
        except ValueError:
            threads = 0
        if threads:
            return CpuPlan(logical, physical, threads, logical - threads, "override")

    if logical <= 2:
        reserved = 1 if logical > 1 else 0
    elif logical <= 8:
        reserved = 2
    else:
        reserved = max(2, logical // 8)
    budget = max(1, logical - reserved)
    threads = max(1, min(physical, budget, 12))
    return CpuPlan(logical, physical, threads, logical - threads, "auto")


def inference_threads() -> int:
    return choose_cpu_plan().inference_threads
