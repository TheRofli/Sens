from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    ram_mb: float
    cpu_percent: float
    threads: int

    @property
    def ram_label(self) -> str:
        if self.ram_mb >= 1024:
            return f"{self.ram_mb / 1024:.2f} GB"
        return f"{self.ram_mb:.0f} MB"

    @property
    def cpu_label(self) -> str:
        return f"{self.cpu_percent:.1f}%"

    @property
    def threads_label(self) -> str:
        return str(self.threads)


class ProcessResourceMonitor:
    def __init__(self) -> None:
        self._process = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)
        except Exception:
            self._process = None

    def snapshot(self) -> ResourceSnapshot:
        if self._process is None:
            return ResourceSnapshot(0.0, 0.0, 0)
        try:
            memory = self._process.memory_info().rss / (1024 * 1024)
            cpu = self._process.cpu_percent(interval=None)
            threads = self._process.num_threads()
            return ResourceSnapshot(memory, cpu, threads)
        except Exception:
            return ResourceSnapshot(0.0, 0.0, 0)

