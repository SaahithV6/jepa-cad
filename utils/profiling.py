"""Lightweight profiling helpers for training/debugging."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch


@dataclass
class TimeWindow:
    """Track average duration over a sliding window."""

    window: int = 50
    count: int = 0
    total: float = 0.0

    def update(self, dt: float) -> float:
        self.total += dt
        self.count += 1
        if self.count < self.window:
            return self.total / self.count
        avg = self.total / self.count
        # reset window
        self.count = 0
        self.total = 0.0
        return avg


@contextlib.contextmanager
def timer() -> float:
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return end - start  # pragma: no cover


def cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def bytes_to_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"


def get_peak_cuda_mem(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated(device=device)),
        "reserved": int(torch.cuda.memory_reserved(device=device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device=device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device=device)),
    }

