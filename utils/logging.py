"""Training logging utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(self, log_dir: str | Path, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self._step = 0
        self._window_start = time.perf_counter()
        self._window_steps = 0

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        self._step = step
        record = {"step": step, "time": time.time(), **metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        parts = [f"step={step}"]
        for key in ("loss", "lr", "grad_norm", "embed_norm", "embed_std", "samples_per_sec"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.6f}" if isinstance(metrics[key], float) else f"{key}={metrics[key]}")
        print(" | ".join(parts))

    def tick_batch(self, batch_size: int) -> float | None:
        self._window_steps += 1
        if self._window_steps < 10:
            return None
        elapsed = time.perf_counter() - self._window_start
        sps = (self._window_steps * batch_size) / max(elapsed, 1e-6)
        self._window_start = time.perf_counter()
        self._window_steps = 0
        return sps

    def warn(self, message: str) -> None:
        print(f"WARNING: {message}")
