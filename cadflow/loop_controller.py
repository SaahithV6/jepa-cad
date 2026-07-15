"""Repeated execution controller for the verified-data flywheel loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Sequence
import json

from .flywheel_loop import FlywheelLoopResult, run_flywheel_loop


@dataclass(frozen=True, slots=True)
class LoopControllerResult:
    iterations: int
    stop_reason: str
    repeat: int
    interval_seconds: float
    stop_file: str | None
    out_dir: str
    history_path: str
    latest_path: str
    results: tuple[FlywheelLoopResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "repeat": self.repeat,
            "interval_seconds": self.interval_seconds,
            "stop_file": self.stop_file,
            "out_dir": self.out_dir,
            "history_path": self.history_path,
            "latest_path": self.latest_path,
            "results": [result.to_dict() for result in self.results],
            "ok": self.ok,
        }


def _append_history(history_path: Path, payload: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_latest(latest_path: Path, payload: dict[str, Any]) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sleep_with_stop(interval_seconds: float, stop_path: Path | None) -> bool:
    if interval_seconds <= 0:
        return stop_path is not None and stop_path.exists()
    remaining = interval_seconds
    while remaining > 0:
        if stop_path is not None and stop_path.exists():
            return True
        chunk = min(1.0, remaining)
        sleep(chunk)
        remaining -= chunk
    return stop_path is not None and stop_path.exists()


def run_loop_controller(
    raw_dirs: Sequence[str | Path] | None,
    out_dir: str | Path,
    *,
    repeat: int = 1,
    interval_seconds: float = 0.0,
    stop_file: str | Path | None = None,
    **loop_kwargs: Any,
) -> LoopControllerResult:
    """Run ``run_flywheel_loop`` repeatedly until a stop condition is met.

    ``repeat`` controls the upper bound on cycles. ``repeat=0`` means keep
    running until ``stop_file`` appears or a cycle fails.
    """

    if repeat < 0:
        raise ValueError("repeat must be >= 0")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be >= 0")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "history.jsonl"
    latest_path = root / "latest.json"
    stop_path = Path(stop_file) if stop_file is not None else None
    cycles_dir = root / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)

    results: list[FlywheelLoopResult] = []
    iterations = 0
    stop_reason = "repeat-exhausted"

    while True:
        if stop_path is not None and stop_path.exists():
            stop_reason = "stop-file"
            break
        if repeat > 0 and iterations >= repeat:
            stop_reason = "repeat-exhausted"
            break

        result = run_flywheel_loop(raw_dirs, root, **loop_kwargs)
        results.append(result)
        iterations += 1

        payload = result.to_dict()
        _append_history(history_path, payload)
        _write_latest(latest_path, payload)
        cycle_path = cycles_dir / f"{iterations:04d}_{result.cycle_id}.json"
        cycle_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if not result.ok:
            stop_reason = "cycle-failed"
            break
        if stop_path is not None and stop_path.exists():
            stop_reason = "stop-file"
            break
        if repeat > 0 and iterations >= repeat:
            stop_reason = "repeat-exhausted"
            break
        if _sleep_with_stop(interval_seconds, stop_path):
            stop_reason = "stop-file"
            break

    return LoopControllerResult(
        iterations=iterations,
        stop_reason=stop_reason,
        repeat=repeat,
        interval_seconds=interval_seconds,
        stop_file=str(stop_path) if stop_path is not None else None,
        out_dir=str(root),
        history_path=str(history_path),
        latest_path=str(latest_path),
        results=tuple(results),
    )
