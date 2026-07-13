"""Small data flywheel utility for keeping useful CAD/CAE runs around."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .manifest import RunRecord
from .solver import SolverResult
from .verification import VerificationReport


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class FlywheelEntry:
    run: RunRecord
    solver_result: SolverResult
    verification: VerificationReport
    recorded_at: str = field(default_factory=_utc_now)

    @property
    def manifest(self):
        return self.run.manifest

    @property
    def manifest_fingerprint(self) -> str:
        return self.run.manifest_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "solver_result": self.solver_result.to_dict(),
            "verification": self.verification.to_dict(),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlywheelEntry":
        return cls(
            run=RunRecord.from_dict(payload["run"]),
            solver_result=SolverResult.from_dict(payload["solver_result"]),
            verification=VerificationReport.from_dict(payload["verification"]),
            recorded_at=str(payload.get("recorded_at", _utc_now())),
        )


class DataFlywheel:
    """Append-only JSONL store with light ranking helpers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        run: RunRecord,
        solver_result: SolverResult,
        verification: VerificationReport,
    ) -> FlywheelEntry:
        entry = FlywheelEntry(run=run, solver_result=solver_result, verification=verification)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return entry

    def load_entries(self) -> Iterable[FlywheelEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield FlywheelEntry.from_dict(json.loads(line))

    def best_runs(self, limit: int = 5) -> list[FlywheelEntry]:
        entries = list(self.load_entries())

        def key(entry: FlywheelEntry) -> tuple[int, float, str]:
            objective = entry.solver_result.objective
            objective_score = float(objective) if objective is not None else float("inf")
            return (
                1 if entry.verification.passed else 0,
                -objective_score if entry.verification.passed else objective_score,
                entry.recorded_at,
            )

        return sorted(entries, key=key, reverse=True)[:limit]
