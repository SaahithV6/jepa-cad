"""Heuristic solver-result-driven design loop for existing or intake'd projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Callable, Mapping, Sequence

from .backends import CadBackend, get_backend
from .flywheel import DataFlywheel
from .manifest import JobManifest
from .pipeline import PipelineResult, run_pipeline
from .project import intake_project
from .runtime import SolverRuntime


@dataclass(frozen=True, slots=True)
class DesignCycleResult:
    cycle_index: int
    manifest: JobManifest
    pipeline: PipelineResult
    target_key: str | None
    target_value: float | None
    objective: float | None
    decision: str
    mutation: dict[str, Any]
    workdir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "manifest": self.manifest.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "target_key": self.target_key,
            "target_value": self.target_value,
            "objective": self.objective,
            "decision": self.decision,
            "mutation": self.mutation,
            "workdir": self.workdir,
        }


@dataclass(frozen=True, slots=True)
class DesignLoopResult:
    project_root: str
    out_dir: str
    repeat: int
    tolerance: float
    target_key: str | None
    target_value: float | None
    stop_reason: str
    intake_manifest: str
    latest_manifest: str
    latest_result: str
    best_objective: float | None
    cycles: tuple[DesignCycleResult, ...]

    @property
    def ok(self) -> bool:
        return self.stop_reason in {"target-met", "repeat-exhausted"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "out_dir": self.out_dir,
            "repeat": self.repeat,
            "tolerance": self.tolerance,
            "target_key": self.target_key,
            "target_value": self.target_value,
            "stop_reason": self.stop_reason,
            "intake_manifest": self.intake_manifest,
            "latest_manifest": self.latest_manifest,
            "latest_result": self.latest_result,
            "best_objective": self.best_objective,
            "cycles": [cycle.to_dict() for cycle in self.cycles],
            "ok": self.ok,
        }


def _target_from_manifest(manifest: JobManifest) -> tuple[str | None, float | None]:
    raw_targets = manifest.parameters.get("targets")
    targets = raw_targets if isinstance(raw_targets, dict) else {}
    for key, value in targets.items():
        if isinstance(value, (int, float)):
            return str(key), float(value)
    objective = manifest.parameters.get("objective")
    if isinstance(objective, (int, float)):
        return "objective", float(objective)
    return None, None


def _scale_geometry(geometry: dict[str, Any], factor: float, cycle_index: int) -> dict[str, Any]:
    kind = str(geometry.get("kind") or "box").lower()
    geom = dict(geometry)
    if kind == "box":
        for key in ("width", "height", "depth"):
            if key in geom:
                geom[key] = max(float(geom[key]) * factor, 1e-3)
    elif kind in {"extrude", "extrusion", "extrude_profile"}:
        if "height" in geom:
            geom["height"] = max(float(geom["height"]) * factor, 1e-3)
        profile = geom.get("profile")
        if isinstance(profile, list):
            geom["profile"] = [(float(x) * factor, float(y) * factor) for x, y in profile]
    elif kind in {"assembly", "union"}:
        parts = geom.get("parts")
        if isinstance(parts, list):
            geom["parts"] = [_scale_geometry(dict(part), factor, cycle_index) for part in parts]
    else:
        for key in ("width", "height", "depth", "radius"):
            if key in geom:
                geom[key] = max(float(geom[key]) * factor, 1e-3)

    features = list(geom.get("features") or [])
    if cycle_index > 0:
        features.append({"op": "fillet", "radius": round(0.01 * (cycle_index + 1), 5)})
    if features:
        geom["features"] = features
    return geom


def _mutation_factor(solver: str, objective: float | None, target_value: float | None, cycle_index: int) -> float:
    if objective is None or target_value is None or target_value <= 0:
        return 1.0
    if objective <= target_value:
        return 1.0
    if solver == "openfoam":
        return 0.94
    if solver == "fea":
        return 1.08
    if solver == "mbd":
        return 0.97
    return 1.02


def run_design_loop(
    *,
    manifest: JobManifest | None = None,
    project_root: str | Path | None = None,
    goal: str | None = None,
    family: str = "space",
    solver: str | None = None,
    material: str | None = None,
    targets: Mapping[str, Any] | None = None,
    out_dir: str | Path,
    repeat: int = 3,
    tolerance: float = 0.05,
    notes: str | None = None,
    backend: CadBackend | None = None,
    flywheel: DataFlywheel | None = None,
    runtime: SolverRuntime | None = None,
    allow_solver_fallback: bool = True,
    prefer_real_cad: bool = True,
    solver_payload_factory: Callable[[JobManifest, int], Mapping[str, Any] | None] | None = None,
) -> DesignLoopResult:
    """Iteratively run solver/verification cycles and adjust the manifest between cycles.

    The loop is intentionally deterministic and heuristic:
    - FEA targets enlarge structural envelope when stress is above the bound.
    - CFD targets shrink frontal area when drag is above the bound.
    - If the project was intake'd from an existing directory, the derived envelope
      from the source geometry is used as the starting point.
    """

    if manifest is None:
        if project_root is None or goal is None:
            raise ValueError("either manifest or project_root+goal must be provided")
        intake = intake_project(
            project_root,
            goal=goal,
            family=family,
            solver=solver,
            material=material,
            targets=targets,
            notes=notes,
            out_dir=Path(out_dir) / "intake",
        )
        manifest = intake.manifest
    target_key, target_value = _target_from_manifest(manifest)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    backend = backend or get_backend(prefer_real=prefer_real_cad)

    cycles: list[DesignCycleResult] = []
    best_objective: float | None = None
    best_cycle_manifest: JobManifest | None = None
    latest_manifest = manifest
    latest_result_path = root / "latest_result.json"
    latest_manifest_path = root / "latest_manifest.json"
    history_path = root / "history.jsonl"
    stop_reason = "repeat-exhausted"

    for cycle_index in range(max(1, repeat)):
        cycle_dir = root / f"cycle_{cycle_index:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        payload = solver_payload_factory(latest_manifest, cycle_index) if solver_payload_factory is not None else None
        pipeline = run_pipeline(
            latest_manifest,
            backend=backend,
            workdir=cycle_dir,
            flywheel=flywheel,
            solver_kind=solver or str(latest_manifest.parameters.get("solver", "fea")),
            solver_payload=payload,
            prefer_real_cad=prefer_real_cad,
            allow_solver_fallback=allow_solver_fallback,
            runtime=runtime,
        )
        objective = pipeline.solver_result.objective
        if isinstance(objective, (int, float)):
            objective = float(objective)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_cycle_manifest = latest_manifest
        mutation: dict[str, Any] = {}
        decision = "continue"
        if target_value is not None and objective is not None and objective <= target_value * (1.0 + tolerance):
            decision = "target-met"
            stop_reason = decision
        elif cycle_index + 1 >= repeat:
            decision = "repeat-exhausted"
            stop_reason = decision
        elif not pipeline.ok:
            decision = "pipeline-failed"
            stop_reason = decision
        else:
            factor = _mutation_factor(str(latest_manifest.parameters.get("solver", solver or "fea")), objective, target_value, cycle_index)
            geometry = dict(latest_manifest.inputs.get("geometry") or {"kind": "box", "width": 1.0, "height": 1.0, "depth": 1.0})
            next_geometry = _scale_geometry(geometry, factor, cycle_index + 1)
            mutation = {"factor": factor, "geometry": next_geometry}
            latest_manifest = JobManifest(
                name=latest_manifest.name,
                inputs={**latest_manifest.inputs, "geometry": next_geometry},
                parameters=dict(latest_manifest.parameters),
                tags=latest_manifest.tags,
                notes=latest_manifest.notes,
                artifacts=latest_manifest.artifacts,
            )
            latest_manifest_path.write_text(json.dumps(latest_manifest.to_dict(), indent=2), encoding="utf-8")

        cycle_result = DesignCycleResult(
            cycle_index=cycle_index,
            manifest=latest_manifest,
            pipeline=pipeline,
            target_key=target_key,
            target_value=target_value,
            objective=objective if isinstance(objective, (int, float)) else None,
            decision=decision,
            mutation=mutation,
            workdir=str(cycle_dir),
        )
        cycles.append(cycle_result)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_result.to_dict(), sort_keys=True) + "\n")
        latest_result_path.write_text(json.dumps(cycle_result.to_dict(), indent=2), encoding="utf-8")

        if decision in {"target-met", "pipeline-failed"}:
            break

    intake_manifest_path = root / "intake_manifest.json"
    intake_manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    latest_manifest_path.write_text(json.dumps(latest_manifest.to_dict(), indent=2), encoding="utf-8")
    if best_cycle_manifest is None:
        best_cycle_manifest = latest_manifest

    return DesignLoopResult(
        project_root=str(project_root or manifest.inputs.get("project_root") or ""),
        out_dir=str(root),
        repeat=repeat,
        tolerance=float(tolerance),
        target_key=target_key,
        target_value=target_value,
        stop_reason=stop_reason,
        intake_manifest=str(intake_manifest_path),
        latest_manifest=str(latest_manifest_path),
        latest_result=str(latest_result_path),
        best_objective=best_objective,
        cycles=tuple(cycles),
    )
