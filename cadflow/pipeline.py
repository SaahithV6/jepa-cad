"""Thin end-to-end CAD/CAE orchestration path.

manifest -> geometry build -> export -> solver -> verification -> flywheel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends import CadBackend, build_from_spec, get_backend
from .flywheel import DataFlywheel, FlywheelEntry
from .manifest import JobManifest, ProvenanceRecord, RunRecord
from .solver import (
    SolverResult,
    probe_fea,
    probe_mbd,
    probe_openfoam,
    run_fallback_solver,
    wrap_solver_result,
)
from .verification import VerificationReport, render_verification_report, verify_solid


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run: RunRecord
    shape: Any
    solver_result: SolverResult
    verification: VerificationReport
    artifacts: tuple[str, ...]
    flywheel_entry: FlywheelEntry | None
    report_text: str

    @property
    def ok(self) -> bool:
        return self.verification.passed and self.solver_result.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "solver_result": self.solver_result.to_dict(),
            "verification": self.verification.to_dict(),
            "artifacts": list(self.artifacts),
            "flywheel_recorded": self.flywheel_entry is not None,
            "report_text": self.report_text,
            "ok": self.ok,
        }


def _select_solver_probe(kind: str):
    mapping = {
        "openfoam": probe_openfoam,
        "cfd": probe_openfoam,
        "fea": probe_fea,
        "structural": probe_fea,
        "mbd": probe_mbd,
        "dynamics": probe_mbd,
    }
    return mapping.get(kind.lower(), probe_fea)()


def run_pipeline(
    manifest: JobManifest,
    *,
    backend: CadBackend | None = None,
    workdir: str | Path | None = None,
    flywheel: DataFlywheel | None = None,
    source: str = "cadflow.pipeline",
    solver_kind: str | None = None,
    solver_payload: Mapping[str, Any] | None = None,
    prefer_real_cad: bool = True,
) -> PipelineResult:
    """Execute a minimal deterministic CAD/CAE orchestration loop."""

    backend = backend or get_backend(prefer_real=prefer_real_cad)
    workdir = Path(workdir or "artifacts") / manifest.fingerprint
    workdir.mkdir(parents=True, exist_ok=True)

    geometry_spec = dict(manifest.inputs.get("geometry") or manifest.parameters.get("geometry") or {})
    if not geometry_spec:
        # Sensible default primitive when planner omitted explicit geometry.
        geometry_spec = {
            "kind": "box",
            "width": float(manifest.parameters.get("width", 1.0)),
            "height": float(manifest.parameters.get("height", 1.0)),
            "depth": float(manifest.parameters.get("depth", 1.0)),
        }

    shape = build_from_spec(geometry_spec, backend=backend)
    step_path = backend.export_step(shape, workdir / "geometry.step")
    stl_path = backend.export_stl(shape, workdir / "geometry.stl")
    artifacts = (str(step_path), str(stl_path))

    kind = solver_kind or str(manifest.parameters.get("solver", "fea"))
    probe = _select_solver_probe(kind)
    if solver_payload is not None:
        solver_result = wrap_solver_result(solver_payload, probe=probe)
    elif probe.available:
        # Native binary present but full case decks are not wired yet — keep
        # an explicit fallback with probe metadata for accountability.
        solver_result = run_fallback_solver(
            backend=kind,
            objective=float(manifest.parameters.get("objective", 1.0)),
            metadata={"note": "native probe available; using calibrated fallback until case decks land"},
            probe=probe,
        )
    else:
        solver_result = run_fallback_solver(
            backend=kind,
            objective=float(manifest.parameters.get("objective", 1.0)),
            probe=probe,
        )

    verification = verify_solid(shape, backend=backend)
    report_text = render_verification_report(verification)

    status = "verified" if verification.passed and solver_result.ok else "failed"
    provenance = ProvenanceRecord.for_manifest(
        manifest,
        source=source,
        details={"backend": backend.name, "solver": kind},
        artifact_refs=artifacts,
    )
    run = RunRecord(
        manifest=manifest.with_artifacts(artifacts),
        provenance=provenance,
        status=status,
        solver_result=solver_result.to_dict(),
        verification=verification.to_dict(),
        artifact_refs=artifacts,
    )

    entry = None
    if flywheel is not None:
        # Only verified outputs enter the flywheel training path by default.
        entry = flywheel.record(run, solver_result, verification, only_verified=True)

    return PipelineResult(
        run=run,
        shape=shape,
        solver_result=solver_result,
        verification=verification,
        artifacts=artifacts,
        flywheel_entry=entry,
        report_text=report_text,
    )
