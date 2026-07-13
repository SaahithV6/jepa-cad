"""Thin end-to-end CAD/CAE orchestration path.

manifest -> geometry build -> export -> solver adapter -> verification -> flywheel
Optional: promote verified runs into curated training shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adapters import run_solver
from .backends import CadBackend, build_from_spec, get_backend
from .flywheel import DataFlywheel, FlywheelEntry
from .manifest import JobManifest, ProvenanceRecord, RunRecord
from .promotion import PromotionResult, promote_verified_to_dataset
from .solver import SolverResult, wrap_solver_result
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
    promotion: PromotionResult | None = None

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
            "promotion": self.promotion.to_dict() if self.promotion is not None else None,
        }


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
    allow_solver_fallback: bool = True,
    promote_to: str | Path | None = None,
    promote_limit: int = 5,
) -> PipelineResult:
    """Execute a deterministic CAD/CAE orchestration loop with hard gates."""

    backend = backend or get_backend(prefer_real=prefer_real_cad)
    workdir = Path(workdir or "artifacts") / manifest.fingerprint
    workdir.mkdir(parents=True, exist_ok=True)

    geometry_spec = dict(manifest.inputs.get("geometry") or manifest.parameters.get("geometry") or {})
    if not geometry_spec:
        geometry_spec = {
            "kind": "box",
            "width": float(manifest.parameters.get("width", 1.0)),
            "height": float(manifest.parameters.get("height", 1.0)),
            "depth": float(manifest.parameters.get("depth", 1.0)),
        }

    shape = build_from_spec(geometry_spec, backend=backend)
    step_path = backend.export_step(shape, workdir / "geometry.step")
    stl_path = backend.export_stl(shape, workdir / "geometry.stl")
    artifacts: list[str] = [str(step_path), str(stl_path)]

    # Geometry gate before solvers.
    verification = verify_solid(shape, backend=backend)
    report_text = render_verification_report(verification)
    (workdir / "verification.txt").write_text(report_text + "\n", encoding="utf-8")
    artifacts.append(str(workdir / "verification.txt"))

    kind = solver_kind or str(manifest.parameters.get("solver", "fea"))
    if not verification.passed:
        solver_result = SolverResult(
            status="failed",
            metadata={"reason": "geometry_verification_failed"},
            logs=("solver skipped: geometry verification failed",),
        )
    elif solver_payload is not None:
        solver_result = wrap_solver_result(solver_payload)
    else:
        materials = tuple(manifest.inputs.get("materials") or manifest.parameters.get("materials") or [])
        solver_result = run_solver(
            kind,
            job_id=manifest.fingerprint,
            geometry_path=str(stl_path),
            workdir=workdir / "solver",
            parameters=dict(manifest.parameters),
            materials=materials,
            allow_fallback=allow_solver_fallback,
        )
        artifacts.extend(list(solver_result.artifacts))

    status = "verified" if verification.passed and solver_result.ok else "failed"
    provenance = ProvenanceRecord.for_manifest(
        manifest,
        source=source,
        details={
            "backend": backend.name,
            "solver": kind,
            "solver_mode": solver_result.metadata.get("mode"),
        },
        artifact_refs=artifacts,
    )
    run = RunRecord(
        manifest=manifest.with_artifacts(artifacts),
        provenance=provenance,
        status=status,
        solver_result=solver_result.to_dict(),
        verification=verification.to_dict(),
        artifact_refs=tuple(artifacts),
    )

    entry = None
    if flywheel is not None:
        entry = flywheel.record(run, solver_result, verification, only_verified=True)

    promotion = None
    if promote_to is not None and flywheel is not None:
        promotion = promote_verified_to_dataset(
            flywheel,
            promote_to,
            limit=promote_limit,
        )

    return PipelineResult(
        run=run,
        shape=shape,
        solver_result=solver_result,
        verification=verification,
        artifacts=tuple(artifacts),
        flywheel_entry=entry,
        report_text=report_text,
        promotion=promotion,
    )
