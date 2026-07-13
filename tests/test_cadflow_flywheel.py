from __future__ import annotations

from cadflow.backends import CadQueryBackend
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord
from cadflow.solver import SolverResult
from cadflow.verification import VerificationReport


def test_flywheel_persists_and_reloads_entries(tmp_path) -> None:
    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    manifest = JobManifest(name="solve-bracket", parameters={"backend": "cadquery"})
    provenance = ProvenanceRecord.for_manifest(manifest, source="unit-test")
    run = RunRecord(manifest=manifest, provenance=provenance, status="finished")
    solver = SolverResult(status="optimal", objective=0.5, iterations=3)
    verification = VerificationReport(name="solve-bracket", passed=True, findings=(), metrics={"volume": 6.0})

    entry = flywheel.record(run, solver_result=solver, verification=verification)
    reloaded = list(flywheel.load_entries())

    assert entry is not None
    assert entry.manifest_fingerprint == manifest.fingerprint
    assert len(reloaded) == 1
    assert reloaded[0].verification.passed is True
    assert reloaded[0].solver_result.ok is True
    assert reloaded[0].verified is True


def test_flywheel_suggests_best_past_case(tmp_path) -> None:
    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    backend = CadQueryBackend()
    solid = backend.box(1.0, 1.0, 1.0)

    flywheel.record(
        RunRecord(JobManifest(name="case-a"), ProvenanceRecord(source="unit-test")),
        solver_result=SolverResult(status="failed", objective=None),
        verification=VerificationReport(name="case-a", passed=False, findings=("solver failed",), metrics={"volume": 1.0}),
    )
    flywheel.record(
        RunRecord(JobManifest(name="case-b"), ProvenanceRecord(source="unit-test")),
        solver_result=SolverResult(status="optimal", objective=0.0),
        verification=VerificationReport(name="case-b", passed=True, findings=(), metrics={"volume": backend.volume(solid)}),
    )

    best = flywheel.best_runs(limit=1)

    assert len(best) == 1
    assert best[0].manifest.name == "case-b"


def test_flywheel_only_verified_record_and_promote(tmp_path) -> None:
    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    bad = flywheel.record(
        RunRecord(JobManifest(name="bad"), ProvenanceRecord(source="t")),
        SolverResult(status="failed"),
        VerificationReport(name="bad", passed=False),
        only_verified=True,
    )
    assert bad is None
    assert list(flywheel.load_entries()) == []

    flywheel.record(
        RunRecord(JobManifest(name="good-high"), ProvenanceRecord(source="t")),
        SolverResult(status="optimal", objective=5.0),
        VerificationReport(name="good-high", passed=True),
    )
    flywheel.record(
        RunRecord(JobManifest(name="good-low"), ProvenanceRecord(source="t")),
        SolverResult(status="optimal", objective=0.1),
        VerificationReport(name="good-low", passed=True),
    )
    promoted = flywheel.promote_best(limit=1)
    assert promoted[0].manifest.name == "good-low"
