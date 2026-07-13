from __future__ import annotations

from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord


def test_job_manifest_fingerprint_round_trip() -> None:
    manifest = JobManifest(
        name="generate-box",
        inputs={"dimensions": [1, 2, 3]},
        parameters={"backend": "cadquery"},
        tags=("smoke", "cad"),
        artifacts=("geometry.step",),
    )

    again = JobManifest.from_dict(manifest.to_dict())

    assert again == manifest
    assert len(manifest.fingerprint) == 16
    assert manifest.fingerprint == again.fingerprint
    assert again.artifacts == ("geometry.step",)


def test_run_record_includes_manifest_and_provenance() -> None:
    manifest = JobManifest(name="solve-foo")
    provenance = ProvenanceRecord.for_manifest(
        manifest,
        source="unit-test",
        parent_fingerprints=("abc",),
        artifact_refs=("out.stl",),
        tags=("ci",),
    )
    record = RunRecord(manifest=manifest, provenance=provenance, status="queued", artifact_refs=("out.stl",))

    payload = record.to_dict()
    restored = RunRecord.from_dict(payload)

    assert restored.manifest == manifest
    assert restored.provenance.source == "unit-test"
    assert restored.status == "queued"
    assert restored.manifest_fingerprint == manifest.fingerprint
    assert restored.artifact_refs == ("out.stl",)
    assert restored.provenance.parent_fingerprints == ("abc",)
    assert "fingerprint" in restored.provenance.to_dict()


def test_run_record_with_results_round_trip() -> None:
    manifest = JobManifest(name="job")
    provenance = ProvenanceRecord.for_manifest(manifest, source="test")
    run = RunRecord(manifest=manifest, provenance=provenance).with_results(
        status="verified",
        solver_result={"status": "optimal", "objective": 1.0},
        verification={"name": "solid", "passed": True},
        artifact_refs=["a.step"],
    )
    restored = RunRecord.from_dict(run.to_dict())
    assert restored.status == "verified"
    assert restored.solver_result["objective"] == 1.0
    assert restored.verification["passed"] is True
