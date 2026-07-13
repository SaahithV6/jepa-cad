from __future__ import annotations

from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord


def test_job_manifest_fingerprint_round_trip() -> None:
    manifest = JobManifest(
        name="generate-box",
        inputs={"dimensions": [1, 2, 3]},
        parameters={"backend": "cadquery"},
        tags=("smoke", "cad"),
    )

    again = JobManifest.from_dict(manifest.to_dict())

    assert again == manifest
    assert len(manifest.fingerprint) == 16
    assert manifest.fingerprint == again.fingerprint


def test_run_record_includes_manifest_and_provenance() -> None:
    manifest = JobManifest(name="solve-foo")
    provenance = ProvenanceRecord.for_manifest(manifest, source="unit-test")
    record = RunRecord(manifest=manifest, provenance=provenance, status="queued")

    payload = record.to_dict()
    restored = RunRecord.from_dict(payload)

    assert restored.manifest == manifest
    assert restored.provenance.source == "unit-test"
    assert restored.status == "queued"
    assert restored.manifest_fingerprint == manifest.fingerprint
