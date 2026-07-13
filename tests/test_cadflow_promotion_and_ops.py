"""Tests for richer CAD editing ops and flywheel promotion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cadflow.backends import MockCadBackend, build_from_spec
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord
from cadflow.promotion import promote_verified_to_dataset
from cadflow.solver import SolverResult
from cadflow.verification import VerificationReport


def test_boolean_cut_union_fillet_mock() -> None:
    backend = MockCadBackend()
    box = backend.box(10, 10, 10)
    tool = backend.cylinder(2, 10)
    cut = backend.boolean_cut(box, tool)
    assert backend.volume(cut) < backend.volume(box)
    fillet = backend.fillet(box, 0.5)
    assert backend.face_count(fillet) > backend.face_count(box)
    assembly = backend.boolean_union(box, tool)
    assert assembly.kind == "assembly"
    assert backend.volume(assembly) > 0


def test_build_from_spec_with_features() -> None:
    shape = build_from_spec(
        {
            "kind": "box",
            "width": 8,
            "height": 8,
            "depth": 8,
            "features": [
                {"op": "cut", "tool": {"kind": "cylinder", "radius": 1, "height": 8}},
                {"op": "fillet", "radius": 0.2},
            ],
        },
        backend=MockCadBackend(),
    )
    assert "boolean_cut" in shape.ops
    assert "fillet" in shape.ops


def test_promote_verified_writes_shards(tmp_path: Path) -> None:
    backend = MockCadBackend()
    solid = backend.box(2, 2, 2)
    stl = backend.export_stl(solid, tmp_path / "geom.stl")

    flywheel = DataFlywheel(tmp_path / "fw.jsonl")
    manifest = JobManifest(name="promo", artifacts=(str(stl),), tags=("curated",))
    run = RunRecord(
        manifest=manifest,
        provenance=ProvenanceRecord.for_manifest(manifest, source="test"),
        status="verified",
        artifact_refs=(str(stl),),
    )
    flywheel.record(
        run,
        SolverResult(status="optimal", objective=0.2),
        VerificationReport(name="solid", passed=True, metrics={"volume": 8.0}),
    )

    out = tmp_path / "curated"
    result = promote_verified_to_dataset(flywheel, out, limit=5, num_points=64, num_fields=3)
    assert result.promoted == 1
    assert Path(result.shard_paths[0]).exists()
    data = np.load(result.shard_paths[0])
    assert data["points"].shape == (64, 3)
    assert (out / "manifest.json").exists()
