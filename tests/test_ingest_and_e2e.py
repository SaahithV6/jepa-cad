"""Smoke tests for ingesting mixed sources and running the end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

from cadflow.backends import MockCadBackend
from cadflow.e2e import run_end_to_end
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord
from cadflow.solver import SolverResult
from cadflow.verification import VerificationReport
from data.ingest import ingest_sources


def _write_ascii_stl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "solid box",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 1 1 0",
                "    endloop",
                "  endfacet",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 1 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                "endsolid box",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_ingest_sources_combines_raw_and_verified_flywheel(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    stl = raw_dir / "part.stl"
    _write_ascii_stl(stl)

    backend = MockCadBackend()
    solid = backend.box(2, 2, 2)
    geom = backend.export_stl(solid, tmp_path / "geom.stl")
    flywheel = DataFlywheel(tmp_path / "fw.jsonl")
    manifest = JobManifest.from_dict({"name": "promo", "artifacts": [str(geom)]})
    run = RunRecord(
        manifest,
        ProvenanceRecord.for_manifest(manifest, source="test"),
        status="verified",
        solver_result={"status": "optimal", "objective": 0.2},
        verification={"name": "solid", "passed": True, "metrics": {"volume": 8.0}},
    )
    flywheel.record(
        run,
        SolverResult(status="optimal", objective=0.2),
        VerificationReport(name="solid", passed=True, metrics={"volume": 8.0}),
    )

    out = tmp_path / "curated"
    result = ingest_sources([raw_dir], out, flywheel_path=flywheel.path, num_points=32, num_fields=3)
    assert result.ingested == 2
    assert (out / "manifest.json").exists()
    assert (out / "ingestion_manifest.json").exists()


def test_end_to_end_smoke_trains_on_ingested_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_ascii_stl(raw_dir / "part.stl")

    out = tmp_path / "dataset"
    result = run_end_to_end([raw_dir], out, max_steps=1, data_source="real")
    assert result.ingestion.ingested == 1
    assert result.train_returncode == 0
    assert (out / "manifest.json").exists()
