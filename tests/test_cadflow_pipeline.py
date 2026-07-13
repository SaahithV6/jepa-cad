"""End-to-end smoke test for cadflow orchestration."""

from __future__ import annotations

from pathlib import Path

from cadflow.backends import MockCadBackend
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest
from cadflow.pipeline import run_pipeline


def test_pipeline_manifest_to_flywheel(tmp_path: Path) -> None:
    manifest = JobManifest(
        name="bracket-smoke",
        inputs={
            "geometry": {
                "kind": "box",
                "width": 2.0,
                "height": 1.0,
                "depth": 0.5,
            }
        },
        parameters={"solver": "fea", "objective": 0.4},
        tags=("smoke", "e2e"),
    )
    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    result = run_pipeline(
        manifest,
        backend=MockCadBackend(),
        workdir=tmp_path / "work",
        flywheel=flywheel,
        prefer_real_cad=False,
    )

    assert result.ok is True
    assert result.verification.passed is True
    assert result.solver_result.ok is True
    assert len(result.artifacts) == 2
    assert all(Path(p).exists() for p in result.artifacts)
    assert result.flywheel_entry is not None
    assert "PASSED" in result.report_text
    assert len(list(flywheel.load_entries())) == 1
    assert flywheel.promote_best(1)[0].manifest.name == "bracket-smoke"


def test_pipeline_rejects_unverified_from_flywheel(tmp_path: Path) -> None:
    # Force a volume mismatch by verifying against unexpected expected volume indirectly:
    # Use a solver failure payload so flywheel only_verified skips recording.
    manifest = JobManifest(
        name="fail-case",
        inputs={"geometry": {"kind": "box", "width": 1, "height": 1, "depth": 1}},
        parameters={"solver": "fea"},
    )
    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    result = run_pipeline(
        manifest,
        backend=MockCadBackend(),
        workdir=tmp_path / "work",
        flywheel=flywheel,
        solver_payload={"status": "failed", "objective": None},
    )
    assert result.ok is False
    assert result.flywheel_entry is None
    assert list(flywheel.load_entries()) == []
