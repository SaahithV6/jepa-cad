from __future__ import annotations

from pathlib import Path

from cadflow.cloud import build_cloud_training_plan
from cadflow.manifest import JobManifest
from cadflow.project import intake_project


def test_project_intake_detects_assemblies_fasteners_and_space_assets(tmp_path: Path) -> None:
    project = tmp_path / "cassini-bracket"
    (project / "subassembly").mkdir(parents=True)
    (project / "subassembly" / "rocket_nozzle.step").write_text("STEP", encoding="utf-8")
    (project / "fasteners").mkdir()
    (project / "fasteners" / "m8_bolt.stl").write_text("solid x\nendsolid x\n", encoding="utf-8")
    (project / "fasteners" / "washer.obj").write_text("o washer\n", encoding="utf-8")
    (project / "docs").mkdir()
    (project / "docs" / "cassini_assembly_notes.txt").write_text("notes", encoding="utf-8")

    result = intake_project(
        project,
        goal="reduce peak stress on spacecraft bracket assembly",
        family="space",
        material="Al 6061-T6",
        out_dir=tmp_path / "intake",
    )

    assert result.recommended_solver == "fea"
    assert result.detected_fasteners
    assert result.detected_assemblies
    assert result.detected_subsystems
    assert "structures" in result.detected_subsystems or "propulsion" in result.detected_subsystems or "tanks_and_feed" in result.detected_subsystems
    assert (tmp_path / "intake" / "project_manifest.json").exists()
    assert (tmp_path / "intake" / "project_intake.json").exists()
    assert result.manifest.parameters["solver"] == "fea"
    assert result.manifest.parameters["materials"] == ["Al 6061-T6"]
    assert any("structural target" in question.lower() for question in result.questions)


def test_cloud_plan_prefers_modal_and_space_datasets(tmp_path: Path) -> None:
    manifest = JobManifest(
        name="spacecraft-bracket",
        inputs={"project_root": str(tmp_path), "goal": "space bracket stress reduction", "source_paths": ["cassini.step"]},
        parameters={"solver": "fea", "family": "space", "targets": {"max_stress_mpa": 180.0}},
        tags=("space", "assembly"),
        notes="space bracket assembly",
    )

    plan = build_cloud_training_plan(manifest, family="space")

    assert plan.primary_provider == "Modal"
    assert plan.secondary_provider == "Fireworks"
    assert plan.dataset_sources
    keys = {source.key for source in plan.dataset_sources}
    assert {"nasa_3d_resources", "abc_dataset"}.intersection(keys)
    assert any(step.startswith("python train.py") for step in plan.training_steps)
    assert any("Modal" in note for note in plan.notes)
