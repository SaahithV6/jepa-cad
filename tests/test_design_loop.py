from __future__ import annotations

from pathlib import Path

from cadflow.design_loop import run_design_loop
from cadflow.manifest import JobManifest


def test_design_loop_mutates_geometry_until_target_met(tmp_path: Path) -> None:
    manifest = JobManifest(
        name="bracket-loop",
        inputs={"geometry": {"kind": "box", "width": 1.0, "height": 1.0, "depth": 1.0}},
        parameters={"solver": "fea", "targets": {"max_stress_mpa": 0.5}},
        tags=("space", "assembly"),
    )

    def fake_solver_payload(current_manifest: JobManifest, cycle_index: int):
        objective = 1.0 if cycle_index == 0 else 0.4
        return {"status": "success", "objective": objective}

    result = run_design_loop(
        manifest=manifest,
        out_dir=tmp_path / "loop",
        repeat=3,
        tolerance=0.05,
        solver_payload_factory=fake_solver_payload,
    )

    assert result.stop_reason == "target-met"
    assert len(result.cycles) == 2
    assert result.cycles[0].objective == 1.0
    assert result.cycles[0].mutation["factor"] > 1.0
    assert result.cycles[0].mutation["geometry"]["width"] > 1.0
    assert result.cycles[1].objective == 0.4
    assert (tmp_path / "loop" / "history.jsonl").exists()
    assert (tmp_path / "loop" / "intake_manifest.json").exists()
