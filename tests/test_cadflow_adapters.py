"""Tests for solver adapters and case-deck generation."""

from __future__ import annotations

from pathlib import Path

from cadflow.adapters import FEAAdapter, MBDAdapter, OpenFOAMAdapter, SolverJob, run_solver


def test_fea_adapter_writes_deck_and_falls_back(tmp_path: Path) -> None:
    geom = tmp_path / "g.stl"
    geom.write_text("solid x\nendsolid x\n", encoding="utf-8")
    adapter = FEAAdapter()
    job = SolverJob(
        job_id="j1",
        geometry_path=str(geom),
        workdir=tmp_path / "fea",
        parameters={"max_stress_mpa": 120.0, "objective": 120.0},
        materials=("Al6061",),
        allow_fallback=True,
    )
    result = adapter.run(job)
    assert (tmp_path / "fea" / "job.inp").exists()
    assert "*ELASTIC" in (tmp_path / "fea" / "job.inp").read_text()
    assert result.ok is True
    assert result.metadata.get("mode") in {"fallback", "fallback_after_native_error", "native", "native_stub"}
    assert result.objective == 120.0 or result.metadata.get("max_von_mises_mpa") == 120.0


def test_openfoam_and_mbd_case_decks(tmp_path: Path) -> None:
    geom = tmp_path / "g.stl"
    geom.write_text("solid x\nendsolid x\n", encoding="utf-8")
    foam = OpenFOAMAdapter().run(
        SolverJob("j2", str(geom), tmp_path / "of", {"Cd_guess": 0.22, "objective": 0.22}, allow_fallback=True)
    )
    assert (tmp_path / "of" / "openfoam_case" / "system" / "controlDict").exists()
    assert foam.ok is True

    mbd = MBDAdapter().run(
        SolverJob("j3", str(geom), tmp_path / "mbd", {"peak_torque": 33.0, "objective": 33.0}, allow_fallback=True)
    )
    assert (tmp_path / "mbd" / "mbd_model.json").exists()
    assert mbd.ok is True


def test_run_solver_helper(tmp_path: Path) -> None:
    geom = tmp_path / "g.stl"
    geom.write_text("solid x\nendsolid x\n", encoding="utf-8")
    result = run_solver(
        "fea",
        job_id="x",
        geometry_path=str(geom),
        workdir=tmp_path / "s",
        parameters={"objective": 10.0},
        allow_fallback=True,
    )
    assert result.ok is True
