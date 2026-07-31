"""Tests for MSH2 -> CalculiX solid mesh conversion."""

from pathlib import Path

from cadflow.msh_to_calculix import (
    generate_fea_case_inp,
    parse_frd_summary,
    parse_msh2_solid,
    run_calculix_case,
)

SAMPLE_CASE = Path("artifacts/fea_final/0024da0fdb17d12c")
KNOWN_GOOD_FRD = Path("artifacts/fea_final/446ab87ed8e9c21c/case.frd")


def test_parse_msh2_solid_extracts_tets_only():
    mesh = parse_msh2_solid(SAMPLE_CASE / "mesh.msh")
    assert len(mesh.nodes) == 17031
    assert len(mesh.elements) == 88960
    assert all(len(nodes) == 4 for _, nodes in mesh.elements)


def test_generate_fea_case_inp_writes_decks():
    setup = generate_fea_case_inp(SAMPLE_CASE)
    assert setup.mesh_inp.exists()
    assert setup.case_inp.exists()
    assert setup.fixed_nodes > 0
    assert setup.loaded_nodes > 0
    mesh_text = setup.mesh_inp.read_text(encoding="utf-8")
    assert "*NODE" in mesh_text
    assert "C3D4" in mesh_text
    assert "T3D2" not in mesh_text
    assert "CPS3" not in mesh_text
    case_text = setup.case_inp.read_text(encoding="utf-8")
    assert "mesh_solid.inp" in case_text
    assert "mesh_filtered.inp" not in case_text


def test_parse_frd_summary_reads_disp_and_stress():
    assert KNOWN_GOOD_FRD.exists()
    summary = parse_frd_summary(KNOWN_GOOD_FRD)
    assert summary is not None
    assert summary.max_displacement_mm > 0
    assert summary.max_von_mises_mpa > 0
    assert summary.node_count > 1000


def test_calculix_produces_frd():
    generate_fea_case_inp(SAMPLE_CASE)
    result = run_calculix_case(SAMPLE_CASE, timeout=120)
    assert result.converged
    assert result.frd_bytes > 1_000_000
