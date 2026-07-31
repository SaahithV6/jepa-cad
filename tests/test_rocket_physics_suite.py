"""Tests for OpenRocket hardware CalculiX + OpenFOAM physics suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadflow.msh_to_calculix import case_has_valid_frd, parse_frd_summary, parse_msh2_solid
from cadflow.rocket_physics_suite import (
    domain_from_stl_mm,
    ensure_parts_in_graph,
    ingest_cfd_to_graph,
    ingest_fea_to_graph,
    load_manifest,
    material_elastic_props,
    mesh_stl_volume,
    run_cfd_for_entry,
    run_fea_for_entry,
    scale_load_n,
    select_entries,
)

CORPUS = Path("data/openrocket_hardware_8k")
SAMPLE_STL = CORPUS / "parts/nose_cone_00000.stl"


@pytest.fixture(scope="module")
def sample_entry() -> dict:
    manifest = load_manifest(CORPUS)
    entry = next(e for e in manifest if e["part_id"] == "nose_cone_00000")
    return entry


def test_manifest_loadable():
    manifest = load_manifest(CORPUS)
    assert len(manifest) >= 8000
    assert {"nose_cone", "fin", "nozzle"} <= {e["family"] for e in manifest}


def test_material_and_load_scaling(sample_entry):
    e_pa, nu = material_elastic_props(sample_entry)
    assert e_pa == pytest.approx(68.9e9)
    assert 0.2 < nu < 0.4
    load = scale_load_n(sample_entry, e_pa)
    assert 50.0 <= load <= 5e5


def test_mesh_stl_produces_tets(tmp_path, sample_entry):
    if not SAMPLE_STL.exists():
        pytest.skip("corpus STL missing")
    out = tmp_path / "mesh.msh"
    result = mesh_stl_volume(SAMPLE_STL, out, cl_max_mm=5.0, cl_min_mm=1.5)
    assert result.success, result.error
    assert result.tet_count > 100
    mesh = parse_msh2_solid(out)
    assert len(mesh.elements) == result.tet_count
    # SI meters: nose cone ~60 mm → ~0.06 m
    zs = [p[2] for p in mesh.nodes.values()]
    assert max(zs) < 0.2


def test_fea_end_to_end(tmp_path, sample_entry):
    if not SAMPLE_STL.exists():
        pytest.skip("corpus STL missing")
    if not Path("/home/best/.local/bin/ccx").exists():
        pytest.skip("CalculiX missing")
    fea_root = tmp_path / "fea"
    # Point geometry at real corpus via a shallow entry copy
    entry = dict(sample_entry)
    result = run_fea_for_entry(entry, CORPUS, fea_root, timeout=180, cl_max_mm=5.0)
    assert result.success, result.error
    case = fea_root / entry["part_id"]
    assert case_has_valid_frd(case, min_bytes=10_000)
    summary = parse_frd_summary(case / "case.frd", min_bytes=10_000)
    assert summary is not None
    assert summary.max_von_mises_mpa > 0
    assert summary.max_displacement_mm > 0


def test_cfd_domain_and_run(tmp_path, sample_entry):
    if not SAMPLE_STL.exists():
        pytest.skip("corpus STL missing")
    U, Lx, Ly, Lz = domain_from_stl_mm(SAMPLE_STL)
    assert U > 0 and Lx > 0 and Ly > 0 and Lz > 0
    # Domain should be meter-scale for mm STLs
    assert Lx <= 10.0
    assert Lx >= 0.05

    of_bin = Path.home() / ".local/cadflow-solvers/openfoam_1912.200626-2build3_amd64/usr/bin/simpleFoam"
    if not of_bin.exists():
        pytest.skip("OpenFOAM missing")

    cfd_root = tmp_path / "cfd"
    result = run_cfd_for_entry(sample_entry, CORPUS, cfd_root, timeout=180)
    assert result.success, result.error
    assert result.metrics is not None
    assert result.metrics["U_mag_max"] > 0.1


def test_graph_register_and_ingest(tmp_path, sample_entry):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "material:al-6061-t6",
                        "type": "Material",
                        "label": "Al 6061-T6",
                        "properties": {},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    stats = ensure_parts_in_graph(graph_path, [sample_entry])
    assert stats["parts_added"] == 1
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    part = next(n for n in graph["nodes"] if n["type"] == "Part")
    assert part["id"] == "part:rocket:nose_cone_00000"
    assert any(e["type"] == "MADE_OF" for e in graph["edges"])

    # Fake FEA/CFD ingest
    fea_root = tmp_path / "fea" / "nose_cone_00000"
    fea_root.mkdir(parents=True)
    # Minimal path: ingest via results list
    linked = ingest_fea_to_graph(
        graph_path,
        tmp_path / "fea",
        [
            {
                "part_id": "nose_cone_00000",
                "success": True,
                "metrics": {
                    "max_stress_mpa": 12.3,
                    "mean_stress_mpa": 4.0,
                    "max_displacement_mm": 0.01,
                    "frd_bytes": 100000,
                    "solver": "calculix",
                },
            }
        ],
    )
    assert linked == 1
    linked_cfd = ingest_cfd_to_graph(
        graph_path,
        [
            {
                "part_id": "nose_cone_00000",
                "success": True,
                "metrics": {"U_mag_max": 3.3, "p_mean": 0.01},
            }
        ],
    )
    assert linked_cfd == 1
    part = next(
        n
        for n in json.loads(graph_path.read_text(encoding="utf-8"))["nodes"]
        if n["type"] == "Part"
    )
    assert part["has_fea"] and part["has_cfd"]
    assert part["physics_verified"]
    assert part["simulation_results_fea"]["max_stress_mpa"] == 12.3


def test_select_entries_families():
    manifest = load_manifest(CORPUS)
    selected = select_entries(manifest, limit=10, families=["fin", "nozzle"])
    assert len(selected) == 10
    assert all(e["family"] in {"fin", "nozzle"} for e in selected)
