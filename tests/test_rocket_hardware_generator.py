"""Smoke tests for OpenRocket-style rocket hardware meshes."""
from __future__ import annotations

from cadflow.rocket_hardware_generator import (
    build_mesh,
    iter_part_specs,
    mesh_fin,
    mesh_nose_cone,
    mesh_nozzle,
    write_ork_rocket,
    PartSpec,
)


def test_nose_and_fin_have_faces() -> None:
    nose = mesh_nose_cone(diameter_mm=80, length_mm=160, shape="ogive")
    fin = mesh_fin(height_mm=90, root_chord_mm=110, tip_chord_mm=40, thickness_mm=3)
    assert len(nose.faces) > 100
    assert len(fin.faces) >= 8
    assert fin.is_watertight


def test_build_mesh_families() -> None:
    for family, params in [
        ("nozzle", {"throat_diameter_mm": 20, "expansion_ratio": 16, "length_mm": 100}),
        ("tank", {"diameter_mm": 200, "length_mm": 600, "shape": "cylinder"}),
        ("body_tube", {"diameter_mm": 100, "length_mm": 800, "wall_mm": 2}),
        ("transition", {"fore_diameter_mm": 80, "aft_diameter_mm": 120, "length_mm": 60}),
    ]:
        mesh = build_mesh(PartSpec("t", family, params))
        assert len(mesh.faces) > 0


def test_catalog_balanced_near_target() -> None:
    specs = iter_part_specs(8000)
    assert len(specs) == 8000
    families = {s.family for s in specs}
    assert {"nose_cone", "fin", "tank", "nozzle", "fairing"} <= families


def test_ork_zip_writable(tmp_path) -> None:
    path = tmp_path / "demo.ork"
    write_ork_rocket(
        path,
        name="demo",
        diameter_m=0.1,
        nose_length_m=0.15,
        body_length_m=0.8,
        fin_height_m=0.08,
        fin_root_m=0.1,
        fin_tip_m=0.04,
    )
    assert path.exists() and path.stat().st_size > 100
