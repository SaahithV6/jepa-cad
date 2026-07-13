from __future__ import annotations

import math
from pathlib import Path

import pytest

from cadflow.backends import CadQueryBackend, MockCadBackend, build_from_spec, get_backend


def test_cadquery_backend_builds_real_solid() -> None:
    backend = CadQueryBackend()
    solid = backend.box(1.0, 2.0, 3.0)

    assert backend.name == "cadquery"
    assert math.isclose(backend.volume(solid), 6.0, rel_tol=1e-6)
    assert backend.describe(solid)["type"] == "cadquery"
    assert backend.face_count(solid) >= 6
    assert backend.is_valid(solid)
    assert backend.is_watertight(solid)
    bbox = backend.bounding_box(solid)
    assert bbox[3] - bbox[0] == pytest.approx(1.0, rel=1e-6)


def test_mock_backend_is_available_as_fallback() -> None:
    backend = get_backend(prefer_real=False)
    solid = backend.box(1.0, 2.0, 3.0)

    assert isinstance(backend, MockCadBackend)
    assert backend.describe(solid)["type"] == "mock"
    assert math.isclose(backend.volume(solid), 6.0, rel_tol=1e-6)


def test_mock_primitives_cylinder_sphere_extrude_and_sculpt(tmp_path: Path) -> None:
    backend = MockCadBackend()
    cyl = backend.cylinder(1.0, 2.0)
    sph = backend.sphere(1.0)
    ext = backend.extrude_profile([(0, 0), (2, 0), (2, 1), (0, 1)], height=3.0)
    sculpted = backend.sculpt_offset(backend.box(1, 1, 1), 0.1)

    assert backend.volume(cyl) == pytest.approx(math.pi * 1.0**2 * 2.0, rel=1e-6)
    assert backend.volume(sph) == pytest.approx((4 / 3) * math.pi, rel=1e-6)
    assert backend.volume(ext) == pytest.approx(6.0, rel=1e-6)
    assert backend.volume(sculpted) > 1.0
    assert backend.is_watertight(cyl)

    step = backend.export_step(cyl, tmp_path / "cyl.step")
    stl = backend.export_stl(sph, tmp_path / "sph.stl")
    assert step.exists() and "ISO-10303-21" in step.read_text()
    assert stl.exists() and "solid" in stl.read_text()


def test_cadquery_primitives_and_export(tmp_path: Path) -> None:
    backend = CadQueryBackend()
    cyl = backend.cylinder(1.0, 2.0)
    sph = backend.sphere(1.0)
    ext = backend.extrude_profile([(0, 0), (2, 0), (2, 1), (0, 1)], height=3.0)

    assert backend.volume(cyl) > 0
    assert backend.volume(sph) > 0
    assert backend.volume(ext) == pytest.approx(6.0, rel=1e-5)
    assert backend.export_step(cyl, tmp_path / "c.step").exists()
    assert backend.export_stl(sph, tmp_path / "s.stl").exists()


def test_build_from_spec_rejects_llm_mesh_kind() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_from_spec({"kind": "llm_mesh", "vertices": []}, backend=MockCadBackend())


def test_build_from_spec_box() -> None:
    shape = build_from_spec(
        {"kind": "box", "width": 2, "height": 3, "depth": 4},
        backend=MockCadBackend(),
    )
    assert MockCadBackend().volume(shape) == pytest.approx(24.0)
