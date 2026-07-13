"""Smoke tests for the CAD/geometry toolchain."""

from __future__ import annotations

import cadquery as cq
import trimesh


def test_cadquery_box_volume():
    box = cq.Workplane("XY").box(10, 20, 30)
    assert box.val().Volume() == 6000.0  # type: ignore[attr-defined]


def test_trimesh_box_volume():
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    assert mesh.volume == 6.0
