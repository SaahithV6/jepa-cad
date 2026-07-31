"""Parametric OpenRocket-style rocket hardware mesh generator.

Produces watertight STL parts (nose cones, fins, tubes, tanks, nozzles,
transitions, mounts, fairings) plus companion OpenRocket .ork design files
for full vehicle profiles. Intended for LatticeZero / JEPA spaceflight
training corpora — not for replacing CalculiX FEA ownership.
"""
from __future__ import annotations

import json
import math
import uuid
import zipfile
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

import numpy as np
import trimesh
from trimesh.creation import cylinder


MM = 1.0  # all coordinates in millimeters


def _densify_polygon_2d(
    points_xy: list[tuple[float, float]],
    *,
    max_seg_mm: float = 3.0,
) -> list[tuple[float, float]]:
    """Insert edge midpoints so extruded fins have enough facets for volume meshing."""
    if len(points_xy) < 3:
        return list(points_xy)
    out: list[tuple[float, float]] = []
    n = len(points_xy)
    max_seg = max(float(max_seg_mm), 0.5)
    for i in range(n):
        x0, y0 = points_xy[i]
        x1, y1 = points_xy[(i + 1) % n]
        out.append((float(x0), float(y0)))
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        nseg = max(1, int(math.ceil(dist / max_seg)))
        for k in range(1, nseg):
            t = k / nseg
            out.append((float(x0 + t * dx), float(y0 + t * dy)))
    return out


def _extrude_convex_polygon_2d(
    points_xy: list[tuple[float, float]],
    height: float,
) -> trimesh.Trimesh:
    """Extrude an XY planform polygon along +Z (thickness).

    Prefers shapely triangulation (avoids zero-area fan tris on densified
    elliptical outlines that make Gmsh throw Singular matrix / overlapping facets).
    """
    h = float(height)
    if h <= 0:
        raise ValueError("extrude height must be > 0")
    pts = [(float(x), float(y)) for x, y in points_xy]
    if len(pts) < 3:
        raise ValueError("need >=3 polygon points")
    try:
        from shapely.geometry import Polygon

        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            raise ValueError("empty planform")
        mesh = trimesh.creation.extrude_polygon(poly, height=h)
    except Exception:
        # Fallback: fan caps (OK for true trapezoid corners; weak on dense curves).
        arr = np.asarray(pts, dtype=np.float64)
        n = len(arr)
        bottom = np.column_stack([arr, np.zeros(n)])
        top = np.column_stack([arr, np.full(n, h)])
        verts = np.vstack([bottom, top])
        faces: list[list[int]] = []
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])
            faces.append([n, n + i, n + i + 1])
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices()
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


@dataclass(frozen=True)
class PartSpec:
    part_id: str
    family: str
    params: dict[str, Any]
    tags: tuple[str, ...] = ()


def _revolve_profile(zs: np.ndarray, rs: np.ndarray, sections: int = 48) -> trimesh.Trimesh:
    """Revolve a (z, r) polyline into a solid of revolution about +Z."""
    zs = np.asarray(zs, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    assert len(zs) == len(rs) and len(zs) >= 2
    theta = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    verts: list[list[float]] = []
    for z, r in zip(zs, rs):
        for t in theta:
            verts.append([r * math.cos(t), r * math.sin(t), float(z)])
    verts_arr = np.asarray(verts, dtype=np.float64)
    faces: list[list[int]] = []
    rings = len(zs)
    for i in range(rings - 1):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            c = (i + 1) * sections + (j + 1) % sections
            d = (i + 1) * sections + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    # Cap bottom if r[0] > 0
    if rs[0] > 1e-9:
        center = len(verts_arr)
        verts_arr = np.vstack([verts_arr, [[0.0, 0.0, float(zs[0])]]])
        for j in range(sections):
            # outward normal -Z
            faces.append([center, (j + 1) % sections, j])
    # Cap tip
    if rs[-1] <= 1e-9:
        tip = len(verts_arr)
        verts_arr = np.vstack([verts_arr, [[0.0, 0.0, float(zs[-1])]]])
        base = (rings - 1) * sections
        for j in range(sections):
            faces.append([tip, base + j, base + (j + 1) % sections])
    else:
        center = len(verts_arr)
        verts_arr = np.vstack([verts_arr, [[0.0, 0.0, float(zs[-1])]]])
        base = (rings - 1) * sections
        for j in range(sections):
            faces.append([center, base + j, base + (j + 1) % sections])
    mesh = trimesh.Trimesh(
        vertices=verts_arr,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def mesh_nose_cone(
    *,
    diameter_mm: float,
    length_mm: float,
    shape: str = "ogive",
    power: float = 0.5,
    sections: int = 48,
    profile_pts: int = 40,
) -> trimesh.Trimesh:
    r0 = diameter_mm / 2.0
    zs = np.linspace(0.0, length_mm, profile_pts)
    x = zs / max(length_mm, 1e-9)
    if shape == "conical":
        rs = r0 * (1.0 - x)
    elif shape == "parabolic":
        rs = r0 * (1.0 - x * x)
    elif shape == "elliptical":
        rs = r0 * np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))
    elif shape == "power":
        rs = r0 * np.power(np.clip(1.0 - x, 0.0, 1.0), power)
    else:  # tangent ogive approximation
        rho = (r0 * r0 + length_mm * length_mm) / (2.0 * r0)
        rs = np.sqrt(np.clip(rho * rho - (length_mm - zs) ** 2, 0.0, None)) - (rho - r0)
        rs = np.clip(rs, 0.0, None)
    rs[-1] = 0.0
    return _revolve_profile(zs, rs, sections=sections)


def mesh_fin(
    *,
    height_mm: float,
    root_chord_mm: float,
    tip_chord_mm: float,
    thickness_mm: float,
    sweep_mm: float = 0.0,
    shape: str = "trapezoidal",
    span_stations: int = 16,
    section_pts: int = 20,
) -> trimesh.Trimesh:
    """Model a real sounding/HPR-style fin: planform + tapered plate airfoil.

    Not a flat 8-triangle wafer. Uses:
    - trapezoidal / clipped-delta / elliptical planforms (OpenRocket families)
    - leading-edge sweep from ``sweep_mm``
    - thickness taper root→tip (~60% at tip, like plywood/G10 fins)
    - symmetric beveled LE/TE cross-section (flat-plate airfoil with chamfers)
    """
    height_mm = max(float(height_mm), 5.0)
    root_chord_mm = max(float(root_chord_mm), 5.0)
    tip_chord_mm = max(float(tip_chord_mm), 1.0)
    # Real fins are thin, but keep a solver-safe floor so Gmsh/OCC can tet-mesh.
    t_root = max(float(thickness_mm), 1.5)
    t_tip = max(t_root * 0.55, 1.0)
    # Tip chord must stay well above thickness or lofted bevels self-intersect.
    tip_floor = max(0.18 * root_chord_mm, 4.0 * t_tip, 4.0)
    tip_chord_mm = max(tip_chord_mm, tip_floor * 0.35)
    sweep_mm = float(sweep_mm)
    # Fixed section topology (no chord-dependent densify) → equal rings, clean loft.
    n_stations = max(int(span_stations), 8)
    _ = section_pts  # kept for API compat; densify disabled for meshability

    def planform_x(y: float) -> tuple[float, float]:
        """Return (x_le, x_te) at span station y ∈ [0, height]."""
        s = float(np.clip(y / height_mm, 0.0, 1.0))
        if shape == "elliptical":
            # Semi-ellipse with clipped tip (no zero-area spike → Gmsh overlap).
            chord = root_chord_mm * math.sqrt(max(1.0 - s * s, 0.0))
            chord = max(chord, tip_floor)
            x_mid = root_chord_mm * 0.5
            return x_mid - 0.5 * chord, x_mid + 0.5 * chord
        if shape == "delta":
            # Clipped delta: tip chord small, LE sweeps back.
            x_le = sweep_mm * s
            chord = root_chord_mm * (1.0 - s) + tip_chord_mm * s
            return x_le, x_le + max(chord, tip_floor)
        # Trapezoidal / clipped (most common model & HPR fin)
        x_le = sweep_mm * s
        chord = root_chord_mm * (1.0 - s) + tip_chord_mm * s
        return x_le, x_le + max(chord, tip_floor * 0.5)

    def airfoil_xz(chord: float, thickness: float) -> list[tuple[float, float]]:
        """Symmetric flat-plate section with LE/TE bevels (fixed 6 pts)."""
        c = max(chord, tip_floor * 0.5)
        t = max(thickness, 0.8)
        # Keep bevel small enough that LE/TE apexes cannot fold through the plate.
        bevel = min(0.12 * c, 0.35 * c, max(0.25 * c, 1.5 * t))
        bevel = min(bevel, 0.4 * c)
        return [
            (0.0, 0.0),  # LE apex
            (bevel, 0.5 * t),
            (c - bevel, 0.5 * t),
            (c, 0.0),  # TE apex
            (c - bevel, -0.5 * t),
            (bevel, -0.5 * t),
        ]

    def _extrude_planform() -> trimesh.Trimesh:
        # Robust plate: planform polygon extruded in thickness.
        # Keep point count modest so fan triangulation stays non-degenerate
        # (mapbox-earcut may be unavailable in this env).
        if shape == "elliptical":
            n = max(n_stations, 16)
            outline: list[tuple[float, float]] = []
            for y in np.linspace(0.0, height_mm, n):
                x_le, _x_te = planform_x(float(y))
                outline.append((x_le, float(y)))
            for y in np.linspace(height_mm, 0.0, n)[1:]:
                _x_le, x_te = planform_x(float(y))
                outline.append((x_te, float(y)))
            # No densify — dense outlines create zero-area fan tris / PLC errors.
            mesh = _extrude_convex_polygon_2d(outline, t_root)
        else:
            x_le0, x_te0 = planform_x(0.0)
            x_le1, x_te1 = planform_x(height_mm)
            pts = [(x_le0, 0.0), (x_te0, 0.0), (x_te1, height_mm), (x_le1, height_mm)]
            pts = _densify_polygon_2d(pts, max_seg_mm=max(3.0, height_mm / 10.0))
            mesh = _extrude_convex_polygon_2d(pts, t_root)
        try:
            if float(mesh.volume) < 0:
                mesh.invert()
            mesh.fix_normals()
        except Exception:
            pass
        return mesh

    # Elliptical loft tips are the usual Gmsh overlap offenders — prefer extrusion.
    if shape == "elliptical":
        return _extrude_planform()

    ys = np.linspace(0.0, height_mm, n_stations)
    rings: list[np.ndarray] = []
    for y in ys:
        x_le, x_te = planform_x(float(y))
        chord = max(x_te - x_le, tip_floor * 0.5)
        s = float(y / height_mm)
        thick = t_root * (1.0 - s) + t_tip * s
        sec = airfoil_xz(chord, thick)
        ring = np.array([[x_le + x, float(y), z] for x, z in sec], dtype=np.float64)
        rings.append(ring)

    n_sec = len(rings[0])
    verts = np.vstack(rings)
    faces: list[list[int]] = []
    for i in range(len(rings) - 1):
        for j in range(n_sec):
            a = i * n_sec + j
            b = i * n_sec + (j + 1) % n_sec
            c = (i + 1) * n_sec + (j + 1) % n_sec
            d = (i + 1) * n_sec + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    for ring_i, reverse in ((0, True), (len(rings) - 1, False)):
        base = ring_i * n_sec
        center = verts[base : base + n_sec].mean(axis=0)
        cid = len(verts)
        verts = np.vstack([verts, center])
        for j in range(n_sec):
            a = base + j
            b = base + (j + 1) % n_sec
            faces.append([cid, b, a] if reverse else [cid, a, b])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    except Exception:
        pass
    if (not mesh.is_watertight) or float(getattr(mesh, "volume", 0.0) or 0.0) <= 0.0:
        mesh = _extrude_planform()
    return mesh



def mesh_body_tube(*, diameter_mm: float, length_mm: float, wall_mm: float = 2.0) -> trimesh.Trimesh:
    outer = cylinder(radius=diameter_mm / 2.0, height=length_mm, sections=48)
    # Solid tube proxy (hollow subtract is optional / slow); keep solid for JEPA shape prior
    _ = wall_mm
    return outer


def mesh_tank(
    *,
    diameter_mm: float,
    length_mm: float,
    shape: str = "cylinder",
) -> trimesh.Trimesh:
    r = diameter_mm / 2.0
    if shape == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    if shape == "capsule":
        body = cylinder(radius=r, height=max(length_mm - 2.0 * r, r * 0.5), sections=48)
        # Approximate with cylinder + two sphere caps via concat (not boolean)
        s1 = trimesh.creation.icosphere(subdivisions=2, radius=r)
        s2 = s1.copy()
        s1.apply_translation([0, 0, body.bounds[1, 2]])
        s2.apply_translation([0, 0, body.bounds[0, 2]])
        return trimesh.util.concatenate([body, s1, s2])
    return cylinder(radius=r, height=length_mm, sections=48)


def mesh_nozzle(
    *,
    throat_diameter_mm: float,
    expansion_ratio: float,
    length_mm: float,
    sections: int = 48,
) -> trimesh.Trimesh:
    throat_r = throat_diameter_mm / 2.0
    exit_r = throat_r * math.sqrt(max(expansion_ratio, 1.0))
    # Bell-ish: quadratic radius growth
    zs = np.linspace(0.0, length_mm, 36)
    t = zs / max(length_mm, 1e-9)
    rs = throat_r + (exit_r - throat_r) * (t * t)
    # Outer wall ~ +thickness
    thickness = max(1.5, throat_r * 0.15)
    outer = _revolve_profile(zs, rs + thickness, sections=sections)
    return outer


def mesh_transition(
    *,
    fore_diameter_mm: float,
    aft_diameter_mm: float,
    length_mm: float,
) -> trimesh.Trimesh:
    zs = np.linspace(0.0, length_mm, 24)
    rs = np.linspace(fore_diameter_mm / 2.0, aft_diameter_mm / 2.0, 24)
    return _revolve_profile(zs, rs, sections=48)


def mesh_engine_mount(*, motor_diameter_mm: float, length_mm: float) -> trimesh.Trimesh:
    return cylinder(radius=motor_diameter_mm / 2.0 + 2.0, height=length_mm, sections=32)


def mesh_fairing(*, diameter_mm: float, length_mm: float, nose_shape: str = "ogive") -> trimesh.Trimesh:
    tube_len = length_mm * 0.65
    nose_len = length_mm - tube_len
    tube = cylinder(radius=diameter_mm / 2.0, height=tube_len, sections=48)
    tube.apply_translation([0, 0, tube_len / 2.0])
    nose = mesh_nose_cone(diameter_mm=diameter_mm, length_mm=nose_len, shape=nose_shape)
    nose.apply_translation([0, 0, tube_len])
    return trimesh.util.concatenate([tube, nose])


def mesh_tps_tile(
    *,
    side_mm: float,
    thickness_mm: float,
    shape: str = "hex",
    densified_cap_mm: float = 0.0,
) -> trimesh.Trimesh:
    """Shuttle-class TPS tile proxy: hex/square/rect silica block (+ optional densified face)."""
    if shape == "hex":
        r = side_mm
        pts = [
            (r * math.cos(math.pi / 3 * i + math.pi / 6), r * math.sin(math.pi / 3 * i + math.pi / 6))
            for i in range(6)
        ]
    elif shape == "rect":
        w, h = side_mm, side_mm * 0.7
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    else:  # square
        s = side_mm / 2.0
        pts = [(-s, -s), (s, -s), (s, s), (-s, s)]
    body = _extrude_convex_polygon_2d(pts, thickness_mm)
    if densified_cap_mm > 0:
        cap = _extrude_convex_polygon_2d(
            [(x * 0.92, y * 0.92) for x, y in pts],
            densified_cap_mm,
        )
        cap.apply_translation([0, 0, thickness_mm])
        return trimesh.util.concatenate([body, cap])
    return body


def mesh_blanket(*, width_mm: float, length_mm: float, thickness_mm: float = 8.0) -> trimesh.Trimesh:
    return _extrude_convex_polygon_2d(
        [
            (-width_mm / 2, -length_mm / 2),
            (width_mm / 2, -length_mm / 2),
            (width_mm / 2, length_mm / 2),
            (-width_mm / 2, length_mm / 2),
        ],
        thickness_mm,
    )


def mesh_solar_panel(*, width_mm: float, length_mm: float, thickness_mm: float = 4.0) -> trimesh.Trimesh:
    return mesh_blanket(width_mm=width_mm, length_mm=length_mm, thickness_mm=thickness_mm)


def mesh_antenna_dish(*, diameter_mm: float, depth_mm: float) -> trimesh.Trimesh:
    """Shallow parabolic dish approximation via solid of revolution."""
    r0 = diameter_mm / 2.0
    zs = np.linspace(0.0, depth_mm, 28)
    # z = a r^2  => r = sqrt(z/a); set a so r(depth)=r0
    a = depth_mm / max(r0 * r0, 1e-9)
    rs = np.sqrt(np.clip(zs / max(a, 1e-12), 0.0, None))
    rs[0] = 0.0
    return _revolve_profile(zs, rs, sections=48)


def mesh_ring_frame(*, diameter_mm: float, height_mm: float, radial_width_mm: float = 20.0) -> trimesh.Trimesh:
    outer = cylinder(radius=diameter_mm / 2.0, height=height_mm, sections=64)
    # Solid ring proxy (true hollow needs boolean); JEPA cares about extents/topology prior
    _ = radial_width_mm
    return outer


def mesh_bulkhead(*, diameter_mm: float, thickness_mm: float = 8.0) -> trimesh.Trimesh:
    return cylinder(radius=diameter_mm / 2.0, height=thickness_mm, sections=64)


def mesh_strut(*, length_mm: float, diameter_mm: float) -> trimesh.Trimesh:
    return cylinder(radius=diameter_mm / 2.0, height=length_mm, sections=24)


def build_mesh(spec: PartSpec) -> trimesh.Trimesh:
    p = spec.params
    fam = spec.family
    if fam == "nose_cone":
        return mesh_nose_cone(
            diameter_mm=p["diameter_mm"],
            length_mm=p["length_mm"],
            shape=p.get("shape", "ogive"),
            power=float(p.get("power", 0.5)),
        )
    if fam == "fin":
        return mesh_fin(
            height_mm=p["height_mm"],
            root_chord_mm=p["root_chord_mm"],
            tip_chord_mm=p.get("tip_chord_mm", p["root_chord_mm"] * 0.4),
            thickness_mm=p["thickness_mm"],
            sweep_mm=p.get("sweep_mm", 0.0),
            shape=p.get("shape", "trapezoidal"),
        )
    if fam == "body_tube":
        return mesh_body_tube(
            diameter_mm=p["diameter_mm"],
            length_mm=p["length_mm"],
            wall_mm=p.get("wall_mm", 2.0),
        )
    if fam == "tank":
        return mesh_tank(
            diameter_mm=p["diameter_mm"],
            length_mm=p["length_mm"],
            shape=p.get("shape", "cylinder"),
        )
    if fam == "nozzle":
        return mesh_nozzle(
            throat_diameter_mm=p["throat_diameter_mm"],
            expansion_ratio=p["expansion_ratio"],
            length_mm=p["length_mm"],
        )
    if fam == "transition":
        return mesh_transition(
            fore_diameter_mm=p["fore_diameter_mm"],
            aft_diameter_mm=p["aft_diameter_mm"],
            length_mm=p["length_mm"],
        )
    if fam == "engine_mount":
        return mesh_engine_mount(
            motor_diameter_mm=p["motor_diameter_mm"],
            length_mm=p["length_mm"],
        )
    if fam == "fairing":
        return mesh_fairing(
            diameter_mm=p["diameter_mm"],
            length_mm=p["length_mm"],
            nose_shape=p.get("nose_shape", "ogive"),
        )
    if fam == "tps_tile":
        return mesh_tps_tile(
            side_mm=p["side_mm"],
            thickness_mm=p["thickness_mm"],
            shape=p.get("shape", "hex"),
            densified_cap_mm=float(p.get("densified_cap_mm", 0.0)),
        )
    if fam == "blanket":
        return mesh_blanket(
            width_mm=p["width_mm"],
            length_mm=p["length_mm"],
            thickness_mm=p.get("thickness_mm", 8.0),
        )
    if fam == "solar_panel":
        return mesh_solar_panel(
            width_mm=p["width_mm"],
            length_mm=p["length_mm"],
            thickness_mm=p.get("thickness_mm", 4.0),
        )
    if fam == "antenna":
        return mesh_antenna_dish(diameter_mm=p["diameter_mm"], depth_mm=p["depth_mm"])
    if fam == "ring_frame":
        return mesh_ring_frame(
            diameter_mm=p["diameter_mm"],
            height_mm=p["height_mm"],
            radial_width_mm=p.get("radial_width_mm", 20.0),
        )
    if fam == "bulkhead":
        return mesh_bulkhead(diameter_mm=p["diameter_mm"], thickness_mm=p.get("thickness_mm", 8.0))
    if fam == "strut":
        return mesh_strut(length_mm=p["length_mm"], diameter_mm=p["diameter_mm"])
    raise ValueError(f"unknown family {fam}")


def iter_part_specs(target: int = 8000) -> list[PartSpec]:
    """Enumerate a diverse OpenRocket-style hardware catalog (~target parts)."""
    buckets: dict[str, list[PartSpec]] = {
        "nose_cone": [],
        "fin": [],
        "body_tube": [],
        "tank": [],
        "nozzle": [],
        "transition": [],
        "engine_mount": [],
        "fairing": [],
    }

    def add(family: str, params: dict[str, Any], *tags: str) -> None:
        buckets[family].append(
            PartSpec(part_id="pending", family=family, params=params, tags=tags)
        )

    for d, L, shape, power in product(
        range(35, 250, 15),
        range(60, 560, 25),
        ("ogive", "conical", "parabolic", "elliptical", "power"),
        (0.35, 0.5, 0.75),
    ):
        if shape != "power" and power != 0.5:
            continue
        add("nose_cone", {"diameter_mm": d, "length_mm": L, "shape": shape, "power": power}, "openrocket", "aero")

    for h, root, tip, thk, shape, sweep_frac in product(
        range(30, 300, 30),
        range(50, 400, 35),
        (15, 30, 50, 80, 120, 160),
        (1.0, 2.0, 3.0, 5.0),
        ("trapezoidal", "delta", "elliptical"),
        (0.0, 0.2, 0.4),
    ):
        add(
            "fin",
            {
                "height_mm": h,
                "root_chord_mm": root,
                "tip_chord_mm": tip,
                "thickness_mm": thk,
                "sweep_mm": root * sweep_frac,
                "shape": shape,
            },
            "openrocket",
            "aero",
        )

    for d, L, wall in product(range(35, 250, 15), range(150, 3000, 150), (1.0, 1.5, 2.0, 3.0, 4.0)):
        add("body_tube", {"diameter_mm": d, "length_mm": L, "wall_mm": wall}, "openrocket", "airframe")

    for d, L, shape in product(range(60, 700, 30), range(150, 3600, 180), ("cylinder", "capsule", "sphere")):
        add("tank", {"diameter_mm": d, "length_mm": L, "shape": shape}, "propellant", "pressure")

    for throat, er, L in product(range(5, 70, 5), range(4, 90, 4), range(30, 280, 25)):
        add(
            "nozzle",
            {"throat_diameter_mm": throat, "expansion_ratio": er, "length_mm": L},
            "engine",
            "propulsion",
        )

    for fore, aft, L in product(range(35, 220, 25), range(50, 280, 25), range(30, 240, 30)):
        if abs(fore - aft) < 10:
            continue
        add(
            "transition",
            {"fore_diameter_mm": fore, "aft_diameter_mm": aft, "length_mm": L},
            "openrocket",
            "airframe",
        )

    for md, L in product((13, 18, 24, 29, 38, 54, 75, 98, 120), range(60, 560, 30)):
        add("engine_mount", {"motor_diameter_mm": md, "length_mm": L}, "openrocket", "propulsion")

    for d, L, ns in product(
        range(150, 1400, 80),
        range(300, 2800, 150),
        ("ogive", "conical", "parabolic", "elliptical"),
    ):
        add("fairing", {"diameter_mm": d, "length_mm": L, "nose_shape": ns}, "payload", "aero")

    # Balanced quotas for LatticeZero diversity (sums to target)
    quotas = {
        "nose_cone": 1200,
        "fin": 1400,
        "body_tube": 1000,
        "tank": 1100,
        "nozzle": 1100,
        "transition": 700,
        "engine_mount": 500,
        "fairing": 1000,
    }
    # normalize if target != 8000
    scale = target / sum(quotas.values())
    quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}

    specs: list[PartSpec] = []
    for family, quota in quotas.items():
        pool = buckets[family]
        if not pool:
            continue
        if len(pool) <= quota:
            chosen = pool
        else:
            step = len(pool) / quota
            chosen = [pool[int(i * step)] for i in range(quota)]
        for s in chosen:
            specs.append(s)

    # top up if rounding left us short
    if len(specs) < target:
        extras = [s for fam in buckets.values() for s in fam]
        i = 0
        while len(specs) < target and extras:
            specs.append(extras[i % len(extras)])
            i += 1

    specs = specs[:target]
    return [
        PartSpec(part_id=f"{s.family}_{i:05d}", family=s.family, params=s.params, tags=s.tags)
        for i, s in enumerate(specs)
    ]


def iter_tps_spacecraft_specs(target: int = 2500) -> list[PartSpec]:
    """TPS tiles, blankets, solar panels, antennas, frames — LatticeZero spacecraft diversity."""
    buckets: dict[str, list[PartSpec]] = {
        "tps_tile": [],
        "blanket": [],
        "solar_panel": [],
        "antenna": [],
        "ring_frame": [],
        "bulkhead": [],
        "strut": [],
    }

    def add(family: str, params: dict[str, Any], *tags: str) -> None:
        buckets[family].append(PartSpec(part_id="pending", family=family, params=params, tags=tags))

    for side, thk, shape, cap in product(
        range(80, 320, 20),
        (20, 30, 40, 50, 65, 80, 100),
        ("hex", "square", "rect"),
        (0.0, 2.0, 4.0),
    ):
        add(
            "tps_tile",
            {"side_mm": side, "thickness_mm": thk, "shape": shape, "densified_cap_mm": cap},
            "tps",
            "thermal",
            "reentry",
        )

    for w, L, thk in product(range(200, 1600, 100), range(200, 2000, 150), (4, 8, 12, 20)):
        add("blanket", {"width_mm": w, "length_mm": L, "thickness_mm": thk}, "tps", "mlti", "thermal")

    for w, L, thk in product(range(300, 2000, 150), range(600, 4000, 250), (2, 4, 6)):
        add("solar_panel", {"width_mm": w, "length_mm": L, "thickness_mm": thk}, "power", "deployable")

    for d, depth in product(range(200, 2400, 100), range(40, 400, 40)):
        add("antenna", {"diameter_mm": d, "depth_mm": depth}, "rf", "comms")

    for d, h, rw in product(range(300, 3000, 150), range(20, 120, 20), (10, 20, 35)):
        add("ring_frame", {"diameter_mm": d, "height_mm": h, "radial_width_mm": rw}, "structure")

    for d, thk in product(range(200, 2500, 100), (4, 8, 12, 20, 30)):
        add("bulkhead", {"diameter_mm": d, "thickness_mm": thk}, "structure", "pressure")

    for L, d in product(range(200, 3000, 150), range(10, 80, 8)):
        add("strut", {"length_mm": L, "diameter_mm": d}, "structure")

    quotas = {
        "tps_tile": 900,
        "blanket": 350,
        "solar_panel": 350,
        "antenna": 250,
        "ring_frame": 250,
        "bulkhead": 200,
        "strut": 200,
    }
    scale = target / sum(quotas.values())
    quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}

    specs: list[PartSpec] = []
    for family, quota in quotas.items():
        pool = buckets[family]
        if len(pool) <= quota:
            chosen = pool
        else:
            step = len(pool) / quota
            chosen = [pool[int(i * step)] for i in range(quota)]
        specs.extend(chosen)

    while len(specs) < target:
        specs.append(buckets["tps_tile"][len(specs) % len(buckets["tps_tile"])])

    specs = specs[:target]
    # Offset ids so they don't collide with existing openrocket_* ids when merged
    return [
        PartSpec(part_id=f"{s.family}_{i:05d}", family=s.family, params=s.params, tags=s.tags)
        for i, s in enumerate(specs)
    ]


def write_stl(mesh: trimesh.Trimesh, path: Path) -> None:
    """Write an STL Gmsh will actually load.

    Trimesh's default binary export uses an all-zero 80-byte header; Gmsh rejects
    those with ``Error loading``. ASCII avoids the issue and stays tiny for our
    parametric parts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="stl_ascii")


def write_ork_rocket(
    path: Path,
    *,
    name: str,
    diameter_m: float,
    nose_length_m: float,
    body_length_m: float,
    fin_height_m: float,
    fin_root_m: float,
    fin_tip_m: float,
    fin_count: int = 4,
) -> None:
    """Write a minimal OpenRocket 1.8 .ork (zipped XML) vehicle profile."""
    cfg = str(uuid.uuid4())
    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<openrocket version="1.8" creator="jepa-cad-rocket-hardware-generator">
  <rocket>
    <name>{escape(name)}</name>
    <referencetype>maximum</referencetype>
    <motorconfiguration configid="{cfg}" default="true">
      <stage number="0" active="true"/>
    </motorconfiguration>
    <subcomponents>
      <stage>
        <name>Sustainer</name>
        <subcomponents>
          <nosecone>
            <name>Nose cone</name>
            <shape>ogive</shape>
            <length>{nose_length_m:.4f}</length>
            <aftradius>{diameter_m/2:.4f}</aftradius>
            <thickness>0.002</thickness>
          </nosecone>
          <bodytube>
            <name>Body tube</name>
            <length>{body_length_m:.4f}</length>
            <radius>{diameter_m/2:.4f}</radius>
            <thickness>0.002</thickness>
            <subcomponents>
              <trapezoidfinset>
                <name>Fins</name>
                <fincount>{fin_count}</fincount>
                <rootchord>{fin_root_m:.4f}</rootchord>
                <tipchord>{fin_tip_m:.4f}</tipchord>
                <height>{fin_height_m:.4f}</height>
                <sweeplength>{fin_root_m*0.25:.4f}</sweeplength>
                <thickness>0.003</thickness>
              </trapezoidfinset>
              <innertube>
                <name>Motor mount</name>
                <length>{min(body_length_m*0.4, 0.3):.4f}</length>
                <outerradius>{max(diameter_m/2*0.4, 0.012):.4f}</outerradius>
                <thickness>0.002</thickness>
              </innertube>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("rocket.ork", xml)


def iter_ork_profiles(count: int = 500) -> Iterable[dict[str, Any]]:
    i = 0
    for d_mm, nose_mm, body_mm, fin_h, fin_root, fins in product(
        range(40, 160, 20),
        range(80, 320, 40),
        range(400, 2000, 200),
        range(40, 200, 40),
        range(60, 240, 40),
        (3, 4),
    ):
        yield {
            "id": f"ork_rocket_{i:05d}",
            "name": f"JEPA-OR-{i:05d}",
            "diameter_m": d_mm / 1000.0,
            "nose_length_m": nose_mm / 1000.0,
            "body_length_m": body_mm / 1000.0,
            "fin_height_m": fin_h / 1000.0,
            "fin_root_m": fin_root / 1000.0,
            "fin_tip_m": fin_root * 0.4 / 1000.0,
            "fin_count": fins,
        }
        i += 1
        if i >= count:
            break


def generate_corpus(
    out_dir: Path,
    *,
    target_parts: int = 8000,
    ork_profiles: int = 500,
    workers: int = 1,
) -> dict[str, Any]:
    """Generate STL hardware corpus + OpenRocket profiles. Returns summary stats."""
    del workers  # reserved; generation is CPU-light enough serially / batched
    parts_dir = out_dir / "parts"
    ork_dir = out_dir / "openrocket_designs"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ork_dir.mkdir(parents=True, exist_ok=True)

    specs = iter_part_specs(target=target_parts)
    manifest: list[dict[str, Any]] = []
    ok = 0
    failed = 0
    family_counts: dict[str, int] = {}

    for i, spec in enumerate(specs):
        try:
            mesh = build_mesh(spec)
            if mesh.is_empty or len(mesh.faces) == 0:
                raise ValueError("empty mesh")
            stl_path = parts_dir / f"{spec.part_id}.stl"
            write_stl(mesh, stl_path)
            rec = {
                **asdict(spec),
                "stl": str(stl_path.relative_to(out_dir)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "extents_mm": mesh.extents.tolist(),
            }
            manifest.append(rec)
            ok += 1
            family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
        except Exception as exc:  # noqa: BLE001 — keep batch running
            failed += 1
            manifest.append({"part_id": spec.part_id, "family": spec.family, "error": str(exc)})
        if (i + 1) % 500 == 0:
            print(f"  parts [{i+1}/{len(specs)}] ok={ok} failed={failed}", flush=True)

    ork_manifest: list[dict[str, Any]] = []
    for prof in iter_ork_profiles(ork_profiles):
        path = ork_dir / f"{prof['id']}.ork"
        write_ork_rocket(
            path,
            name=prof["name"],
            diameter_m=prof["diameter_m"],
            nose_length_m=prof["nose_length_m"],
            body_length_m=prof["body_length_m"],
            fin_height_m=prof["fin_height_m"],
            fin_root_m=prof["fin_root_m"],
            fin_tip_m=prof["fin_tip_m"],
            fin_count=prof["fin_count"],
        )
        ork_manifest.append({**prof, "ork": str(path.relative_to(out_dir))})

    summary = {
        "parts_ok": ok,
        "parts_failed": failed,
        "parts_target": target_parts,
        "family_counts": family_counts,
        "ork_profiles": len(ork_manifest),
        "output_dir": str(out_dir),
        "note": "OpenRocket-style hardware for LatticeZero; FEA owned by other agent",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "ork_manifest.json").write_text(json.dumps(ork_manifest, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
