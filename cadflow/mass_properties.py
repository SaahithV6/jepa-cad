"""Solid mass properties from closed STL + material density.

Uses trimesh for polyhedral volume / COM / inertia of the watertight solid.
Inertia is about the center of mass, SI units (kg, m, kg·m²).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MassProperties:
    mass_kg: float
    volume_m3: float
    density_kg_m3: float
    center_of_mass_m: tuple[float, float, float]
    center_of_mass_mm: tuple[float, float, float]
    inertia_kg_m2: tuple[float, float, float, float, float, float]
    """Ixx, Iyy, Izz, Ixy, Ixz, Iyz about COM (kg·m²)."""
    principal_inertia_kg_m2: tuple[float, float, float]
    watertight: bool
    method: str = "trimesh_stl_solid"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center_of_mass_m"] = list(self.center_of_mass_m)
        d["center_of_mass_mm"] = list(self.center_of_mass_mm)
        d["inertia_kg_m2"] = {
            "Ixx": self.inertia_kg_m2[0],
            "Iyy": self.inertia_kg_m2[1],
            "Izz": self.inertia_kg_m2[2],
            "Ixy": self.inertia_kg_m2[3],
            "Ixz": self.inertia_kg_m2[4],
            "Iyz": self.inertia_kg_m2[5],
        }
        d["principal_inertia_kg_m2"] = list(self.principal_inertia_kg_m2)
        return d


def mass_properties_from_stl(
    stl_path: Path | str,
    density_kg_m3: float,
    *,
    extents_mm: list[float] | tuple[float, ...] | None = None,
) -> MassProperties | None:
    """Return mass / COM / inertia for a solid STL at the given density."""
    import numpy as np
    import trimesh

    path = Path(stl_path)
    if not path.is_file():
        return None
    density = float(density_kg_m3)
    if density <= 0:
        return None

    try:
        mesh = trimesh.load(str(path), force="mesh", process=True)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
        return None

    # STL is authored in millimeters in this corpus.
    mesh.apply_scale(1e-3)
    watertight = bool(mesh.is_watertight)
    if not watertight:
        try:
            mesh.fill_holes()
            watertight = bool(mesh.is_watertight)
        except Exception:  # noqa: BLE001
            pass

    mesh.density = density
    try:
        vol = float(mesh.volume)
    except Exception:  # noqa: BLE001
        vol = 0.0

    # Fallback: bbox fill for open/degenerate shells (still better than nothing).
    method = "trimesh_stl_solid"
    if (not np.isfinite(vol)) or abs(vol) < 1e-12:
        ext = list(extents_mm or [])
        if len(ext) < 3:
            # derive extents from mesh bounds (already in meters after scale)
            bounds = np.asarray(mesh.bounds, dtype=float)
            ext_m = bounds[1] - bounds[0]
            ext = [float(x) * 1e3 for x in ext_m]
        if len(ext) >= 3:
            # assume ~15% solid fraction for thin-wallish / open shells
            dx, dy, dz = (max(float(ext[i]), 0.5) * 1e-3 for i in range(3))
            vol = dx * dy * dz * 0.15
            method = "bbox_fill_0.15"
            com = np.array([dx, dy, dz], dtype=float) * 0.5
            # crude box inertia about COM
            m = density * vol
            Ixx = m * (dy * dy + dz * dz) / 12.0
            Iyy = m * (dx * dx + dz * dz) / 12.0
            Izz = m * (dx * dx + dy * dy) / 12.0
            inertia = (float(Ixx), float(Iyy), float(Izz), 0.0, 0.0, 0.0)
            principal = (float(Ixx), float(Iyy), float(Izz))
            mass = float(m)
            com_t = (float(com[0]), float(com[1]), float(com[2]))
            return MassProperties(
                mass_kg=mass,
                volume_m3=float(vol),
                density_kg_m3=density,
                center_of_mass_m=com_t,
                center_of_mass_mm=(com_t[0] * 1e3, com_t[1] * 1e3, com_t[2] * 1e3),
                inertia_kg_m2=inertia,
                principal_inertia_kg_m2=principal,
                watertight=False,
                method=method,
            )
        return None

    vol = abs(vol)
    mass = float(density * vol)
    com = np.asarray(mesh.center_mass, dtype=float).reshape(3)
    inertia_matrix = np.asarray(mesh.moment_inertia, dtype=float).reshape(3, 3)
    Ixx, Iyy, Izz = float(inertia_matrix[0, 0]), float(inertia_matrix[1, 1]), float(inertia_matrix[2, 2])
    Ixy, Ixz, Iyz = float(inertia_matrix[0, 1]), float(inertia_matrix[0, 2]), float(inertia_matrix[1, 2])
    try:
        principal = tuple(float(x) for x in np.linalg.eigvalsh(inertia_matrix))
        principal = tuple(sorted(principal, reverse=True))  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        principal = (Ixx, Iyy, Izz)

    com_t = (float(com[0]), float(com[1]), float(com[2]))
    return MassProperties(
        mass_kg=mass,
        volume_m3=float(vol),
        density_kg_m3=density,
        center_of_mass_m=com_t,
        center_of_mass_mm=(com_t[0] * 1e3, com_t[1] * 1e3, com_t[2] * 1e3),
        inertia_kg_m2=(Ixx, Iyy, Izz, Ixy, Ixz, Iyz),
        principal_inertia_kg_m2=(principal[0], principal[1], principal[2]),
        watertight=watertight,
        method=method,
    )
