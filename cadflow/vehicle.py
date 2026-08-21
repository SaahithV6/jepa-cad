"""Whole-vehicle mass properties from its parts.

Components have been analysed one at a time, but the intent is a rocket, and a
rocket's behaviour depends on quantities no single component has: where the
centre of gravity sits, and how much the whole stack resists being rotated.
Those set static stability against the centre of pressure, and they set how the
vehicle responds to a gimbal command.

Parts are combined with the parallel-axis theorem rather than by booleaning
them into one solid. Booleaning six thin shells is slow and can fail on
coincident faces, and it would double-count nothing that the theorem does not
handle exactly -- the parts are disjoint, which is the theorem's only condition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placed:
    """A part's own mass properties plus where it sits on the vehicle axis."""
    name: str
    mass_kg: float
    #: centroid in the part's own frame, metres
    cx_m: float
    cy_m: float
    cz_m: float
    #: moments about the part's own centroid, kg m^2
    Ixx_kg_m2: float
    Iyy_kg_m2: float
    Izz_kg_m2: float
    #: where the part's origin is placed on the vehicle, metres
    station_z_m: float = 0.0


def combine(parts: list[Placed]) -> dict:
    """Vehicle mass, centre of gravity, and moments about that centre of gravity.

    z is the vehicle axis, so Izz is the roll inertia and Ixx/Iyy are the pitch
    and yaw inertias -- the large ones, and the ones the parallel-axis shift
    dominates for a long thin vehicle.
    """
    if not parts:
        raise ValueError("a vehicle needs at least one part")

    total = sum(p.mass_kg for p in parts)
    if total <= 0.0:
        raise ValueError("total mass must be positive")

    def part_z(p: Placed) -> float:
        return p.station_z_m + p.cz_m

    cg_x = sum(p.mass_kg * p.cx_m for p in parts) / total
    cg_y = sum(p.mass_kg * p.cy_m for p in parts) / total
    cg_z = sum(p.mass_kg * part_z(p) for p in parts) / total

    ixx = iyy = izz = 0.0
    for p in parts:
        dx = p.cx_m - cg_x
        dy = p.cy_m - cg_y
        dz = part_z(p) - cg_z
        # I_axis = I_own + m * (squared distance from the axis through the CG)
        ixx += p.Ixx_kg_m2 + p.mass_kg * (dy * dy + dz * dz)
        iyy += p.Iyy_kg_m2 + p.mass_kg * (dx * dx + dz * dz)
        izz += p.Izz_kg_m2 + p.mass_kg * (dx * dx + dy * dy)

    return {
        "mass_kg": total,
        "cg_x_m": cg_x,
        "cg_y_m": cg_y,
        "cg_z_m": cg_z,
        "Ixx_kg_m2": ixx,
        "Iyy_kg_m2": iyy,
        "Izz_kg_m2": izz,
    }


def static_margin(cg_z_m: float, cp_z_m: float, body_diameter_m: float) -> float:
    """Separation of centre of pressure and centre of gravity, in calibers.

    The standard rocketry stability measure. Positive means the centre of
    pressure is aft of the centre of gravity, which is the stable arrangement:
    an angle of attack then produces a moment that reduces it. Convention here
    is +z forward, so a stable vehicle has cp *behind* cg, i.e. cp_z < cg_z.
    """
    if body_diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    return (cg_z_m - cp_z_m) / body_diameter_m
