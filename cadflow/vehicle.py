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

import math
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


#: Bulk density of a LOX/RP-1 load at a typical mixture ratio, kg/m^3. LOX is
#: 1141 and RP-1 about 810; at O/F 2.56 the mass-weighted bulk figure is ~1020.
PROPELLANT_BULK_DENSITY = 1020.0


def cylinder_inertia(mass_kg: float, radius_m: float,
                     length_m: float) -> tuple[float, float]:
    """(Ixx about the centroid, Izz about the axis) for a uniform cylinder."""
    m, r, ell = float(mass_kg), float(radius_m), float(length_m)
    return m * (3.0 * r * r + ell * ell) / 12.0, m * r * r / 2.0


def flight_vehicle_properties(
    stages,
    payload_kg: float,
    radius_m: float,
    bulk_density: float = PROPELLANT_BULK_DENSITY,
    payload_length_m: float | None = None,
) -> dict:
    """Mass properties of the vehicle the trajectory actually flies.

    This is deliberately separate from the mass properties of the analysed CAD.
    The CAD is a set of scaled coupons: body radius is clamped to 50 mm so the
    parts stay meshable, while the trajectory's reference diameter for this
    mission is 569 mm -- a factor of 8 in radius. The coupons carry 672 g of
    structure against the 151 kg the planner sized. Both numbers are correct
    about different objects, and reporting the coupon figure as the vehicle's
    would be wrong.

    Stage lengths come from propellant volume at the given bulk density, and
    each stage is treated as a uniform cylinder of its full wet mass. That is
    coarse -- a real stage has domes, a dry engine at one end and a moving
    liquid level -- but it is the right order and it uses only numbers the
    planner actually produced.
    """
    r = float(radius_m)
    if r <= 0.0:
        raise ValueError("radius must be positive")

    parts: list[Placed] = []
    station = 0.0
    # aft to forward: stage 1 first
    for i, st in enumerate(stages):
        wet = float(st.prop_mass_kg) + float(st.struct_mass_kg)
        volume = float(st.prop_mass_kg) / float(bulk_density)
        length = max(0.1, volume / (math.pi * r * r))
        ixx, izz = cylinder_inertia(wet, r, length)
        parts.append(Placed(
            name=f"stage {i+1}", mass_kg=wet,
            cx_m=0.0, cy_m=0.0, cz_m=0.0,
            Ixx_kg_m2=ixx, Iyy_kg_m2=ixx, Izz_kg_m2=izz,
            station_z_m=station + length / 2.0))
        station += length

    pay_len = payload_length_m if payload_length_m is not None else 2.0 * r
    ixx, izz = cylinder_inertia(float(payload_kg), r, pay_len)
    parts.append(Placed(
        name="payload", mass_kg=float(payload_kg),
        cx_m=0.0, cy_m=0.0, cz_m=0.0,
        Ixx_kg_m2=ixx, Iyy_kg_m2=ixx, Izz_kg_m2=izz,
        station_z_m=station + pay_len / 2.0))
    station += pay_len

    out = combine(parts)
    out["length_m"] = station
    out["radius_m"] = r
    out["sections"] = [(p.name, p.mass_kg, p.station_z_m) for p in parts]
    return out
