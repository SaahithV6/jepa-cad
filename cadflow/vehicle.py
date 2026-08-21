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


def nose_center_of_pressure(profile, base_radius_m: float) -> float:
    """Distance from the nose tip to its centre of pressure, in profile units.

    Slender-body theory puts it at

        X_cp = L - V_nose / A_base

    which needs only the nose volume -- a quantity computed exactly from the
    meridian -- rather than a per-family coefficient looked up from a table.

    That it is right is checked against the two families whose values are exact
    constants: a cone gives 2L/3 and a von Karman ogive gives L/2, both to the
    last digit. The tangent ogive is the interesting case. Tables quote a single
    0.466 L, but the true value depends on fineness -- 0.4300 at fineness 1,
    0.4606 at 2.5, 0.4661 at 8 -- and 0.466 is the slender limit. So the formula
    is not merely as good as the table, it is better below fineness 4 or so,
    which is where sounding-rocket noses actually live.

    Not valid for a blunt nose. An elliptical nose meets the axis with infinite
    slope and the formula returns 0.333 L against a tabulated 0.5 -- the same
    validity boundary that makes its wave drag meaningless, showing up again.
    """
    from .profiles import profile_volume

    r = float(base_radius_m)
    if r <= 0.0:
        raise ValueError("base radius must be positive")
    pts = list(profile)
    length = max(z for _, z in pts) - min(z for _, z in pts)
    volume = profile_volume(pts)
    return length - volume / (math.pi * r * r)


def body_alone_static_margin(
    nose_profile_pts,
    body_radius_m: float,
    nose_tip_station_m: float,
    cg_z_m: float,
) -> float:
    """Static margin in calibers for a finless vehicle, +z forward.

    A cylinder generates no normal force in slender-body theory, so a finless
    vehicle's centre of pressure is its nose's. Such a vehicle is almost always
    unstable -- the nose CP sits well forward of any realistic CG -- which is
    exactly why fins exist, and a negative number here is the expected answer
    rather than a bug.
    """
    x_from_tip = nose_center_of_pressure(nose_profile_pts, body_radius_m)
    cp_z = nose_tip_station_m - x_from_tip
    return static_margin(cg_z_m, cp_z, 2.0 * body_radius_m)


def fin_normal_force_slope(
    n_fins: int,
    span_m: float,
    root_chord_m: float,
    tip_chord_m: float,
    sweep_m: float,
    body_radius_m: float,
) -> float:
    """Fin-set CN_alpha per radian, referenced to body frontal area.

    The Barrowman form,

        CNa = 4 n (s/d)^2 / (1 + sqrt(1 + (2 l / (Cr + Ct))^2))

    with l the mid-chord line length, multiplied by the body-fin interference
    factor Kfb = 1 + r/(s + r).

    I had refused to implement this on the grounds that the Barrowman set is
    semi-empirical and I had no way to check its constants. That was wrong:
    referenced to fin area rather than body area, the expression converges onto
    Jones' slender-wing theorem C_La = pi AR / 2 as aspect ratio goes to zero --
    ratio 0.9999 at AR 0.02, 0.9994 at 0.05, 0.9975 at 0.1. The constant is not
    remembered, it is forced by a theorem, and the departure above AR ~ 0.3 is
    the finite-aspect-ratio correction the theorem does not contain.
    """
    n = int(n_fins)
    s, cr, ct = float(span_m), float(root_chord_m), float(tip_chord_m)
    m, r = float(sweep_m), float(body_radius_m)
    if n <= 0 or s <= 0.0 or cr <= 0.0 or r <= 0.0:
        raise ValueError("fin set needs positive count, span, chord and body radius")
    if ct < 0.0:
        raise ValueError("tip chord cannot be negative")

    d = 2.0 * r
    mid_chord = math.sqrt(s * s + (m + ct / 2.0 - cr / 2.0) ** 2)
    base = 4.0 * n * (s / d) ** 2 / (
        1.0 + math.sqrt(1.0 + (2.0 * mid_chord / (cr + ct)) ** 2)
    )
    interference = 1.0 + r / (s + r)
    return base * interference


def fin_center_of_pressure(
    root_chord_m: float, tip_chord_m: float, sweep_m: float
) -> float:
    """Distance from the fin root leading edge to the fin CP, along the axis.

    Checked against the case with an exact answer: an unswept rectangular fin
    must put its centre of pressure at the quarter chord, and this returns
    exactly Cr/4 for Ct = Cr and zero sweep.
    """
    cr, ct, m = float(root_chord_m), float(tip_chord_m), float(sweep_m)
    if cr <= 0.0 or ct < 0.0:
        raise ValueError("chords must be positive")
    total = cr + ct
    return (m * (cr + 2.0 * ct)) / (3.0 * total) + (
        total - cr * ct / total
    ) / 6.0


def vehicle_center_of_pressure(
    nose_profile_pts,
    body_radius_m: float,
    nose_tip_station_m: float,
    fin_root_le_station_m: float,
    n_fins: int,
    fin_span_m: float,
    fin_root_chord_m: float,
    fin_tip_chord_m: float,
    fin_sweep_m: float,
) -> dict:
    """Whole-vehicle centre of pressure, +z forward.

    Nose and fins are combined by weighting each contribution's station with its
    CN_alpha, which is what "centre of pressure" means. A cylindrical body
    section contributes no normal force in this theory and so does not appear.
    """
    x_nose = nose_center_of_pressure(nose_profile_pts, body_radius_m)
    cp_nose = float(nose_tip_station_m) - x_nose
    cna_nose = 2.0                      # slender-body: 2 per radian, any nose

    cna_fins = fin_normal_force_slope(n_fins, fin_span_m, fin_root_chord_m,
                                      fin_tip_chord_m, fin_sweep_m,
                                      body_radius_m)
    x_fins = fin_center_of_pressure(fin_root_chord_m, fin_tip_chord_m,
                                    fin_sweep_m)
    cp_fins = float(fin_root_le_station_m) - x_fins

    total = cna_nose + cna_fins
    return {
        "cp_z_m": (cna_nose * cp_nose + cna_fins * cp_fins) / total,
        "cna_total": total,
        "cna_nose": cna_nose,
        "cna_fins": cna_fins,
        "cp_nose_z_m": cp_nose,
        "cp_fins_z_m": cp_fins,
    }


#: Conventional design target for a finned rocket, in calibers. Below about 1
#: the vehicle is marginal; much above 2 it weathercocks into the wind and
#: loses altitude to it.
TARGET_STATIC_MARGIN = 1.5


def size_fins_for_margin(
    nose_profile_pts,
    body_radius_m: float,
    nose_tip_station_m: float,
    cg_z_m: float,
    fin_root_le_station_m: float,
    n_fins: int = 4,
    target_margin: float = TARGET_STATIC_MARGIN,
    root_chord_ratio: float = 2.0,
    taper_ratio: float = 0.5,
    sweep_ratio: float = 0.6,
    max_span_ratio: float = 4.0,
) -> dict:
    """Find the fin span that puts the static margin on target.

    This is the design direction rather than the analysis direction: the vehicle
    and its centre of gravity are already fixed by the mission, so the free
    variable is fin size, and stability is the requirement it has to meet.

    Chord, taper and sweep are held in fixed proportion to the body and to each
    other, so span is the single unknown and the solve is a bisection on a
    monotone function -- bigger fins always move the centre of pressure aft.
    """
    r = float(body_radius_m)
    if r <= 0.0:
        raise ValueError("body radius must be positive")
    root_chord = root_chord_ratio * r

    def margin_for(span: float) -> float:
        cp = vehicle_center_of_pressure(
            nose_profile_pts, r, nose_tip_station_m, fin_root_le_station_m,
            n_fins, span, root_chord, root_chord * taper_ratio,
            root_chord * sweep_ratio)
        return static_margin(cg_z_m, cp["cp_z_m"], 2.0 * r)

    lo, hi = 1e-4 * r, max_span_ratio * r
    if margin_for(hi) < target_margin:
        # Even the largest fins allowed cannot stabilise it; report the best
        # available rather than silently returning a span that misses.
        span = hi
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if margin_for(mid) < target_margin:
                lo = mid
            else:
                hi = mid
        span = 0.5 * (lo + hi)

    cp = vehicle_center_of_pressure(
        nose_profile_pts, r, nose_tip_station_m, fin_root_le_station_m,
        n_fins, span, root_chord, root_chord * taper_ratio,
        root_chord * sweep_ratio)
    return {
        "span_m": span,
        "root_chord_m": root_chord,
        "tip_chord_m": root_chord * taper_ratio,
        "sweep_m": root_chord * sweep_ratio,
        "n_fins": int(n_fins),
        "static_margin_cal": static_margin(cg_z_m, cp["cp_z_m"], 2.0 * r),
        "target_margin_cal": float(target_margin),
        "met": abs(static_margin(cg_z_m, cp["cp_z_m"], 2.0 * r)
                   - target_margin) < 1e-3,
        **cp,
    }
