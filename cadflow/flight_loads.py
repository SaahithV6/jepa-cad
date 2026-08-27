"""Shear and bending moment along the assembled vehicle at max-Q.

Every structural check in this project so far has been a component check. Each
part is meshed alone, gripped at one end, pushed on the other, and asked whether
it survives an axial load. That answers a question about a coupon. It does not
answer the question a launch vehicle actually fails: whether the *stack* holds
together while flying at incidence through maximum dynamic pressure.

The load case is standard and it is the one that sizes barrel sections. At
max-Q the vehicle is at a few degrees of angle of attack -- wind shear and
guidance error put it there -- and the nose and fins develop side loads. The
vehicle is free, so it does not react those loads at a support: it accelerates
laterally and in pitch, and every element of mass along the body pushes back
through its own inertia. The bending moment is the running mismatch between
where the air pushes and where the mass resists, and it peaks somewhere in the
middle of the vehicle, far from either.

A component analysis cannot see this. The interstage that passes a 170 kN axial
check may still be the station carrying the largest bending moment in flight,
and nothing in a per-part model would say so.

What makes the result trustworthy is that the load set has to balance. The
vehicle is in d'Alembert equilibrium, so integrating the net load from nose to
tail must return shear and moment to zero at the aft end. If it does not, the
load distribution is wrong -- and a wrong distribution still produces a smooth,
plausible-looking moment curve. That closure is checked and reported rather than
assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Ultimate factor applied to limit flight loads. NASA-STD-5001B for structures
#: qualified by test; the same factor the component path uses, kept in one place
#: so the two cannot drift apart.
ULTIMATE_FACTOR = 1.4


@dataclass(frozen=True)
class PointLoad:
    """A concentrated aerodynamic side load and where it acts."""

    name: str
    station_m: float          # from the nose tip, positive aft
    force_n: float            # lateral, positive in the direction the air pushes


@dataclass
class LoadsResult:
    stations_m: list[float]
    shear_n: list[float]
    moment_nm: list[float]
    peak_moment_nm: float
    peak_moment_station_m: float
    closure_shear_n: float
    closure_moment_nm: float
    balanced: bool
    lateral_accel_m_s2: float
    pitch_accel_rad_s2: float
    pitch_inertia_kg_m2: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "peak_moment_nm": round(self.peak_moment_nm, 1),
            "peak_moment_station_m": round(self.peak_moment_station_m, 3),
            "closure_shear_n": round(self.closure_shear_n, 6),
            "closure_moment_nm": round(self.closure_moment_nm, 6),
            "balanced": self.balanced,
            "lateral_accel_m_s2": round(self.lateral_accel_m_s2, 4),
            "pitch_accel_rad_s2": round(self.pitch_accel_rad_s2, 6),
            "pitch_inertia_kg_m2": round(self.pitch_inertia_kg_m2, 1),
            "notes": list(self.notes),
        }


def mass_per_length(section_extents, stations: list[float]) -> list[float]:
    """Mass per unit length at each station, kg/m.

    Each section is treated as uniform over its own extent, which is what the
    mass properties model already assumes when it computes section inertia as a
    uniform cylinder. Using the section centroids instead would concentrate
    every stage at a point and produce a moment curve made of straight lines
    between them -- smooth, plausible, and wrong by the amount the distribution
    actually matters.
    """
    out = []
    for x in stations:
        mu = 0.0
        for _name, z0, z1, mass in section_extents:
            span = float(z1) - float(z0)
            if span > 0 and float(z0) <= x <= float(z1):
                mu += float(mass) / span
        out.append(mu)
    return out


def solve(vehicle, point_loads: list[PointLoad], *, n_stations: int = 401
          ) -> LoadsResult:
    """Integrate shear and moment along the vehicle under a balanced load set.

    ``vehicle`` is the dict from ``flight_vehicle_properties``; it must carry
    ``section_extents`` so the mass distribution is available rather than only
    the centre of gravity.
    """
    extents = vehicle.get("section_extents")
    if not extents:
        raise ValueError(
            "vehicle has no section_extents; the mass distribution is required "
            "and a centre of gravity alone cannot substitute for it")
    length = float(vehicle["length_m"])
    total_mass = float(vehicle["mass_kg"])
    if length <= 0 or total_mass <= 0:
        raise ValueError("vehicle must have positive length and mass")

    stations = [length * i / (n_stations - 1) for i in range(n_stations)]
    dx = length / (n_stations - 1)
    mu = mass_per_length(extents, stations)

    # Mass and centre of gravity of the *discretised* body, not the analytic
    # one. The integration below can only balance against the distribution it
    # actually sees; checking closure against a centre of gravity computed some
    # other way would test the bookkeeping rather than the integration.
    #
    # Every integral here uses the trapezoid rule, the same rule the shear and
    # moment integration below uses. That consistency is not cosmetic. Computing
    # the mass and inertia with rectangle sums while integrating the load with
    # trapezoids leaves the two disagreeing about the same body, and the load
    # set then cannot balance: the first version of this left 466 N of shear on
    # 100 kN applied, and produced a perfectly smooth moment curve while doing
    # it. The mass distribution is a step function, so the two rules differ
    # exactly at the section boundaries, which is where the error came from.
    def trapz(values: list[float]) -> float:
        return dx * (sum(values) - 0.5 * (values[0] + values[-1]))

    m_disc = trapz(mu)
    if m_disc <= 0:
        raise ValueError("discretised mass is zero; section extents are empty")
    cg = trapz([m * x for m, x in zip(mu, stations)]) / m_disc
    i_pitch = trapz([m * (x - cg) ** 2 for m, x in zip(mu, stations)])

    notes: list[str] = []
    drift = abs(m_disc - total_mass) / total_mass
    if drift > 0.02:
        notes.append(
            f"discretised mass differs from the reported vehicle mass by "
            f"{100.0 * drift:.1f}%; sections may overlap or leave gaps")

    # Free-flight response. The vehicle reacts the aerodynamic side load by
    # accelerating, not by pushing against a support, so there is no reaction
    # force anywhere and the inertial relief is distributed by mass.
    f_total = sum(p.force_n for p in point_loads)
    m_cg = sum(p.force_n * (p.station_m - cg) for p in point_loads)
    a_lat = f_total / m_disc
    theta_dd = (m_cg / i_pitch) if i_pitch > 0 else 0.0

    # Net running load: air pushing minus mass resisting.
    w = [-mu_i * (a_lat + theta_dd * (x - cg)) for mu_i, x in zip(mu, stations)]

    # Concentrated loads are split between the two nodes that bracket them,
    # weighted by distance. Snapping each to its nearest node instead moves it
    # by up to half a cell, and a 100 kN load displaced 1 mm is 100 N m of
    # moment that never integrates away -- it showed up as a residual that
    # refused to shrink under refinement while the shear closed to 1e-11.
    # Splitting preserves the force and its first moment exactly, so both close.
    #
    # Each share is divided by the quadrature weight of the node it lands on,
    # not by dx alone. The trapezoid rule counts the two end nodes at half
    # weight, so a load placed exactly at the nose tip or the tail enters the
    # integral at half strength -- a tip load on a uniform rod came out at
    # -25,006 N m against a closed-form 7,407, and the shear failed to close
    # along with it. Interior loads were unaffected, which is precisely why
    # this needed an analytic case to catch rather than a plausibility check.
    def node_weight(i: int) -> float:
        return 0.5 if i in (0, n_stations - 1) else 1.0

    for p in point_loads:
        s = min(length, max(0.0, p.station_m))
        lo = min(n_stations - 2, int(s / dx))
        frac = (s - lo * dx) / dx
        w[lo] += p.force_n * (1.0 - frac) / (dx * node_weight(lo))
        w[lo + 1] += p.force_n * frac / (dx * node_weight(lo + 1))

    shear, moment = [0.0], [0.0]
    for i in range(1, n_stations):
        shear.append(shear[-1] + 0.5 * (w[i] + w[i - 1]) * dx)
        moment.append(moment[-1] + 0.5 * (shear[-1] + shear[-2]) * dx)

    peak_i = max(range(n_stations), key=lambda i: abs(moment[i]))
    # Scaled against the loads applied, not their resultant. A self-equilibrated
    # set -- equal and opposite loads at the two ends -- has zero net force and
    # bends the body hard, and scaling the tolerance by the resultant demanded
    # closure to within a thousandth of one newton for it.
    applied = sum(abs(p.force_n) for p in point_loads)
    ref_force = max(applied, 1.0)
    ref_moment = max(applied * length, 1.0)
    closure_v, closure_m = shear[-1], moment[-1]
    balanced = (abs(closure_v) < 1e-3 * ref_force
                and abs(closure_m) < 1e-3 * ref_moment)
    if not balanced:
        notes.append(
            f"load set does not close: shear {closure_v:.3g} N and moment "
            f"{closure_m:.3g} N m remain at the aft end against {ref_force:.3g} N "
            f"applied. The distribution is not in equilibrium and the moment "
            f"curve below is not a valid load set.")

    return LoadsResult(
        stations_m=stations, shear_n=shear, moment_nm=moment,
        peak_moment_nm=moment[peak_i], peak_moment_station_m=stations[peak_i],
        closure_shear_n=closure_v, closure_moment_nm=closure_m,
        balanced=balanced, lateral_accel_m_s2=a_lat,
        pitch_accel_rad_s2=theta_dd, pitch_inertia_kg_m2=i_pitch,
        notes=tuple(notes))


def skin_stress_mpa(moment_nm: float, axial_n: float, radius_m: float,
                    wall_m: float) -> dict:
    """Combined axial and bending stress in a thin monocoque skin, in MPa.

    A launch vehicle barrel is a thin shell in compression, and the two effects
    add on the windward side: thrust compresses the whole section while bending
    compresses one half of it and relieves the other. Sizing against either one
    alone is how a section that passes both checks separately still fails.
    """
    r, t = float(radius_m), float(wall_m)
    if r <= 0 or t <= 0:
        raise ValueError("radius and wall thickness must be positive")
    area = 2.0 * math.pi * r * t
    # Thin-walled circular section: I = pi r^3 t, so the extreme-fibre modulus
    # is I/r = pi r^2 t.
    section_modulus = math.pi * r * r * t
    axial = float(axial_n) / area
    bending = abs(float(moment_nm)) / section_modulus
    return {
        "axial_mpa": axial / 1e6,
        "bending_mpa": bending / 1e6,
        "combined_mpa": (axial + bending) / 1e6,
        "bending_share": bending / max(axial + bending, 1e-9),
    }
