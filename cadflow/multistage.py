"""Multi-stage trajectory integration.

The single-stage path cannot represent high-energy missions: everything the
vehicle carries to burnout it carries the whole way, so the mass ratio needed
grows exponentially and the structure eventually cannot be built. Staging drops
spent structure, which is why real vehicles reaching orbit are staged.

Concretely, the single-stage solver in solve_mission_corpus.py runs out at a few
hundred kilometres of apogee -- its 1600 km specifications fail -- because the
mass ratio required exceeds what a fixed structural coefficient allows.

Same physics as generate_propulsion_trajectory_corpus.integrate_trajectory:
gravity turn over an exponential atmosphere, RK4, drag and mass depletion.
The difference is that thrust and mass come from whichever stage is live, and
the spent stage's structure leaves the vehicle at separation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from generate_propulsion_trajectory_corpus import (
    G0, R_EARTH, atmosphere, nozzle_performance,
)


@dataclass
class Stage:
    prop_mass_kg: float
    struct_mass_kg: float
    throat_area_m2: float
    chamber_pressure_pa: float
    expansion_ratio: float
    gamma: float
    chamber_temp: float
    mol_mass: float


def integrate_stack(
    stages: list[Stage],
    payload_kg: float,
    *,
    cd: float,
    ref_area_m2: float,
    pitchover_time: float = 8.0,
    pitchover_angle: float = math.radians(3.0),
    dt: float = 0.2,
    t_max: float = 4000.0,
    coast_between_stages_s: float = 2.0,
) -> dict:
    """Fly a stack. Spent structure is jettisoned at each separation."""
    # Mass above stage i is payload plus every stage on top of it.
    above = [payload_kg]
    for s in reversed(stages[:-1] if len(stages) > 1 else []):
        above.append(above[-1] + s.prop_mass_kg + s.struct_mass_kg)
    above = list(reversed(above)) if len(stages) > 1 else [payload_kg]
    while len(above) < len(stages):
        above.append(payload_kg)

    total = payload_kg + sum(s.prop_mass_kg + s.struct_mass_kg for s in stages)

    idx = 0
    stage = stages[0]
    prop_left = stage.prop_mass_kg
    m = total
    v, fpa, h, x = 1e-3, math.pi / 2.0, 0.0, 0.0
    t = 0.0
    max_q = 0.0
    sep_times: list[float] = []
    coasting_until = -1.0
    pitched = False

    while t < t_max:
        rho, p_amb, _ = atmosphere(h)
        g = G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2

        burning = prop_left > 1e-9 and idx < len(stages) and t >= coasting_until
        if burning:
            perf = nozzle_performance(
                chamber_pressure=stage.chamber_pressure_pa,
                chamber_temp=stage.chamber_temp,
                expansion_ratio=stage.expansion_ratio,
                throat_area=stage.throat_area_m2,
                gamma=stage.gamma, mol_mass=stage.mol_mass,
                ambient_pressure=p_amb)
            thrust, mdot = perf["thrust"], perf["mdot"]
        else:
            thrust, mdot = 0.0, 0.0

        v_safe = max(v, 1e-3)
        drag = 0.5 * rho * v_safe * v_safe * cd * ref_area_m2
        max_q = max(max_q, 0.5 * rho * v_safe * v_safe)

        if not pitched and t >= pitchover_time:
            fpa = math.pi / 2.0 - pitchover_angle
            pitched = True

        dv = thrust / m - g * math.sin(fpa) - drag / m
        dfpa = -(g / v_safe - v_safe / (R_EARTH + max(h, 0.0))) * math.cos(fpa)

        v += dv * dt
        fpa += dfpa * dt
        h += v * math.sin(fpa) * dt
        x += v * math.cos(fpa) * R_EARTH / (R_EARTH + max(h, 0.0)) * dt
        burned = mdot * dt
        m -= burned
        prop_left -= burned
        t += dt

        # separation: drop the spent stage's structure and light the next
        if prop_left <= 1e-9 and idx < len(stages) - 1:
            m -= stage.struct_mass_kg
            sep_times.append(t)
            idx += 1
            stage = stages[idx]
            prop_left = stage.prop_mass_kg
            coasting_until = t + coast_between_stages_s

        if h < 0.0 and t > 1.0:
            break
        if idx >= len(stages) - 1 and prop_left <= 1e-9 and v * math.sin(fpa) < 0.0:
            break

    # ballistic apogee from the state at burnout, so a suborbital arc that is
    # still climbing when integration stops is not truncated
    apogee_m = h + max(0.0, (v * math.sin(fpa)) ** 2 / (2 * 9.80665))
    return {
        "apogee_m": apogee_m,
        "downrange_m": x,
        "max_q_pa": max_q,
        "burnout_s": t,
        "v_final": v,
        "separations": sep_times,
        "final_mass_kg": m,
    }
