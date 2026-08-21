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

MU_EARTH = 3.986004418e14


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


try:  # aeroheating is an addition, not a gate on flying a trajectory
    from cadflow.thermal import skin_temperature as _skin_temperature
except Exception:  # noqa: BLE001
    _skin_temperature = None


def integrate_stack(
    stages: list[Stage],
    payload_kg: float,
    *,
    cd: float,
    ref_area_m2: float,
    ref_length_m: float = 4.0,
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
    # Peak axial acceleration per stage, not one global figure. The global peak
    # happens at final burnout when the vehicle is lightest, by which point the
    # lower stages have separated -- sizing stage 1 for it would be sizing it
    # for a load it never sees.
    max_axial = 0.0
    max_axial_by_stage = [0.0] * len(stages)
    liftoff_thrust = 0.0
    # Peak skin temperature on the way up. The structural analysis uses
    # room-temperature allowables, and the skin does not stay at room
    # temperature -- so this is the number that says whether that assumption
    # holds. max_skin_temp_K was a conditioning slot with nothing behind it.
    max_skin_temp = 0.0
    skin_temp_alt = 0.0
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

        if _skin_temperature is not None and rho > 1e-6 and v > 50.0:
            _, _, t_amb = atmosphere(h)
            skin = _skin_temperature(t_amb, rho, v, ref_length_m)["skin_temp_k"]
            if skin > max_skin_temp:
                max_skin_temp, skin_temp_alt = skin, h

        if not pitched and t >= pitchover_time:
            fpa = math.pi / 2.0 - pitchover_angle
            pitched = True

        # Peak axial acceleration is what sizes the structure: every stage
        # below a given station has to react it. The integrator had this number
        # in hand at every step and threw it away, so component sizing assumed a
        # flat 4.5 g instead of using the trajectory's own answer.
        if thrust > 0.0 and liftoff_thrust == 0.0:
            liftoff_thrust = thrust
        axial_g = (thrust - drag) / m / G0
        max_axial = max(max_axial, axial_g)
        if burning and idx < len(max_axial_by_stage):
            max_axial_by_stage[idx] = max(max_axial_by_stage[idx], axial_g)

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
        # Once the stack is out of propellant and above the sensible
        # atmosphere, the rest of the arc is a two-body conic and the analytic
        # apogee below is exact -- integrating further only accumulates step
        # error. Stopping here also removes the dependence on t_max: high-energy
        # flights used to run into the 4000 s limit and get extrapolated from
        # whatever mid-flight state they happened to be in, which made apogee
        # jump discontinuously (25,094 km at 1100 kg of propellant, escape at
        # 1200 kg) and left targets in between unreachable.
        if idx >= len(stages) - 1 and prop_left <= 1e-9 and h > 200_000.0:
            break

    # Ballistic apogee from the state at cutoff, so an arc still climbing when
    # integration stops is not truncated.
    #
    # This used h + v_vertical^2 / 2g, which assumes constant gravity and only
    # counts the vertical component. Both fail high up: g falls as 1/r^2, and
    # the horizontal component carries angular momentum that raises the apogee.
    # It under-reported badly enough that near-escape missions looked
    # unreachable.
    #
    # Conserve energy and angular momentum instead:
    #   eps = v^2/2 - mu/r,   L = r v cos(fpa)
    #   r_apo = (-mu + sqrt(mu^2 + 2 eps L^2)) / (2 eps)      for eps < 0
    r = R_EARTH + max(h, 0.0)
    eps = v * v / 2.0 - MU_EARTH / r
    if eps >= 0.0:
        apogee_m = float("inf")           # escape trajectory
    else:
        L = r * v * math.cos(fpa)
        disc = MU_EARTH * MU_EARTH + 2.0 * eps * L * L
        # Apsides are the roots of 2 eps r^2 + 2 mu r - L^2 = 0. For a bound
        # orbit eps < 0, so the denominator is negative and the +sqrt root is
        # PERIGEE while the -sqrt root is APOGEE. Taking +sqrt reported the
        # perigee: apogee appeared pinned just above the current altitude and
        # then jumped straight to escape, leaving every target in between
        # unreachable. At r=6571 km and v=10.2 km/s the two roots are 6,571 km
        # and 39,561 km -- the difference between "200 km" and "33,190 km".
        r_apo = (-MU_EARTH - math.sqrt(max(0.0, disc))) / (2.0 * eps)
        apogee_m = max(h, r_apo - R_EARTH)
    return {
        "apogee_m": apogee_m,
        "downrange_m": x,
        "max_q_pa": max_q,
        "burnout_s": t,
        "v_final": v,
        "separations": sep_times,
        "final_mass_kg": m,
        "max_axial_g": max_axial,
        "max_axial_g_by_stage": max_axial_by_stage,
        "liftoff_thrust_n": liftoff_thrust,
        "max_skin_temp_k": max_skin_temp,
        "max_skin_temp_altitude_m": skin_temp_alt,
    }
