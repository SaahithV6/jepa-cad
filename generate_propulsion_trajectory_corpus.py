#!/usr/bin/env python3
"""Deterministic propulsion + trajectory corpus generator.

Closes the discipline gap in the training corpus. Before this, the only physics
in the graph was FEA (structural) and CFD (aerodynamic fields), so a
specification like "x kg payload to y km" had nothing to condition on: neither
burn profiles nor trajectory outcomes existed anywhere in the 153-d conditioning
vector.

Everything here is analytic or numerically integrated -- no solver required, so
a full sweep runs in seconds rather than the hours a CalculiX/OpenFOAM batch
takes. That matches the propulsion spec in docs/propulsion-nozzle-generator-spec.md,
which calls for analytic backends (isentropic nozzle relations, Bartz-style
estimates) with solver verification as an optional later step.

Physics
-------
Nozzle: ideal-rocket isentropic relations. Expansion ratio -> exit Mach via the
area-Mach relation (supersonic branch, solved by bisection), then exit pressure
from the isentropic pressure ratio, then thrust coefficient C_F and
characteristic velocity c*. Gives thrust, mass flow, Isp (vacuum and sea level),
and burn time.

Trajectory: 2-D gravity turn over an exponential atmosphere, integrated with
RK4. Vertical rise, pitchover kick, then gravity-driven turn. Tracks drag,
dynamic pressure, and mass depletion through burnout, then coasts to apogee.
Gives apogee, downrange, max-Q, burnout velocity, and ideal delta-v.

Output
------
Shards use the same contract as the FEA/CFD shards -- points (N,3) and fields
(N,8) -- so data/graph_dataset.py loads them with no changes:

    points = (t_norm, altitude_norm, downrange_norm)
    fields = thrust, mass, drag, mach, dynamic_pressure, accel, velocity, altitude

Fields are normalised per shard to [-1,1], matching the existing convention
(and its documented cost: absolute magnitudes are not recoverable from the
shard alone, which is why the unnormalised engineering values are also written
to the manifest and the record JSON).
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
import math

from cadflow.wave_drag import TANGENT_SHAPES, cd_multiplier, shape_factor
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
G0 = 9.80665          # standard gravity, m/s^2
R_EARTH = 6_371_000.0  # m
RHO0 = 1.225           # sea-level density, kg/m^3
H_SCALE = 8_500.0      # atmospheric scale height, m
P0 = 101_325.0         # sea-level pressure, Pa
R_UNIV = 8314.462      # J/(kmol K)
GAMMA_SOUND = 1.4      # air, for Mach number
R_AIR = 287.05

N_POINTS = 2048        # match the FEA/CFD shard point count
N_FIELDS = 8


# ------------------------------------------------------------ nozzle physics
def exit_mach_from_area_ratio(eps: float, gamma: float) -> float:
    """Invert the area-Mach relation for the supersonic branch.

    eps = (1/M) * [ (2/(g+1)) * (1 + (g-1)/2 * M^2) ] ^ ((g+1)/(2(g-1)))

    Monotonic in M for M > 1, so bisection is safe and needs no derivative.
    """
    if eps <= 1.0:
        return 1.0

    def area_ratio(m: float) -> float:
        t = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)
        return (1.0 / m) * t ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))

    lo, hi = 1.0000001, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid) < eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def area_ratio_from_mach(m: float, gamma: float) -> float:
    t = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)
    return (1.0 / m) * t ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def separation_limited_ratio(chamber_pressure: float, ambient: float, gamma: float) -> float:
    """Largest expansion ratio that stays attached at this ambient pressure.

    Summerfield criterion: an over-expanded nozzle separates once the exit
    pressure falls below roughly 0.4 x ambient. Past that the flow detaches
    from the wall, so the nozzle behaves as if it ended at the separation
    plane. Without this, ideal theory keeps charging the full (p_e - p_a)*A_e
    pressure debt and reports absurd sea-level Isp -- e.g. 64 s for an eps=76
    nozzle that in reality would simply separate.
    """
    # The exponential atmosphere underflows to a denormal at extreme altitude:
    # nonzero, so a plain `<= 0` vacuum check passes it through, but small
    # enough that the pressure ratio below underflows to exactly zero and
    # `0 ** negative` raises ZeroDivisionError. Treat anything that small as
    # vacuum, which is what it physically is.
    if ambient <= 1e-12:
        return float("inf")
    p_sep = 0.4 * ambient
    if p_sep >= chamber_pressure:
        return 1.0
    pr = p_sep / chamber_pressure
    if pr <= 1e-300:
        return float("inf")
    m_sep = math.sqrt(max(0.0, (2.0 / (gamma - 1.0)) * (pr ** (-(gamma - 1.0) / gamma) - 1.0)))
    if m_sep <= 1.0:
        return 1.0
    return area_ratio_from_mach(m_sep, gamma)


def nozzle_performance(
    *,
    chamber_pressure: float,   # Pa
    chamber_temp: float,       # K
    expansion_ratio: float,
    throat_area: float,        # m^2
    gamma: float,
    mol_mass: float,           # kg/kmol
    ambient_pressure: float,   # Pa
    divergence_efficiency: float = 1.0,
) -> dict:
    """Ideal-rocket performance, with flow separation when over-expanded."""
    r_specific = R_UNIV / mol_mass

    # An over-expanded nozzle separates rather than paying the full pressure
    # debt; the effective exit plane moves upstream to the separation point.
    eps_max = separation_limited_ratio(chamber_pressure, ambient_pressure, gamma)
    separated = expansion_ratio > eps_max
    eps_eff = min(expansion_ratio, eps_max)

    mach_e = exit_mach_from_area_ratio(eps_eff, gamma)
    # isentropic static/stagnation pressure ratio at the exit
    pe_pc = (1.0 + 0.5 * (gamma - 1.0) * mach_e * mach_e) ** (-gamma / (gamma - 1.0))
    p_exit = pe_pc * chamber_pressure
    expansion_ratio = eps_eff

    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    vandenkerckhove = math.sqrt(gamma) * (2.0 / gp1) ** (gp1 / (2.0 * gm1))

    # characteristic velocity
    c_star = math.sqrt(r_specific * chamber_temp) / vandenkerckhove

    # momentum term of the thrust coefficient
    cf_mom = math.sqrt(
        (2.0 * gamma * gamma / gm1)
        * (2.0 / gp1) ** (gp1 / gm1)
        * max(0.0, 1.0 - pe_pc ** (gm1 / gamma))
    )
    # pressure term, the part that makes Isp altitude-dependent
    cf = cf_mom + (p_exit - ambient_pressure) / chamber_pressure * expansion_ratio

    # Divergence loss. A real nozzle throws some of its exhaust sideways, and
    # only the axial component pushes: lambda = (1 + cos theta_e)/2. This was
    # absent, so nozzle *shape* had no consequence anywhere -- the contour could
    # be a 40-degree cone or an optimised bell and the thrust came out the same.
    # Defaults to 1.0 so callers that do not have a contour are unaffected.
    thrust = cf * chamber_pressure * throat_area * float(divergence_efficiency)
    mdot = chamber_pressure * throat_area / c_star
    isp = thrust / (mdot * G0) if mdot > 0 else 0.0

    return {
        "mach_exit": mach_e,
        "p_exit": p_exit,
        "c_star": c_star,
        "cf": cf,
        "thrust": thrust,
        "mdot": mdot,
        "isp": isp,
        "separated": separated,
        "eps_effective": eps_eff,
    }


# -------------------------------------------------------------- atmosphere
def atmosphere(h: float) -> tuple[float, float, float]:
    """Exponential atmosphere: (density, pressure, speed of sound)."""
    h = max(0.0, h)
    rho = RHO0 * math.exp(-h / H_SCALE)
    p = P0 * math.exp(-h / H_SCALE)
    # crude but monotone temperature profile, enough for Mach bookkeeping
    temp = max(180.0, 288.15 - 0.0065 * min(h, 11_000.0))
    a = math.sqrt(GAMMA_SOUND * R_AIR * temp)
    return rho, p, a


# -------------------------------------------------------------- trajectory
def integrate_trajectory(
    *,
    m0: float,           # liftoff mass, kg
    m_prop: float,       # propellant mass, kg
    throat_area: float,
    chamber_pressure: float,
    chamber_temp: float,
    expansion_ratio: float,
    gamma: float,
    mol_mass: float,
    cd: float,
    ref_area: float,     # m^2
    pitchover_time: float,
    pitchover_angle: float,   # radians off vertical
    dt: float = 0.05,
    t_max: float = 2000.0,
) -> dict:
    """2-D gravity turn, RK4, integrated through burnout and coast to apogee."""

    def derivs(state: np.ndarray, burning: bool) -> tuple[np.ndarray, dict]:
        v, fpa, h, x, m = state
        rho, p_amb, a_snd = atmosphere(h)
        g = G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2

        if burning and m > (m0 - m_prop):
            perf = nozzle_performance(
                chamber_pressure=chamber_pressure,
                chamber_temp=chamber_temp,
                expansion_ratio=expansion_ratio,
                throat_area=throat_area,
                gamma=gamma,
                mol_mass=mol_mass,
                ambient_pressure=p_amb,
            )
            thrust, mdot = perf["thrust"], perf["mdot"]
        else:
            thrust, mdot = 0.0, 0.0

        v_safe = max(v, 1e-3)
        drag = 0.5 * rho * v_safe * v_safe * cd * ref_area
        q = 0.5 * rho * v_safe * v_safe

        dv = thrust / m - g * math.sin(fpa) - drag / m
        # gravity turn; no steering authority applied, gravity does the turning
        dfpa = -(g / v_safe - v_safe / (R_EARTH + max(h, 0.0))) * math.cos(fpa)
        dh = v * math.sin(fpa)
        dx = v * math.cos(fpa) * R_EARTH / (R_EARTH + max(h, 0.0))
        dm = -mdot

        aux = {"thrust": thrust, "drag": drag, "q": q, "mach": v / a_snd, "accel": dv}
        return np.array([dv, dfpa, dh, dx, dm]), aux

    state = np.array([1e-3, math.pi / 2.0, 0.0, 0.0, m0])
    t = 0.0
    burnout_t = None
    samples: list[list[float]] = []
    max_q = 0.0
    max_mach = 0.0

    while t < t_max:
        burning = state[4] > (m0 - m_prop) + 1e-9
        if burnout_t is None and not burning:
            burnout_t = t

        # pitchover kick: one-time small rotation off vertical
        if burnout_t is None and abs(t - pitchover_time) < dt / 2:
            state[1] = math.pi / 2.0 - pitchover_angle

        k1, aux = derivs(state, burning)
        k2, _ = derivs(state + 0.5 * dt * k1, burning)
        k3, _ = derivs(state + 0.5 * dt * k2, burning)
        k4, _ = derivs(state + dt * k3, burning)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        state[4] = max(state[4], m0 - m_prop)
        max_q = max(max_q, aux["q"])
        max_mach = max(max_mach, aux["mach"])

        samples.append([
            t, state[2], state[3],                       # time, altitude, downrange
            aux["thrust"], state[4], aux["drag"],
            aux["mach"], aux["q"], aux["accel"],
            state[0],                                     # velocity
        ])

        # stop at apogee (vertical velocity turns negative) or on impact
        if state[2] < 0.0 and t > 1.0:
            break
        if burnout_t is not None and state[0] * math.sin(state[1]) < 0.0:
            break

    arr = np.array(samples, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("empty trajectory")

    apogee = float(arr[:, 1].max())
    downrange = float(arr[:, 2].max())
    v_burnout = 0.0
    if burnout_t is not None:
        idx = int(min(len(arr) - 1, burnout_t / dt))
        v_burnout = float(arr[idx, 9])

    return {
        "samples": arr,
        "apogee_m": apogee,
        "downrange_m": downrange,
        "max_q_pa": max_q,
        "max_mach": max_mach,
        "burnout_s": burnout_t if burnout_t is not None else float(t),
        "v_burnout": v_burnout,
        "flight_time_s": float(t),
    }


# ------------------------------------------------------------ shard writing
def to_shard(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resample the trajectory onto the fixed (N,3)+(N,8) shard contract."""
    arr = traj["samples"]
    idx = np.linspace(0, len(arr) - 1, N_POINTS).astype(int)
    s = arr[idx]

    def norm(col: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi - lo < 1e-12:
            return np.zeros_like(col)
        return 2.0 * (col - lo) / (hi - lo) - 1.0

    points = np.stack([norm(s[:, 0]), norm(s[:, 1]), norm(s[:, 2])], axis=1)
    fields = np.stack([
        norm(s[:, 3]),   # thrust
        norm(s[:, 4]),   # mass
        norm(s[:, 5]),   # drag
        norm(s[:, 6]),   # mach
        norm(s[:, 7]),   # dynamic pressure
        norm(s[:, 8]),   # acceleration
        norm(s[:, 9]),   # velocity
        norm(s[:, 1]),   # altitude
    ], axis=1)
    return points.astype(np.float32), fields.astype(np.float32)


# ------------------------------------------------------------------ sweep
PROPELLANTS = {
    # gamma, chamber temp K, molecular mass kg/kmol
    "lox_rp1":    (1.20, 3_600.0, 23.0),
    "lox_lh2":    (1.24, 3_300.0, 12.0),
    "lox_ch4":    (1.20, 3_500.0, 21.0),
    "n2o4_mmh":   (1.24, 3_200.0, 22.0),
    "solid_apcp": (1.18, 3_000.0, 27.0),
}


#: Mixture-ratio sampling window per propellant, as a fraction of the reference
#: value. Wide enough that the trade is visible in the data, narrow enough that
#: every sample is a mixture an engine could actually run.
_OF_WINDOW = (0.65, 1.45)


def _sample_chamber(rng: random.Random, prop: str):
    """Sample a mixture ratio and solve the chamber for it.

    Falls back to the old constant table if the chemistry is unavailable, so a
    machine without Cantera still generates a corpus -- one without an O/F axis,
    but a valid one.
    """
    try:
        from cadflow.combustion import REFERENCE, bulk_density, chamber_equilibrium

        of_ref = REFERENCE[prop][0]
        lo, hi = _OF_WINDOW
        of = round(of_ref * rng.uniform(lo, hi) / 0.05) * 0.05
        state = chamber_equilibrium(prop, of)
        return (of, state.gamma, state.chamber_temp_k, state.molecular_mass,
                state.c_star_m_s, bulk_density(prop, of))
    except Exception:  # noqa: BLE001 - chemistry is an enhancement, not a gate
        gamma, tc, mol = PROPELLANTS[prop]
        return None, gamma, tc, mol, None, 1030.0


def _thermal_state(prop: str, of_ratio, throat_area_m2: float,
                   chamber_pressure_pa: float, c_star):
    """Throat heat flux and whether the fuel flow can carry it away."""
    if of_ratio is None or not c_star:
        return None
    try:
        from cadflow.combustion import COMBINATIONS, chamber_equilibrium
        from cadflow.thermal import chamber_heat_load, regenerative_cooling

        state = chamber_equilibrium(prop, of_ratio, chamber_pressure_pa)
        mdot = chamber_pressure_pa * throat_area_m2 / state.c_star_m_s
        load = chamber_heat_load(state, throat_area_m2, mdot)
        fuel = COMBINATIONS[prop][1]
        cool = regenerative_cooling(load["q_total_w"], mdot / (1.0 + of_ratio),
                                    fuel)
        return {
            "throat_heat_flux_MWm2": load["throat"].heat_flux_mw_m2,
            "wall_temp_max_K": load["throat"].wall_temp_k,
            "recovery_temp_K": load["throat"].recovery_temp_k,
            "coolant_outlet_K": cool["outlet_temp_k"],
            "cooling_margin_K": cool["margin_k"],
            "cooling_feasible": bool(cool["feasible"]),
        }
    except Exception:  # noqa: BLE001
        return None


# Yield strength of Al-6061-T6, the default airframe material in the corpus.
ALLOWABLE_MPA = 276.0

_COUPLING: dict | None = None


def load_coupling(path: Path | None = None) -> dict:
    """Load solver-derived aero/structural distributions, if extracted.

    Without this the generator invents its own drag coefficient and structural
    coefficient from uniform distributions, leaving the disciplines running in
    parallel: thousands of CFD and FEA shards exist but nothing in the
    trajectory depends on either. With it, drag comes from CFD-derived Cd and
    the structural mass fraction is driven by real CalculiX von Mises results.
    """
    global _COUPLING
    if _COUPLING is not None:
        return _COUPLING
    p = path or Path("artifacts/coupling/aero_structural_library.json")
    if p.exists():
        _COUPLING = json.loads(p.read_text())
        n_cd = len(_COUPLING.get("cd", {}).get("values", []))
        n_vm = len(_COUPLING.get("fea_max_von_mises_mpa", {}).get("values", []))
        print(f"coupling library: {n_cd} Cd values, {n_vm} FEA von Mises values")
    else:
        _COUPLING = {}
        print("coupling library absent -- falling back to uncoupled sampling")
    return _COUPLING


def _draw(rng: random.Random, key: str, fallback_lo: float, fallback_hi: float) -> tuple[float, bool]:
    """Draw from a harvested distribution; report whether it was coupled."""
    vals = load_coupling().get(key, {}).get("values") or []
    if vals:
        return float(rng.choice(vals)), True
    return rng.uniform(fallback_lo, fallback_hi), False


def sample_design(rng: random.Random, idx: int) -> dict:
    prop = rng.choice(list(PROPELLANTS))

    # MIXTURE RATIO. The corpus carried one chamber condition per propellant
    # and no O/F axis at all, so the central trade in a liquid engine -- rich
    # for specific impulse against lean for density -- was absent from the
    # training data entirely. It is now a sampled design variable and the
    # chamber conditions follow from equilibrium chemistry at that ratio.
    #
    # Quantised to 0.05 because the equilibrium solve is memoised on a rounded
    # ratio; the answer varies far more slowly with O/F than that.
    of_ratio, gamma, tc, mol, c_star, rho_bulk = _sample_chamber(rng, prop)

    # Mass budget at FIXED gross liftoff mass, so payload competes with
    # propellant the way it does on a real vehicle. Sampling payload and mass
    # ratio independently (the obvious approach) makes a heavier payload merely
    # imply a proportionally bigger rocket, leaving delta-v unchanged -- the
    # payload/range trade, which is the whole point of this corpus, then simply
    # is not present in the data.
    m0 = 10.0 ** rng.uniform(3.0, 5.5)            # 1 t .. ~300 t gross
    payload_frac = rng.uniform(0.004, 0.070)      # payload as fraction of gross
    payload = m0 * payload_frac

    # STRUCTURAL COUPLING: the structural mass coefficient is driven by real
    # CalculiX results rather than drawn at random. A part running closer to
    # the material allowable needs more material to carry its load, so higher
    # utilisation buys a heavier structure and eats propellant mass.
    #
    # von Mises in the corpus spans five orders of magnitude (p10 0.095 MPa,
    # p90 2776 MPa); the upper tail is mesh singularities at re-entrant
    # corners rather than real load, so utilisation is clamped before use.
    vm_mpa, vm_coupled = _draw(rng, "fea_max_von_mises_mpa", 10.0, 200.0)
    utilisation = min(1.0, max(0.05, vm_mpa / ALLOWABLE_MPA))
    struct_coeff = 0.06 + 0.10 * utilisation      # spans the same 0.06-0.16 band

    m_struct = struct_coeff * (m0 - payload)
    m_prop = m0 - payload - m_struct
    m_dry = m0 - m_prop
    mass_ratio = m0 / max(m_dry, 1e-6)            # falls as payload rises

    diameter = max(0.2, (m0 / 1000.0) ** (1.0 / 3.0) * rng.uniform(0.5, 1.2))
    ref_area = math.pi * (diameter / 2.0) ** 2

    # AERO COUPLING: drag comes from the CFD corpus rather than a guess.
    # Only the slender-body band is used -- see extract_coupling_library.py for
    # why the raw Cd_proxy distribution cannot be taken at face value.
    cd, cd_coupled = _draw(rng, "cd", 0.25, 0.55)

    # NOSE SHAPE. The corpus previously had no nose at all, so shape could not
    # be learned from it. The drawn Cd is taken to describe a tangent ogive,
    # which is what cd_multiplier normalises to, so an ogive record keeps
    # exactly the drag it would have had before.
    #
    # Only tangent noses are sampled. A cone meets the body with a slope break,
    # and slender-body wave drag is logarithmically divergent there -- its Cd
    # grows without bound as the quadrature refines. Training on that would be
    # training on the resolution of an integrator.
    nose_shape = rng.choice(list(TANGENT_SHAPES))
    nose_fineness = rng.uniform(1.5, 5.5)
    cd = cd * cd_multiplier(nose_shape, nose_fineness)

    pc = rng.uniform(2.0e6, 20.0e6)
    # Expansion ratio is a design choice tied to where the stage operates.
    # Sea-level stages run low eps to stay attached in atmosphere; vacuum
    # stages run high eps because there is no ambient pressure to fight.
    stage_role = rng.choice(["booster", "booster", "sustainer", "vacuum"])
    eps = {
        "booster": lambda: rng.uniform(5.0, 22.0),
        "sustainer": lambda: rng.uniform(15.0, 45.0),
        "vacuum": lambda: rng.uniform(40.0, 100.0),
    }[stage_role]()
    # size the throat so thrust/weight lands in a flyable band
    twr = rng.uniform(1.3, 2.6)
    perf_guess = nozzle_performance(
        chamber_pressure=pc, chamber_temp=tc, expansion_ratio=eps,
        throat_area=1.0, gamma=gamma, mol_mass=mol, ambient_pressure=P0,
    )
    throat_area = (twr * m0 * G0) / max(perf_guess["thrust"], 1e-6)

    return {
        "index": idx,
        "propellant": prop,
        "stage_role": stage_role,
        "gamma": gamma,
        "chamber_temp": tc,
        "mol_mass": mol,
        "payload_kg": float(payload),
        "payload_fraction": float(payload_frac),
        "struct_coeff": float(struct_coeff),
        "dry_mass_kg": float(m_dry),
        "prop_mass_kg": float(m_prop),
        "liftoff_mass_kg": float(m0),
        "mass_ratio": float(mass_ratio),
        "diameter_m": float(diameter),
        "ref_area_m2": float(ref_area),
        "chamber_pressure_pa": float(pc),
        "expansion_ratio": float(eps),
        "throat_area_m2": float(throat_area),
        **( {} if (_th := _thermal_state(prop, of_ratio, float(throat_area),
                                         float(pc), c_star)) is None
            else {"thermal": _th} ),
        "target_twr": float(twr),
        "cd": float(cd),
        "cd_coupled": bool(cd_coupled),
        # Chemistry: the mixture ratio this design runs at, and the chamber
        # conditions equilibrium gives for it. Previously one row per
        # propellant, so the model had no O/F axis to learn on.
        "mixture_ratio": (float(of_ratio) if of_ratio is not None else None),
        "bulk_density_kgm3": float(rho_bulk),
        "c_star_m_s": (float(c_star) if c_star else None),
        "nose_shape": str(nose_shape),
        "nose_fineness": float(nose_fineness),
        # Named to match the conditioning slots so they reach the model.
        "fineness_ratio": float(nose_fineness),
        "nose_wave_factor": float(shape_factor(nose_shape, nose_fineness)),
        "fea_max_von_mises_mpa": float(vm_mpa),
        "structural_utilisation": float(utilisation),
        "struct_coupled": bool(vm_coupled),
        "pitchover_time_s": float(rng.uniform(8.0, 25.0)),
        "pitchover_angle_deg": float(rng.uniform(1.0, 6.0)),
    }


def run_case(design: dict) -> dict | None:
    gamma, tc, mol = design["gamma"], design["chamber_temp"], design["mol_mass"]

    vac = nozzle_performance(
        chamber_pressure=design["chamber_pressure_pa"], chamber_temp=tc,
        expansion_ratio=design["expansion_ratio"], throat_area=design["throat_area_m2"],
        gamma=gamma, mol_mass=mol, ambient_pressure=0.0,
    )
    sl = nozzle_performance(
        chamber_pressure=design["chamber_pressure_pa"], chamber_temp=tc,
        expansion_ratio=design["expansion_ratio"], throat_area=design["throat_area_m2"],
        gamma=gamma, mol_mass=mol, ambient_pressure=P0,
    )
    if sl["thrust"] <= design["liftoff_mass_kg"] * G0:
        return None  # cannot lift off; drop rather than emit a bogus trajectory

    burn_time = design["prop_mass_kg"] / vac["mdot"]
    delta_v_ideal = vac["isp"] * G0 * math.log(design["mass_ratio"])

    traj = integrate_trajectory(
        m0=design["liftoff_mass_kg"], m_prop=design["prop_mass_kg"],
        throat_area=design["throat_area_m2"],
        chamber_pressure=design["chamber_pressure_pa"], chamber_temp=tc,
        expansion_ratio=design["expansion_ratio"], gamma=gamma, mol_mass=mol,
        cd=design["cd"], ref_area=design["ref_area_m2"],
        pitchover_time=design["pitchover_time_s"],
        pitchover_angle=math.radians(design["pitchover_angle_deg"]),
    )

    return {
        "isp_vac_s": vac["isp"], "isp_sl_s": sl["isp"],
        "thrust_vac_n": vac["thrust"], "thrust_sl_n": sl["thrust"],
        "mdot_kg_s": vac["mdot"], "c_star_m_s": vac["c_star"], "cf_vac": vac["cf"],
        "mach_exit": vac["mach_exit"], "p_exit_pa": vac["p_exit"],
        "burn_time_s": burn_time, "delta_v_ideal_m_s": delta_v_ideal,
        "apogee_km": traj["apogee_m"] / 1000.0,
        "downrange_km": traj["downrange_m"] / 1000.0,
        "max_q_pa": traj["max_q_pa"], "max_mach": traj["max_mach"],
        "v_burnout_m_s": traj["v_burnout"], "flight_time_s": traj["flight_time_s"],
        "_traj": traj,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/physics_shards/traj"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("artifacts/physics_shards/traj_manifest.jsonl"))
    ap.add_argument("--records", type=Path,
                    default=Path("artifacts/propulsion_trajectory/records.jsonl"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.records.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to run alongside another generation. Two concurrent runs write the
    # same manifest and records, and the one that finishes last wins -- so a
    # long run started before a fix silently reverted the corpus to the state
    # the fix had corrected, after a shorter corrected run had already written
    # it. The conditioning contract test caught it, but nothing should have had
    # to.
    lock = args.manifest.with_suffix(args.manifest.suffix + ".lock")
    try:
        lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"another generation holds {lock}; refusing to run.\n"
              f"if no generator is running, remove it.", file=sys.stderr)
        return 2
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))

    rng = random.Random(args.seed)
    ok = skipped = 0
    manifest_lines: list[str] = []
    record_lines: list[str] = []

    for i in range(args.count):
        design = sample_design(rng, i)
        try:
            res = run_case(design)
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            print(f"  [{i}] failed: {exc}")
            skipped += 1
            continue
        if res is None:
            skipped += 1
            continue

        traj = res.pop("_traj")
        points, fields = to_shard(traj)

        case_id = hashlib.sha1(
            json.dumps(design, sort_keys=True).encode()
        ).hexdigest()[:16]
        shard_path = args.out / f"{case_id}.npz"
        np.savez_compressed(
            shard_path,
            points=points,
            fields=fields,
            max_stress=np.float32(res["max_q_pa"]),
        )

        manifest_lines.append(json.dumps({
            "part_id": f"part:traj_{case_id}",
            "case_id": case_id,
            "kind": "traj",
            "shard_path": str(shard_path),
            "metrics": {
                "isp_vac_s": res["isp_vac_s"],
                "thrust_sl_n": res["thrust_sl_n"],
                "burn_time_s": res["burn_time_s"],
                "delta_v_ideal_m_s": res["delta_v_ideal_m_s"],
                "apogee_km": res["apogee_km"],
                "downrange_km": res["downrange_km"],
                "max_q_pa": res["max_q_pa"],
                "payload_kg": design["payload_kg"],
                # carried so the graph ingest can populate conditioning slots
                "chamber_pressure_pa": design["chamber_pressure_pa"],
                "expansion_ratio": design["expansion_ratio"],
                "mdot_kg_s": res["mdot_kg_s"],
                "mass_fraction": design["prop_mass_kg"] / design["liftoff_mass_kg"],
                "cd": design["cd"],
                # Nose shape reaches the model through here or not at all. The
                # graph ingest seeds conditioning from a node's own properties
                # by slot name, so a slot whose name never appears in these
                # metrics stays at zero however carefully it was computed --
                # which is what nose_wave_factor was doing.
                "fineness_ratio": design["fineness_ratio"],
                "nose_wave_factor": design["nose_wave_factor"],
                # Slot-named duplicates. The conditioning slots are delta_v_ms
                # and max_dynamic_pressure_kpa; this corpus was emitting
                # delta_v_ideal_m_s and max_q_pa, so the lookup by slot name
                # found nothing and both conditioned on zero -- including the
                # mission delta-v, which is the single most descriptive number
                # about a launch vehicle. The kPa one is also a unit change,
                # which a name-only fix would have got wrong by 1000x.
                "delta_v_ms": res["delta_v_ideal_m_s"],
                "max_dynamic_pressure_kpa": res["max_q_pa"] / 1000.0,
                # Chemistry and thermal, under their conditioning slot names.
                # These slots existed and were populated only by values sampled
                # from family ranges; they now carry numbers computed for this
                # specific design, which is a different kind of training signal.
                **({} if design.get("mixture_ratio") is None
                   else {"mixture_ratio": design["mixture_ratio"]}),
                **({} if not design.get("thermal") else {
                    "throat_heat_flux_MWm2":
                        design["thermal"]["throat_heat_flux_MWm2"],
                    "wall_temp_max_K": design["thermal"]["wall_temp_max_K"],
                }),
                "channels": "thrust,mass,drag,mach,q,accel,velocity,altitude",
            },
        }))
        record_lines.append(json.dumps({"case_id": case_id, "design": design, "outcomes": res}))
        ok += 1

    # Write via a temporary file and rename, so a reader never sees a
    # half-written corpus and an interrupted run leaves the old one intact.
    for path, lines in ((args.manifest, manifest_lines),
                        (args.records, record_lines)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        tmp.replace(path)

    print(f"generated  : {ok}")
    print(f"skipped    : {skipped} (thrust below liftoff weight)")
    print(f"shards     : {args.out}")
    print(f"manifest   : {args.manifest}")
    print(f"records    : {args.records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
