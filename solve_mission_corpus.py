#!/usr/bin/env python3
"""Solve for the canonical vehicle that meets each mission specification.

The rejection-sampled corpus gives uniform coverage of mission outcomes, but it
still pairs each specification with *one of many* vehicles that happen to reach
that altitude: at a given apogee, log mass ratio spans 0.41 to 2.18 because
payload, propellant, chamber pressure and drag all vary independently.

Mean-squared error converges to the conditional mean of those designs, and
apogee is nonlinear in the parameters, so the mean of designs that each achieve
X does not itself achieve X. The training target is a relation, not a function,
and no loss can fix that -- the model is being asked to predict a set.

This makes it a function. Everything that varies is stated in the prompt, and
the one remaining degree of freedom -- mass ratio -- is *solved* by bisection
until the vehicle actually flies the requested altitude. The result is one
design per specification, and it is the design that meets it.

Bisection is possible because apogee increases monotonically in mass ratio at
fixed everything-else, and the trajectory integrator is cheap enough to call
~40 times per specification.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generate_propulsion_trajectory_corpus import (  # noqa: E402
    G0, P0, PROPELLANTS, integrate_trajectory, load_coupling, nozzle_performance,
)

# Fixed by convention so the specification fully determines the design.
TWR = 5.0
STRUCT_COEFF = 0.15
CD = 0.42


def fly_for(log_mr: float, payload: float, pc: float, eps: float,
            prop: str) -> tuple[float, dict]:
    """Apogee (km) for a vehicle with this log mass ratio."""
    gamma, tc, mol = PROPELLANTS[prop]
    # m_dry = struct + payload, and struct is a fixed fraction of (m0 - payload)
    # m0 = m_dry * MR  ->  solve for the consistent split
    mr = math.exp(log_mr)
    # m_dry = payload + s*(m0 - payload); m0 = m_dry*mr
    #  => m0 = (payload + s*m0 - s*payload) * mr
    #  => m0 (1 - s*mr) = payload*(1-s)*mr
    denom = 1.0 - STRUCT_COEFF * mr
    if denom <= 1e-6:
        return float("inf"), {}
    m0 = payload * (1.0 - STRUCT_COEFF) * mr / denom
    m_dry = m0 / mr
    m_struct = m_dry - payload
    m_prop = m0 - m_dry
    if m_prop <= 0 or m_struct <= 0:
        return float("inf"), {}

    unit = nozzle_performance(chamber_pressure=pc, chamber_temp=tc,
                             expansion_ratio=eps, throat_area=1.0,
                             gamma=gamma, mol_mass=mol, ambient_pressure=P0)
    at = (TWR * m0 * G0) / max(unit["thrust"], 1e-9)
    dia = max(0.08, (m0 / 1000.0) ** (1.0 / 3.0) * 0.5)
    ref = math.pi * (dia / 2) ** 2

    t = integrate_trajectory(m0=m0, m_prop=m_prop, throat_area=at,
        chamber_pressure=pc, chamber_temp=tc, expansion_ratio=eps,
        gamma=gamma, mol_mass=mol, cd=CD, ref_area=ref,
        pitchover_time=8.0, pitchover_angle=math.radians(3.0),
        # Coarser step for the search loop: bisection calls this ~20 times per
        # specification, and targeting accuracy is bounded by the 2% tolerance
        # well before it is bounded by integration step size.
        dt=0.2)
    return t["apogee_m"] / 1000.0, {
        "m0": m0, "m_prop": m_prop, "m_struct": m_struct,
        "throat_area_mm2": at * 1e6, "max_q_pa": t["max_q_pa"],
        "downrange_km": t["downrange_m"] / 1000.0,
    }


def solve(target_km: float, payload: float, pc: float, eps: float,
          prop: str, tol: float = 0.02) -> dict | None:
    """Bisect log mass ratio until the vehicle flies the requested altitude."""
    lo, hi = 0.05, 2.7                      # MR 1.05 .. ~15
    a_lo, _ = fly_for(lo, payload, pc, eps, prop)
    a_hi, _ = fly_for(hi, payload, pc, eps, prop)
    if not (a_lo <= target_km <= a_hi):
        return None                          # target outside reachable band
    best = None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        a, aux = fly_for(mid, payload, pc, eps, prop)
        if not aux:
            hi = mid
            continue
        best = (mid, a, aux)
        if abs(a - target_km) / target_km < tol:
            break
        if a < target_km:
            lo = mid
        else:
            hi = mid
    if best is None:
        return None
    log_mr, achieved, aux = best
    if abs(achieved - target_km) / target_km > 0.10:
        return None                          # did not converge tightly enough
    return {"log_mass_ratio": log_mr, "achieved_km": achieved, **aux}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/mission_designs/solved_corpus.jsonl")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    load_coupling()
    rng = random.Random(args.seed)

    payloads = [1.0, 3.0, 8.0, 20.0, 45.0]
    apogees = [20, 35, 60, 95, 150, 250, 400, 650, 1000, 1600]
    pressures = [35.0, 55.0]
    props = ["lox_rp1", "lox_ch4", "solid_apcp"]

    rows: list[str] = []
    tried = solved = 0
    t0 = time.time()
    for pay in payloads:
        for apo in apogees:
            for pc_bar in pressures:
                for prop in props:
                    tried += 1
                    eps = 12.0
                    r = solve(float(apo), pay, pc_bar * 1e5, eps, prop)
                    if r is None:
                        continue
                    solved += 1
                    if solved % 25 == 0:
                        print(f"  solved {solved}/{tried} tried  "
                              f"{(time.time()-t0)/60:.1f} min", flush=True)
                    m0 = r["m0"]
                    # airframe derived deterministically from vehicle size
                    body_r = max(20.0, min(50.0, 18.0 * (m0 / 100.0) ** (1 / 3)))
                    rows.append(json.dumps({
                        "prompt": (
                            f"deliver {pay:.0f} kg payload to {apo:.0f} km apogee "
                            f"using {prop.replace('_','/')} at {pc_bar:.0f} bar "
                            f"chamber pressure"),
                        "params": {
                            "body_radius_mm": round(body_r, 2),
                            "body_height_mm": round(body_r * 3.4, 2),
                            "nose_radius_mm": round(body_r, 2),
                            "nose_height_mm": round(body_r * 1.3, 2),
                            "fin_span_mm": round(body_r * 0.8, 2),
                            "fin_thickness_mm": 5.0,
                            "fin_chord_mm": round(body_r * 1.0, 2),
                            "fillet_radius_mm": 2.0,
                            "chamber_pressure_bar": pc_bar,
                            "expansion_ratio": eps,
                            "log_mass_ratio": r["log_mass_ratio"],
                            "struct_mass_kg": r["m_struct"],
                            "payload_kg": pay,
                            "throat_area_mm2": r["throat_area_mm2"],
                        },
                        "mission": {
                            "payload_kg": pay, "target_apogee_km": apo,
                            "achieved_apogee_km": r["achieved_km"],
                            "downrange_km": r["downrange_km"],
                            "propellant": prop, "gross_kg": m0,
                        },
                    }))
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"specifications tried : {tried}")
    print(f"solved               : {solved}")
    print(f"elapsed              : {(time.time()-t0)/60:.1f} min")
    print(f"corpus               : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
