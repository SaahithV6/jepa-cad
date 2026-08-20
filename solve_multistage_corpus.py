#!/usr/bin/env python3
"""Solve for two-stage vehicles meeting a mission specification.

The single-stage solver runs out around a few hundred kilometres: everything
carried to burnout is carried the whole way, so the required mass ratio grows
exponentially and the 1600 km specifications simply had no solution at a fixed
structural coefficient. That is why the single-stage model failed at the top of
its range -- not a training problem, a physics one.

Staging drops spent structure. Same solving approach as the single-stage
corpus: everything that varies is stated in the prompt, and the one remaining
degree of freedom -- total mass ratio, with a fixed inter-stage split -- is
bisected until the stack actually flies the requested altitude. One design per
specification, and it is the design that meets it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cadflow.multistage import Stage, integrate_stack  # noqa: E402
from generate_propulsion_trajectory_corpus import (  # noqa: E402
    G0, P0, PROPELLANTS, load_coupling, nozzle_performance,
)

STRUCT_COEFF = 0.14      # per stage, structure / (structure + propellant)
STAGE1_FRACTION = 0.72   # share of total propellant in the booster
TWR1, TWR2 = 4.5, 3.0
EPS1, EPS2 = 12.0, 30.0  # booster sea-level-ish, upper stage vacuum
CD = 0.42


def build_stack(total_prop: float, payload: float, pc: float, prop: str):
    gamma, tc, mol = PROPELLANTS[prop]
    p1 = total_prop * STAGE1_FRACTION
    p2 = total_prop - p1
    s1_struct = p1 * STRUCT_COEFF / (1 - STRUCT_COEFF)
    s2_struct = p2 * STRUCT_COEFF / (1 - STRUCT_COEFF)

    m_above_2 = payload
    m_above_1 = payload + p2 + s2_struct
    gross = m_above_1 + p1 + s1_struct

    def throat(pc_, eps, m_supported, twr):
        u = nozzle_performance(chamber_pressure=pc_, chamber_temp=tc,
                               expansion_ratio=eps, throat_area=1.0,
                               gamma=gamma, mol_mass=mol, ambient_pressure=P0)
        return (twr * m_supported * G0) / max(u["thrust"], 1e-9)

    s1 = Stage(p1, s1_struct, throat(pc, EPS1, gross, TWR1), pc, EPS1, gamma, tc, mol)
    s2 = Stage(p2, s2_struct, throat(pc, EPS2, m_above_1, TWR2), pc, EPS2, gamma, tc, mol)
    return [s1, s2], gross, (p1, s1_struct, p2, s2_struct)


def fly(total_prop: float, payload: float, pc: float, prop: str):
    stages, gross, split = build_stack(total_prop, payload, pc, prop)
    dia = max(0.10, (gross / 1000.0) ** (1.0 / 3.0) * 0.55)
    r = integrate_stack(stages, payload, cd=CD,
                        ref_area_m2=math.pi * (dia / 2) ** 2, dt=0.2)
    return r["apogee_m"] / 1000.0, gross, split, r


def solve(target_km: float, payload: float, pc: float, prop: str,
          tol: float = 0.02) -> dict | None:
    lo, hi = payload * 0.5, payload * 4000.0
    a_lo, *_ = fly(lo, payload, pc, prop)
    a_hi, *_ = fly(hi, payload, pc, prop)
    if not (a_lo <= target_km <= a_hi):
        return None
    # Keep the closest iterate, not the latest: a probe landing on an escape
    # trajectory has infinite apogee and would otherwise discard a converged
    # solve.
    best = None
    best_err = float("inf")
    for _ in range(30):
        mid = math.sqrt(lo * hi)            # geometric bisection: mass spans decades
        a, gross, split, r = fly(mid, payload, pc, prop)
        err = abs(a - target_km) / target_km if math.isfinite(a) else float("inf")
        if err < best_err:
            best_err, best = err, (mid, a, gross, split, r)
        if err < tol:
            break
        if a < target_km:
            lo = mid
        else:
            hi = mid
    if best is None or best_err > 0.10:
        return None
    tp, achieved, gross, split, r = best
    p1, s1, p2, s2 = split
    stages_built, _, _ = build_stack(tp, payload, pc, prop)
    return {
        "total_prop_kg": tp, "achieved_km": achieved, "gross_kg": gross,
        "stage1_prop_kg": p1, "stage1_struct_kg": s1,
        "stage2_prop_kg": p2, "stage2_struct_kg": s2,
        "log_mass_ratio": math.log(gross / max(gross - tp, 1e-6)),
        "downrange_km": r["downrange_m"] / 1000.0,
        "separations": r["separations"],
        # Real throat areas. These were emitted as a 0.0 placeholder with a
        # "filled below" comment and never filled, so every record in the
        # corpus taught the decoder that the throat area is zero. The
        # evaluation did not catch it because the flight path recomputes the
        # throat from thrust-to-weight and ignores the predicted value.
        "stage1_throat_mm2": stages_built[0].throat_area_m2 * 1e6,
        "stage2_throat_mm2": stages_built[1].throat_area_m2 * 1e6,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/mission_designs/multistage_corpus.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    load_coupling()

    payloads = [1.0, 3.0, 8.0, 20.0, 45.0, 100.0]
    # Extended past 12,000 km. The trained model failed badly exactly at the
    # old top of range (12,000 km asked, 318 km flown) because that point was
    # the edge of the corpus and it was extrapolating. Everything inside the
    # range landed within 22%.
    apogees = [50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 25000, 50000]
    pressures = [35.0, 55.0, 80.0]
    props = ["lox_rp1", "lox_ch4", "solid_apcp"]

    rows: list[str] = []
    tried = solved = 0
    t0 = time.time()
    for pay in payloads:
        for apo in apogees:
            for pc_bar in pressures:
                for prop in props:
                    tried += 1
                    r = solve(float(apo), pay, pc_bar * 1e5, prop)
                    if r is None:
                        continue
                    solved += 1
                    if solved % 50 == 0:
                        print(f"  solved {solved}/{tried}  "
                              f"{(time.time()-t0)/60:.1f} min", flush=True)
                    body_r = max(20.0, min(50.0, 16.0 * (r["gross_kg"] / 100.0) ** (1/3)))
                    rows.append(json.dumps({
                        "prompt": (
                            f"two-stage vehicle delivering {pay:.0f} kg payload to "
                            f"{apo:.0f} km apogee using {prop.replace('_','/')} at "
                            f"{pc_bar:.0f} bar chamber pressure"),
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
                            "expansion_ratio": EPS1,
                            "log_mass_ratio": r["log_mass_ratio"],
                            "struct_mass_kg": r["stage1_struct_kg"] + r["stage2_struct_kg"],
                            "payload_kg": pay,
                            "log_throat_area_mm2": math.log(
                                max(r["stage1_throat_mm2"], 1e-6)),
                            "stage1_prop_kg": r["stage1_prop_kg"],
                            "stage2_prop_kg": r["stage2_prop_kg"],
                        },
                        "mission": {
                            "payload_kg": pay, "target_apogee_km": apo,
                            "achieved_apogee_km": r["achieved_km"],
                            "downrange_km": r["downrange_km"],
                            "gross_kg": r["gross_kg"], "propellant": prop,
                            "stages": 2, "separations_s": r["separations"],
                        },
                    }))
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"\nspecifications tried : {tried}")
    print(f"solved               : {solved}")
    print(f"elapsed              : {(time.time()-t0)/60:.1f} min")
    print(f"corpus               : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
