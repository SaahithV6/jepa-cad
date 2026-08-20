#!/usr/bin/env python3
"""Build a corpus of (mission specification -> complete vehicle) pairs.

The confirmed-design corpus answers "make me an airframe section with these
dimensions". This one answers the question the project actually exists to
answer: "deliver x kg to y km", where the design that comes back has to include
the propulsion sizing and mass budget that decide whether the mission closes,
not just geometry.

Each record is verified on two disciplines:

  trajectory  the vehicle is integrated through a gravity turn over an
              exponential atmosphere and must actually reach the altitude
              claimed in its own specification
  structural  the airframe section is meshed and solved in CalculiX under the
              vehicle's own liftoff thrust

Scale is sounding-rocket class (payload 1-50 kg, gross 20-500 kg) so the
airframe stays in the same size range as the structural corpus. A 300 t
launcher would need a metre-scale airframe, which the CAD path and the
decoder's output scales are not set up for.
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
    G0,
    P0,
    PROPELLANTS,
    integrate_trajectory,
    load_coupling,
    nozzle_performance,
    _draw,
)
from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402


def sample_vehicle(rng: random.Random) -> dict:
    prop = rng.choice(list(PROPELLANTS))
    gamma, tc, mol = PROPELLANTS[prop]

    m0 = rng.uniform(20.0, 500.0)                 # gross liftoff, kg
    # Wide payload and structural fractions on purpose. Narrow ranges
    # (payload 1-10%, structure 10-25%) only ever produce high mass ratios, so
    # every vehicle is energetic and the corpus has no low-apogee designs to
    # learn from -- a 95 km request then has almost no neighbours. Heavy
    # payload or heavy structure gives genuinely low-delta-v vehicles.
    payload = m0 * rng.uniform(0.01, 0.35)
    struct_coeff = rng.uniform(0.10, 0.50)
    m_struct = struct_coeff * (m0 - payload)
    m_prop = m0 - payload - m_struct
    if m_prop <= 0.05 * m0:
        m_prop = 0.05 * m0
        m_struct = m0 - payload - m_prop

    pc = rng.uniform(1.5e6, 8.0e6)
    eps = rng.uniform(4.0, 25.0)                  # sea-level-ish stages
    twr = rng.uniform(3.0, 8.0)                   # sounding rockets run high T/W

    guess = nozzle_performance(
        chamber_pressure=pc, chamber_temp=tc, expansion_ratio=eps,
        throat_area=1.0, gamma=gamma, mol_mass=mol, ambient_pressure=P0,
    )
    throat_area = (twr * m0 * G0) / max(guess["thrust"], 1e-6)

    cd, _ = _draw(rng, "cd", 0.25, 0.55)          # CFD-derived drag
    diameter = max(0.08, (m0 / 1000.0) ** (1.0 / 3.0) * rng.uniform(0.35, 0.7))

    return {
        "propellant": prop, "gamma": gamma, "chamber_temp": tc, "mol_mass": mol,
        "liftoff_mass_kg": m0, "payload_kg": payload,
        "struct_mass_kg": m_struct, "prop_mass_kg": m_prop,
        "chamber_pressure_pa": pc, "expansion_ratio": eps,
        "throat_area_m2": throat_area, "cd": cd,
        "diameter_m": diameter, "ref_area_m2": math.pi * (diameter / 2) ** 2,
        # airframe section, independently sampled (see the confirmed-design
        # corpus: dimensions derived from one another cannot be specified)
        "body_radius_mm": round(rng.uniform(20.0, 50.0), 2),
        "body_height_mm": round(rng.uniform(70.0, 200.0), 2),
        "nose_height_mm": round(rng.uniform(20.0, 70.0), 2),
        "fin_span_mm": round(rng.uniform(12.0, 45.0), 2),
        "fin_thickness_mm": round(rng.uniform(3.0, 8.0), 2),
        "fin_chord_mm": round(rng.uniform(15.0, 55.0), 2),
        "fillet_radius_mm": round(rng.uniform(1.2, 3.0), 2),
    }


def fly(v: dict) -> dict | None:
    sl = nozzle_performance(
        chamber_pressure=v["chamber_pressure_pa"], chamber_temp=v["chamber_temp"],
        expansion_ratio=v["expansion_ratio"], throat_area=v["throat_area_m2"],
        gamma=v["gamma"], mol_mass=v["mol_mass"], ambient_pressure=P0,
    )
    if sl["thrust"] <= v["liftoff_mass_kg"] * G0:
        return None
    traj = integrate_trajectory(
        m0=v["liftoff_mass_kg"], m_prop=v["prop_mass_kg"],
        throat_area=v["throat_area_m2"], chamber_pressure=v["chamber_pressure_pa"],
        chamber_temp=v["chamber_temp"], expansion_ratio=v["expansion_ratio"],
        gamma=v["gamma"], mol_mass=v["mol_mass"], cd=v["cd"],
        ref_area=v["ref_area_m2"],
        pitchover_time=8.0, pitchover_angle=math.radians(3.0),
    )
    return {
        "thrust_sl_n": sl["thrust"], "isp_sl_s": sl["isp"],
        "apogee_km": traj["apogee_m"] / 1000.0,
        "downrange_km": traj["downrange_m"] / 1000.0,
        "burn_time_s": v["prop_mass_kg"] / max(sl["mdot"], 1e-9),
        "max_q_pa": traj["max_q_pa"],
    }


def build_prompt(v: dict, o: dict) -> str:
    return (
        f"deliver {v['payload_kg']:.0f} kg payload to {o['apogee_km']:.0f} km apogee, "
        f"{o['downrange_km']:.0f} km downrange, using {v['propellant'].replace('_','/')} "
        f"at {v['chamber_pressure_pa']/1e5:.0f} bar chamber pressure"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--workdir", type=Path, default=ROOT / "artifacts/mission_designs/work")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/mission_designs/corpus.jsonl")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    load_coupling()
    rng = random.Random(args.seed)
    rows: list[str] = []
    n_ok = n_nofly = n_struct = n_err = n_binfull = 0
    t0 = time.time()

    # Rejection-sample so mission outcomes are roughly uniform in log apogee.
    #
    # Sampling designs uniformly and keeping whatever mission falls out gives a
    # corpus skewed across 2.6 decades -- 12.6 to 5,219 km, median 437 -- which
    # puts a 95 km request at the 7.9th percentile with almost no neighbours to
    # learn from. The head then answers such requests from the bulk of the
    # distribution and overshoots by 7-10x. Going from design->outcome data to
    # outcome->design capability requires the outcomes to be covered evenly,
    # not the design parameters.
    #
    # The trajectory integration is cheap and runs first, so rejected designs
    # cost no solver time.
    n_bins = 12
    lo_l, hi_l = math.log10(15.0), math.log10(3000.0)
    per_bin = max(1, args.count // n_bins)
    bins: dict[int, int] = {}
    attempts = 0
    max_attempts = args.count * 40

    while n_ok < args.count and attempts < max_attempts:
        attempts += 1
        i = n_ok
        v = sample_vehicle(rng)
        o = fly(v)
        if o is None:
            n_nofly += 1
            continue

        apo = o["apogee_km"]
        if not (15.0 <= apo <= 3000.0):
            n_binfull += 1
            continue
        b = min(n_bins - 1, int((math.log10(apo) - lo_l) / (hi_l - lo_l) * n_bins))
        if bins.get(b, 0) >= per_bin:
            n_binfull += 1
            continue

        # structural check on the airframe under the vehicle's own thrust
        geom = {
            "body_radius_mm": v["body_radius_mm"], "body_height_mm": v["body_height_mm"],
            "nose_radius_mm": v["body_radius_mm"], "nose_height_mm": v["nose_height_mm"],
            "fin_span_mm": v["fin_span_mm"], "fin_thickness_mm": v["fin_thickness_mm"],
            "fin_chord_mm": v["fin_chord_mm"], "fillet_radius_mm": v["fillet_radius_mm"],
            "cl_max_mm": 8.0, "cl_min_mm": 2.0,
        }
        try:
            rep = run_confirmed(
                params_mm=geom, out=args.workdir / f"m{i:05d}",
                max_stress_mpa=200.0, max_disp_mm=3.0, max_iters=1,
                load_n=o["thrust_sl_n"], prompt=build_prompt(v, o),
            )
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"  [{i}] error: {type(exc).__name__}: {exc}")
            continue

        acc = rep.get("accepted") or {}
        if not acc.get("params_mm"):
            n_struct += 1
            continue

        params = dict(acc["params_mm"])
        params.update({
            "chamber_pressure_bar": v["chamber_pressure_pa"] / 1e5,
            "expansion_ratio": v["expansion_ratio"],
            # log mass ratio rather than propellant mass: see models/cad_decoder.py
            # for why parameter error must be proportional to delta-v error.
            "log_mass_ratio": math.log(
                v["liftoff_mass_kg"] / max(v["struct_mass_kg"] + v["payload_kg"], 1e-6)),
            "struct_mass_kg": v["struct_mass_kg"],
            "payload_kg": v["payload_kg"],
            # Without throat area the vehicle cannot be flown from its own
            # parameters -- thrust-to-weight has to be assumed, and it drives
            # apogee strongly.
            "throat_area_mm2": v["throat_area_m2"] * 1e6,
        })

        rows.append(json.dumps({
            "prompt": build_prompt(v, o),
            "params": params,
            "mission": {
                "payload_kg": v["payload_kg"], "apogee_km": o["apogee_km"],
                "downrange_km": o["downrange_km"], "burn_time_s": o["burn_time_s"],
                "thrust_sl_n": o["thrust_sl_n"], "isp_sl_s": o["isp_sl_s"],
                "propellant": v["propellant"],
            },
            "structural": {
                "max_von_mises_mpa": acc.get("max_von_mises_mpa"),
                "max_displacement_mm": acc.get("max_displacement_mm"),
                "solver_mode": acc.get("solver_mode"),
            },
        }))
        n_ok += 1
        bins[b] = bins.get(b, 0) + 1
        if n_ok % 20 == 0:
            print(f"  {n_ok}/{args.count} attempts={attempts} nofly={n_nofly} "
                  f"struct={n_struct} binfull={n_binfull} err={n_err}  "
                  f"{n_ok/max(time.time()-t0,1e-6):.2f}/s")

    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"\nverified missions : {n_ok}")
    print(f"attempts          : {attempts}")
    print(f"rejected (bin full/range): {n_binfull}")
    print("apogee bin counts : " + ", ".join(f"{k}:{v}" for k, v in sorted(bins.items())))
    print(f"could not lift off: {n_nofly}")
    print(f"structure failed  : {n_struct}")
    print(f"errors            : {n_err}")
    print(f"elapsed           : {(time.time()-t0)/60:.1f} min")
    print(f"corpus            : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
