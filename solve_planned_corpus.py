#!/usr/bin/env python3
"""Build the mission corpus through the planner, not a second solver.

solve_multistage_corpus.py carried its own simplified solver: fixed two stages,
fixed 0.72 propellant split, fixed 3 degree pitchover, and a propellant bracket
of payload x 4000. cadflow/planner.py meanwhile chooses the stage count,
searches the split and the ascent profile, and brackets far wider.

Two solvers for the same problem is how they drift apart, and they had: the
corpus generator could not solve anything above ~6,000 km, so the corpus topped
out there while the planner reached 200,000 km. The model then failed exactly at
the corpus edge -- 12,000 km asked, 318 km flown -- because it was extrapolating
past what its training data could express.

This generates the corpus by calling the planner, so the corpus inherits
whatever the planner can do, and there is one implementation to fix.
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

from cadflow.planner import plan  # noqa: E402
from generate_propulsion_trajectory_corpus import load_coupling  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/mission_designs/planned_corpus.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    load_coupling()

    payloads = [1.0, 3.0, 8.0, 20.0, 45.0, 100.0]
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
                    p = plan(float(apo), pay, propellant=prop,
                             chamber_bar=pc_bar)
                    if p is None:
                        continue
                    solved += 1
                    if solved % 40 == 0:
                        print(f"  solved {solved}/{tried}  "
                              f"{(time.time()-t0)/60:.1f} min", flush=True)

                    total_prop = sum(s.prop_mass_kg for s in p.stack)
                    total_struct = sum(s.struct_mass_kg for s in p.stack)
                    m_dry = total_struct + pay
                    body_r = max(20.0, min(50.0,
                                           16.0 * (p.gross_kg / 100.0) ** (1 / 3)))
                    rows.append(json.dumps({
                        "prompt": (
                            f"{p.stages}-stage vehicle delivering {pay:.0f} kg payload "
                            f"to {apo:.0f} km apogee using {prop.replace('_','/')} at "
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
                            "expansion_ratio": p.stack[0].expansion_ratio,
                            "log_mass_ratio": math.log(
                                p.gross_kg / max(m_dry, 1e-6)),
                            "struct_mass_kg": total_struct,
                            "payload_kg": pay,
                            "log_throat_area_mm2": math.log(
                                max(p.stack[0].throat_area_m2 * 1e6, 1e-6)),
                        },
                        "mission": {
                            "payload_kg": pay, "target_apogee_km": apo,
                            "achieved_apogee_km": p.achieved_km,
                            "gross_kg": p.gross_kg, "propellant": prop,
                            "stages": p.stages, "split": p.split,
                            "separations_s": p.trajectory.get("separations", []),
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
