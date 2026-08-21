#!/usr/bin/env python3
"""Arbitrary mission specification -> architecture -> vehicle -> verified packet.

Chains every layer that exists:

  planner       chooses stage count and split from the mission's delta-v
  sizing        bisects propellant until the stack flies the requested altitude
  decomposition splits the vehicle into major structural components
  simulation    solves each component in CalculiX under the load it carries
  verification  margin and pass/fail per component, plus the whole-assembly flight
  reporting     writes the packet

Unlike the grid-trained path this takes any payload and altitude, because the
architecture is decided and the vehicle solved at request time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadflow.planner import plan  # noqa: E402
from cadflow.structural_sizing import size_wall  # noqa: E402
from generate_propulsion_trajectory_corpus import load_coupling  # noqa: E402
from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402

ALLOWABLE_MPA = 200.0


def frd_stress_percentiles(case_dir: Path) -> dict | None:
    """Von Mises percentiles from the FRD.

    The absolute maximum is the wrong acceptance metric for a tet mesh with
    sharp re-entrant corners: stress there does not converge with refinement and
    does not relieve when the wall is thickened. Sizing the thrust structure
    from 1.44 to 2.44 mm moved the reported peak from 333 to 3,948 MPa on a
    116 MPa nominal wall -- the peak is tracking the corner, not the load. A
    high percentile describes the structure; the peak flags where a fillet or
    doubler is needed.
    """
    import math as _m
    frds = sorted(case_dir.rglob("case.frd"))
    if not frds:
        return None
    vals = []
    in_stress = False
    for line in open(frds[0], errors="ignore"):
        if "STRESS" in line:
            in_stress = True
            continue
        if in_stress:
            if line.startswith(" -3"):
                break
            if line.startswith(" -1"):
                parts = line.split()
                try:
                    sxx, syy, szz, sxy, syz, szx = (float(x) for x in parts[2:8])
                except (ValueError, IndexError):
                    continue
                vm = _m.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2)
                             + 3.0 * (sxy**2 + syz**2 + szx**2))
                vals.append(vm / 1e6)
    if not vals:
        return None
    vals.sort()
    def pct(q):
        return vals[min(len(vals) - 1, int(q * len(vals)))]
    return {"median": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": vals[-1], "n": len(vals),
            "frac_over_yield": sum(1 for v in vals if v > 276.0) / len(vals)}


def component_specs(body_r_mm: float, stages, gross_kg: float, max_q_pa: float,
                    payload_kg: float):
    frontal = math.pi * (body_r_mm / 1000.0) ** 2
    thrust1 = gross_kg * 9.80665 * 4.5
    specs = [
        ("nose cone", "aerodynamic pressure at max-Q",
         max(200.0, max_q_pa * frontal * 1.8),
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 1.2,
              nose_height_mm=body_r_mm * 1.6, fin_thickness_mm=4.0)),
        ("thrust structure", "engine thrust into the aft ring",
         thrust1 * 1.3,
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 1.2,
              nose_height_mm=body_r_mm * 0.5, fin_thickness_mm=6.5)),
        ("fin set", "aerodynamic side load at max-Q",
         max(200.0, max_q_pa * frontal * 0.9),
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 2.0,
              nose_height_mm=body_r_mm * 0.7, fin_thickness_mm=5.0)),
    ]
    # one tank and one interstage per stage
    for i, st in enumerate(stages):
        supported = payload_kg + sum(s.prop_mass_kg + s.struct_mass_kg
                                     for s in stages[i:])
        specs.append((f"stage {i+1} tank",
                      f"carries {st.prop_mass_kg:.1f} kg propellant under thrust",
                      supported * 9.80665 * 4.5,
                      dict(body_radius_mm=body_r_mm * (1.0 - 0.08 * i),
                           body_height_mm=body_r_mm * 3.5,
                           nose_height_mm=body_r_mm * 0.8,
                           fin_thickness_mm=5.5)))
        if i < len(stages) - 1:
            specs.append((f"interstage {i+1}/{i+2}",
                          "transmits lower-stage thrust to the stage above",
                          supported * 9.80665 * 4.5,
                          dict(body_radius_mm=body_r_mm * (1.0 - 0.08 * i),
                               body_height_mm=body_r_mm * 1.5,
                               nose_height_mm=body_r_mm * 0.6,
                               fin_thickness_mm=5.5)))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload-kg", type=float, required=True)
    ap.add_argument("--apogee-km", type=float, required=True)
    ap.add_argument("--propellant", type=str, default="lox_rp1")
    ap.add_argument("--chamber-bar", type=float, default=55.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/plan_packet")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_coupling()

    spec = (f"deliver {args.payload_kg:.0f} kg payload to {args.apogee_km:.0f} km "
            f"apogee using {args.propellant.replace('_','/')} at "
            f"{args.chamber_bar:.0f} bar chamber pressure")

    p = plan(args.apogee_km, args.payload_kg,
             propellant=args.propellant, chamber_bar=args.chamber_bar)
    if p is None:
        print(f"# Design packet\n\n**Specification:** {spec}\n")
        print("No architecture up to 3 stages closes this mission.")
        return 1

    body_r = max(20.0, min(50.0, 16.0 * (p.gross_kg / 100.0) ** (1 / 3)))
    results = []
    for name, why, load, geo in component_specs(
            body_r, p.stack, p.gross_kg, p.trajectory["max_q_pa"], args.payload_kg):
        # Size the wall first, then mesh THAT shell. Previously the component
        # was meshed solid and the FEA reported margins of 200-300x, which is a
        # property of a billet rather than of the structure being designed.
        wall = size_wall(load, geo["body_radius_mm"] / 1000.0,
                         geo["body_height_mm"] / 1000.0)
        geom = {"body_radius_mm": geo["body_radius_mm"],
                "body_height_mm": geo["body_height_mm"],
                "nose_radius_mm": geo["body_radius_mm"],
                "nose_height_mm": geo["nose_height_mm"],
                "fin_span_mm": 14.0, "fin_thickness_mm": geo["fin_thickness_mm"],
                "fin_chord_mm": 18.0, "fillet_radius_mm": 2.5,
                "wall_thickness_mm": wall.thickness_m * 1000.0,
                # Mesh sizing has to follow the wall: a thin shell needs about
                # three elements through the thickness or gmsh fails with
                # "PLC Error: a segment and a facet intersect". At 8/2 mm on a
                # 1.11 mm wall it failed outright; at 1.5/0.4 it meshes to
                # 51,403 nodes.
                # Element size floors. Scaling purely with the wall drove
                # cl_min to 0.27 mm on a 0.8 mm shell, which on a 70 mm part is
                # millions of tets -- CalculiX then died with "Failed during
                # initial partitioning" and the component returned nothing.
                # 0.4 mm still gives two elements through the thinnest wall
                # here while keeping the model solvable.
                "cl_max_mm": max(0.8, wall.thickness_m * 1000.0 * 1.4),
                "cl_min_mm": max(0.25, wall.thickness_m * 1000.0 / 3.0)}
        try:
            rep = run_confirmed(params_mm=geom,
                out=args.out / "components" / name.replace(" ", "_").replace("/", "-"),
                max_stress_mpa=ALLOWABLE_MPA, max_disp_mm=3.0, max_iters=1,
                load_n=load, prompt=name)
            acc = rep.get("accepted") or {}
            vm, ok = acc.get("max_von_mises_mpa"), bool(acc.get("targets_met"))
            # If the mesher fell back to a convex hull the geometry solved was a
            # solid billet, not this shell, so the stress means nothing here.
            comp_dir = args.out / "components" / name.replace(" ", "_").replace("/", "-")
            if any(comp_dir.rglob("MESH_IS_CONVEX_HULL")):
                vm, ok, hulled = None, False, True
            else:
                hulled = False
            err = None
        except Exception as exc:  # noqa: BLE001
            # Record why. Swallowing this left two components reporting a bare
            # "-" with empty output directories and no way to tell a mesh
            # failure from a solver failure.
            vm, ok, hulled = None, False, False
            err = f"{type(exc).__name__}: {exc}"
            print(f"  [{name}] {err}", flush=True)
        comp_dir = args.out / "components" / name.replace(" ", "_").replace("/", "-")
        dist = None if hulled else frd_stress_percentiles(comp_dir)
        if dist is not None:
            # judge on the field, not the corner
            ok = dist["p99"] <= ALLOWABLE_MPA
            vm = dist["p99"]
        results.append({"name": name, "why": why, "load_n": load,
                        "error": err,
                        "mesh_was_hull": hulled, "stress_dist": dist,
                        "shell_von_mises_mpa": vm, "coupon_passed": ok,
                        "coupon_margin": (ALLOWABLE_MPA / vm) if vm else None,
                        "wall_mm": wall.thickness_m * 1000.0,
                        "wall_mass_kg": wall.mass_kg,
                        "wall_driver": wall.driver,
                        "buckling_margin": wall.margin_buckling,
                        "passed": ok and wall.margin_buckling >= 1.0})

    L = [f"# Design packet\n", f"**Specification:** {spec}\n", "## Architecture\n"]
    for line in p.rationale:
        L.append(f"- {line}")
    L.append(f"\n## Vehicle: {p.stages} stage(s)\n")
    L.append("| stage | propellant | structure | expansion |")
    L.append("|---|---|---|---|")
    for i, st in enumerate(p.stack):
        L.append(f"| {i+1} | {st.prop_mass_kg:.2f} kg | "
                 f"{st.struct_mass_kg:.2f} kg | {st.expansion_ratio:.0f} |")
    L.append(f"\npayload {args.payload_kg:.1f} kg, gross {p.gross_kg:.1f} kg\n")
    L.append("## Mission verification\n")
    err = abs(p.achieved_km - args.apogee_km) / args.apogee_km * 100
    seps = ", ".join(f"{s:.1f} s" for s in p.trajectory["separations"]) or "none"
    L.append(f"Flown apogee **{p.achieved_km:.1f} km** against "
             f"{args.apogee_km:.0f} km requested (**{err:.1f}%** error); "
             f"downrange {p.trajectory['downrange_m']/1000:.1f} km, "
             f"max-Q {p.trajectory['max_q_pa']/1000:.1f} kPa, "
             f"separations {seps}.\n")
    L.append("## Component verification\n")
    L.append("| component | load case | load | wall | driver | buckling margin |"
             " shell p99 | peak | status |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        vm = (f"{r['shell_von_mises_mpa']:.1f} MPa"
              if r["shell_von_mises_mpa"] else "-")
        d = r.get("stress_dist")
        peak = f"{d['max']:.0f} MPa" if d else ("hull" if r["mesh_was_hull"] else "-")
        L.append(f"| {r['name']} | {r['why']} | {r['load_n']:.0f} N | "
                 f"{r['wall_mm']:.2f} mm | {r['wall_driver']} | "
                 f"{r['buckling_margin']:.2f}x | {vm} | {peak} | "
                 f"{'PASS' if r['passed'] else 'FAIL'} |")
    L.append("\nWall thickness and buckling margin size the thin shell; the "
             "shell FEA column is CalculiX on that same hollow geometry, so the "
             "meshed part is the part being designed rather than a solid billet "
             "with the same outer dimensions.")
    allp = all(r["passed"] for r in results)
    L.append(f"\nAllowable {ALLOWABLE_MPA:.0f} MPa. "
             f"All {len(results)} components passed: **{allp}**\n")

    (args.out / "PACKET.md").write_text("\n".join(L))
    (args.out / "PACKET.json").write_text(json.dumps({
        "specification": spec, "stages": p.stages, "split": p.split,
        "gross_kg": p.gross_kg, "achieved_km": p.achieved_km,
        "error_pct": err, "rationale": p.rationale,
        "components": results, "all_passed": allp}, indent=2))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
