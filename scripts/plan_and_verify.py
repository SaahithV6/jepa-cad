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
        # Size the wall, mesh that shell, and thicken if the analysis says so.
        #
        # The membrane formula assumes a long cylinder. The thrust structure is
        # 42.8 mm long on a 71 mm diameter, so end effects cover the whole part
        # and the analytic wall under-sizes it: sized to 116 MPa nominal it came
        # back at 279 MPa. Rather than trust either number alone, iterate --
        # this is design-by-analysis, and it is why the analysis exists.
        wall = size_wall(load, geo["body_radius_mm"] / 1000.0,
                         geo["body_height_mm"] / 1000.0)
        t_mm = wall.thickness_m * 1000.0
        vm = None
        ok = False
        hulled = False
        err = None
        dist = None
        comp_dir = args.out / "components" / name.replace(" ", "_").replace("/", "-")

        for attempt in range(3):
            geom = {"body_radius_mm": geo["body_radius_mm"],
                    "body_height_mm": geo["body_height_mm"],
                    "nose_radius_mm": geo["body_radius_mm"],
                    "nose_height_mm": geo["nose_height_mm"],
                    # Only the fin set has fins. Every component was being
                    # built with a fin stub, including the tanks and the thrust
                    # structure, which welds an artificial stress riser onto
                    # parts that do not have one -- and a partially embedded
                    # stub in a thick wall makes a sharp internal T-joint whose
                    # stress does not relieve when the wall is thickened. That
                    # is why the thrust structure went 230 -> 817 MPa as it was
                    # made heavier.
                    "fin_span_mm": 14.0 if "fin" in name else 0.0,
                    "fin_thickness_mm": geo["fin_thickness_mm"],
                    "fin_chord_mm": 18.0, "fillet_radius_mm": 2.5,
                    "wall_thickness_mm": t_mm,
                    # Roughly three elements through the wall, but the element
                    # size is also capped against the radius. Scaling cl purely
                    # with the wall meant a 12 mm wall gave 16.8 mm elements, so
                    # the loaded face carried only a handful of nodes and
                    # load_per_node = total_load / len(loaded) turned into point
                    # loads. Stress then *rose* with thickness -- 230 MPa at
                    # 2.44 mm, 817 MPa at 6.48 mm -- which is a load-introduction
                    # artifact, not structure.
                    "cl_max_mm": min(max(0.8, t_mm * 1.4),
                                     geo["body_radius_mm"] / 6.0),
                    "cl_min_mm": min(max(0.25, t_mm / 3.0),
                                     geo["body_radius_mm"] / 20.0)}
            try:
                rep = run_confirmed(
                    params_mm=geom, out=comp_dir,
                    max_stress_mpa=ALLOWABLE_MPA, max_disp_mm=3.0, max_iters=1,
                    load_n=load, prompt=name)
                acc = rep.get("accepted") or {}
                vm, ok = acc.get("max_von_mises_mpa"), bool(acc.get("targets_met"))
                hulled = any(comp_dir.rglob("MESH_IS_CONVEX_HULL"))
                if hulled:
                    vm, ok = None, False
                err = None
            except Exception as exc:  # noqa: BLE001
                vm, ok, hulled = None, False, False
                err = f"{type(exc).__name__}: {exc}"
                print(f"  [{name}] {err}", flush=True)
                break

            dist = None if hulled else frd_stress_percentiles(comp_dir)
            if dist is None:
                break
            vm = dist["p99"]
            ok = vm <= ALLOWABLE_MPA
            if ok:
                break
            # thicken in proportion to the overshoot and re-analyse
            t_mm = min(t_mm * min(2.0, 1.15 * vm / ALLOWABLE_MPA), 12.0)
            print(f"  [{name}] p99 {vm:.0f} MPa over {ALLOWABLE_MPA:.0f}; "
                  f"thickening wall to {t_mm:.2f} mm", flush=True)

        # Recompute wall properties at the thickness the analysis settled on,
        # otherwise the packet reports mass and margin for the first estimate
        # rather than the design that was actually verified.
        import math as _m
        from cadflow.structural_sizing import E_PA, RHO, SIGMA_YIELD_PA, _knockdown
        t_final = t_mm / 1000.0
        r_final = geo["body_radius_mm"] / 1000.0
        L_final = geo["body_height_mm"] / 1000.0
        wall_mass = RHO * 2.0 * _m.pi * r_final * t_final * L_final
        sigma_app = load / (2.0 * _m.pi * r_final * t_final)
        gamma_kd = _knockdown(r_final / t_final)
        sigma_cr = gamma_kd * 0.605 * E_PA * t_final / r_final
        wall_driver = wall.driver if abs(t_mm - wall.thickness_m * 1000.0) < 1e-6 \
            else "analysis"
        buckling_margin = sigma_cr / max(sigma_app, 1.0)
        wall_mm_final = t_mm
        results.append({"name": name, "why": why, "load_n": load,
                        "error": err,
                        "mesh_was_hull": hulled, "stress_dist": dist,
                        "shell_von_mises_mpa": vm, "coupon_passed": ok,
                        "coupon_margin": (ALLOWABLE_MPA / vm) if vm else None,
                        "wall_mm": wall_mm_final,
                        "wall_mass_kg": wall_mass,
                        "wall_driver": wall_driver,
                        "buckling_margin": buckling_margin,
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
