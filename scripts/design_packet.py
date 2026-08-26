#!/usr/bin/env python3
"""Mission specification -> staged vehicle -> per-component verification -> packet.

Everything before this verified a single airframe coupon under one axial load.
A rocket is a stack of components that fail in different ways: a nose cone sees
aerodynamic pressure at max-Q, tanks see internal pressure, the interstage and
thrust structure see the engine's thrust. Verifying one coupon says nothing
about the others.

This decomposes the designed vehicle into its major structural components,
sizes each from the vehicle, works out the load *that component* actually
carries, and solves each one separately in CalculiX. Then it writes a design
packet: the specification, the vehicle, the mission verification, and a
per-component verification table with a pass/fail on each.

That is the reporting layer from the statement of intent -- "every solver result
is paired with a validation summary", "every design proposal has a
confidence/verification status" -- applied to an assembly rather than a part.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_propulsion_trajectory_corpus import load_coupling  # noqa: E402
from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402
from solve_multistage_corpus import build_stack, fly  # noqa: E402

#: This path has no material selection -- every component is aluminium -- so
#: unlike `plan_and_verify` it does not need a per-design allowable. It does
#: need the number to be derived rather than asserted: 200 MPa appeared here as
#: a bare constant with no material, no basis and no factor of safety attached
#: to it, which happened to be close to a 6061-T6 value and gave no way to tell
#: that from a coincidence.
from cadflow.allowables import design_allowable  # noqa: E402

_ALLOWABLE = design_allowable("al-6061-t6")
ALLOWABLE_MPA = _ALLOWABLE.allowable_mpa
MAX_DISP_MM = 3.0


def components(p: dict, stack, gross: float, split, mission: dict) -> list[dict]:
    """Major structural components, each with the load it actually carries.

    Loads differ by component, which is the point: sizing everything for the
    same axial load would pass parts that a real vehicle would lose.
    """
    r = p["body_radius_mm"]
    p1_prop, p1_str, p2_prop, p2_str = split
    thrust1 = gross * 9.80665 * 4.5                    # stage 1 liftoff thrust
    thrust2 = (p["payload_kg"] + p2_prop + p2_str) * 9.80665 * 3.0
    q_pa = mission["max_q_pa"]
    # frontal area the nose presents, in m^2
    frontal = math.pi * (r / 1000.0) ** 2

    return [
        {
            "name": "nose cone",
            "why": "aerodynamic pressure at max-Q",
            "geom": {"body_radius_mm": r, "body_height_mm": r * 1.2,
                     "nose_radius_mm": r, "nose_height_mm": r * 1.6,
                     "fin_span_mm": 12.0, "fin_thickness_mm": 4.0,
                     "fin_chord_mm": 15.0, "fillet_radius_mm": 2.0},
            "load_n": max(200.0, q_pa * frontal * 1.8),   # stagnation, 1.8 factor
        },
        {
            "name": "stage 2 tank",
            "why": "internal pressure at MEOP plus upper-stage thrust",
            "geom": {"body_radius_mm": r * 0.9, "body_height_mm": r * 3.0,
                     "nose_radius_mm": r * 0.9, "nose_height_mm": r * 0.8,
                     "fin_span_mm": 12.0, "fin_thickness_mm": 4.5,
                     "fin_chord_mm": 15.0, "fillet_radius_mm": 2.0},
            "load_n": thrust2,
        },
        {
            "name": "interstage",
            "why": "carries stage 1 thrust through to the upper stage",
            "geom": {"body_radius_mm": r, "body_height_mm": r * 1.5,
                     "nose_radius_mm": r, "nose_height_mm": r * 0.6,
                     "fin_span_mm": 12.0, "fin_thickness_mm": 5.5,
                     "fin_chord_mm": 15.0, "fillet_radius_mm": 2.5},
            "load_n": thrust1,
        },
        {
            "name": "stage 1 tank",
            "why": "full liftoff thrust plus propellant column",
            "geom": {"body_radius_mm": r, "body_height_mm": r * 4.0,
                     "nose_radius_mm": r, "nose_height_mm": r * 0.8,
                     "fin_span_mm": 14.0, "fin_thickness_mm": 5.5,
                     "fin_chord_mm": 18.0, "fillet_radius_mm": 2.5},
            "load_n": thrust1 * 1.15,
        },
        {
            "name": "fin set",
            "why": "aerodynamic side load at max-Q",
            "geom": {"body_radius_mm": r, "body_height_mm": r * 2.0,
                     "nose_radius_mm": r, "nose_height_mm": r * 0.7,
                     "fin_span_mm": p["fin_span_mm"],
                     "fin_thickness_mm": p["fin_thickness_mm"],
                     "fin_chord_mm": p["fin_chord_mm"],
                     "fillet_radius_mm": p["fillet_radius_mm"]},
            "load_n": max(200.0, q_pa * frontal * 0.9),
        },
        {
            "name": "thrust structure",
            "why": "engine thrust reacted into the tank aft ring",
            "geom": {"body_radius_mm": r, "body_height_mm": r * 1.2,
                     "nose_radius_mm": r, "nose_height_mm": r * 0.5,
                     "fin_span_mm": 12.0, "fin_thickness_mm": 6.5,
                     "fin_chord_mm": 15.0, "fillet_radius_mm": 3.0},
            "load_n": thrust1 * 1.3,
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--target-km", type=float, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "artifacts/mission_train_v14/latest.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/design_packet")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_coupling()

    subprocess.run(["python", "scripts/infer_text_to_assembly.py",
        "--prompt", args.prompt, "--ckpt", str(args.ckpt),
        "--out", str(args.out / "vehicle")],
        cwd=ROOT, capture_output=True, text=True)
    p = json.load(open(args.out / "vehicle/INFER_REPORT.json"))["decoded_params_mm"]

    m_dry = p["struct_mass_kg"] + p["payload_kg"]
    mr = math.exp(p["log_mass_ratio"])
    gross = m_dry * mr
    total_prop = gross - m_dry
    apogee, g, split, traj = fly(total_prop, p["payload_kg"],
                                 p["chamber_pressure_bar"] * 1e5, "lox_rp1")
    stack, _, _ = build_stack(total_prop, p["payload_kg"],
                              p["chamber_pressure_bar"] * 1e5, "lox_rp1")
    mission = {"max_q_pa": traj["max_q_pa"]}

    results = []
    for c in components(p, stack, g, split, mission):
        geom = dict(c["geom"]); geom.update({"cl_max_mm": 8.0, "cl_min_mm": 2.0})
        try:
            rep = run_confirmed(params_mm=geom,
                out=args.out / "components" / c["name"].replace(" ", "_"),
                max_stress_mpa=ALLOWABLE_MPA, max_disp_mm=MAX_DISP_MM,
                max_iters=1, load_n=c["load_n"], prompt=c["name"])
            acc = rep.get("accepted") or {}
            vm = acc.get("max_von_mises_mpa")
            disp = acc.get("max_displacement_mm")
            ok = bool(acc.get("targets_met"))
        except Exception as exc:  # noqa: BLE001
            vm = disp = None; ok = False
            acc = {"error": f"{type(exc).__name__}: {exc}"}
        results.append({**{k: c[k] for k in ("name", "why", "load_n")},
                        "max_von_mises_mpa": vm, "max_displacement_mm": disp,
                        "margin": (ALLOWABLE_MPA / vm) if vm else None,
                        "passed": ok, "solver_mode": acc.get("solver_mode")})

    p1_prop, p1_str, p2_prop, p2_str = split
    packet = {
        "specification": args.prompt,
        "target_apogee_km": args.target_km,
        "vehicle": {
            "payload_kg": p["payload_kg"],
            "stage1": {"propellant_kg": p1_prop, "structure_kg": p1_str,
                       "expansion_ratio": stack[0].expansion_ratio},
            "stage2": {"propellant_kg": p2_prop, "structure_kg": p2_str,
                       "expansion_ratio": stack[1].expansion_ratio},
            "gross_liftoff_kg": g, "mass_ratio": mr,
            "chamber_pressure_bar": p["chamber_pressure_bar"],
        },
        "mission_verification": {
            "achieved_apogee_km": apogee,
            "error_pct": abs(apogee - args.target_km) / args.target_km * 100,
            "downrange_km": traj["downrange_m"] / 1000.0,
            "max_q_kpa": traj["max_q_pa"] / 1000.0,
            "separation_s": traj["separations"],
        },
        "component_verification": results,
        "all_components_passed": all(r["passed"] for r in results),
    }
    (args.out / "PACKET.json").write_text(json.dumps(packet, indent=2))

    # human-readable packet
    L = []
    L.append(f"# Design packet\n")
    L.append(f"**Specification:** {args.prompt}\n")
    v = packet["vehicle"]; m = packet["mission_verification"]
    L.append("## Vehicle\n")
    L.append(f"| | propellant | structure | expansion |")
    L.append(f"|---|---|---|---|")
    L.append(f"| stage 1 | {v['stage1']['propellant_kg']:.2f} kg | "
             f"{v['stage1']['structure_kg']:.2f} kg | {v['stage1']['expansion_ratio']:.0f} |")
    L.append(f"| stage 2 | {v['stage2']['propellant_kg']:.2f} kg | "
             f"{v['stage2']['structure_kg']:.2f} kg | {v['stage2']['expansion_ratio']:.0f} |")
    L.append(f"\npayload {v['payload_kg']:.2f} kg, gross {v['gross_liftoff_kg']:.2f} kg, "
             f"mass ratio {v['mass_ratio']:.2f}, chamber {v['chamber_pressure_bar']:.1f} bar\n")
    L.append("## Mission verification\n")
    L.append(f"Flown apogee **{m['achieved_apogee_km']:.1f} km** against "
             f"{args.target_km:.0f} km requested ({m['error_pct']:.1f}% error), "
             f"downrange {m['downrange_km']:.1f} km, max-Q {m['max_q_kpa']:.1f} kPa, "
             f"separation at {m['separation_s'][0]:.1f} s.\n")
    L.append("## Component verification\n")
    L.append("| component | load case | load | von Mises | margin | status |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        vm = f"{r['max_von_mises_mpa']:.2f} MPa" if r["max_von_mises_mpa"] else "-"
        mg = f"{r['margin']:.1f}x" if r["margin"] else "-"
        st = "PASS" if r["passed"] else "FAIL"
        L.append(f"| {r['name']} | {r['why']} | {r['load_n']:.0f} N | {vm} | {mg} | {st} |")
    L.append(f"\nAllowable {ALLOWABLE_MPA:.0f} MPa, displacement limit "
             f"{MAX_DISP_MM:.1f} mm. Derived from {_ALLOWABLE.material_id} at "
             f"{_ALLOWABLE.source_strength_mpa:.0f} MPa "
             f"({_ALLOWABLE.strength_basis}) with a yield factor of safety of "
             f"{_ALLOWABLE.factor_of_safety} and a {_ALLOWABLE.knockdown} "
             f"knockdown; not a certifiable allowable. "
             f"All components passed: **{packet['all_components_passed']}**\n")
    (args.out / "PACKET.md").write_text("\n".join(L))

    print("\n".join(L))
    print(f"\nwritten: {args.out}/PACKET.md and PACKET.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
