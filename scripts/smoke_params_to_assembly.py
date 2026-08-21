#!/usr/bin/env python3
"""Smoke: user params/constraints → CAD assembly → solid verify → solver.

This is the non-neural MVP path (CadQuery/Mock + FEA/CFD adapter).
Does not claim JEPA text→CAD decode; proves the classical assembly loop.

Usage:
  python3 scripts/smoke_params_to_assembly.py
  python3 scripts/smoke_params_to_assembly.py --out artifacts/smoke_params_assembly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadflow.manifest import JobManifest  # noqa: E402
from cadflow.pipeline import run_pipeline  # noqa: E402
from cadflow.profiles import (  # noqa: E402
    NOSE_SHAPES, centred, fin_planform, nose_profile)


def constraints_to_geometry(constraints: dict) -> dict:
    """Map typed constraints → assembly spec (not free-form NL)."""
    body_r = float(constraints.get("body_radius_mm", 40.0))
    body_h = float(constraints.get("body_height_mm", 200.0))
    nose_r = float(constraints.get("nose_radius_mm", body_r))
    nose_h = float(constraints.get("nose_height_mm", body_r * 1.5))
    fin_span = float(constraints.get("fin_span_mm", body_r * 0.8))
    fin_thick = float(constraints.get("fin_thickness_mm", 3.0))
    fin_chord = float(constraints.get("fin_chord_mm", body_r * 1.2))

    # Wall thickness. Airframe sections are thin shells, not billets. Building
    # them solid made every FEA margin 200-300x -- a solid cylinder under an
    # axial load is barely stressed, so the result said nothing about the
    # flight structure. Each cylinder is now hollowed by cutting a coaxial
    # cylinder, which build_from_spec applies per part.
    #
    # Shelling via sculpt_offset does not work here: CadQuery's Solid has no
    # usable shell for this case, the backend silently falls back to an
    # expanded bounding box, and the "shelled" body came out at 135% of the
    # solid volume. A boolean cut is exact -- 0.00% against the analytic tube
    # volume.
    wall = float(constraints.get("wall_thickness_mm", 0.0))

    # The nose is a surface of revolution, not a cylinder. A cylinder has
    # neither the drag nor the load path of a nose cone, and its flat forward
    # face made the FEA answer a question about a shape that would never fly.
    nose_shape = str(constraints.get("nose_shape", "ogive")).lower()
    if nose_shape not in NOSE_SHAPES:
        nose_shape = "ogive"

    def nose(radius: float, height: float, at=None) -> dict:
        outer = centred(nose_profile(radius, height, nose_shape), height)
        part = {"kind": "revolve", "params": {"profile": outer}}
        if at is not None:
            part["params"]["at"] = at
        inner_r, inner_h = radius - wall, height - wall
        if wall > 0.0 and inner_r > 0.5 and inner_h > wall:
            # The cut tool is the same curve one wall in, pushed below the base
            # so the aft end opens into the body it sits on. What is left at the
            # tip is a solid plug, which is what a real nose cone has anyway.
            pad = max(0.5, min(2.0, radius * 0.05))
            prof = centred(nose_profile(inner_r, inner_h, nose_shape), inner_h)
            dz = -height / 2.0 - pad + inner_h / 2.0
            part["features"] = [{
                "op": "cut",
                "params": {"tool": {"kind": "revolve", "params": {
                    "profile": prof, "at": [0.0, 0.0, dz]}}},
            }]
        return part

    def tube(radius: float, height: float, at=None) -> dict:
        params = {"radius": radius, "height": height}
        if at is not None:
            params["at"] = at
        part = {"kind": "cylinder", "params": params}
        if wall > 0.0 and radius - wall > 0.5:
            part["features"] = [{
                "op": "cut",
                "params": {"tool": {"kind": "cylinder", "params": {
                    "radius": radius - wall, "height": height * 1.05}}},
            }]
        return part

    # Parts are now placed, not all stacked at the origin. Concentric parts at
    # one origin only ever formed a single solid because they overlapped; once
    # hollowed, the inner ones floated free inside the void and the assembly
    # stopped being one watertight body, so nothing could be meshed. Nose sits
    # on top of the body, fins attach to the outside of the wall at the aft end.
    # Overlap the stacked sections rather than butting them face to face.
    # A coincident-face union leaves an internal interface that tessellates into
    # near-duplicate facets, and gmsh then either rejects the mesh or falls back
    # to a convex hull -- which for a hollow part is a solid billet.
    stack_overlap = max(1.0, min(3.0, body_r * 0.05))
    nose_z = (body_h + nose_h) / 2.0 - stack_overlap
    # The planform starts at its root rather than being centred, so the root is
    # embedded in the skin and the fin grows outward from there. Embed by half
    # the wall: enough overlap for a clean boolean union, but never through it.
    #
    # This was min(2.0, wall + 1.0), which is deeper than the wall for any wall
    # under 2 mm. On a 0.8 mm shell it put the fin root 1.0 mm *past* the inner
    # surface, so the fin punched through and left a sliver tab dangling in the
    # cavity. gmsh could not mesh that and fell back to a convex hull -- which
    # for a hollow part is a solid billet, so the component reported no usable
    # result at all. It was marginal rather than always broken: the same design
    # meshed at one tessellation and hulled at another.
    fin_root_x = body_r - (wall * 0.5 if wall > 0.0 else min(2.0, body_r * 0.05))
    fin_z = -body_h / 2.0 + fin_chord / 4.0

    return {
        "kind": "assembly",
        "parts": [
            tube(body_r, body_h),
            tube(nose_r, nose_h, at=[0.0, 0.0, nose_z]),
        ] + ([] if fin_span <= 0.0 else [
            {
                # A real trapezoid, not a box. Sketched in XY with x radial and
                # y aft, extruded through the thickness along Z, then turned
                # 90 degrees about X so the chord lies along the vehicle axis
                # and the thickness across it.
                "kind": "extrude",
                "params": {
                    "profile": fin_planform(fin_span, fin_chord),
                    "height": fin_thick,
                    "rotate": ["x", 90.0],
                    "at": [fin_root_x, 0.0, fin_z],
                },
            },
        ]),
        # Fillet solid parts only. The operation rounds *every* edge, which on
        # a thin shell means rounding the wall edges themselves and leaving
        # slivers: the nose cone came out at 54 faces against 12 for the same
        # assembly unfilleted, and gmsh then ran past a 900 s timeout on it.
        # Clamping the radius did not help because the problem is the number of
        # sliver faces, not their size. The tank only ever worked because the
        # fillet silently failed there and fell back to the clean solid.
        "features": ([] if wall > 0.0 else
                     [{"op": "fillet",
                       "params": {"radius": min(1.5, fin_thick * 0.4)}}]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "smoke_params_assembly")
    ap.add_argument("--solver", default="fea", choices=["fea", "cfd", "none"])
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    constraints = {
        "family": "rocket_stack_proxy",
        "body_radius_mm": 42.0,
        "body_height_mm": 220.0,
        "nose_height_mm": 70.0,
        "fin_span_mm": 55.0,
        "fin_thickness_mm": 3.0,
        "max_stress_mpa": 120.0,
        "material": "Al6061",
        "text_prompt": "small aluminum sounding-rocket body with nose and single fin stub",
    }
    geometry = constraints_to_geometry(constraints)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "constraints.json").write_text(json.dumps(constraints, indent=2) + "\n")
    (args.out / "geometry_spec.json").write_text(json.dumps(geometry, indent=2) + "\n")

    manifest = JobManifest(
        name="smoke_params_to_assembly",
        inputs={"geometry": geometry, "materials": [constraints["material"]]},
        parameters={
            "solver": args.solver if args.solver != "none" else "fea",
            "family": constraints["family"],
            "constraints": constraints,
            "text_prompt": constraints["text_prompt"],
            "max_stress_mpa": constraints["max_stress_mpa"],
        },
        tags=("smoke", "params_to_assembly"),
        notes="Classical params→assembly→verify fixture",
    )

    t0 = time.time()
    result = run_pipeline(
        manifest,
        workdir=args.out,
        source="scripts.smoke_params_to_assembly",
        solver_kind=None if args.solver != "none" else "fea",
        allow_solver_fallback=not args.no_fallback,
        prefer_real_cad=True,
    )
    elapsed = time.time() - t0

    report = {
        "ok": bool(result.ok),
        "status": result.run.status,
        "elapsed_s": round(elapsed, 3),
        "backend": (result.run.provenance.details or {}).get("backend")
        if result.run.provenance
        else None,
        "solver_mode": result.solver_result.metadata.get("mode")
        if result.solver_result
        else None,
        "verification_passed": bool(result.verification.passed),
        "artifacts": list(result.artifacts),
        "workdir": str(args.out / manifest.fingerprint),
        "note": (
            "Classical params→assembly→verify path. "
            "JEPA still lacks semantic text encoder + CAD decoder for neural oneshot."
        ),
    }
    out_path = args.out / "SMOKE_REPORT.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
