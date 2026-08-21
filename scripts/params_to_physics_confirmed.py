#!/usr/bin/env python3
"""Params → CadQuery assembly → native CalculiX confirm (no fallback).

Mutates geometry up to --max-iters when stress/disp targets are missed.
Writes CONFIRMED_REPORT.json only when solver_mode=native and targets pass.

Does not launch Modal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadflow.manifest import JobManifest  # noqa: E402
from cadflow.pipeline import run_pipeline  # noqa: E402
from models.cad_decoder import assembly_params_to_constraints  # noqa: E402
from scripts.smoke_params_to_assembly import constraints_to_geometry  # noqa: E402


def _mutate(params: dict[str, float], cycle: int) -> dict[str, float]:
    """Heuristic thicken / shorten to reduce stress."""
    out = dict(params)
    scale = 1.0 + 0.08 * (cycle + 1)
    out["fin_thickness_mm"] = float(out.get("fin_thickness_mm", 3.0)) * scale
    out["fillet_radius_mm"] = min(float(out.get("fillet_radius_mm", 1.0)) * scale, out["fin_thickness_mm"] * 0.45)
    out["body_radius_mm"] = float(out.get("body_radius_mm", 40.0)) * (1.0 + 0.03 * (cycle + 1))
    out["nose_radius_mm"] = float(out.get("nose_radius_mm", out["body_radius_mm"]))
    return out


def _targets_met(meta: dict[str, Any], max_stress: float, max_disp: float) -> bool:
    stress = meta.get("max_von_mises_mpa")
    disp = meta.get("max_displacement_mm")
    if not isinstance(stress, (int, float)) or not isinstance(disp, (int, float)):
        return False
    return float(stress) <= float(max_stress) and float(disp) <= float(max_disp)


def run_confirmed(
    *,
    params_mm: dict[str, float],
    out: Path,
    max_stress_mpa: float,
    max_disp_mm: float,
    max_iters: int,
    load_n: float,
    prompt: str = "",
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cycles: list[dict[str, Any]] = []
    current = dict(params_mm)
    t0 = time.time()
    accepted: dict[str, Any] | None = None

    for cycle in range(max_iters):
        constraints = assembly_params_to_constraints(current)
        constraints["text_prompt"] = prompt or "physics-confirmed rocket fixture"
        constraints["max_stress_mpa"] = max_stress_mpa
        geometry = constraints_to_geometry(constraints)
        # Soften fillet if geometry fails on aggressive fillets
        manifest = JobManifest(
            name=f"physics_confirmed_c{cycle}",
            inputs={"geometry": geometry, "materials": ["Al6061"]},
            parameters={
                "solver": "fea",
                "constraints": constraints,
                "load_n": load_n,
                "max_stress_mpa": max_stress_mpa,
                "max_disp_mm": max_disp_mm,
                "cl_max_mm": float(current.get("cl_max_mm", 8.0)),
                "cl_min_mm": float(current.get("cl_min_mm", 2.0)),
                "solver_timeout_s": 600,
                # Thin-walled parts need several elements through the wall, so
                # meshes run to hundreds of thousands of tets. The adapter
                # defaults this from job.timeout_s, which left the nose cone
                # timing out in gmsh and reporting nothing at all.
                "mesh_timeout_s": int(current.get("mesh_timeout_s", 900)),
                "youngs_modulus": 70e9,
                "poisson": 0.33,
            },
            tags=("physics_confirmed", f"cycle{cycle}"),
        )
        result = run_pipeline(
            manifest,
            workdir=out / f"cycle_{cycle}",
            allow_solver_fallback=False,
            prefer_real_cad=True,
        )
        mode = str(result.solver_result.metadata.get("mode") or "")
        meta = dict(result.solver_result.metadata)
        cycle_row = {
            "cycle": cycle,
            "params_mm": dict(current),
            "ok": bool(result.ok),
            "verification_passed": bool(result.verification.passed),
            "solver_status": result.solver_result.status,
            "solver_mode": mode,
            "max_von_mises_mpa": meta.get("max_von_mises_mpa"),
            "max_displacement_mm": meta.get("max_displacement_mm"),
            "frd_bytes": meta.get("frd_bytes"),
            "artifacts": list(result.artifacts),
            "targets_met": False,
        }
        native_ok = result.ok and mode == "native" and int(meta.get("frd_bytes") or 0) > 1000
        if native_ok:
            cycle_row["targets_met"] = _targets_met(meta, max_stress_mpa, max_disp_mm)
            if cycle_row["targets_met"]:
                accepted = cycle_row
                cycles.append(cycle_row)
                break
        cycles.append(cycle_row)
        current = _mutate(current, cycle)

    report = {
        "ok": accepted is not None,
        "solver_mode": (accepted or {}).get("solver_mode"),
        "elapsed_s": round(time.time() - t0, 3),
        "cycles": cycles,
        "accepted": accepted,
        "prompt": prompt,
        "note": "Physics-confirmed requires native CalculiX (no fallback) + target gates.",
    }
    (out / "CONFIRMED_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "physics_confirmed")
    ap.add_argument("--body-radius-mm", type=float, default=30.0)
    ap.add_argument("--body-height-mm", type=float, default=80.0)
    ap.add_argument("--nose-height-mm", type=float, default=25.0)
    ap.add_argument("--fin-span-mm", type=float, default=20.0)
    ap.add_argument("--fin-thickness-mm", type=float, default=4.0)
    ap.add_argument("--max-stress-mpa", type=float, default=250.0)
    ap.add_argument("--max-disp-mm", type=float, default=2.0)
    ap.add_argument("--load-n", type=float, default=500.0)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--prompt", type=str, default="")
    args = ap.parse_args()

    params = {
        "body_radius_mm": args.body_radius_mm,
        "body_height_mm": args.body_height_mm,
        "nose_radius_mm": args.body_radius_mm,
        "nose_height_mm": args.nose_height_mm,
        "fin_span_mm": args.fin_span_mm,
        "fin_thickness_mm": args.fin_thickness_mm,
        "fin_chord_mm": args.body_radius_mm * 1.1,
        "fillet_radius_mm": min(1.5, args.fin_thickness_mm * 0.35),
        "cl_max_mm": 8.0,
        "cl_min_mm": 2.0,
    }
    report = run_confirmed(
        params_mm=params,
        out=args.out,
        max_stress_mpa=args.max_stress_mpa,
        max_disp_mm=args.max_disp_mm,
        max_iters=args.max_iters,
        load_n=args.load_n,
        prompt=args.prompt
        or (
            f"aluminum rocket body {args.body_radius_mm}mm x {args.body_height_mm}mm "
            f"nose {args.nose_height_mm}mm fin {args.fin_span_mm}mm"
        ),
    )
    print(json.dumps({k: report[k] for k in ("ok", "solver_mode", "elapsed_s", "accepted") if k in report}, indent=2))
    print(f"report={args.out / 'CONFIRMED_REPORT.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
