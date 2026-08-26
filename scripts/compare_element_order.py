"""C3D4 against C3D10 on real corpus parts, at the same mesh.

The Lame verification case established that linear tetrahedra converge
first-order in stress and read 9.8% low on surface stress at the production
element budget, while quadratic ones do better with fifty times fewer elements.
That case is a smooth, singularity-free cylinder chosen so peak stress is a
convergent quantity.

Production parts are not that. They are STL-derived, carry re-entrant corners,
and are the reason the design loop accepts on p95 rather than peak. So the
question this script answers is narrower and more useful: on the actual parts,
at the mesh the pipeline actually builds, how much does the element type move
the number the design loop reads?

Both runs use identical geometry, identical characteristic lengths and
identical loads. The only difference is `setOrder(2)`, so any change in the
reported stress is the element formulation and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentiles(case_dir: Path) -> dict | None:
    """Von Mises percentiles from the newest FRD under ``case_dir``."""
    from scripts.plan_and_verify import frd_stress_percentiles

    return frd_stress_percentiles(case_dir)


def run_one(entry: dict, corpus: Path, work: Path, order: int,
            cl_max: float, cl_min: float, timeout: int) -> dict:
    from cadflow.msh_to_calculix import (
        generate_fea_case_inp, parse_msh2_solid, run_calculix_case,
        write_solid_mesh_inp)
    from cadflow.rocket_physics_suite import material_elastic_props, mesh_stl_volume

    case = work / f"{entry['part_id']}_o{order}"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mr = mesh_stl_volume(corpus / entry["stl"], case / "mesh.msh",
                         cl_max_mm=cl_max, cl_min_mm=cl_min,
                         mesh_timeout_s=300, element_order=order)
    if not mr.success:
        return {"order": order, "error": f"mesh:{mr.error}"}

    mesh = parse_msh2_solid(case / "mesh.msh")
    e_pa, nu = material_elastic_props(entry)
    try:
        # This both writes the include deck and builds the case, so the C3D10
        # permutation and its geometric check run inside it.
        generate_fea_case_inp(case, mesh_filename="mesh_solid.inp",
                              case_filename="case.inp", youngs_modulus=e_pa,
                              poisson=nu, total_load=50_000.0)
    except ValueError as exc:
        return {"order": order, "error": f"deck:{exc}"}
    run = run_calculix_case(case, job_name="case", timeout=timeout)
    dist = percentiles(case)
    elapsed = time.time() - t0
    if dist is None:
        # "no_frd" on its own is not a diagnosis, and reporting it as one sent
        # this comparison chasing a C3D10 defect that did not exist: the same
        # case solved cleanly when run by hand. Carry the solver's own last
        # words out with the failure.
        log = case / "ccx.log"
        tail = ""
        if log.exists():
            lines = [ln.strip() for ln in log.read_text(errors="ignore").splitlines()
                     if ln.strip()]
            tail = " | ".join(lines[-3:])[:300]
        return {"order": order, "error": "no_frd", "seconds": round(elapsed, 1),
                "converged": bool(getattr(run, "converged", False)),
                "frd_bytes": int(getattr(run, "frd_bytes", 0) or 0),
                "solver_tail": tail}
    return {
        "order": order,
        "elements": len(mesh.elements),
        "nodes": len(mesh.nodes),
        "seconds": round(elapsed, 1),
        **{k: dist[k] for k in ("median", "p95", "p99", "max", "n")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "data/openrocket_hardware_8k")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/verification/element_order_ab.json")
    ap.add_argument("--per-family", type=int, default=1)
    ap.add_argument("--families", nargs="+",
                    default=["body_tube", "tank", "nose_cone", "fin"])
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    picks: list[dict] = []
    for fam in args.families:
        fam_entries = [e for e in manifest if e.get("family") == fam]
        picks.extend(fam_entries[: args.per_family])

    from cadflow.rocket_physics_suite import cl_for_target_tets

    work = Path(tempfile.mkdtemp(prefix="elorder_"))
    rows: list[dict] = []
    try:
        for entry in picks:
            cl_max, cl_min = cl_for_target_tets(entry.get("extents_mm"),
                                                target_tets=12_000)
            print(f"\n{entry['part_id']} ({entry.get('family')}) "
                  f"cl={cl_max:.2f}/{cl_min:.2f} mm", flush=True)
            row = {"part_id": entry["part_id"], "family": entry.get("family"),
                   "cl_max_mm": cl_max, "cl_min_mm": cl_min, "runs": []}
            for order in (1, 2):
                r = run_one(entry, args.corpus, work, order, cl_max, cl_min,
                            args.timeout)
                row["runs"].append(r)
                if "error" in r:
                    print(f"  order {order}: {r['error']} " + str(r.get('solver_tail',''))[:200], flush=True)
                else:
                    print(f"  {'C3D4 ' if order == 1 else 'C3D10'}: "
                          f"{r['elements']:>7d} el  median {r['median']:8.2f}  "
                          f"p95 {r['p95']:8.2f}  p99 {r['p99']:9.2f}  "
                          f"max {r['max']:10.2f} MPa  ({r['seconds']:.0f}s)",
                          flush=True)
            ok = [r for r in row["runs"] if "error" not in r]
            if len(ok) == 2:
                lin, quad = ok[0], ok[1]
                # p95 is the quantity the design loop sizes against, so it is
                # the one whose movement matters. The peak is reported too, but
                # it tracks the mesh singularity rather than the structure.
                row["p95_shift_pct"] = 100.0 * (quad["p95"] - lin["p95"]) / lin["p95"]
                row["median_shift_pct"] = (
                    100.0 * (quad["median"] - lin["median"]) / lin["median"])
                row["cost_ratio"] = quad["seconds"] / max(lin["seconds"], 1e-6)
                print(f"  p95 shift {row['p95_shift_pct']:+.1f}%, "
                      f"median shift {row['median_shift_pct']:+.1f}%, "
                      f"cost x{row['cost_ratio']:.1f}", flush=True)
            rows.append(row)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    shifts = [r["p95_shift_pct"] for r in rows if "p95_shift_pct" in r]
    summary = {
        "parts_compared": len(shifts),
        "p95_shift_pct_median": (round(statistics.median(shifts), 2)
                                 if shifts else None),
        "p95_shift_pct_range": ([round(min(shifts), 2), round(max(shifts), 2)]
                                if shifts else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "parts": rows}, indent=1))
    print(f"\n{json.dumps(summary, indent=1)}\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
