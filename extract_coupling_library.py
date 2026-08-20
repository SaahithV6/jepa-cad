#!/usr/bin/env python3
"""Harvest real aero and structural values from the TAO graph.

The propulsion/trajectory generator originally invented its own drag
coefficient and structural coefficient from uniform distributions, which left
the disciplines running in parallel rather than coupled: 3,267 CFD shards and
9,029 FEA shards existed, but nothing in the trajectory depended on either.

This pulls the real distributions out of the graph once and caches them, so
the generator can draw from solver-derived values instead of inventing them:

  aero       Cd / Cd_proxy on Part nodes, derived from bodyfit and internal
             CFD runs, with fineness ratio where available
  structural stress and safety-factor results from CalculiX FRD parses, used
             to set the structural mass coefficient rather than guessing it

Writes a compact JSON so the generator never has to load the 407 MB graph.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _f(v):
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=Path("artifacts/jepa-train-bundle/graph.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/coupling/aero_structural_library.json"))
    args = ap.parse_args()

    # --- real solver output, straight from the shard manifests ----------
    # Preferred over the graph: graph Part nodes carry physics *targets* from
    # physics_targets.py (max_stress_mpa has just 9 distinct values -- 100,
    # 110 ... 180 -- i.e. sampled family windows, not measurements), whereas
    # the FEA manifest carries values parsed out of actual CalculiX FRDs.
    fea_vm: list[float] = []
    fea_disp: list[float] = []
    fea_scale: list[float] = []
    fea_manifest = Path("artifacts/physics_shards/fea_manifest.jsonl")
    if fea_manifest.exists():
        for line in fea_manifest.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line).get("metrics", {})
            v = _f(m.get("max_von_mises_mpa"))
            if v is not None and 0.0 < v <= 5_000.0:
                fea_vm.append(v)
            v = _f(m.get("max_disp_mm"))
            if v is not None and 0.0 <= v <= 1_000.0:
                fea_disp.append(v)
            v = _f(m.get("geom_scale_m"))
            if v is not None and 0.0 < v <= 100.0:
                fea_scale.append(v)
        print(f"FEA manifest: {len(fea_vm)} real von Mises values")

    print(f"loading graph: {args.graph} ({args.graph.stat().st_size/1e6:.0f} MB)")
    payload = json.loads(args.graph.read_text())
    nodes = payload.get("nodes", [])
    print(f"nodes: {len(nodes)}")

    cds: list[float] = []
    fineness: list[float] = []
    stresses: list[float] = []
    safety: list[float] = []
    massfrac: list[float] = []

    for n in nodes:
        props = n.get("properties", {}) or {}
        if not isinstance(props, dict):
            continue

        # --- aero -------------------------------------------------------
        cfd = props.get("simulation_results_cfd")
        cfd = cfd if isinstance(cfd, dict) else {}
        for src in (props.get("Cd"), props.get("Cd_proxy"),
                    cfd.get("Cd"), cfd.get("Cd_proxy")):
            v = _f(src)
            # Restricted to the slender-body range. The raw Cd_proxy
            # distribution is bimodal -- a credible cluster around 0.35-0.52
            # and a second at 1.4-1.6 holding 65% of the mass. A launch
            # vehicle does not have Cd 1.5; that upper cluster is a proxy
            # artefact, not a drag coefficient, and feeding it to a trajectory
            # integrator would produce confidently wrong ranges.
            if v is not None and 0.15 <= v <= 0.80:
                cds.append(v)
                break

        v = _f(props.get("fineness_ratio"))
        if v is not None and 1.0 <= v <= 20.0:
            fineness.append(v)

        # --- structural -------------------------------------------------
        fea = props.get("simulation_results_fea")
        fea = fea if isinstance(fea, dict) else {}
        v = _f(fea.get("max_stress_mpa") or props.get("max_stress_mpa"))
        if v is not None and 0.0 < v <= 5_000.0:
            stresses.append(v)
        v = _f(fea.get("safety_factor") or props.get("safety_factor"))
        if v is not None and 0.1 <= v <= 20.0:
            safety.append(v)
        v = _f(props.get("mass_fraction"))
        if v is not None and 0.0 < v < 1.0:
            massfrac.append(v)

    def summarise(name: str, xs: list[float]) -> dict:
        if not xs:
            print(f"  {name:12s} NONE FOUND")
            return {"count": 0, "values": []}
        xs_sorted = sorted(xs)
        q = lambda p: xs_sorted[min(len(xs_sorted) - 1, int(p * len(xs_sorted)))]
        print(f"  {name:12s} n={len(xs):6d}  p10={q(0.10):.4g}  median={q(0.50):.4g}  p90={q(0.90):.4g}")
        # keep a bounded sample the generator can draw from directly
        step = max(1, len(xs_sorted) // 2000)
        return {
            "count": len(xs),
            "p10": q(0.10), "median": q(0.50), "p90": q(0.90),
            "mean": statistics.fmean(xs_sorted),
            "values": xs_sorted[::step][:2000],
        }

    print("harvested:")
    lib = {
        # aero: CFD-derived, filtered to the physically valid band
        "cd": summarise("Cd", cds),
        "fineness_ratio": summarise("fineness", fineness),
        # structural: real CalculiX FRD parses from the shard manifest
        "fea_max_von_mises_mpa": summarise("vm_mpa", fea_vm),
        "fea_max_disp_mm": summarise("disp_mm", fea_disp),
        "fea_geom_scale_m": summarise("geom_m", fea_scale),
        # graph-side values kept for reference but NOT used for coupling:
        # physics targets, not measurements (only 9 distinct stress values)
        "graph_target_stress_mpa": summarise("tgt_stress", stresses),
        "safety_factor": summarise("safety", safety),
        "mass_fraction": summarise("mass_frac", massfrac),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lib, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
