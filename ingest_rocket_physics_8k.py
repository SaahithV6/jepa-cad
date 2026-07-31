#!/usr/bin/env python3.12
"""Final FEA+CFD ingest + summary for rocket physics suite (run after solvers finish)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cadflow.msh_to_calculix import case_has_valid_frd, parse_frd_summary
from cadflow.rocket_physics_suite import (
    DEFAULT_CFD_ROOT,
    DEFAULT_FEA_ROOT,
    DEFAULT_GRAPH,
    DEFAULT_SUMMARY,
    ingest_cfd_to_graph,
    ingest_fea_to_graph,
)
from run_cfd_5k_proper import summarize_fields


def main() -> int:
    fea_root = DEFAULT_FEA_ROOT
    cfd_root = DEFAULT_CFD_ROOT
    graph = DEFAULT_GRAPH

    fea_dirs = [d for d in fea_root.iterdir() if d.is_dir()] if fea_root.exists() else []
    fea_ok = sum(1 for d in fea_dirs if case_has_valid_frd(d, min_bytes=50_000))
    print(f"FEA on disk: {fea_ok}/{len(fea_dirs)} valid", flush=True)

    print("Ingesting FEA...", flush=True)
    fea_linked = ingest_fea_to_graph(graph, fea_root)
    print(f"FEA linked={fea_linked}", flush=True)

    cfd_results = []
    cfd_dirs = [d for d in cfd_root.iterdir() if d.is_dir()] if cfd_root.exists() else []
    for case_dir in cfd_dirs:
        metrics = summarize_fields(case_dir)
        if not metrics or metrics.get("U_mag_max", 0) < 1e-6:
            continue
        cfd_results.append({"part_id": case_dir.name, "success": True, "metrics": metrics})
    print(f"CFD on disk: {len(cfd_results)}/{len(cfd_dirs)} with fields", flush=True)

    print("Ingesting CFD...", flush=True)
    cfd_linked = ingest_cfd_to_graph(graph, cfd_results)
    print(f"CFD linked={cfd_linked}", flush=True)

    g = json.loads(graph.read_text(encoding="utf-8"))
    rocket = [n for n in g["nodes"] if n.get("type") == "Part" and str(n.get("id", "")).startswith("part:rocket:")]
    summary = {
        "pipeline": "rocket-stl-volume-msh2-calculix-simplefoam",
        "parts_registered": len(rocket),
        "fea_dirs": len(fea_dirs),
        "fea_valid": fea_ok,
        "fea_graph_linked": fea_linked,
        "cfd_dirs": len(cfd_dirs),
        "cfd_with_fields": len(cfd_results),
        "cfd_graph_linked": cfd_linked,
        "rocket_has_fea": sum(1 for p in rocket if p.get("has_fea")),
        "rocket_has_cfd": sum(1 for p in rocket if p.get("has_cfd")),
        "rocket_physics_verified": sum(1 for p in rocket if p.get("physics_verified")),
        "fea_root": str(fea_root),
        "cfd_root": str(cfd_root),
        "graph": str(graph),
    }
    DEFAULT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["rocket_physics_verified"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
