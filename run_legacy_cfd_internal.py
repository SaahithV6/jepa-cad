#!/usr/bin/env python3
"""Run recipe-routed internal OpenFOAM CFD for legacy non-aero Parts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cadflow.legacy_cfd_internal import (
    CFD_ROOT,
    GRAPH_PATH,
    SUMMARY_PATH,
    ingest_internal,
    run_batch,
    select_internal_parts,
)
from cadflow.legacy_cfd_routes import recipe_for_part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="0 = all selected")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--recipe", default="", help="Filter one recipe_id")
    ap.add_argument("--timeout-mesh", type=int, default=360)
    ap.add_argument("--timeout-solve", type=int, default=300)
    args = ap.parse_args()

    graph = json.loads(Path(GRAPH_PATH).read_text(encoding="utf-8"))
    parts = select_internal_parts(graph, limit=0)
    if args.recipe:
        parts = [p for p in parts if recipe_for_part(p).recipe_id == args.recipe]
    # Prefer diversity for pilots: one-of-each then fill
    if args.pilot > 0:
        picked: list[dict] = []
        seen: set[str] = set()
        for p in parts:
            rid = recipe_for_part(p).recipe_id
            if rid not in seen:
                picked.append(p)
                seen.add(rid)
            if len(picked) >= args.pilot:
                break
        if len(picked) < args.pilot:
            for p in parts:
                if p in picked:
                    continue
                picked.append(p)
                if len(picked) >= args.pilot:
                    break
        parts = picked

    print("=" * 72)
    print("LEGACY INTERNAL CFD (duct snappy + recipe solvers)")
    print("=" * 72)
    print(f"parts={len(parts)} workers={args.workers} root={CFD_ROOT}")
    print("recipes:", dict(Counter(recipe_for_part(p).recipe_id for p in parts)))

    if args.ingest_only:
        # Re-ingest from meta.json
        results = []
        for meta in CFD_ROOT.glob("*/meta.json"):
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("metrics", {}).get("U_mag_max", 0) > 1e-6:
                    results.append(
                        {
                            "part_id": m["part_id"],
                            "case_id": m.get("case_id") or meta.parent.name,
                            "recipe_id": m.get("recipe_id"),
                            "family": m.get("family"),
                            "success": True,
                            "metrics": m["metrics"],
                        }
                    )
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        linked = ingest_internal(Path(GRAPH_PATH), results)
        print(f"ingest_only linked={linked}")
        return

    results = run_batch(
        parts,
        workers=max(1, args.workers),
        force=args.force,
        timeout_mesh=args.timeout_mesh,
        timeout_solve=args.timeout_solve,
    )
    ok = [r for r in results if r.get("success")]
    linked = ingest_internal(Path(GRAPH_PATH), ok)
    by_recipe = Counter(r.get("recipe_id") for r in ok)
    fail = Counter(r.get("error") for r in results if not r.get("success"))
    summary = {
        "attempted": len(results),
        "ok": len(ok),
        "linked": linked,
        "ok_by_recipe": dict(by_recipe),
        "failures": dict(fail),
        "root": str(CFD_ROOT),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
