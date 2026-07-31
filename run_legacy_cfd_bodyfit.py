#!/usr/bin/env python3.12
"""Body-fitted snappyHexMesh + simpleFoam for legacy TAO aero Parts.

Does NOT touch artifacts/rocket_cfd_bodyfit or the OpenRocket 8k corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadflow.legacy_cfd_bodyfit import (
    CFD_ROOT,
    GRAPH_PATH,
    SUMMARY_PATH,
    ingest_legacy_bodyfit,
    ingest_legacy_bodyfit_from_disk,
    run_batch_legacy_bodyfit,
    run_legacy_bodyfit_case,
    select_aero_parts,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0, help="0 = all aero Parts")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cfd-root", type=Path, default=CFD_ROOT)
    ap.add_argument("--graph", type=Path, default=GRAPH_PATH)
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--no-ingest", action="store_true")
    args = ap.parse_args()

    if args.ingest_only:
        summary = ingest_legacy_bodyfit_from_disk(args.graph, args.cfd_root)
        print(summary, flush=True)
        SUMMARY_PATH.write_text(json.dumps({"mode": "ingest_only", **summary}, indent=2) + "\n")
        return 0 if summary.get("linked", 0) > 0 else 1

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    parts = select_aero_parts(graph, limit=args.pilot)
    print("=" * 72)
    print("LEGACY BODY-FITTED CFD (snappyHexMesh + simpleFoam)")
    print("=" * 72)
    print(f"aero_parts={len(parts)} workers={args.workers} root={args.cfd_root}", flush=True)
    if not parts:
        print("ERROR: no aero parts with STL", file=sys.stderr)
        return 2

    if len(parts) == 1:
        results = [run_legacy_bodyfit_case(parts[0], args.cfd_root, force=args.force)]
        print(results[0], flush=True)
    else:
        results = run_batch_legacy_bodyfit(
            parts,
            args.cfd_root,
            workers=args.workers,
            force=args.force,
        )

    ok = sum(1 for r in results if r.get("success"))
    print(f"ok={ok}/{len(results)}", flush=True)
    linked = 0
    if not args.no_ingest:
        linked = ingest_legacy_bodyfit(args.graph, results)
        disk = ingest_legacy_bodyfit_from_disk(args.graph, args.cfd_root)
        print(f"graph linked={linked} disk_reingest={disk}", flush=True)

    summary = {
        "pipeline": "legacy_snappyHexMesh_external_simpleFoam",
        "total": len(results),
        "ok": ok,
        "graph_linked": linked,
        "cfd_root": str(args.cfd_root),
        "failures": [
            {"part_id": r.get("part_id"), "case_id": r.get("case_id"), "error": r.get("error")}
            for r in results
            if not r.get("success")
        ][:40],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if ok >= max(1, int(0.5 * len(results))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
