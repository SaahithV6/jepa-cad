#!/usr/bin/env python3.12
"""Body-fitted snappyHexMesh + simpleFoam CFD for the rocket STL corpus.

This is the real-geometry path (STL as a wall in external flow). It replaces
the old empty blockMesh channel proxy under artifacts/rocket_cfd_8k.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadflow.rocket_cfd_bodyfit import (
    DEFAULT_CFD_ROOT,
    DEFAULT_CORPUS,
    DEFAULT_GRAPH,
    ingest_bodyfit_to_graph,
    run_batch_bodyfit,
    run_bodyfit_case,
)
from cadflow.rocket_cfd_curate import (
    ROCKET_CFD_FAMILIES,
    curate_rocket_cfd_entries,
    demote_duplicate_and_non_curated_cfd,
    write_curated_manifest,
)
from cadflow.rocket_physics_suite import load_manifest, select_entries

CURATED_PATH = Path("artifacts/rocket_cfd_curated.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--cfd-root", type=Path, default=DEFAULT_CFD_ROOT)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--pilot", type=int, default=5, help="0 = all selected")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--families",
        type=str,
        default=",".join(sorted(ROCKET_CFD_FAMILIES)),
        help="Ignored when --curated is set",
    )
    ap.add_argument(
        "--curated",
        action="store_true",
        help="Deduped classic rocket parts only (skip tiles/panels/degenerate boxes)",
    )
    ap.add_argument("--curate-only", action="store_true", help="Write curated list and exit")
    ap.add_argument("--demote-non-curated", action="store_true")
    ap.add_argument("--single", type=str, default="", help="Run one part_id")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="Skip parts with existing meta.json")
    args = ap.parse_args()

    manifest = load_manifest(args.corpus)
    if args.single:
        entries = [e for e in manifest if e["part_id"] == args.single]
        if not entries:
            print(f"part not found: {args.single}", file=sys.stderr)
            return 2
    elif args.curated or args.curate_only or args.demote_non_curated:
        entries, stats = curate_rocket_cfd_entries(manifest, args.corpus)
        write_curated_manifest(entries, CURATED_PATH, stats)
        printable = {k: v for k, v in stats.items() if k != "dupe_canonical"}
        print(json.dumps(printable, indent=2), flush=True)
        print(f"wrote {CURATED_PATH} entries={len(entries)}", flush=True)
        if args.demote_non_curated:
            d = demote_duplicate_and_non_curated_cfd(
                args.graph,
                {e["part_id"] for e in entries},
                stats.get("dupe_canonical") or {},
            )
            print(f"graph demote={d}", flush=True)
        if args.curate_only:
            return 0
        if args.offset:
            entries = entries[args.offset :]
        if args.pilot > 0:
            entries = entries[: args.pilot]
    else:
        families = [f.strip() for f in args.families.split(",") if f.strip()] or None
        entries = select_entries(
            manifest, limit=args.pilot, families=families, offset=args.offset
        )

    if args.skip_done:
        before = len(entries)
        entries = [
            e
            for e in entries
            if not (args.cfd_root / e["part_id"] / "meta.json").exists()
        ]
        print(f"skip_done: {before} -> {len(entries)}", flush=True)

    print("=" * 72)
    print("BODY-FITTED CFD (snappyHexMesh + simpleFoam)")
    print("=" * 72)
    print(f"entries={len(entries)} workers={args.workers} root={args.cfd_root}", flush=True)

    if args.single or len(entries) == 1:
        r = run_bodyfit_case(entries[0], args.corpus, args.cfd_root, force=args.force)
        results = [
            {
                "part_id": r.part_id,
                "success": r.success,
                "metrics": r.metrics,
                "error": r.error,
                "cached": r.cached,
            }
        ]
        print(results[0], flush=True)
    else:
        results = run_batch_bodyfit(
            entries,
            args.corpus,
            args.cfd_root,
            workers=args.workers,
            force=args.force,
        )

    ok = sum(1 for r in results if r["success"])
    print(f"ok={ok}/{len(results)}", flush=True)
    if not args.no_ingest:
        linked = ingest_bodyfit_to_graph(args.graph, results)
        print(f"graph linked={linked}", flush=True)

    summary = {
        "pipeline": "snappyHexMesh_external_simpleFoam",
        "total": len(results),
        "ok": ok,
        "cfd_root": str(args.cfd_root),
        "failures": [r for r in results if not r["success"]][:20],
    }
    Path("data/rocket_cfd_bodyfit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
