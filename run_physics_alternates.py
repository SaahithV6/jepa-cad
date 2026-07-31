#!/usr/bin/env python3.12
"""Run class-conditioned FEA/CFD alternates on existing ~2k Parts.

Does NOT touch the physics-8k rocket corpus lane
(``artifacts/rocket_fea_8k``, ``artifacts/rocket_cfd_8k``).
"""
from __future__ import annotations

import argparse
import sys

from cadflow.physics_alternates import ingest_alternates_from_disk, run_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0, help="Limit parts (0=all with mesh)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fea-only", action="store_true")
    ap.add_argument("--cfd-only", action="store_true")
    ap.add_argument(
        "--ingest-only",
        action="store_true",
        help="Ingest existing fea_alt/cfd_alt results into TAO graph.json",
    )
    args = ap.parse_args()
    if args.ingest_only:
        summary = ingest_alternates_from_disk()
        print(summary, flush=True)
        return 0 if summary.get("graph_linked", 0) > 0 else 1
    summary = run_batch(
        pilot=args.pilot,
        workers=args.workers,
        force=args.force,
        fea_only=args.fea_only,
        cfd_only=args.cfd_only,
    )
    ok = int(summary.get("successful") or 0)
    # Always re-ingest from disk so TAO matches artifacts even if memory path missed some.
    disk = ingest_alternates_from_disk()
    print(f"disk_reingest {disk}", flush=True)
    return 0 if ok > 0 or disk.get("graph_linked", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
