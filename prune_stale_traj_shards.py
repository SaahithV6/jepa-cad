#!/usr/bin/env python3
"""Delete trajectory shards not referenced by the current manifest.

The generator writes into artifacts/physics_shards/traj/ without clearing it,
and case ids are content hashes of the design, so regenerating with changed
physics (for example switching from invented drag to CFD-derived drag) writes
a fresh set alongside the old one rather than replacing it.

Left alone the corpus becomes a silent mixture of coupled and uncoupled
trajectories, which weakens exactly the correlations the coupling was added to
create -- and nothing about the file listing would reveal it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, default=Path("artifacts/physics_shards/traj"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("artifacts/physics_shards/traj_manifest.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep = {
        json.loads(line)["case_id"]
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    }
    on_disk = sorted(args.shards.glob("*.npz"))
    stale = [p for p in on_disk if p.stem not in keep]

    print(f"manifest references : {len(keep)}")
    print(f"shards on disk      : {len(on_disk)}")
    print(f"stale (unreferenced): {len(stale)}")

    if args.dry_run:
        print("dry run -- nothing deleted")
        return 0

    freed = 0
    for p in stale:
        freed += p.stat().st_size
        p.unlink()

    print(f"deleted {len(stale)} shards, freed {freed/1e6:.1f} MB")
    print(f"remaining           : {len(list(args.shards.glob('*.npz')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
