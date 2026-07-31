#!/usr/bin/env python3.12
"""Generate ~8000 OpenRocket-style rocket hardware samples for LatticeZero/JEPA.

Owns geometry corpus expansion only. Does not touch FEA / Modal training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadflow.rocket_hardware_generator import generate_corpus, iter_part_specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/openrocket_hardware_8k"),
        help="Output corpus directory",
    )
    parser.add_argument("--parts", type=int, default=8000, help="Target STL part count")
    parser.add_argument("--ork", type=int, default=500, help="OpenRocket .ork vehicle profiles")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned family counts")
    args = parser.parse_args()

    if args.dry_run:
        specs = iter_part_specs(target=args.parts)
        from collections import Counter

        c = Counter(s.family for s in specs)
        print(f"planned parts={len(specs)}")
        for k, v in sorted(c.items()):
            print(f"  {k}: {v}")
        print(f"ork profiles: {args.ork}")
        return 0

    print("=" * 72)
    print("OPENROCKET-STYLE ROCKET HARDWARE CORPUS")
    print("=" * 72)
    print(f"out={args.out} parts={args.parts} ork={args.ork}")
    summary = generate_corpus(args.out, target_parts=args.parts, ork_profiles=args.ork)
    print(summary)
    print("done")
    return 0 if summary.get("parts_ok", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
