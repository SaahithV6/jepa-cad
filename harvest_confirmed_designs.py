#!/usr/bin/env python3
"""Rebuild the confirmed-design corpus from per-design reports.

generate_confirmed_design_corpus.py only writes corpus.jsonl once the whole
sweep finishes, so an interrupted run loses every design it solved -- and a
long sweep cannot be used until it is completely done. Each design does write
its own CONFIRMED_REPORT.json as it completes, and those carry the prompt,
the accepted parameters and the solver outcome, so the corpus can be rebuilt
from them at any point.

Use this to recover a killed sweep, or to start training on a partial corpus
while the rest is still solving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path,
                    default=Path("artifacts/confirmed_designs/work"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/confirmed_designs/corpus.jsonl"))
    args = ap.parse_args()

    reports = sorted(args.workdir.glob("*/CONFIRMED_REPORT.json"))
    rows: list[str] = []
    rejected = 0

    for r in reports:
        try:
            data = json.loads(r.read_text())
        except Exception:  # noqa: BLE001 - a report mid-write is not fatal
            continue
        acc = data.get("accepted") or {}
        params = acc.get("params_mm")
        prompt = data.get("prompt")
        if not (isinstance(params, dict) and prompt):
            rejected += 1
            continue
        rows.append(json.dumps({
            "prompt": prompt,
            "params": params,
            "outcomes": {
                "max_von_mises_mpa": acc.get("max_von_mises_mpa"),
                "max_displacement_mm": acc.get("max_displacement_mm"),
                "solver_mode": acc.get("solver_mode"),
                "frd_bytes": acc.get("frd_bytes"),
            },
        }))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))

    print(f"reports found : {len(reports)}")
    print(f"accepted      : {len(rows)}")
    print(f"no accepted   : {rejected}")
    print(f"corpus        : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
