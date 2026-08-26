"""Aggregate probe runs across seeds, because one run cannot answer this.

A single probe run reports a bootstrap interval on the gain over random
projections. That interval resamples the held-out rows but not which corpus
rows were drawn, nor how they were split -- the script that produces it says so
in its own help text. What it leaves out turns out to dominate.

The same three checkpoints, probed four times, gave gains from +0.142 to
-0.281. One run called step 1500 significantly negative; another called the
whole thing a plateau. Neither was wrong about its own sample. Both were wrong
to be read as the answer.

So the statistic that belongs in a report is the spread across seeds: if the
sign is not stable across independent draws, there is no gain to claim,
whatever any individual interval says. This script collects the runs and says
that plainly.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: A checkpoint row: name, step, R^2, [lo, hi], gain, verdict.
_ROW = re.compile(
    r"^\s*(\S+\.pt|random init)\s+(\d+)\s+([-\d.]+)\s+"
    r"(?:\(reference\)|\[\s*([-+\d.]+),\s*([-+\d.]+)\])\s+([-+\d.]+)\s*(\w+)?\s*$")

_POOL = re.compile(r"(\d+) usable samples \((\d+) train / (\d+) held out")
_SEED_IN_NAME = re.compile(r"seed(\d+)")


def load_run(path: Path) -> dict | None:
    """Read one probe run, preferring its JSON over scraping its stdout.

    Log scraping is the fallback rather than the route because it was the only
    route once, and that cost four completed seed sweeps: the runs existed
    solely as text under /tmp, and a machine restart took them. Results that
    matter belong in the repository.
    """
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return {
            "log": path.name,
            "seed": payload.get("seed"),
            "usable": payload.get("usable"),
            "held_out": payload.get("held_out"),
            "steps": {int(c["step"]): {"checkpoint": None, "r2": c["r2"],
                                       "gain": c["gain"], "ci": c["ci"],
                                       "verdict": c["verdict"]}
                      for c in payload.get("checkpoints", [])},
            "split_spread": payload.get("split_spread", {}),
        }
    return parse_log(path)


def parse_log(path: Path) -> dict | None:
    """Pull the checkpoint table out of one probe run's stdout."""
    text = path.read_text(errors="ignore")
    pool = _POOL.search(text)
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        name, step, r2, lo, hi, gain, verdict = m.groups()
        rows[int(step)] = {
            "checkpoint": name,
            "r2": float(r2),
            "gain": float(gain),
            "ci": [float(lo), float(hi)] if lo is not None else None,
            "verdict": verdict or ("reference" if name == "random init" else None),
        }
    if not rows:
        return None
    seed = _SEED_IN_NAME.search(path.stem)
    return {
        "log": path.name,
        "seed": int(seed.group(1)) if seed else None,
        "usable": int(pool.group(1)) if pool else None,
        "held_out": int(pool.group(3)) if pool else None,
        "steps": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", type=Path, nargs="+",
                    help="probe run outputs; .json preferred, .log tolerated")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/verification/probe_across_seeds.json")
    ap.add_argument("--min-runs", type=int, default=3,
                    help="below this many runs, refuse to summarise -- a sign "
                         "that flips between two runs cannot be characterised "
                         "by either of them")
    args = ap.parse_args()

    runs = [r for r in (load_run(p) for p in args.logs if p.exists()) if r]
    if not runs:
        print("no parsable probe runs")
        return 1

    print(f"{len(runs)} runs\n")
    print(f"{'log':26s} {'seed':>5s} {'n':>6s} " +
          " ".join(f"{s:>9s}" for s in ("step200", "step1000", "step1500")))
    steps_of_interest = [200, 1000, 1500]
    for r in runs:
        cells = []
        for s in steps_of_interest:
            cells.append(f"{r['steps'][s]['gain']:+9.4f}" if s in r["steps"]
                         else f"{'-':>9s}")
        print(f"{r['log'][:26]:26s} {str(r['seed']):>5s} "
              f"{str(r['usable']):>6s} " + " ".join(cells))

    summary = {}
    print()
    for s in steps_of_interest:
        gains = [r["steps"][s]["gain"] for r in runs if s in r["steps"]]
        if len(gains) < 2:
            continue
        pos = sum(1 for g in gains if g > 0)
        stable = pos == len(gains) or pos == 0
        entry = {
            "runs": len(gains),
            "mean_gain": round(statistics.mean(gains), 4),
            "min_gain": round(min(gains), 4),
            "max_gain": round(max(gains), 4),
            "positive_runs": pos,
            "sign_stable": stable,
            "enough_runs": len(gains) >= args.min_runs,
        }
        # A mean is only meaningful once the sign holds. Reporting "mean gain
        # -0.09" across runs that span -0.28 to +0.14 would describe none of
        # them, and would read as a measured degradation rather than as an
        # unresolved measurement.
        if not stable:
            entry["conclusion"] = (
                "no gain can be claimed: the sign is not stable across seeds")
        elif len(gains) < args.min_runs:
            # Two runs agreeing on a sign is not consistency, it is a coin
            # landing the same way twice. Saying "consistently worse" from a
            # pair would repeat the error this script exists to correct.
            entry["conclusion"] = (
                f"sign agrees across {len(gains)} runs, which is too few to "
                f"call consistent; needs at least {args.min_runs}")
        else:
            entry["conclusion"] = (
                "consistent gain over random projections" if pos == len(gains)
                else "consistently worse than random projections")
        summary[f"step_{s}"] = entry
        print(f"step {s:>5}: {len(gains)} runs, gain "
              f"{min(gains):+.4f} to {max(gains):+.4f}, "
              f"{pos}/{len(gains)} positive -> {entry['conclusion']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs, "summary": summary}, indent=1))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
