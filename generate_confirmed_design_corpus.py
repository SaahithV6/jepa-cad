#!/usr/bin/env python3
"""Build a corpus of physics-confirmed (prompt -> design) pairs.

The generative head is the piece that turns a specification into geometry, and
it was training on three targets. scripts/train_text_cad_confirmed.py sources
them from CONFIRMED_REPORT.json files, each of which holds a single accepted
design, so the head cycled the same three parameter vectors:

    gens = [accepted[i % len(accepted)] for i in range(args.batch_size)]

Its generative loss duly fell to ~1e-5 within 60 steps. That is memorisation of
three vectors, not learning to design -- and it looks exactly like success on a
loss curve.

This sweeps the parameter space, runs each candidate through the same
params -> CAD -> mesh -> CalculiX -> verify loop used by
scripts/params_to_physics_confirmed.py, and keeps the designs that actually
satisfy their stress and displacement constraints. Every retained pair is
solver-backed: the geometry was meshed and solved, not asserted.

Each record carries a natural-language prompt, so the corpus supervises
text -> parameters directly.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402


def sample_params(rng: random.Random) -> dict[str, float]:
    """Sample geometry with the stated dimensions largely independent.

    Deriving secondary dimensions from body radius -- fin span as
    uniform(0.4, 1.2) * body_r and so on -- looks like sensible engineering
    scaling, but it makes those numbers nearly redundant given the radius. A
    model trained on it learns to predict fin span from radius and to ignore
    whatever the prompt asked for, which is the correct inference from that
    data and useless behaviour in a design tool.

    Measured: with radius-derived sampling the head returned ~42 mm fin span
    whether the prompt asked for 30 or anything else, a 38-47% error that no
    amount of extra data or better tokenisation moved. Independent sampling
    makes each stated dimension carry its own information.
    """
    body_r = rng.uniform(15.0, 60.0)
    return {
        "body_radius_mm": round(body_r, 2),
        "body_height_mm": round(rng.uniform(50.0, 220.0), 2),
        "nose_radius_mm": round(body_r, 2),          # coincident by construction
        "nose_height_mm": round(rng.uniform(12.0, 90.0), 2),
        "fin_span_mm": round(rng.uniform(8.0, 60.0), 2),
        "fin_thickness_mm": round(rng.uniform(2.0, 8.0), 2),
        "fin_chord_mm": round(rng.uniform(10.0, 80.0), 2),
        "fillet_radius_mm": round(rng.uniform(0.8, 3.0), 2),
        "cl_max_mm": round(rng.uniform(5.0, 12.0), 2),
        "cl_min_mm": round(rng.uniform(1.0, 4.0), 2),
    }


def build_prompt(p: dict[str, float], load_n: float, max_stress: float) -> str:
    """Natural-language spec. This is the text side of the supervision."""
    return (
        f"rocket airframe section {p['body_radius_mm']:.0f} mm radius and "
        f"{p['body_height_mm']:.0f} mm long, ogive nose {p['nose_height_mm']:.0f} mm tall, "
        f"{p['fin_span_mm']:.0f} mm fin span at {p['fin_thickness_mm']:.1f} mm thickness, "
        f"carrying {load_n:.0f} N axial load below {max_stress:.0f} MPa"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--workdir", type=Path,
                    default=ROOT / "artifacts/confirmed_designs/work")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/confirmed_designs/corpus.jsonl")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    accepted_rows: list[str] = []
    n_ok = n_fail = n_err = 0
    t0 = time.time()

    for i in range(args.count):
        params = sample_params(rng)
        # Loads must be large enough that the stress constraint actually binds.
        # At 200-6000 N these sections come out at 0.2-2 MPa against a 120-260
        # MPa allowable, so every candidate passes and "physics-confirmed"
        # certifies nothing -- the corpus would carry no feasibility boundary
        # for the model to learn. Empirically stress runs ~1e-3 MPa/N here, so
        # this band straddles the allowable and produces real rejections.
        load_n = 10.0 ** rng.uniform(4.3, 5.7)   # ~20 kN .. ~500 kN
        max_stress = rng.uniform(120.0, 260.0)
        max_disp = rng.uniform(0.5, 4.0)
        prompt = build_prompt(params, load_n, max_stress)

        try:
            report = run_confirmed(
                params_mm=params,
                out=args.workdir / f"d{i:05d}",
                max_stress_mpa=max_stress,
                max_disp_mm=max_disp,
                max_iters=args.max_iters,
                load_n=load_n,
                prompt=prompt,
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            n_err += 1
            if n_err <= 3:
                print(f"  [{i}] error: {type(exc).__name__}: {exc}")
            continue

        acc = report.get("accepted")
        if not acc or not acc.get("params_mm"):
            n_fail += 1
            continue

        accepted_rows.append(json.dumps({
            "prompt": prompt,
            "params": acc["params_mm"],
            "constraints": {
                "load_n": load_n,
                "max_stress_mpa": max_stress,
                "max_disp_mm": max_disp,
            },
            "outcomes": {
                "max_von_mises_mpa": acc.get("max_von_mises_mpa"),
                "max_displacement_mm": acc.get("max_displacement_mm"),
                "solver_mode": acc.get("solver_mode"),
                "frd_bytes": acc.get("frd_bytes"),
            },
        }))
        n_ok += 1

        if (i + 1) % 10 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  {i+1}/{args.count}  accepted={n_ok} rejected={n_fail} "
                  f"errors={n_err}  {rate:.2f} designs/s")

    args.out.write_text("\n".join(accepted_rows) + ("\n" if accepted_rows else ""))

    dt = time.time() - t0
    print(f"\naccepted : {n_ok}")
    print(f"rejected : {n_fail} (constraints not met within max_iters)")
    print(f"errors   : {n_err}")
    print(f"elapsed  : {dt/60:.1f} min")
    print(f"corpus   : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
