#!/usr/bin/env python3
"""Evaluate the planner + model pipeline end to end.

The planner chooses the architecture, the model sizes the vehicle, and the
vehicle is flown with its own predicted engine. Prompts carry the stage count
because that is the planner's decision, not the model's -- the model's job is
mass ratio, structure and engine sizing given the architecture.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadflow.multistage import Stage, integrate_stack  # noqa: E402
from cadflow.planner import plan  # noqa: E402
from generate_propulsion_trajectory_corpus import PROPELLANTS, load_coupling  # noqa: E402

STRUCT_COEFF = 0.14
CD = 0.42
EPS_BY_STAGE = [12.0, 30.0, 60.0, 80.0]


def fly_decoded(p: dict, n_stages: int, split: list[float],
                prop: str = "lox_rp1") -> dict:
    gamma, tc, mol = PROPELLANTS[prop]
    m_dry = p["struct_mass_kg"] + p["payload_kg"]
    mr = math.exp(p["log_mass_ratio"])
    gross = m_dry * mr
    total_prop = gross - m_dry

    at1 = math.exp(p["log_throat_area_mm2"]) / 1e6
    pc = p["chamber_pressure_bar"] * 1e5

    props = [total_prop * f for f in split]
    structs = [pp * STRUCT_COEFF / (1 - STRUCT_COEFF) for pp in props]
    supported, running = [], p["payload_kg"]
    for i in range(len(props) - 1, -1, -1):
        running += props[i] + structs[i]
        supported.append(running)
    supported = list(reversed(supported))

    stages = []
    for i, (pp, ss) in enumerate(zip(props, structs)):
        eps = EPS_BY_STAGE[min(i, len(EPS_BY_STAGE) - 1)]
        # upper-stage throats scale with the mass they lift
        at = at1 * supported[i] / max(supported[0], 1e-6)
        stages.append(Stage(pp, ss, at, pc, eps, gamma, tc, mol))

    dia = max(0.10, (gross / 1000.0) ** (1.0 / 3.0) * 0.55)
    r = integrate_stack(stages, p["payload_kg"], cd=CD,
                        ref_area_m2=math.pi * (dia / 2) ** 2, dt=0.2)
    return {"apogee_km": r["apogee_m"] / 1000.0, "gross_kg": gross}


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/mission_train_v16/latest.pt"
    base = "/tmp/claude-1000/-home-lain-iwakura/6a071b7a-7311-4dbb-85f8-8d63c135e9bd/scratchpad"
    load_coupling()
    print(f"{'asked km':>9} {'stages':>7} {'flown km':>11} {'ratio':>8} {'gross kg':>10}")
    errs = []
    for A in (50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 25000, 50000):
        pl = plan(float(A), 8.0)
        if pl is None:
            print(f"{A:9d}   planner: no architecture")
            continue
        out = f"{base}/v16_{A}"
        subprocess.run([
            "python", "scripts/infer_text_to_assembly.py",
            "--prompt", f"{pl.stages}-stage vehicle delivering 8 kg payload to "
                        f"{A} km apogee using lox/rp1 at 55 bar chamber pressure",
            "--ckpt", ckpt, "--out", out],
            cwd=ROOT, capture_output=True, text=True)
        p = json.load(open(f"{out}/INFER_REPORT.json"))["decoded_params_mm"]
        r = fly_decoded(p, pl.stages, pl.split)
        a = r["apogee_km"]
        errs.append(abs(math.log10(max(a, 0.1) / A)))
        av = f"{a:11.1f}" if math.isfinite(a) else "     escape"
        print(f"{A:9d} {pl.stages:7d} {av} {a/A:7.2f}x {r['gross_kg']:10.1f}")
    print()
    print(f"mean |log10 error| : {sum(errs)/len(errs):.3f} decades")
    print(f"within 2x          : {sum(1 for e in errs if e < 0.301)}/{len(errs)}")
    print(f"within 20%         : {sum(1 for e in errs if e < 0.0792)}/{len(errs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
