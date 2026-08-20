#!/usr/bin/env python3
"""Evaluate mission targeting using the model's own engine sizing.

Earlier evaluations rebuilt the throat area from thrust-to-weight and ignored
what the decoder predicted, which is how a corpus of zeros went unnoticed. This
flies the vehicle the model actually specified: mass ratio, structure, payload,
chamber pressure and throat area all come from the decode.
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
from generate_propulsion_trajectory_corpus import PROPELLANTS, load_coupling  # noqa: E402

STAGE1_FRACTION = 0.72
STRUCT_COEFF = 0.14
EPS1, EPS2 = 12.0, 30.0
CD = 0.42


def fly_decoded(p: dict, prop: str = "lox_rp1") -> dict:
    gamma, tc, mol = PROPELLANTS[prop]
    m_dry = p["struct_mass_kg"] + p["payload_kg"]
    mr = math.exp(p["log_mass_ratio"])
    gross = m_dry * mr
    total_prop = gross - m_dry

    p1 = total_prop * STAGE1_FRACTION
    p2 = total_prop - p1
    s1 = p1 * STRUCT_COEFF / (1 - STRUCT_COEFF)
    s2 = p2 * STRUCT_COEFF / (1 - STRUCT_COEFF)

    at1 = math.exp(p["log_throat_area_mm2"]) / 1e6      # model's engine sizing
    at2 = at1 * (p2 + s2 + p["payload_kg"]) / max(gross, 1e-6)

    pc = p["chamber_pressure_bar"] * 1e5
    stages = [Stage(p1, s1, at1, pc, EPS1, gamma, tc, mol),
              Stage(p2, s2, at2, pc, EPS2, gamma, tc, mol)]
    dia = max(0.10, (gross / 1000.0) ** (1.0 / 3.0) * 0.55)
    r = integrate_stack(stages, p["payload_kg"], cd=CD,
                        ref_area_m2=math.pi * (dia / 2) ** 2, dt=0.2)
    return {"apogee_km": r["apogee_m"] / 1000.0, "gross_kg": gross,
            "throat_mm2": at1 * 1e6, "separations": r["separations"],
            "downrange_km": r["downrange_m"] / 1000.0}


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/mission_train_v15/latest.pt"
    base = "/tmp/claude-1000/-home-lain-iwakura/6a071b7a-7311-4dbb-85f8-8d63c135e9bd/scratchpad"
    load_coupling()
    print(f"{'asked km':>9} {'flown km':>11} {'ratio':>8} {'gross kg':>10} "
          f"{'throat mm2':>11}")
    errs = []
    for A in (50, 100, 200, 400, 800, 1500, 3000, 6000, 12000):
        out = f"{base}/v15_{A}"
        subprocess.run([
            "python", "scripts/infer_text_to_assembly.py",
            "--prompt", f"two-stage vehicle delivering 8 kg payload to {A} km "
                        f"apogee using lox/rp1 at 55 bar chamber pressure",
            "--ckpt", ckpt, "--out", out],
            cwd=ROOT, capture_output=True, text=True)
        p = json.load(open(f"{out}/INFER_REPORT.json"))["decoded_params_mm"]
        r = fly_decoded(p)
        a = r["apogee_km"]
        errs.append(abs(math.log10(max(a, 0.1) / A)))
        av = f"{a:11.1f}" if math.isfinite(a) else "     escape"
        print(f"{A:9d} {av} {a/A:7.2f}x {r['gross_kg']:10.1f} {r['throat_mm2']:11.1f}")
    print()
    print(f"mean |log10 error| : {sum(errs)/len(errs):.3f} decades")
    print(f"within 2x          : {sum(1 for e in errs if e < 0.301)}/{len(errs)}")
    print(f"within 20%         : {sum(1 for e in errs if e < 0.0792)}/{len(errs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
