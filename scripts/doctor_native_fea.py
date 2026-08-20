#!/usr/bin/env python3
"""Doctor: native CalculiX must mesh+solve a simple solid and produce FRD.

Exits 0 only when solver_mode=native and case.frd is non-trivial.
Does not launch Modal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadflow.manifest import JobManifest  # noqa: E402
from cadflow.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    out = ROOT / "artifacts" / "doctor_native_fea"
    out.mkdir(parents=True, exist_ok=True)
    manifest = JobManifest(
        name="doctor_native_fea",
        inputs={
            "geometry": {"kind": "box", "params": {"width": 40.0, "height": 12.0, "depth": 20.0}},
            "materials": ["Al6061"],
        },
        parameters={
            "solver": "fea",
            "load_n": 500.0,
            "max_stress_mpa": 500.0,
            "max_disp_mm": 5.0,
            "cl_max_mm": 5.0,
            "cl_min_mm": 1.5,
            "solver_timeout_s": 600,
            "youngs_modulus": 70e9,
            "poisson": 0.33,
        },
        tags=("doctor", "native_fea"),
    )
    result = run_pipeline(
        manifest,
        workdir=out,
        allow_solver_fallback=False,
        prefer_real_cad=True,
    )
    mode = str(result.solver_result.metadata.get("mode") or "")
    frd_bytes = int(result.solver_result.metadata.get("frd_bytes") or 0)
    report = {
        "ok": bool(result.ok and mode == "native" and frd_bytes > 1000),
        "verification_passed": bool(result.verification.passed),
        "solver_status": result.solver_result.status,
        "solver_mode": mode,
        "frd_bytes": frd_bytes,
        "max_von_mises_mpa": result.solver_result.metadata.get("max_von_mises_mpa"),
        "max_displacement_mm": result.solver_result.metadata.get("max_displacement_mm"),
        "artifacts": list(result.artifacts),
        "logs_tail": (result.solver_result.logs[-1] if result.solver_result.logs else "")[-2000:],
    }
    path = out / "DOCTOR_REPORT.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
