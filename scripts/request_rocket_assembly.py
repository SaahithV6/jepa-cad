#!/usr/bin/env python3
"""Request rocket parameters → assembly CAD with physics gate.

Default perfection bar: --no-fallback (native CalculiX required).
Use --allow-fallback only for debugging decks.

Examples:
  .venv/bin/python scripts/request_rocket_assembly.py --no-fallback \\
      --body-radius-mm 30 --body-height-mm 80 --nose-height-mm 25 --fin-span-mm 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.cad_decoder import assembly_params_to_constraints, tensor_to_params_mm  # noqa: E402
from models.jepa import JEPAModel  # noqa: E402
from models.text_encoder import batch_tokenize_texts  # noqa: E402
from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402
from utils.config import load_yaml_with_family  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/requested_rocket_assembly")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--body-radius-mm", type=float, default=None)
    ap.add_argument("--body-height-mm", type=float, default=None)
    ap.add_argument("--nose-height-mm", type=float, default=None)
    ap.add_argument("--fin-span-mm", type=float, default=None)
    ap.add_argument("--fin-thickness-mm", type=float, default=None)
    ap.add_argument("--max-stress-mpa", type=float, default=250.0)
    ap.add_argument("--max-disp-mm", type=float, default=2.0)
    ap.add_argument("--load-n", type=float, default=500.0)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/text_cad_local_train/latest.pt")
    ap.add_argument("--use-neural", action="store_true")
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        default=True,
        help="Require native CalculiX (default true)",
    )
    ap.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Permit adapter fallback (fails perfection bar)",
    )
    args = ap.parse_args()
    require_native = not args.allow_fallback
    args.out.mkdir(parents=True, exist_ok=True)

    typed = {
        k: v
        for k, v in {
            "body_radius_mm": args.body_radius_mm,
            "body_height_mm": args.body_height_mm,
            "nose_height_mm": args.nose_height_mm,
            "fin_span_mm": args.fin_span_mm,
            "fin_thickness_mm": args.fin_thickness_mm,
        }.items()
        if v is not None
    }
    prompt = args.prompt or (
        "aluminum sounding rocket "
        + " ".join(f"{k.replace('_', ' ')} {v}" for k, v in typed.items())
        if typed
        else "aluminum sounding rocket body radius 30 mm height 80 mm nose 25 mm fin span 20 mm"
    )

    source = "typed_params"
    params_mm = {
        "body_radius_mm": float(typed.get("body_radius_mm", 30.0)),
        "body_height_mm": float(typed.get("body_height_mm", 80.0)),
        "nose_radius_mm": float(typed.get("body_radius_mm", 30.0)),
        "nose_height_mm": float(typed.get("nose_height_mm", 25.0)),
        "fin_span_mm": float(typed.get("fin_span_mm", 20.0)),
        "fin_thickness_mm": float(typed.get("fin_thickness_mm", 4.0)),
        "fin_chord_mm": float(typed.get("body_radius_mm", 30.0)) * 1.1,
        "fillet_radius_mm": 1.2,
        "cl_max_mm": 8.0,
        "cl_min_mm": 2.0,
    }

    if args.use_neural and args.ckpt.exists():
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = blob.get("cfg") or load_yaml_with_family(ROOT / "configs/base.yaml")
        cfg["model"]["enable_generative"] = True
        cfg["data"]["graph_metadata_dim"] = 0
        model = JEPAModel.from_config(cfg)
        model.load_state_dict(blob["model"], strict=False)
        model.eval()
        points = torch.rand(1, 256, 3)
        fields = torch.zeros(1, 256, int(cfg["data"].get("num_fields", 3)))
        tokens = batch_tokenize_texts([prompt])
        with torch.no_grad():
            geom = model.target_encoder(points, fields, mask=torch.ones(1, 256, dtype=torch.bool))
            text = model.encode_text(tokens)
            pred = model.decode_assembly(geom["pooled_embedding"], text)[0]
            decoded = tensor_to_params_mm(pred)
        params_mm.update(decoded)
        params_mm.update(typed)
        if "body_radius_mm" in typed:
            params_mm["nose_radius_mm"] = float(typed["body_radius_mm"])
        source = "neural_plus_typed" if typed else "neural"

    (args.out / "request.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "typed": typed,
                "params_mm": params_mm,
                "source": source,
                "require_native": require_native,
            },
            indent=2,
        )
        + "\n"
    )

    if require_native:
        report = run_confirmed(
            params_mm=params_mm,
            out=args.out,
            max_stress_mpa=args.max_stress_mpa,
            max_disp_mm=args.max_disp_mm,
            max_iters=args.max_iters,
            load_n=args.load_n,
            prompt=prompt,
        )
        report["source"] = source
        report["require_native"] = True
        # Fail closed if not native-confirmed
        if not report.get("ok") or report.get("solver_mode") != "native":
            report["ok"] = False
            report["fail_closed"] = "non_native_or_targets_missed"
        (args.out / "REQUEST_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        # Also keep CONFIRMED_REPORT from run_confirmed
        print(json.dumps({k: report.get(k) for k in ("ok", "solver_mode", "source", "accepted", "fail_closed")}, indent=2))
        return 0 if report.get("ok") else 1

    # Debug path: allow fallback via pipeline directly
    from cadflow.manifest import JobManifest
    from cadflow.pipeline import run_pipeline
    from scripts.smoke_params_to_assembly import constraints_to_geometry

    constraints = assembly_params_to_constraints(params_mm)
    constraints["text_prompt"] = prompt
    geometry = constraints_to_geometry(constraints)
    manifest = JobManifest(
        name="request_rocket_assembly_fallback",
        inputs={"geometry": geometry, "materials": ["Al6061"]},
        parameters={"solver": "fea", "constraints": constraints, "text_prompt": prompt},
        tags=("request", "fallback"),
    )
    result = run_pipeline(manifest, workdir=args.out, allow_solver_fallback=True, prefer_real_cad=True)
    report = {
        "ok": bool(result.ok),
        "source": source,
        "require_native": False,
        "solver_mode": result.solver_result.metadata.get("mode"),
        "artifacts": list(result.artifacts),
        "note": "allow_fallback path — does not satisfy perfection bar",
    }
    (args.out / "REQUEST_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
