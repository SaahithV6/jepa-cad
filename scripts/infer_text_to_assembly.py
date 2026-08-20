#!/usr/bin/env python3
"""Infer: text/params prompt → assembly params → CadQuery STEP/STL + solid verify.

Loads checkpoint from scripts/train_text_cad_local.py when present; otherwise
builds an untrained generative JEPA (still exercises the decoder path).
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

from cadflow.backends import build_from_spec, get_backend  # noqa: E402
from cadflow.verification import verify_solid  # noqa: E402
from models.cad_decoder import (  # noqa: E402
    assembly_params_to_constraints,
    tensor_to_params_mm,
)
from models.jepa import JEPAModel  # noqa: E402
from models.text_encoder import batch_tokenize_texts  # noqa: E402
from utils.config import load_yaml_with_family  # noqa: E402


def _constraints_to_geometry(constraints: dict) -> dict:
    body_r = float(constraints.get("body_radius_mm", 40.0))
    body_h = float(constraints.get("body_height_mm", 200.0))
    nose_r = float(constraints.get("nose_radius_mm", body_r))
    nose_h = float(constraints.get("nose_height_mm", body_r * 1.5))
    fin_span = float(constraints.get("fin_span_mm", body_r * 0.8))
    fin_thick = float(constraints.get("fin_thickness_mm", 3.0))
    fin_chord = float(constraints.get("fin_chord_mm", body_r * 1.2))
    fillet = float(constraints.get("fillet_radius_mm", 1.0))
    return {
        "kind": "assembly",
        "parts": [
            {"kind": "cylinder", "params": {"radius": body_r, "height": body_h}},
            {"kind": "cylinder", "params": {"radius": nose_r * 0.85, "height": nose_h}},
            {
                "kind": "box",
                "params": {"width": fin_chord, "height": fin_thick, "depth": fin_span},
            },
        ],
        "features": [{"op": "fillet", "params": {"radius": max(0.1, min(fillet, fin_thick * 0.4))}}],
    }


def _sample_constraints(rng: torch.Generator) -> dict[str, float]:
    def u(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=rng).item())

    body_r = u(25.0, 60.0)
    return {
        "body_radius_mm": body_r,
        "body_height_mm": u(120.0, 320.0),
        "nose_radius_mm": body_r * u(0.7, 1.0),
        "nose_height_mm": u(40.0, 100.0),
        "fin_span_mm": body_r * u(0.5, 1.2),
        "fin_thickness_mm": u(2.0, 6.0),
        "fin_chord_mm": body_r * u(0.8, 1.5),
        "fillet_radius_mm": u(0.5, 2.5),
    }


def _points_from_constraints(c: dict[str, float], num_points: int, rng: torch.Generator) -> torch.Tensor:
    parts = []
    n1 = num_points // 2
    theta = torch.rand(n1, generator=rng) * 6.2832
    z = torch.rand(n1, generator=rng) * (c["body_height_mm"] / 100.0)
    r = c["body_radius_mm"] / 100.0
    parts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1))
    n2 = num_points // 4
    theta = torch.rand(n2, generator=rng) * 6.2832
    z = c["body_height_mm"] / 100.0 + torch.rand(n2, generator=rng) * (c["nose_height_mm"] / 100.0)
    r = (
        c["nose_radius_mm"]
        / 100.0
        * (1.0 - (z - c["body_height_mm"] / 100.0) / max(c["nose_height_mm"] / 100.0, 1e-3))
    )
    parts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1))
    n3 = num_points - n1 - n2
    x = (torch.rand(n3, generator=rng) - 0.5) * (c["fin_chord_mm"] / 100.0)
    y = (torch.rand(n3, generator=rng) - 0.5) * (c["fin_thickness_mm"] / 100.0)
    z = torch.rand(n3, generator=rng) * (c["fin_span_mm"] / 100.0)
    parts.append(torch.stack([x + c["body_radius_mm"] / 100.0, y, z], dim=-1))
    return torch.cat(parts, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/text_cad_local_train/latest.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/text_cad_infer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_with_family(ROOT / "configs/base.yaml")
    cfg["model"]["enable_generative"] = True
    cfg["data"]["graph_metadata_dim"] = 0
    if args.ckpt.exists():
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = blob.get("cfg", cfg)
        cfg["model"]["enable_generative"] = True
        model = JEPAModel.from_config(cfg)
        model.load_state_dict(blob["model"], strict=False)
        trained = True
    else:
        cfg["model"]["embed_dim"] = 64
        cfg["model"]["encoder"]["num_layers"] = 2
        cfg["model"]["predictor"]["num_layers"] = 1
        cfg["model"]["text_encoder"]["num_layers"] = 1
        model = JEPAModel.from_config(cfg)
        trained = False

    model.eval()
    rng = torch.Generator().manual_seed(0)
    # Seed geometry latent from a neutral assembly cloud; text drives the decode head.
    seed_c = _sample_constraints(rng)
    points = _points_from_constraints(seed_c, 256, rng).unsqueeze(0)
    fields = torch.zeros(1, 256, int(cfg["data"]["num_fields"]))
    prompt = args.prompt or (
        "aluminum sounding rocket body radius 42.0 mm height 220.0 mm "
        "nose 70.0 mm fin span 55.0 mm thickness 3.0 mm"
    )
    tokens = batch_tokenize_texts([prompt])

    with torch.no_grad():
        full = model.target_encoder(points, fields, mask=torch.ones(1, 256, dtype=torch.bool))
        text_lat = model.encode_text(tokens)
        pred = model.decode_assembly(full["pooled_embedding"], text_lat)[0]
        params_mm = tensor_to_params_mm(pred)

    constraints = assembly_params_to_constraints(params_mm)
    constraints["text_prompt"] = prompt
    geometry = _constraints_to_geometry(constraints)

    backend = get_backend(prefer_real=True)
    shape = build_from_spec(geometry, backend=backend)
    step = backend.export_step(shape, args.out / "geometry.step")
    stl = backend.export_stl(shape, args.out / "geometry.stl")
    verification = verify_solid(shape, backend=backend)

    report = {
        "ok": bool(verification.passed),
        "trained_checkpoint": trained,
        "ckpt": str(args.ckpt) if trained else None,
        "prompt": prompt,
        "decoded_params_mm": params_mm,
        "backend": backend.name,
        "verification_passed": bool(verification.passed),
        "artifacts": [str(step), str(stl)],
        "has_semantic_text_encoder": model.text_encoder is not None,
        "has_cad_assembly_decoder": model.cad_decoder is not None,
    }
    (args.out / "INFER_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
