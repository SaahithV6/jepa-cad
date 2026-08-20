#!/usr/bin/env python3
"""Local short train: JEPA + semantic text + CAD assembly decoder.

Synthetic parametric assemblies → point clouds + text prompts + param targets.
Writes checkpoint + metrics JSON. Does NOT launch Modal.

Usage:
  .venv/bin/python scripts/train_text_cad_local.py --steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transforms import MaskingConfig, sample_jepa_masks  # noqa: E402
from models.cad_decoder import (  # noqa: E402
    ASSEMBLY_PARAM_KEYS,
    constraints_to_target_tensor,
    tensor_to_params_mm,
)
from models.jepa import JEPAModel  # noqa: E402
from models.text_encoder import batch_tokenize_texts  # noqa: E402
from utils.config import load_yaml_with_family  # noqa: E402


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


def _prompt_for(c: dict[str, float]) -> str:
    return (
        f"aluminum sounding rocket body radius {c['body_radius_mm']:.1f} mm "
        f"height {c['body_height_mm']:.1f} mm nose {c['nose_height_mm']:.1f} mm "
        f"fin span {c['fin_span_mm']:.1f} mm thickness {c['fin_thickness_mm']:.1f} mm"
    )


def _points_from_constraints(c: dict[str, float], num_points: int, rng: torch.Generator) -> torch.Tensor:
    """Crude multi-primitive point cloud proxy for the assembly."""
    parts = []
    # body cylinder
    n1 = num_points // 2
    theta = torch.rand(n1, generator=rng) * 6.2832
    z = torch.rand(n1, generator=rng) * (c["body_height_mm"] / 100.0)
    r = c["body_radius_mm"] / 100.0
    parts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1))
    # nose
    n2 = num_points // 4
    theta = torch.rand(n2, generator=rng) * 6.2832
    z = c["body_height_mm"] / 100.0 + torch.rand(n2, generator=rng) * (c["nose_height_mm"] / 100.0)
    r = c["nose_radius_mm"] / 100.0 * (1.0 - (z - c["body_height_mm"] / 100.0) / max(c["nose_height_mm"] / 100.0, 1e-3))
    parts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1))
    # fin box cloud
    n3 = num_points - n1 - n2
    x = (torch.rand(n3, generator=rng) - 0.5) * (c["fin_chord_mm"] / 100.0)
    y = (torch.rand(n3, generator=rng) - 0.5) * (c["fin_thickness_mm"] / 100.0)
    z = torch.rand(n3, generator=rng) * (c["fin_span_mm"] / 100.0)
    parts.append(torch.stack([x + c["body_radius_mm"] / 100.0, y, z], dim=-1))
    return torch.cat(parts, dim=0)


def make_batch(batch_size: int, num_points: int, num_fields: int, seed: int) -> dict[str, torch.Tensor | list]:
    rng = torch.Generator().manual_seed(seed)
    constraints = [_sample_constraints(rng) for _ in range(batch_size)]
    prompts = [_prompt_for(c) for c in constraints]
    points = torch.stack([_points_from_constraints(c, num_points, rng) for c in constraints], dim=0)
    fields = torch.rand(batch_size, num_points, num_fields, generator=rng)
    targets = torch.stack([constraints_to_target_tensor(c) for c in constraints], dim=0)
    tokens = batch_tokenize_texts(prompts)
    masks = sample_jepa_masks(
        points,
        MaskingConfig(grid_size=(2, 2, 2), num_target_blocks=2, context_ratio=0.5),
        generator=torch.Generator().manual_seed(seed + 7),
    )
    return {
        "points": points,
        "fields": fields,
        "text_tokens": tokens,
        "assembly_targets": targets,
        "prompts": prompts,
        "constraints": constraints,
        **masks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "text_cad_local_train")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_with_family(args.config)
    cfg["model"]["enable_generative"] = True
    cfg["model"]["embed_dim"] = 64
    cfg["model"]["encoder"]["num_layers"] = 2
    cfg["model"]["encoder"]["num_heads"] = 4
    cfg["model"]["predictor"]["num_layers"] = 1
    cfg["model"]["text_encoder"]["num_layers"] = 1
    cfg["model"]["text_encoder"]["num_heads"] = 4
    cfg["data"]["num_fields"] = 3
    cfg["data"]["graph_metadata_dim"] = 0
    cfg["masking"]["grid_size"] = [2, 2, 2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAModel.from_config(cfg).to(device)
    assert model.has_generative_head
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = make_batch(args.batch_size, num_points=256, num_fields=3, seed=1000 + step)
        opt.zero_grad(set_to_none=True)
        out = model(
            batch["points"].to(device),
            batch["fields"].to(device),
            batch["context_mask"].to(device),
            batch["target_masks"].to(device),
            batch["target_block_ids"].to(device),
            text_tokens=batch["text_tokens"].to(device),
            assembly_targets=batch["assembly_targets"].to(device),
        )
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target_encoder()
        row = {
            "step": step,
            "loss": float(out["loss"].item()),
            "jepa_loss": float(out["jepa_loss"].item()),
            "generative_loss": float(out.get("generative_loss", torch.tensor(0.0)).item()),
            "grad_norm": float(grad_norm),
        }
        history.append(row)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(row))

    ckpt = args.out / "latest.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "steps": args.steps}, ckpt)

    # Quick decode sanity on last batch prompt
    model.eval()
    with torch.no_grad():
        pred = out["assembly_pred"][0]
        decoded = tensor_to_params_mm(pred)
    metrics = {
        "ok": True,
        "device": str(device),
        "steps": args.steps,
        "elapsed_s": round(time.time() - t0, 3),
        "final": history[-1],
        "has_generative_head": True,
        "assembly_param_keys": list(ASSEMBLY_PARAM_KEYS),
        "sample_decode_mm": decoded,
        "checkpoint": str(ckpt),
        "history_tail": history[-5:],
    }
    (args.out / "TRAIN_METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
