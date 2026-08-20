#!/usr/bin/env python3
"""Local train: real graph samples (153-d) + generative head on accepted params.

Writes artifacts/text_cad_confirmed_train/TRAIN_METRICS.json + latest.pt.
Does not launch Modal.
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

from data.graph_dataset import GRAPH_METADATA_DIM, GraphBackedCADDataset  # noqa: E402
from data.transforms import MaskingConfig, sample_jepa_masks  # noqa: E402
from models.cad_decoder import ASSEMBLY_PARAM_KEYS, constraints_to_target_tensor  # noqa: E402
from models.jepa import JEPAModel  # noqa: E402
from models.text_encoder import batch_tokenize_texts  # noqa: E402
from utils.config import load_yaml_with_family  # noqa: E402


def _load_accepted_params(path: Path) -> list[dict]:
    rows: list[dict] = []

    # Preferred source: the swept corpus of physics-confirmed designs. Each
    # CONFIRMED_REPORT.json below holds exactly one accepted design, so before
    # this existed the head trained on three parameter vectors cycled by
    # `gens = [accepted[i % len(accepted)] ...]`. Its generative loss reached
    # ~1e-5 within 60 steps, which is memorisation of three targets rather than
    # learning to design -- and indistinguishable from success on a loss curve.
    corpus = ROOT / "artifacts/confirmed_designs/corpus.jsonl"
    if corpus.exists():
        for line in corpus.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            params = rec.get("params")
            if isinstance(params, dict) and rec.get("prompt"):
                rows.append({"params": params, "prompt": rec["prompt"]})
        if rows:
            print(f"[train_text_cad] loaded {len(rows)} confirmed designs from {corpus.name}")
            return rows

    for candidate in (
        path,
        ROOT / "artifacts/physics_confirmed/CONFIRMED_REPORT.json",
        ROOT / "artifacts/requested_rocket_assembly/CONFIRMED_REPORT.json",
    ):
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text())
        acc = data.get("accepted") or {}
        params = acc.get("params_mm")
        if isinstance(params, dict):
            rows.append(
                {
                    "params": params,
                    "prompt": data.get("prompt")
                    or (
                        f"aluminum rocket body {params.get('body_radius_mm')} mm "
                        f"height {params.get('body_height_mm')} mm"
                    ),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/text_cad_confirmed_train")
    ap.add_argument("--graph", type=Path, default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_with_family(ROOT / "configs/base.yaml")
    cfg["model"]["enable_generative"] = True
    cfg["model"]["embed_dim"] = 64
    cfg["model"]["encoder"]["num_layers"] = 2
    cfg["model"]["encoder"]["num_heads"] = 4
    cfg["model"]["predictor"]["num_layers"] = 1
    cfg["model"]["text_encoder"]["num_layers"] = 1
    cfg["model"]["text_encoder"]["num_heads"] = 4
    cfg["data"]["num_fields"] = 3
    cfg["data"]["graph_metadata_dim"] = GRAPH_METADATA_DIM
    cfg["masking"]["grid_size"] = [2, 2, 2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAModel.from_config(cfg).to(device)
    assert model.has_generative_head
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    ds = GraphBackedCADDataset(
        args.graph,
        num_points=256,
        num_fields=3,
        prefer_physics_shards=True,
        extra_search_roots=[ROOT],
        limit=64,
    )
    accepted = _load_accepted_params(ROOT / "artifacts/physics_confirmed/CONFIRMED_REPORT.json")
    if not accepted:
        accepted = [
            {
                "params": {
                    "body_radius_mm": 30.0,
                    "body_height_mm": 80.0,
                    "nose_radius_mm": 30.0,
                    "nose_height_mm": 25.0,
                    "fin_span_mm": 20.0,
                    "fin_thickness_mm": 4.0,
                    "fin_chord_mm": 33.0,
                    "fillet_radius_mm": 1.4,
                },
                "prompt": "aluminum rocket body 30 mm height 80 mm",
            }
        ]

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        # Real graph JEPA batch
        idxs = [((step - 1) * args.batch_size + i) % len(ds) for i in range(args.batch_size)]
        samples = [ds[i] for i in idxs]
        points = torch.stack([s["points"] for s in samples], dim=0).to(device)
        fields = torch.stack([s["fields"] for s in samples], dim=0).to(device)
        meta = torch.stack([s["graph_metadata"] for s in samples], dim=0).to(device)
        masks = sample_jepa_masks(
            points.cpu(),
            MaskingConfig(grid_size=(2, 2, 2), num_target_blocks=2, context_ratio=0.5),
            generator=torch.Generator().manual_seed(step),
        )
        # Generative targets from accepted physics-confirmed params (cycle through)
        gens = [accepted[i % len(accepted)] for i in range(args.batch_size)]
        tokens = batch_tokenize_texts([g["prompt"] for g in gens]).to(device)
        targets = torch.stack([constraints_to_target_tensor(g["params"]) for g in gens], dim=0).to(device)

        opt.zero_grad(set_to_none=True)
        out = model(
            points,
            fields,
            masks["context_mask"].to(device),
            masks["target_masks"].to(device),
            masks["target_block_ids"].to(device),
            graph_metadata=meta,
            text_tokens=tokens,
            assembly_targets=targets,
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
            "graph_metadata_dim": int(meta.shape[-1]),
        }
        history.append(row)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(row))

    ckpt = args.out / "latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "steps": args.steps,
            "graph_metadata_dim": GRAPH_METADATA_DIM,
            "assembly_param_keys": list(ASSEMBLY_PARAM_KEYS),
            "trained_on_confirmed": True,
        },
        ckpt,
    )
    metrics = {
        "ok": True,
        "device": str(device),
        "steps": args.steps,
        "elapsed_s": round(time.time() - t0, 3),
        "final": history[-1],
        "graph_samples": len(ds),
        "graph_metadata_dim": GRAPH_METADATA_DIM,
        "accepted_prompts": len(accepted),
        "has_generative_head": True,
        "checkpoint": str(ckpt),
        "history_tail": history[-5:],
    }
    (args.out / "TRAIN_METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
