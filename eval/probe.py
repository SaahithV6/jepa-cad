"""Linear probe for sanity-checking JEPA encoder embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import build_dataset
from models.encoder import PointCloudEncoder
from utils.checkpoint import load_checkpoint


@dataclass(frozen=True, slots=True)
class ProbeResult:
    checkpoint: str
    data_source: str
    score_name: str
    score: float
    train_mse: float
    val_mse: float | None
    train_samples: int
    val_samples: int
    epochs: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "data_source": self.data_source,
            "score_name": self.score_name,
            "score": self.score,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "epochs": self.epochs,
            "seed": self.seed,
        }


class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def encode_batch(encoder: PointCloudEncoder, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    points = batch["points"].to(device)
    fields = batch["fields"].to(device)
    mask = torch.ones(points.shape[:2], dtype=torch.bool, device=device)
    return encoder(points, fields, mask=mask)["pooled_embedding"]


def _split_dataset(dataset: Any, split_ratio: float, seed: int):
    total = len(dataset)
    if total <= 0:
        raise ValueError("probe dataset is empty")
    if total == 1:
        return random_split(dataset, [1, 0], generator=torch.Generator().manual_seed(seed))

    train_len = int(round(total * split_ratio))
    train_len = max(1, min(train_len, total - 1))
    val_len = total - train_len
    return random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))


def probe_checkpoint(
    cfg: dict,
    checkpoint: str | Path,
    data_source: str,
    device: torch.device,
    *,
    seed: int | None = None,
    verbose: bool = True,
) -> ProbeResult:
    dataset = build_dataset(data_source, cfg)  # type: ignore[arg-type]
    train_ds, val_ds = _split_dataset(dataset, cfg["probe"]["train_split"], seed if seed is not None else int(cfg["train"]["seed"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["probe"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["probe"]["batch_size"], shuffle=False)

    encoder = PointCloudEncoder.from_config(cfg, cfg["data"]["num_fields"])
    load_checkpoint(checkpoint, encoder, device=device)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    probe = LinearProbeHead(cfg["model"]["embed_dim"]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg["probe"]["lr"])
    loss_fn = nn.MSELoss()

    last_train_mse = 0.0
    last_val_mse: float | None = None
    for epoch in range(cfg["probe"]["num_epochs"]):
        probe.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            if "max_stress" not in batch:
                continue
            emb = encode_batch(encoder, batch, device)
            target = batch["max_stress"].to(device)
            pred = probe(emb)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * target.numel()
            n_train += target.numel()

        probe.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if "max_stress" not in batch:
                    continue
                emb = encode_batch(encoder, batch, device)
                target = batch["max_stress"].to(device)
                pred = probe(emb)
                loss = loss_fn(pred, target)
                val_loss += loss.item() * target.numel()
                n_val += target.numel()

        last_train_mse = train_loss / max(n_train, 1)
        last_val_mse = (val_loss / max(n_val, 1)) if n_val else None
        if verbose:
            if last_val_mse is None:
                print(f"epoch={epoch + 1} train_mse={last_train_mse:.6f} val_mse=nan")
            else:
                print(f"epoch={epoch + 1} train_mse={last_train_mse:.6f} val_mse={last_val_mse:.6f}")

    score_name = "val_mse" if last_val_mse is not None else "train_mse"
    score = float(last_val_mse if last_val_mse is not None else last_train_mse)
    return ProbeResult(
        checkpoint=str(checkpoint),
        data_source=data_source,
        score_name=score_name,
        score=score,
        train_mse=float(last_train_mse),
        val_mse=float(last_val_mse) if last_val_mse is not None else None,
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        epochs=int(cfg["probe"]["num_epochs"]),
        seed=int(seed or cfg["train"]["seed"]),
    )


def train_probe(cfg: dict, checkpoint: str, data_source: str, device: torch.device) -> ProbeResult:
    return probe_checkpoint(cfg, checkpoint, data_source, device, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on frozen JEPA encoder")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-source", choices=["real", "synthetic", "mixed"], default="synthetic")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_probe(cfg, args.checkpoint, args.data_source, device)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"checkpoint={result.checkpoint} data_source={result.data_source} "
            f"score_name={result.score_name} score={result.score:.6f} "
            f"train_mse={result.train_mse:.6f} val_mse={result.val_mse if result.val_mse is not None else 'nan'}"
        )


if __name__ == "__main__":
    main()
