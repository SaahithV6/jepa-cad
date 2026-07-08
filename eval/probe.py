"""Linear probe for sanity-checking JEPA encoder embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import build_dataset
from models.encoder import PointCloudEncoder
from utils.checkpoint import load_checkpoint


class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def encode_batch(encoder: PointCloudEncoder, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    points = batch["points"].to(device)
    fields = batch["fields"].to(device)
    mask = torch.ones(points.shape[:2], dtype=torch.bool, device=device)
    return encoder(points, fields, mask=mask)["pooled_embedding"]


def train_probe(cfg: dict, checkpoint: str, data_source: str, device: torch.device) -> None:
    dataset = build_dataset(data_source, cfg)  # type: ignore[arg-type]
    train_len = int(len(dataset) * cfg["probe"]["train_split"])
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=cfg["probe"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["probe"]["batch_size"], shuffle=False)

    encoder = PointCloudEncoder.from_config(cfg, cfg["data"]["num_fields"])
    payload = load_checkpoint(checkpoint, encoder, device=device)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    probe = LinearProbeHead(cfg["model"]["embed_dim"]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg["probe"]["lr"])
    loss_fn = nn.MSELoss()

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

        print(
            f"epoch={epoch + 1} train_mse={train_loss / max(n_train, 1):.6f} "
            f"val_mse={val_loss / max(n_val, 1):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on frozen JEPA encoder")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-source", choices=["real", "synthetic", "mixed"], default="synthetic")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_probe(cfg, args.checkpoint, args.data_source, device)


if __name__ == "__main__":
    main()
