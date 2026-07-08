"""Point-cloud encoder for JEPA context and target branches."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_positions(num_tokens: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_tokens, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class PointCloudEncoder(nn.Module):
    """
    Lightweight point transformer encoder.

    Input per batch:
      points: (B, N, 3)
      fields: (B, N, F) optional simulation annotations
      mask: (B, N) bool — True = visible token
    Output:
      token_embeddings: (B, N, D)
      pooled_embedding: (B, D)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_fields: int = 3,
        use_field_features: bool = True,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = 3 + (num_fields if use_field_features else 0)
        self.use_field_features = use_field_features
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        points: torch.Tensor,
        fields: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.use_field_features:
            if fields is None:
                raise ValueError("fields required when use_field_features=True")
            feats = torch.cat([points, fields], dim=-1)
        else:
            feats = points

        x = self.input_proj(feats)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device).unsqueeze(0)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        if mask is not None and mask.any(dim=1).all():
            masked = x * mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            pooled = masked.sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        return {"token_embeddings": x, "pooled_embedding": pooled}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_fields: int) -> "PointCloudEncoder":
        enc = cfg["model"]["encoder"]
        return cls(
            embed_dim=cfg["model"]["embed_dim"],
            num_fields=num_fields,
            use_field_features=enc.get("use_field_features", True),
            num_layers=enc["num_layers"],
            num_heads=enc["num_heads"],
            mlp_ratio=enc["mlp_ratio"],
            dropout=enc["dropout"],
        )


def pool_block_embeddings(
    token_embeddings: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings over a boolean block mask -> (B, D)."""
    weights = block_mask.unsqueeze(-1).float()
    denom = block_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    return (token_embeddings * weights).sum(dim=1) / denom
