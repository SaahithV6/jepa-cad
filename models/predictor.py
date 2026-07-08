"""Latent predictor for JEPA target embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.encoder import MLP, TransformerBlock


class LatentPredictor(nn.Module):
    """
    Predict target block embeddings from context embeddings + target position.

    Inputs:
      context_tokens: (B, N, D)
      context_mask: (B, N) bool
      target_block_ids: (B, T) long
      grid_size: (gx, gy, gz)
    Output:
      predicted_targets: (B, T, D)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        max_blocks: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_pos_emb = nn.Embedding(max_blocks, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        target_block_ids: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size = context_tokens.shape[0]
        if target_block_ids.dim() == 1:
            target_block_ids = target_block_ids.unsqueeze(1)
        num_targets = target_block_ids.shape[1]

        gx, gy, gz = grid_size
        pos_emb = self.block_pos_emb(target_block_ids.clamp_min(0))

        context_summary = self._context_summary(context_tokens, context_mask)
        queries = self.query_proj(context_summary.unsqueeze(1) + pos_emb)

        x = torch.cat([context_tokens, queries], dim=1)
        mask = torch.cat([context_mask, torch.ones(batch_size, num_targets, dtype=torch.bool, device=x.device)], dim=1)
        key_padding_mask = ~mask

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        predicted = self.out_proj(self.norm(x[:, -num_targets:]))
        return predicted

    def _context_summary(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return (tokens * weights).sum(dim=1) / denom

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LatentPredictor":
        pred = cfg["model"]["predictor"]
        grid = cfg["masking"]["grid_size"]
        max_blocks = int(grid[0]) * int(grid[1]) * int(grid[2])
        return cls(
            embed_dim=cfg["model"]["embed_dim"],
            num_layers=pred["num_layers"],
            num_heads=pred["num_heads"],
            mlp_ratio=pred["mlp_ratio"],
            dropout=pred["dropout"],
            max_blocks=max(max_blocks, 64),
        )
