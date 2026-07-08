"""JEPA module: context encoder, EMA target encoder, and latent predictor."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import PointCloudEncoder, pool_block_embeddings
from models.predictor import LatentPredictor


@torch.no_grad()
def ema_update(target_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    """Exponential moving average: θ_target ← decay·θ_target + (1−decay)·θ_source."""
    for p_t, p_s in zip(target_model.parameters(), source_model.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


class JEPAModel(nn.Module):
    def __init__(
        self,
        context_encoder: PointCloudEncoder,
        target_encoder: PointCloudEncoder,
        predictor: LatentPredictor,
        ema_decay: float = 0.996,
        loss_type: str = "smooth_l1",
        grid_size: tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.loss_type = loss_type
        self.grid_size = grid_size

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(copy.deepcopy(self.context_encoder.state_dict()))

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, self.ema_decay)

    def forward(
        self,
        points: torch.Tensor,
        fields: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
        target_block_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx_out = self.context_encoder(points, fields, mask=context_mask)
        ctx_tokens = ctx_out["token_embeddings"]

        with torch.no_grad():
            tgt_out = self.target_encoder(points, fields, mask=torch.ones_like(context_mask))
            tgt_tokens = tgt_out["token_embeddings"]

        batch_size, num_targets, num_points = target_masks.shape
        target_embeddings = []
        for t in range(num_targets):
            target_embeddings.append(pool_block_embeddings(tgt_tokens, target_masks[:, t]))
        target_embeddings = torch.stack(target_embeddings, dim=1)

        predicted = self.predictor(
            ctx_tokens,
            context_mask,
            target_block_ids,
            self.grid_size,
        )

        loss = self.compute_loss(predicted, target_embeddings)
        return {
            "loss": loss,
            "predicted": predicted,
            "target": target_embeddings,
            "context_pooled": ctx_out["pooled_embedding"],
        }

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "cosine":
            pred_norm = F.normalize(predicted, dim=-1)
            tgt_norm = F.normalize(target, dim=-1)
            return (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()
        return F.smooth_l1_loss(predicted, target)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "JEPAModel":
        num_fields = cfg["data"]["num_fields"]
        context_encoder = PointCloudEncoder.from_config(cfg, num_fields)
        target_encoder = PointCloudEncoder.from_config(cfg, num_fields)
        predictor = LatentPredictor.from_config(cfg)
        grid_size = tuple(cfg["masking"]["grid_size"])
        return cls(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ema_decay=cfg["model"]["ema_decay"],
            loss_type=cfg["model"]["loss_type"],
            grid_size=grid_size,
        )
