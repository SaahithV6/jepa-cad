"""JEPA module: context encoder, EMA target encoder, latent predictor,
optional semantic text encoder + CAD assembly decoder.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cad_decoder import CadAssemblyDecoder
from models.encoder import PointCloudEncoder, pool_block_embeddings
from models.predictor import LatentPredictor
from models.text_encoder import SemanticTextEncoder


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
        metadata_dim: int = 5,
        text_encoder: SemanticTextEncoder | None = None,
        cad_decoder: CadAssemblyDecoder | None = None,
        generative_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.loss_type = loss_type
        self.grid_size = grid_size
        self.metadata_dim = metadata_dim
        self.metadata_proj = nn.Linear(metadata_dim, predictor.embed_dim) if metadata_dim > 0 else None
        self.text_encoder = text_encoder
        self.cad_decoder = cad_decoder
        self.generative_loss_weight = float(generative_loss_weight)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self._init_target_encoder()

    @property
    def has_generative_head(self) -> bool:
        return self.text_encoder is not None and self.cad_decoder is not None

    def _init_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(copy.deepcopy(self.context_encoder.state_dict()))

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, self.ema_decay)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("text_encoder not configured")
        return self.text_encoder(text_tokens)["pooled_embedding"]

    def decode_assembly(
        self,
        geom_latent: torch.Tensor,
        text_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.cad_decoder is None:
            raise RuntimeError("cad_decoder not configured")
        return self.cad_decoder(geom_latent, text_latent)

    def forward(
        self,
        points: torch.Tensor,
        fields: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
        target_block_ids: torch.Tensor,
        graph_metadata: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        assembly_targets: torch.Tensor | None = None,
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

        conditioning = None
        if graph_metadata is not None and self.metadata_proj is not None:
            conditioning = self.metadata_proj(graph_metadata.float())

        # Optional: blend learned text embedding into predictor conditioning.
        text_pooled = None
        if text_tokens is not None and self.text_encoder is not None:
            text_pooled = self.encode_text(text_tokens)
            if conditioning is None:
                conditioning = text_pooled
            else:
                conditioning = conditioning + text_pooled

        predicted = self.predictor(
            ctx_tokens,
            context_mask,
            target_block_ids,
            self.grid_size,
            conditioning=conditioning,
        )

        jepa_loss = self.compute_loss(predicted, target_embeddings)
        loss = jepa_loss
        out: dict[str, torch.Tensor] = {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "predicted": predicted,
            "target": target_embeddings,
            "context_pooled": ctx_out["pooled_embedding"],
        }

        if (
            self.has_generative_head
            and text_tokens is not None
            and text_pooled is not None
        ):
            # Use full-cloud pooled embedding (target encoder) as geometry latent.
            with torch.no_grad():
                full = self.target_encoder(points, fields, mask=torch.ones_like(context_mask))
                geom_latent = full["pooled_embedding"]
            # Detach geom so CAD head trains against a stable JEPA target; text still trains.
            assembly_pred = self.decode_assembly(geom_latent.detach(), text_pooled)
            out["assembly_pred"] = assembly_pred
            if assembly_targets is not None:
                gen_loss = F.smooth_l1_loss(assembly_pred, assembly_targets.float())
                out["generative_loss"] = gen_loss
                loss = jepa_loss + self.generative_loss_weight * gen_loss
                out["loss"] = loss

        return out

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
        metadata_dim = int(cfg["data"].get("graph_metadata_dim", 5))
        mcfg = cfg.get("model", {})
        enable_gen = bool(mcfg.get("enable_generative", False))
        text_encoder = SemanticTextEncoder.from_config(cfg) if enable_gen else None
        cad_decoder = CadAssemblyDecoder.from_config(cfg) if enable_gen else None
        return cls(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ema_decay=mcfg["ema_decay"],
            loss_type=mcfg["loss_type"],
            grid_size=grid_size,
            metadata_dim=metadata_dim,
            text_encoder=text_encoder,
            cad_decoder=cad_decoder,
            generative_loss_weight=float(mcfg.get("generative_loss_weight", 1.0)),
        )
