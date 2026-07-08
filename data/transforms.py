"""JEPA block masking for point cloud geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MaskingConfig:
    grid_size: tuple[int, int, int] = (4, 4, 4)
    context_ratio: float = 0.5
    num_target_blocks: int = 4
    min_target_block_ratio: float = 0.05


def _normalize_points(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map points to [0, 1]^3 per sample for block indexing."""
    mins = points.min(dim=1, keepdim=True).values
    maxs = points.max(dim=1, keepdim=True).values
    span = (maxs - mins).clamp_min(1e-6)
    normalized = (points - mins) / span
    return normalized, mins, span


def _block_indices(normalized_points: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
    """Assign each point to a spatial block id in [0, num_blocks)."""
    gx, gy, gz = grid_size
    coords = (normalized_points.clamp(0, 0.999) * torch.tensor(grid_size, device=normalized_points.device, dtype=normalized_points.dtype)).long()
    bx, by, bz = coords[..., 0], coords[..., 1], coords[..., 2]
    block_ids = bx * (gy * gz) + by * gz + bz
    return block_ids


def points_to_block_ids(points: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
    """Public helper: (B, N, 3) -> (B, N) block ids."""
    normalized, _, _ = _normalize_points(points)
    return _block_indices(normalized, grid_size)


def sample_jepa_masks(
    points: torch.Tensor,
    cfg: MaskingConfig | dict[str, Any],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    Sample context and target block masks for a batch of point clouds.

    Returns:
        context_mask: (B, N) bool — visible to context encoder
        target_masks: (B, T, N) bool — one mask per target block
        target_block_ids: (B, T) long — block index for each target
    """
    if isinstance(cfg, dict):
        cfg = MaskingConfig(
            grid_size=tuple(cfg.get("grid_size", (4, 4, 4))),
            context_ratio=float(cfg.get("context_ratio", 0.5)),
            num_target_blocks=int(cfg.get("num_target_blocks", 4)),
            min_target_block_ratio=float(cfg.get("min_target_block_ratio", 0.05)),
        )

    if points.dim() != 3:
        raise ValueError(f"points must be (B, N, 3), got {points.shape}")

    batch_size, num_points, _ = points.shape
    device = points.device
    gx, gy, gz = cfg.grid_size
    num_blocks = gx * gy * gz

    normalized, _, _ = _normalize_points(points)
    block_ids = _block_indices(normalized, cfg.grid_size)

    context_mask = torch.zeros(batch_size, num_points, dtype=torch.bool, device=device)
    target_masks = torch.zeros(batch_size, cfg.num_target_blocks, num_points, dtype=torch.bool, device=device)
    target_block_ids = torch.full(
        (batch_size, cfg.num_target_blocks), -1, dtype=torch.long, device=device
    )

    for b in range(batch_size):
        unique_blocks = torch.unique(block_ids[b])
        if unique_blocks.numel() == 0:
            continue

        perm = torch.randperm(unique_blocks.numel(), generator=generator, device=device)
        shuffled = unique_blocks[perm]

        n_context = max(1, int(round(cfg.context_ratio * shuffled.numel())))
        n_context = min(n_context, shuffled.numel() - 1) if shuffled.numel() > 1 else 1
        context_blocks = set(shuffled[:n_context].tolist())

        remaining = [blk for blk in shuffled.tolist() if blk not in context_blocks]
        if not remaining:
            remaining = shuffled.tolist()

        target_blocks: list[int] = []
        for candidate in remaining:
            if len(target_blocks) >= cfg.num_target_blocks:
                break
            count = (block_ids[b] == candidate).sum().item()
            if count / num_points >= cfg.min_target_block_ratio:
                target_blocks.append(candidate)

        while len(target_blocks) < cfg.num_target_blocks and remaining:
            for candidate in remaining:
                if candidate not in target_blocks:
                    target_blocks.append(candidate)
                    break
            else:
                break

        if not target_blocks:
            target_blocks = [remaining[0]]

        ctx_mask_b = torch.zeros(num_points, dtype=torch.bool, device=device)
        for blk in context_blocks:
            ctx_mask_b |= block_ids[b] == blk
        if not ctx_mask_b.any():
            ctx_mask_b = torch.ones(num_points, dtype=torch.bool, device=device)
        context_mask[b] = ctx_mask_b

        for t, blk in enumerate(target_blocks[: cfg.num_target_blocks]):
            target_block_ids[b, t] = blk
            target_masks[b, t] = block_ids[b] == blk

    return {
        "context_mask": context_mask,
        "target_masks": target_masks,
        "target_block_ids": target_block_ids,
        "block_ids": block_ids,
    }


def collate_masked_batch(
    batch: list[dict[str, torch.Tensor]],
    masking_cfg: MaskingConfig | dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Collate raw samples and apply JEPA masking."""
    points = torch.stack([item["points"] for item in batch], dim=0)
    fields = torch.stack([item["fields"] for item in batch], dim=0)
    masks = sample_jepa_masks(points, masking_cfg)

    out = {
        "points": points,
        "fields": fields,
        "context_mask": masks["context_mask"],
        "target_masks": masks["target_masks"],
        "target_block_ids": masks["target_block_ids"],
        "block_ids": masks["block_ids"],
    }
    if "max_stress" in batch[0]:
        out["max_stress"] = torch.stack([item["max_stress"] for item in batch])
    if "is_synthetic" in batch[0]:
        out["is_synthetic"] = torch.stack([item["is_synthetic"] for item in batch])
    return out
