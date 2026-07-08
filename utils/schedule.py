"""Reusable LR schedules beyond the inline lambda."""

from __future__ import annotations

import math

import torch


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup for warmup_steps, then cosine decay to min_lr_ratio * base_lr.
    """

    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

