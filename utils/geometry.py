"""Geometry / point-cloud helper subroutines.

These utilities are intentionally framework-agnostic (PyTorch tensors in/out).
They become the backbone for later real-data ingestion and augmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


def center_and_scale_unit(points: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Center points at origin and scale to fit inside the unit sphere.

    Args:
        points: (B, N, 3) or (N, 3)
    Returns:
        normalized points, center, scale
    """
    is_batched = points.dim() == 3
    if not (points.dim() in (2, 3) and points.shape[-1] == 3):
        raise ValueError(f"points must be (N, 3) or (B, N, 3), got {points.shape}")

    if not is_batched:
        pts = points.unsqueeze(0)
    else:
        pts = points

    center = pts.mean(dim=1, keepdim=True)
    centered = pts - center
    scale = centered.norm(dim=-1).max(dim=1, keepdim=True).values.clamp_min(eps).unsqueeze(-1)
    normalized = centered / scale

    if not is_batched:
        return normalized.squeeze(0), center.squeeze(0), scale.squeeze(0)
    return normalized, center, scale


def random_rotation_matrix(generator: torch.Generator | None = None, device: torch.device | None = None) -> torch.Tensor:
    """Sample a random 3D rotation matrix (uniform over SO(3)) via QR trick."""
    device = device or torch.device("cpu")
    a = torch.randn(3, 3, generator=generator, device=device)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r)
    q = q * d.sign()
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_rigid_transform(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
    """Apply x -> xR^T + t. Supports (B,N,3) or (N,3)."""
    if points.shape[-1] != 3:
        raise ValueError("points last dim must be 3")
    if R.shape != (3, 3):
        raise ValueError("R must be (3,3)")
    if t is None:
        t = torch.zeros(3, device=points.device, dtype=points.dtype)
    if t.shape != (3,):
        raise ValueError("t must be (3,)")
    return points @ R.T + t


def jitter(points: torch.Tensor, std: float = 0.01, clip: float = 0.05, generator: torch.Generator | None = None) -> torch.Tensor:
    """Add clipped Gaussian noise."""
    noise = torch.randn_like(points, generator=generator) * std
    noise = noise.clamp(-clip, clip)
    return points + noise


def farthest_point_sample(points: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Farthest point sampling (FPS).

    Args:
        points: (N, 3)
        k: number of samples
    Returns:
        indices: (k,)
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    n = points.shape[0]
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, n)

    device = points.device
    gen = generator
    start = torch.randint(0, n, (1,), generator=gen, device=device).item()
    indices = torch.empty(k, dtype=torch.long, device=device)
    indices[0] = start

    dist = torch.full((n,), float("inf"), device=device)
    last = points[start].unsqueeze(0)
    for i in range(1, k):
        d = torch.cdist(points.unsqueeze(0), last).squeeze(0).squeeze(1)
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist).item()
        indices[i] = farthest
        last = points[farthest].unsqueeze(0)
    return indices


def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    KNN indices for each point (brute force, O(N^2)).
    Useful for debugging and tiny smoke tests.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    dist = torch.cdist(points, points)
    knn = dist.topk(k=k, largest=False).indices
    return knn


@dataclass(frozen=True)
class CropConfig:
    mode: Literal["box", "sphere"] = "box"
    ratio: float = 0.7  # fraction of points to keep (approx)


def random_crop(points: torch.Tensor, cfg: CropConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Random crop that keeps ~ratio points.

    This is a common pretext augmentation for point-cloud SSL.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    n = points.shape[0]
    keep = max(1, int(round(n * float(cfg.ratio))))
    gen = generator

    center_idx = torch.randint(0, n, (1,), generator=gen, device=points.device).item()
    center = points[center_idx]
    d = (points - center).norm(dim=-1)

    if cfg.mode == "sphere":
        _, idx = torch.topk(d, k=keep, largest=False)
        return points[idx]

    # box: choose axis-aligned box by selecting smallest distances in L_inf
    d_inf = (points - center).abs().max(dim=-1).values
    _, idx = torch.topk(d_inf, k=keep, largest=False)
    return points[idx]

