"""Parametric synthetic CAD/CFD/FEA sample generator for smoke tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SyntheticConfig:
    num_points: int = 1024
    num_fields: int = 3
    num_primitives: int = 3
    noise_std: float = 0.05
    field_scale: float = 1.0
    seed: int | None = None


def _sample_sphere(rng: np.random.Generator, n: int, radius: float, center: np.ndarray) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi, size=n)
    phi = np.arccos(rng.uniform(-1, 1, size=n))
    r = radius * rng.uniform(0.0, 1.0, size=n) ** (1 / 3)
    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _sample_box(rng: np.random.Generator, n: int, half_extents: np.ndarray, center: np.ndarray) -> np.ndarray:
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    pts *= half_extents
    pts += center
    return pts


def _sample_cylinder(rng: np.random.Generator, n: int, radius: float, height: float, center: np.ndarray) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0, 1, size=n))
    z = center[2] + rng.uniform(-height / 2, height / 2, size=n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _synthetic_fields(points: np.ndarray, rng: np.random.Generator, num_fields: int, scale: float) -> np.ndarray:
    """Plausible but not physically accurate scalar fields."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=-1) + 1e-6
    fields = []
    # pressure-like radial decay
    fields.append(scale * np.exp(-r))
    # temperature-like gradient along z
    fields.append(scale * (z - z.min()) / (z.max() - z.min() + 1e-6))
    # von Mises stress-like oscillation
    fields.append(scale * (0.5 + 0.5 * np.sin(3 * x) * np.cos(2 * y)))
    while len(fields) < num_fields:
        w = rng.normal(size=3)
        w /= np.linalg.norm(w) + 1e-6
        fields.append(scale * (points @ w))
    return np.stack(fields[:num_fields], axis=-1).astype(np.float32)


def generate_synthetic_sample(
    index: int,
    cfg: SyntheticConfig | dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one synthetic point cloud with per-point simulation fields."""
    if cfg is None:
        cfg = SyntheticConfig()
    elif isinstance(cfg, dict):
        cfg = SyntheticConfig(**{k: v for k, v in cfg.items() if k in SyntheticConfig.__dataclass_fields__})

    seed = (cfg.seed if cfg.seed is not None else 0) + index
    rng = np.random.default_rng(seed)

    chunks: list[np.ndarray] = []
    per_chunk = max(cfg.num_points // cfg.num_primitives, 1)
    for p in range(cfg.num_primitives):
        center = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
        kind = p % 3
        if kind == 0:
            pts = _sample_sphere(rng, per_chunk, float(rng.uniform(0.2, 0.5)), center)
        elif kind == 1:
            half = rng.uniform(0.15, 0.4, size=3).astype(np.float32)
            pts = _sample_box(rng, per_chunk, half, center)
        else:
            pts = _sample_cylinder(
                rng, per_chunk, float(rng.uniform(0.15, 0.35)), float(rng.uniform(0.3, 0.8)), center
            )
        chunks.append(pts)

    points = np.concatenate(chunks, axis=0)
    if points.shape[0] > cfg.num_points:
        idx = rng.choice(points.shape[0], cfg.num_points, replace=False)
        points = points[idx]
    elif points.shape[0] < cfg.num_points:
        pad = cfg.num_points - points.shape[0]
        extra_idx = rng.choice(points.shape[0], pad, replace=True)
        points = np.concatenate([points, points[extra_idx]], axis=0)

    if cfg.noise_std > 0:
        points = points + rng.normal(0, cfg.noise_std, size=points.shape).astype(np.float32)

    fields = _synthetic_fields(points, rng, cfg.num_fields, cfg.field_scale)
    max_stress = float(fields[:, min(2, cfg.num_fields - 1)].max())

    return {
        "points": torch.from_numpy(points),
        "fields": torch.from_numpy(fields),
        "max_stress": torch.tensor(max_stress, dtype=torch.float32),
        "sample_id": torch.tensor(index, dtype=torch.long),
        "is_synthetic": torch.tensor(1, dtype=torch.long),
    }
