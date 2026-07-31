"""Family-aware synthetic CAD/CFD/FEA sample generator.

This generator is intentionally lightweight, but it is no longer nozzle-only.
It can emit distinct geometric / field priors for multiple spaceflight
subfamilies so fallback data and synthetic training runs can cover broader
engineering structure: propulsion, rotating detonation, ion propulsion,
aerospikes, tanks/feed, structures, and thermal layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SPACEFLIGHT_CYCLE = (
    "nozzle",
    "aerospike",
    "rde",
    "ion_thruster",
    "tanks_feed",
    "structures",
    "thermal",
)
FAMILY_CODES = {name: idx for idx, name in enumerate(("generic",) + SPACEFLIGHT_CYCLE)}


@dataclass
class SyntheticConfig:
    num_points: int = 1024
    num_fields: int = 3
    family: str = "generic"
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


def _sample_frustum(
    rng: np.random.Generator,
    n: int,
    radius0: float,
    radius1: float,
    height: float,
    center: np.ndarray,
) -> np.ndarray:
    z = rng.uniform(-height / 2, height / 2, size=n)
    alpha = (z + height / 2) / max(height, 1e-6)
    radius = (1.0 - alpha) * radius0 + alpha * radius1
    theta = rng.uniform(0, 2 * math.pi, size=n)
    r = radius * np.sqrt(rng.uniform(0, 1, size=n))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = center[2] + z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _sample_ring_cylinders(
    rng: np.random.Generator,
    n: int,
    ring_radius: float,
    tube_radius: float,
    count: int,
    center: np.ndarray,
    z_span: float,
) -> np.ndarray:
    chunks = []
    per = max(n // max(count, 1), 1)
    for i in range(count):
        angle = 2 * math.pi * i / max(count, 1)
        offset = np.array([ring_radius * math.cos(angle), ring_radius * math.sin(angle), 0.0], dtype=np.float32)
        chunks.append(_sample_cylinder(rng, per, tube_radius, z_span, center + offset))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _generic_points(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    chunks: list[np.ndarray] = []
    per_chunk = max(cfg.num_points // max(cfg.num_primitives, 1), 1)
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
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _family_points(rng: np.random.Generator, cfg: SyntheticConfig, family: str) -> np.ndarray:
    per_chunk = max(cfg.num_points // max(cfg.num_primitives, 1), 1)
    center = np.zeros(3, dtype=np.float32)

    if family == "nozzle":
        return np.concatenate(
            [
                _sample_frustum(rng, per_chunk, 0.45, 0.18, 0.6, center + np.array([-0.45, 0.0, 0.0], dtype=np.float32)),
                _sample_cylinder(rng, per_chunk, 0.18, 0.25, center + np.array([0.0, 0.0, 0.0], dtype=np.float32)),
                _sample_frustum(rng, per_chunk, 0.18, 0.55, 0.9, center + np.array([0.55, 0.0, 0.0], dtype=np.float32)),
            ],
            axis=0,
        )
    if family == "aerospike":
        return np.concatenate(
            [
                _sample_cylinder(rng, per_chunk, 0.55, 0.35, center + np.array([-0.35, 0.0, 0.0], dtype=np.float32)),
                _sample_frustum(rng, per_chunk, 0.35, 0.03, 0.95, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)),
                _sample_box(rng, per_chunk, np.array([0.18, 0.18, 0.18], dtype=np.float32), center + np.array([-0.7, 0.0, 0.0], dtype=np.float32)),
            ],
            axis=0,
        )
    if family == "rde":
        return np.concatenate(
            [
                _sample_ring_cylinders(rng, per_chunk * 4, 0.55, 0.12, 4, center, 0.35),
                _sample_cylinder(rng, per_chunk, 0.18, 0.4, center),
                _sample_box(rng, per_chunk, np.array([0.22, 0.22, 0.08], dtype=np.float32), center + np.array([0.0, 0.0, 0.3], dtype=np.float32)),
            ],
            axis=0,
        )
    if family == "ion_thruster":
        return np.concatenate(
            [
                _sample_cylinder(rng, per_chunk, 0.14, 0.85, center),
                _sample_box(rng, per_chunk, np.array([0.25, 0.25, 0.05], dtype=np.float32), center + np.array([0.0, 0.0, -0.45], dtype=np.float32)),
                _sample_frustum(rng, per_chunk, 0.08, 0.35, 0.75, center + np.array([0.4, 0.0, 0.0], dtype=np.float32)),
            ],
            axis=0,
        )
    if family == "tanks_feed":
        return np.concatenate(
            [
                _sample_sphere(rng, per_chunk, 0.38, center + np.array([-0.55, 0.0, 0.0], dtype=np.float32)),
                _sample_sphere(rng, per_chunk, 0.38, center + np.array([0.55, 0.0, 0.0], dtype=np.float32)),
                _sample_cylinder(rng, per_chunk, 0.12, 1.15, center),
                _sample_box(rng, per_chunk, np.array([0.18, 0.35, 0.18], dtype=np.float32), center + np.array([0.0, 0.0, 0.55], dtype=np.float32)),
            ],
            axis=0,
        )
    if family == "structures":
        return np.concatenate(
            [
                _sample_box(rng, per_chunk, np.array([0.6, 0.18, 0.18], dtype=np.float32), center),
                _sample_box(rng, per_chunk, np.array([0.18, 0.6, 0.18], dtype=np.float32), center),
                _sample_box(rng, per_chunk, np.array([0.18, 0.18, 0.6], dtype=np.float32), center),
            ],
            axis=0,
        )
    if family == "thermal":
        return np.concatenate(
            [
                _sample_cylinder(rng, per_chunk, 0.45, 0.35, center),
                _sample_cylinder(rng, per_chunk, 0.25, 0.8, center + np.array([0.0, 0.0, 0.45], dtype=np.float32)),
                _sample_box(rng, per_chunk, np.array([0.55, 0.08, 0.55], dtype=np.float32), center + np.array([0.0, 0.0, -0.55], dtype=np.float32)),
            ],
            axis=0,
        )
    return _generic_points(rng, cfg)


def _family_fields(points: np.ndarray, rng: np.random.Generator, family: str, num_fields: int, scale: float) -> np.ndarray:
    """Family-aware, plausible-but-not-physically-accurate scalar fields."""
    centered = points - points.mean(axis=0, keepdims=True)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    r = np.linalg.norm(centered, axis=-1) + 1e-6
    rho = np.linalg.norm(centered[:, :2], axis=-1) + 1e-6
    theta = np.arctan2(y, x)

    fields: list[np.ndarray] = []
    if family in {"nozzle", "aerospike"}:
        throat = np.exp(-((x / 0.25) ** 2))
        plume = np.exp(-2.0 * rho) * (0.6 + 0.4 * np.cos(2.0 * theta))
        fields.extend([
            scale * throat * plume,
            scale * (z - z.min()) / (z.max() - z.min() + 1e-6),
            scale * np.exp(-((rho - np.median(rho)) ** 2) / (np.var(rho) + 1e-6)),
        ])
    elif family == "rde":
        fields.extend([
            scale * (0.5 + 0.5 * np.sin(6.0 * theta + 4.0 * rho)),
            scale * np.exp(-((rho - np.median(rho)) ** 2) / (np.var(rho) + 1e-6)),
            scale * (0.5 + 0.5 * np.cos(3.0 * x) * np.sin(2.0 * y)),
        ])
    elif family == "ion_thruster":
        beam = np.exp(-4.0 * rho) * np.exp(-1.5 * np.abs(x))
        fields.extend([
            scale * beam,
            scale * (x - x.min()) / (x.max() - x.min() + 1e-6),
            scale * np.exp(-((rho - 0.18) ** 2) / 0.03),
        ])
    elif family == "tanks_feed":
        fields.extend([
            scale * (np.exp(-((x - 0.55) ** 2) / 0.08) + np.exp(-((x + 0.55) ** 2) / 0.08)),
            scale * (z - z.min()) / (z.max() - z.min() + 1e-6),
            scale * np.exp(-np.abs(rho - 0.2)),
        ])
    elif family == "structures":
        fields.extend([
            scale * (0.5 + 0.25 * np.sin(2.0 * x) + 0.25 * np.cos(3.0 * y)),
            scale * np.exp(-np.minimum(np.abs(x), np.minimum(np.abs(y), np.abs(z)))),
            scale * (r / (r.max() + 1e-6)),
        ])
    elif family == "thermal":
        fields.extend([
            scale * (z - z.min()) / (z.max() - z.min() + 1e-6),
            scale * np.exp(-r),
            scale * (0.5 + 0.5 * np.cos(4.0 * x) * np.cos(4.0 * y)),
        ])
    else:
        fields.extend([
            scale * np.exp(-r),
            scale * (z - z.min()) / (z.max() - z.min() + 1e-6),
            scale * (0.5 + 0.5 * np.sin(3.0 * x) * np.cos(2.0 * y)),
        ])

    while len(fields) < num_fields:
        w = rng.normal(size=3)
        w /= np.linalg.norm(w) + 1e-6
        fields.append(scale * (centered @ w))
    return np.stack(fields[:num_fields], axis=-1).astype(np.float32)


def _resolve_family(cfg: SyntheticConfig, index: int) -> str:
    family = (cfg.family or "generic").strip().lower()
    if family in {"spaceflight", "mixed_space", "space"}:
        return SPACEFLIGHT_CYCLE[index % len(SPACEFLIGHT_CYCLE)]
    return family


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
    family = _resolve_family(cfg, index)
    family_code = FAMILY_CODES.get(family, 0)

    points = _family_points(rng, cfg, family)
    if points.shape[0] > cfg.num_points:
        idx = rng.choice(points.shape[0], cfg.num_points, replace=False)
        points = points[idx]
    elif points.shape[0] < cfg.num_points:
        pad = cfg.num_points - points.shape[0]
        extra_idx = rng.choice(points.shape[0], pad, replace=True)
        points = np.concatenate([points, points[extra_idx]], axis=0)

    if cfg.noise_std > 0:
        points = points + rng.normal(0, cfg.noise_std, size=points.shape).astype(np.float32)

    fields = _family_fields(points, rng, family, cfg.num_fields, cfg.field_scale)
    max_stress = float(fields[:, min(2, cfg.num_fields - 1)].max())

    return {
        "points": torch.from_numpy(points.astype(np.float32)),
        "fields": torch.from_numpy(fields.astype(np.float32)),
        "max_stress": torch.tensor(max_stress, dtype=torch.float32),
        "sample_id": torch.tensor(index, dtype=torch.long),
        "family_code": torch.tensor(family_code, dtype=torch.long),
        "is_synthetic": torch.tensor(1, dtype=torch.long),
    }
