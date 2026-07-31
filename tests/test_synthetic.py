"""Tests for family-aware synthetic sample generation."""

from __future__ import annotations

import torch

from data.synthetic import SyntheticConfig, generate_synthetic_sample
from data.transforms import collate_masked_batch


def test_spaceflight_synthetic_cycles_across_subfamilies() -> None:
    cfg = SyntheticConfig(num_points=64, num_fields=5, family="spaceflight", seed=7)
    a = generate_synthetic_sample(0, cfg)
    b = generate_synthetic_sample(1, cfg)

    assert a["points"].shape == (64, 3)
    assert a["fields"].shape == (64, 5)
    assert a["is_synthetic"].item() == 1
    assert a["family_code"].item() != b["family_code"].item()


def test_collate_masked_batch_preserves_family_code() -> None:
    cfg = SyntheticConfig(num_points=32, num_fields=4, family="spaceflight", seed=3)
    batch = [generate_synthetic_sample(i, cfg) for i in range(2)]
    out = collate_masked_batch(batch, {"grid_size": [4, 4, 4], "context_ratio": 0.5, "num_target_blocks": 2})

    assert out["points"].shape == (2, 32, 3)
    assert out["fields"].shape == (2, 32, 4)
    assert out["family_code"].shape == (2,)
    assert torch.equal(out["is_synthetic"], torch.ones(2, dtype=torch.long))
