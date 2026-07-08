"""Unit tests for JEPA masking and EMA update."""

import copy

import torch

from data.transforms import MaskingConfig, sample_jepa_masks
from models.jepa import JEPAModel, ema_update
from models.encoder import PointCloudEncoder
from models.predictor import LatentPredictor


def _random_points(batch: int = 2, num_points: int = 256) -> torch.Tensor:
    return torch.rand(batch, num_points, 3)


def test_masks_are_disjoint_from_targets():
    cfg = MaskingConfig(grid_size=(4, 4, 4), context_ratio=0.5, num_target_blocks=3)
    points = _random_points()
    masks = sample_jepa_masks(points, cfg, generator=torch.Generator().manual_seed(0))

    context = masks["context_mask"]
    targets = masks["target_masks"]
    for t in range(targets.shape[1]):
        overlap = (context & targets[:, t]).any(dim=1)
        assert not overlap.any(), "context and target blocks should not overlap"


def test_target_masks_cover_points():
    cfg = MaskingConfig(grid_size=(4, 4, 4), num_target_blocks=4)
    points = _random_points(batch=4, num_points=512)
    masks = sample_jepa_masks(points, cfg, generator=torch.Generator().manual_seed(1))
    target_masks = masks["target_masks"]
    assert target_masks.dtype == torch.bool
    assert target_masks.shape[0] == points.shape[0]
    assert target_masks.shape[2] == points.shape[1]
    assert target_masks.any(dim=2).all(), "each target block should cover at least one point"


def test_context_mask_nonempty():
    points = _random_points()
    masks = sample_jepa_masks(points, MaskingConfig(), generator=torch.Generator().manual_seed(2))
    assert masks["context_mask"].any(dim=1).all()


def test_ema_update_moves_target_toward_source():
    enc = PointCloudEncoder(embed_dim=32, num_fields=3, num_layers=1, num_heads=2)
    tgt = copy.deepcopy(enc)
    before = next(tgt.parameters()).clone()

    for p in enc.parameters():
        p.data.add_(1.0)

    ema_update(tgt, enc, decay=0.9)
    after = next(tgt.parameters())
    diff_before = (before - after).abs().mean().item()
    assert diff_before > 0.0, "EMA should change target weights"


def test_ema_decay_zero_copies_source():
    enc = PointCloudEncoder(embed_dim=16, num_fields=3, num_layers=1, num_heads=2)
    tgt = copy.deepcopy(enc)
    enc.input_proj.weight.data.fill_(3.14)
    ema_update(tgt, enc, decay=0.0)
    assert torch.allclose(tgt.input_proj.weight, enc.input_proj.weight)


def test_jepa_forward_and_ema_integration():
    cfg = {
        "data": {"num_fields": 3},
        "model": {
            "embed_dim": 32,
            "encoder": {"num_layers": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0, "use_field_features": True},
            "predictor": {"num_layers": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0},
            "ema_decay": 0.99,
            "loss_type": "smooth_l1",
        },
        "masking": {"grid_size": [2, 2, 2]},
    }
    model = JEPAModel.from_config(cfg)
    points = _random_points(batch=2, num_points=128)
    fields = torch.rand(2, 128, 3)
    masks = sample_jepa_masks(points, MaskingConfig(grid_size=(2, 2, 2), num_target_blocks=2))
    out = model(points, fields, masks["context_mask"], masks["target_masks"], masks["target_block_ids"])
    assert out["loss"].ndim == 0
    assert out["predicted"].shape == out["target"].shape
    model.update_target_encoder()
