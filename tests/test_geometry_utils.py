import torch

from utils.geometry import CropConfig, center_and_scale_unit, farthest_point_sample, random_crop


def test_center_and_scale_unit_shapes():
    pts = torch.rand(128, 3)
    norm, center, scale = center_and_scale_unit(pts)
    assert norm.shape == pts.shape
    assert center.shape == (1, 3) or center.shape == (3,)
    assert scale.shape[-1] == 1 or scale.shape == (1,)


def test_center_and_scale_unit_unit_sphere():
    pts = torch.randn(256, 3) * 10.0 + 5.0
    norm, _, _ = center_and_scale_unit(pts)
    r = norm.norm(dim=-1).max().item()
    assert r <= 1.0001


def test_fps_unique_indices():
    pts = torch.rand(64, 3)
    idx = farthest_point_sample(pts, k=16, generator=torch.Generator().manual_seed(0))
    assert idx.shape == (16,)
    assert torch.unique(idx).numel() == 16


def test_random_crop_keeps_ratio():
    pts = torch.rand(100, 3)
    cropped = random_crop(pts, cfg=CropConfig(mode="box", ratio=0.5), generator=torch.Generator().manual_seed(0))
    assert 40 <= cropped.shape[0] <= 60

