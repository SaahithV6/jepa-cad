from __future__ import annotations

from utils.config import load_yaml_with_family


def test_space_family_overlays_base_config() -> None:
    cfg = load_yaml_with_family("configs/base.yaml", family="space")
    assert cfg["data"]["num_points"] == 2048
    assert cfg["data"]["num_fields"] == 6
    assert cfg["masking"]["grid_size"] == [6, 6, 6]
    assert cfg["model"]["embed_dim"] == 192
    assert cfg["train"]["batch_size"] == 4
    assert cfg["logging"]["experiment_name"] == "jepa-cad-space"


def test_space_24b_family_is_available() -> None:
    cfg = load_yaml_with_family("configs/base.yaml", family="space_24b")
    assert cfg["model"]["embed_dim"] == 512
    assert cfg["model"]["gradient_checkpointing"] is True
    # Assert the effective batch, not the split. This asserted
    # grad_accum_steps == 16, which matched neither the original 1 x 32 nor the
    # 8 x 4 it was retuned to: batch_size 1 makes embed_std -- std over the
    # batch dimension -- NaN, so the collapse guard could never fire. What must
    # hold is the effective batch the schedule is built around; how it is split
    # is free, subject to batch_size > 1 keeping embed_std meaningful.
    assert cfg["train"]["batch_size"] > 1
    assert cfg["train"]["batch_size"] * cfg["train"]["grad_accum_steps"] == 32
    assert cfg["logging"]["experiment_name"] == "jepa-cad-space-24b"
