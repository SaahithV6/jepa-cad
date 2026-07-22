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
    assert cfg["train"]["grad_accum_steps"] == 16
    assert cfg["logging"]["experiment_name"] == "jepa-cad-space-24b"
