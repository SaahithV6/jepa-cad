"""Tests for JEPA training scale knobs."""

from __future__ import annotations

import torch

from models.jepa import JEPAModel
from train import resolve_precision


def _base_cfg() -> dict:
    return {
        "data": {"num_fields": 3},
        "masking": {"grid_size": [4, 4, 4]},
        "model": {
            "embed_dim": 64,
            "ema_decay": 0.996,
            "loss_type": "smooth_l1",
            "gradient_checkpointing": True,
            "encoder": {
                "num_layers": 2,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "use_field_features": True,
            },
            "predictor": {
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
        },
        "train": {
            "precision": "auto",
            "amp": False,
        },
    }


def test_resolve_precision_handles_cpu_auto_and_cuda_modes() -> None:
    cfg = _base_cfg()

    precision_dtype, use_scaler, mode = resolve_precision(cfg, torch.device("cpu"))
    assert precision_dtype is None
    assert use_scaler is False
    assert mode == "fp32"

    cfg["train"]["precision"] = "bf16"
    precision_dtype, use_scaler, mode = resolve_precision(cfg, torch.device("cuda"))
    assert precision_dtype == torch.bfloat16
    assert use_scaler is False
    assert mode == "bf16"

    cfg["train"]["precision"] = "fp16"
    precision_dtype, use_scaler, mode = resolve_precision(cfg, torch.device("cuda"))
    assert precision_dtype == torch.float16
    assert use_scaler is True
    assert mode == "fp16"


def test_jepa_from_config_enables_gradient_checkpointing() -> None:
    cfg = _base_cfg()
    model = JEPAModel.from_config(cfg)

    assert model.context_encoder.gradient_checkpointing is True
    assert model.target_encoder.gradient_checkpointing is True
    assert model.predictor.gradient_checkpointing is True
