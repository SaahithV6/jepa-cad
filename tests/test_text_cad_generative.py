"""Tests for semantic text encoder + CAD assembly decoder generative head."""

import torch

from data.transforms import MaskingConfig, sample_jepa_masks
from models.cad_decoder import ASSEMBLY_PARAM_DIM, constraints_to_target_tensor, tensor_to_params_mm
from models.jepa import JEPAModel
from models.text_encoder import SemanticTextEncoder, batch_tokenize_texts, tokenize_text


def test_tokenize_stable_and_padded():
    a = tokenize_text("Rocket Fin SPAN 12.5")
    b = tokenize_text("Rocket Fin SPAN 12.5")
    assert torch.equal(a, b)
    assert a.shape == (64,)
    assert (a == 0).any()


def test_semantic_text_encoder_forward():
    enc = SemanticTextEncoder(embed_dim=32, vocab_size=512, max_tokens=16, num_layers=1, num_heads=2, dropout=0.0)
    tokens = batch_tokenize_texts(["aluminum body 40mm", "fin span 20"], max_tokens=16, vocab_size=512)
    out = enc(tokens)
    assert out["pooled_embedding"].shape == (2, 32)


def test_jepa_generative_forward_and_decode():
    cfg = {
        "data": {"num_fields": 3, "graph_metadata_dim": 0},
        "model": {
            "embed_dim": 32,
            "encoder": {
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "use_field_features": True,
            },
            "predictor": {"num_layers": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0},
            "ema_decay": 0.99,
            "loss_type": "smooth_l1",
            "enable_generative": True,
            "generative_loss_weight": 1.0,
            "text_encoder": {
                "vocab_size": 512,
                "max_tokens": 16,
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
            "cad_decoder": {"param_dim": ASSEMBLY_PARAM_DIM},
        },
        "masking": {"grid_size": [2, 2, 2]},
    }
    model = JEPAModel.from_config(cfg)
    assert model.has_generative_head
    points = torch.rand(2, 64, 3)
    fields = torch.rand(2, 64, 3)
    masks = sample_jepa_masks(points, MaskingConfig(grid_size=(2, 2, 2), num_target_blocks=2))
    tokens = batch_tokenize_texts(["body 40mm", "nose 20mm"], max_tokens=16, vocab_size=512)
    targets = torch.stack(
        [
            constraints_to_target_tensor({"body_radius_mm": 40.0, "body_height_mm": 200.0}),
            constraints_to_target_tensor({"body_radius_mm": 30.0, "body_height_mm": 150.0}),
        ]
    )
    out = model(
        points,
        fields,
        masks["context_mask"],
        masks["target_masks"],
        masks["target_block_ids"],
        text_tokens=tokens,
        assembly_targets=targets,
    )
    assert "generative_loss" in out
    assert out["assembly_pred"].shape == (2, ASSEMBLY_PARAM_DIM)
    params = tensor_to_params_mm(out["assembly_pred"][0])
    assert "body_radius_mm" in params
