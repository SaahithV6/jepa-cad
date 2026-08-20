"""Learned semantic text encoder (replaces fixed MD5 bag as the generative path).

Tokenizes UTF-8 text into a bounded vocab via stable hashing into learned
embeddings, then pools with a small Transformer. This is trainable semantics —
not the 32-d non-learned hash bag used only as JEPA conditioning.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch
import torch.nn as nn

from models.encoder import TransformerBlock, sinusoidal_positions

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\-]+|[^\s]", re.UNICODE)


# Ids [1, _NUM_BUCKETS] are reserved for numbers; hashed word tokens start
# above them. Hashing destroys magnitude: "42" and "44" blake2b to unrelated
# slots, so nothing in the model can learn that they are close, that they are
# numbers at all, or what an unseen "43" should mean. For a specification like
# "42 mm radius ... below 200 MPa" that is the whole signal. Numbers are
# therefore bucketed on a log scale, so nearby magnitudes share or neighbour a
# bucket and unseen values land sensibly between trained ones.
_NUM_BUCKETS = 256
_NUM_LOG_MIN = -2.0   # 0.01
_NUM_LOG_MAX = 7.0    # 10,000,000
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _numeric_token_id(value: float) -> int:
    """Log-scale bucket, so ordinal structure survives tokenisation."""
    if value == 0.0:
        return 1
    mag = math.log10(abs(value))
    frac = (mag - _NUM_LOG_MIN) / (_NUM_LOG_MAX - _NUM_LOG_MIN)
    frac = min(1.0, max(0.0, frac))
    return 1 + min(_NUM_BUCKETS - 1, int(frac * (_NUM_BUCKETS - 1)))


def _stable_token_id(tok: str, vocab_size: int) -> int:
    if _NUMERIC_RE.match(tok):
        try:
            return _numeric_token_id(float(tok))
        except ValueError:
            pass
    digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
    # offset past the reserved numeric range
    span = vocab_size - 1 - _NUM_BUCKETS
    return _NUM_BUCKETS + 1 + int.from_bytes(digest, "little") % span


def tokenize_text(text: str, *, max_tokens: int = 64, vocab_size: int = 4096) -> torch.Tensor:
    """Map text → int64 token ids in ``[1, vocab_size-1]`` (0 = pad)."""
    tokens = _TOKEN_RE.findall(text.lower()) if text else []
    ids: list[int] = []
    for tok in tokens[:max_tokens]:
        ids.append(_stable_token_id(tok, vocab_size))
    if not ids:
        ids = [1]  # unknown/empty sentinel
    out = torch.zeros(max_tokens, dtype=torch.long)
    out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def batch_tokenize_texts(texts: list[str], *, max_tokens: int = 64, vocab_size: int = 4096) -> torch.Tensor:
    return torch.stack([tokenize_text(t, max_tokens=max_tokens, vocab_size=vocab_size) for t in texts], dim=0)


class SemanticTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 4096,
        max_tokens: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        token_ids: (B, T) long
        returns token_embeddings (B,T,D) and pooled (B,D)
        """
        pad_mask = token_ids == 0  # True = pad (key_padding_mask)
        x = self.token_emb(token_ids)
        x = x + sinusoidal_positions(token_ids.shape[1], self.embed_dim, token_ids.device).unsqueeze(0)
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        x = self.norm(x)
        # Mean pool over non-pad tokens.
        weights = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {"token_embeddings": x, "pooled_embedding": pooled}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SemanticTextEncoder":
        m = cfg.get("model", {})
        t = m.get("text_encoder", {})
        return cls(
            embed_dim=int(m.get("embed_dim", 128)),
            vocab_size=int(t.get("vocab_size", 4096)),
            max_tokens=int(t.get("max_tokens", 64)),
            num_layers=int(t.get("num_layers", 2)),
            num_heads=int(t.get("num_heads", m.get("encoder", {}).get("num_heads", 4))),
            mlp_ratio=float(t.get("mlp_ratio", 2.0)),
            dropout=float(t.get("dropout", 0.1)),
        )
