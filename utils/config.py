"""Config loading, validation, and CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a dict at top-level, got {type(cfg)}")
    return cfg


def deep_get(cfg: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Missing config key: {dotted_key}")
        cur = cur[part]
    return cur


def deep_set(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_override(expr: str) -> tuple[str, Any]:
    """
    Parse a CLI override expression like:
      train.lr=3e-4
      masking.grid_size=[6,6,6]
      model.loss_type=cosine

    We use YAML parsing for the RHS so lists, bools, floats work naturally.
    """
    if "=" not in expr:
        raise ConfigError(f"Override must be key=value, got: {expr}")
    key, rhs = expr.split("=", 1)
    key = key.strip()
    rhs = rhs.strip()
    if not key:
        raise ConfigError(f"Override missing key: {expr}")
    try:
        value = yaml.safe_load(rhs)
    except Exception as e:  # pragma: no cover
        raise ConfigError(f"Failed to parse override RHS as YAML: {rhs}") from e
    return key, value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    for expr in overrides:
        key, value = parse_override(expr)
        deep_set(cfg, key, value)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    # Minimal schema checks: fail early with good errors.
    required = [
        "data.num_points",
        "data.num_fields",
        "masking.grid_size",
        "masking.context_ratio",
        "masking.num_target_blocks",
        "model.embed_dim",
        "model.ema_decay",
        "train.batch_size",
        "train.lr",
        "train.weight_decay",
        "train.warmup_steps",
        "train.grad_clip",
    ]
    for k in required:
        deep_get(cfg, k)

    grid = deep_get(cfg, "masking.grid_size")
    if not (isinstance(grid, (list, tuple)) and len(grid) == 3 and all(int(x) > 0 for x in grid)):
        raise ConfigError("masking.grid_size must be [gx, gy, gz] with positive ints")

    ctx_ratio = float(deep_get(cfg, "masking.context_ratio"))
    if not (0.0 < ctx_ratio < 1.0):
        raise ConfigError("masking.context_ratio must be in (0, 1)")

    ema = float(deep_get(cfg, "model.ema_decay"))
    if not (0.0 <= ema < 1.0):
        raise ConfigError("model.ema_decay must be in [0, 1)")

    mix = float(cfg.get("data", {}).get("mix_ratio", 0.0))
    if not (0.0 <= mix <= 1.0):
        raise ConfigError("data.mix_ratio must be in [0, 1]")


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = False


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

