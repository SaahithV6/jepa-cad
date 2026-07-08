"""Dataset for lazy-loading processed CAD/CFD/FEA shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic import SyntheticConfig, generate_synthetic_sample


class CADSimulationDataset(Dataset):
    """
    Lazy-loading dataset for point-cloud geometry + per-point simulation fields.

    Canonical representation (v1): point cloud.
      - points: (N, 3) — xyz coordinates
      - fields: (N, F) — per-point scalars (pressure, temperature, stress, ...)

    Tradeoffs noted for later:
      - Voxel grid: regular structure, easy conv3d, but memory-heavy and loses detail.
      - Mesh: preserves topology, but variable connectivity complicates batching.
      - Point cloud (chosen): simple, memory-efficient, works with PointNet/transformer encoders.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        synthetic: bool = False,
        synthetic_cfg: dict[str, Any] | SyntheticConfig | None = None,
        num_synthetic: int = 256,
        shard_format: str = "npz",
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.synthetic = synthetic
        self.synthetic_cfg = synthetic_cfg or {}
        self.shard_format = shard_format
        self.shard_paths: list[Path] = []

        if synthetic:
            self._length = num_synthetic
        else:
            if self.data_dir is None or not self.data_dir.exists():
                raise FileNotFoundError(
                    f"Real data directory not found: {data_dir}. "
                    "Use synthetic=True or run data/prepare_data.py first."
                )
            pattern = "*.npz" if shard_format == "npz" else "*.pt"
            self.shard_paths = sorted(self.data_dir.glob(pattern))
            manifest = self.data_dir / "manifest.json"
            if manifest.exists():
                with open(manifest) as f:
                    meta = json.load(f)
                self.shard_paths = [self.data_dir / p for p in meta.get("shards", [])]
            if not self.shard_paths:
                raise FileNotFoundError(f"No shards found in {self.data_dir}")
            self._length = len(self.shard_paths)

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, path: Path) -> dict[str, torch.Tensor]:
        if path.suffix == ".npz":
            data = np.load(path)
            sample = {
                "points": torch.from_numpy(data["points"].astype(np.float32)),
                "fields": torch.from_numpy(data["fields"].astype(np.float32)),
                "is_synthetic": torch.tensor(0, dtype=torch.long),
            }
            if "max_stress" in data:
                sample["max_stress"] = torch.tensor(float(data["max_stress"]), dtype=torch.float32)
            else:
                stress_col = min(2, sample["fields"].shape[-1] - 1)
                sample["max_stress"] = sample["fields"][:, stress_col].max()
            return sample

        obj = torch.load(path, weights_only=True)
        obj["is_synthetic"] = torch.tensor(0, dtype=torch.long)
        return obj

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.synthetic:
            return generate_synthetic_sample(index, self.synthetic_cfg)
        path = self.shard_paths[index]
        sample = self._load_shard(path)
        sample["sample_id"] = torch.tensor(index, dtype=torch.long)
        return sample


class MixedDataset(Dataset):
    """Dataset that indexes real and synthetic sources with configurable lengths."""

    def __init__(
        self,
        real_dataset: CADSimulationDataset | None,
        synthetic_dataset: CADSimulationDataset,
        real_length: int | None = None,
        synthetic_length: int | None = None,
    ):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.real_length = len(real_dataset) if real_dataset is not None else 0
        self.synthetic_length = len(synthetic_dataset) if synthetic_dataset is not None else 0

        if real_length is not None:
            self.real_length = real_length
        if synthetic_length is not None:
            self.synthetic_length = synthetic_length

        self._length = self.real_length + self.synthetic_length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < self.real_length:
            if self.real is None:
                raise IndexError("No real dataset configured")
            return self.real[index % len(self.real)]
        synth_idx = index - self.real_length
        return self.synthetic[synth_idx % len(self.synthetic)]


def build_dataset(
    data_source: Literal["real", "synthetic", "mixed"],
    cfg: dict[str, Any],
) -> Dataset:
    """Factory for real / synthetic / mixed datasets."""
    data_cfg = cfg["data"]
    synth_cfg = {
        "num_points": data_cfg["num_points"],
        "num_fields": data_cfg["num_fields"],
        **cfg.get("synthetic", {}),
    }

    synthetic_ds = CADSimulationDataset(
        synthetic=True,
        synthetic_cfg=synth_cfg,
        num_synthetic=256,
        shard_format=data_cfg.get("shard_format", "npz"),
    )

    if data_source == "synthetic":
        return synthetic_ds

    real_ds = CADSimulationDataset(
        data_dir=data_cfg["data_dir"],
        synthetic=False,
        shard_format=data_cfg.get("shard_format", "npz"),
    )

    if data_source == "real":
        return real_ds

    mix_ratio = float(data_cfg.get("mix_ratio", 0.7))
    real_len = max(1, int(256 * mix_ratio))
    synth_len = max(1, 256 - real_len)
    return MixedDataset(real_ds, synthetic_ds, real_length=real_len, synthetic_length=synth_len)
