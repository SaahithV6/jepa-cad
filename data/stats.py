"""Dataset inspection and integrity checks.

Useful before committing to long runs: verify shapes, NaNs, value ranges,
and field statistics over a subset of shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataset import CADSimulationDataset


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "nan_frac": float(torch.isnan(x).float().mean().item()),
    }


def inspect_dataset(ds: CADSimulationDataset, limit: int = 32) -> None:
    n = min(len(ds), limit)
    pts_stats = []
    fld_stats = []
    max_stress = []
    for i in range(n):
        s = ds[i]
        pts_stats.append(summarize_tensor(s["points"]))
        fld_stats.append(summarize_tensor(s["fields"]))
        if "max_stress" in s:
            max_stress.append(float(s["max_stress"].item()))

    def agg(stats: list[dict[str, float]]) -> dict[str, float]:
        keys = stats[0].keys()
        return {k: float(np.mean([d[k] for d in stats])) for k in keys}

    print(f"Samples inspected: {n}")
    print("points avg stats:", agg(pts_stats))
    print("fields avg stats:", agg(fld_stats))
    if max_stress:
        print(
            f"max_stress: mean={np.mean(max_stress):.4f} std={np.std(max_stress):.4f} "
            f"min={np.min(max_stress):.4f} max={np.max(max_stress):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect processed shards for sanity checks")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--format", type=str, default="npz", choices=["npz", "pt"])
    args = parser.parse_args()

    ds = CADSimulationDataset(data_dir=Path(args.data_dir), synthetic=False, shard_format=args.format)
    inspect_dataset(ds, limit=args.limit)


if __name__ == "__main__":
    main()

