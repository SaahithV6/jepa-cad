"""Convert raw CAD + CFD/FEA files into cached tensor shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.parsers import ParseError, parse_raw_file


def _load_raw_sample(
    raw_path: Path,
    num_points: int,
    num_fields: int,
    *,
    allow_synthetic_fallback: bool = False,
) -> dict[str, np.ndarray]:
    """Load CAD/mesh/VTK/NPZ inputs into points + fields arrays."""

    sample = parse_raw_file(
        raw_path,
        num_points=num_points,
        num_fields=num_fields,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    return sample.to_arrays()


def process_sample(
    raw_path: Path,
    num_points: int,
    num_fields: int,
    *,
    allow_synthetic_fallback: bool = False,
) -> dict[str, np.ndarray]:
    data = _load_raw_sample(
        raw_path,
        num_points,
        num_fields,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    points = data["points"]
    fields = data["fields"]

    if points.shape[0] != num_points:
        rng = np.random.default_rng(hash(raw_path.name) % 2**32)
        if points.shape[0] > num_points:
            idx = rng.choice(points.shape[0], num_points, replace=False)
        else:
            idx = rng.choice(points.shape[0], num_points, replace=True)
        points = points[idx]
        fields = fields[idx]

    stress_col = min(2, fields.shape[-1] - 1)
    max_stress = float(fields[:, stress_col].max())
    return {
        "points": points.astype(np.float32),
        "fields": fields.astype(np.float32),
        "max_stress": np.array(max_stress, dtype=np.float32),
    }


def save_shard(shard: dict[str, np.ndarray], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        np.savez_compressed(out_path, **shard)
    else:
        import torch

        torch.save({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in shard.items()}, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CAD/CFD/FEA tensor shards")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory of raw input files")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output shard directory")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-fields", type=int, default=3)
    parser.add_argument("--format", choices=["npz", "pt"], default="npz")
    parser.add_argument("--dry-run", action="store_true", help="Process 5 samples and print stats only")
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="If set, unsupported formats become synthetic samples (off by default)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_files = sorted([p for p in raw_dir.iterdir() if p.is_file()])
    if not raw_files:
        raise SystemExit(f"No files found in {raw_dir}")

    limit = 5 if args.dry_run else len(raw_files)
    manifest_shards: list[str] = []
    errors: list[str] = []

    for i, raw_path in enumerate(raw_files[:limit]):
        try:
            shard = process_sample(
                raw_path,
                args.num_points,
                args.num_fields,
                allow_synthetic_fallback=args.allow_synthetic_fallback,
            )
        except ParseError as exc:
            errors.append(f"{raw_path.name}: {exc}")
            print(f"[{i}] SKIP {raw_path.name}: {exc}")
            continue
        print(f"[{i}] {raw_path.name}")
        print(f"  points: {shard['points'].shape}, fields: {shard['fields'].shape}")
        print(f"  point range: [{shard['points'].min():.3f}, {shard['points'].max():.3f}]")
        print(
            f"  field mean/std: {shard['fields'].mean():.4f} / {shard['fields'].std():.4f}, "
            f"max_stress: {float(shard['max_stress']):.4f}"
        )

        if not args.dry_run:
            out_path = out_dir / f"sample_{i:06d}.{args.format}"
            save_shard(shard, out_path, args.format)
            manifest_shards.append(out_path.name)

    if args.dry_run:
        print("\nDry run complete — no shards written.")
        if errors:
            print(f"Skipped {len(errors)} files due to parse errors.")
        return

    if not manifest_shards:
        raise SystemExit(f"No shards written. Errors: {errors}")

    manifest = {
        "num_points": args.num_points,
        "num_fields": args.num_fields,
        "format": args.format,
        "shards": manifest_shards,
        "errors": errors,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest_shards)} shards to {out_dir}")


if __name__ == "__main__":
    main()
