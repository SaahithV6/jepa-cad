"""Promote verified flywheel runs into curated JEPA training shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from cadflow.flywheel import DataFlywheel, FlywheelEntry
from data.parsers import ParseError, parse_raw_file


@dataclass(frozen=True, slots=True)
class PromotionResult:
    promoted: int
    skipped: int
    shard_paths: tuple[str, ...]
    manifest_path: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "skipped": self.skipped,
            "shard_paths": list(self.shard_paths),
            "manifest_path": self.manifest_path,
            "reasons": list(self.reasons),
        }


def _geometry_artifact(entry: FlywheelEntry) -> Path | None:
    for ref in entry.run.artifact_refs:
        path = Path(ref)
        if path.suffix.lower() in {".stl", ".obj", ".ply", ".step", ".stp", ".npz"} and path.exists():
            return path
    for ref in entry.manifest.artifacts:
        path = Path(ref)
        if path.suffix.lower() in {".stl", ".obj", ".ply", ".step", ".stp", ".npz"} and path.exists():
            return path
    return None


def entry_to_shard_arrays(
    entry: FlywheelEntry,
    *,
    num_points: int = 1024,
    num_fields: int = 3,
) -> dict[str, np.ndarray]:
    """Convert a verified flywheel entry into points/fields/max_stress arrays."""

    geom = _geometry_artifact(entry)
    if geom is None:
        raise ParseError("no geometry artifact available for promotion")

    sample = parse_raw_file(geom, num_points=num_points, num_fields=num_fields, allow_synthetic_fallback=False)
    fields = sample.fields.copy()
    # Inject solver objective into last field channel as a weak physics prior when present.
    if entry.solver_result.objective is not None and fields.shape[1] > 0:
        fields[:, -1] = fields[:, -1] * 0.5 + float(entry.solver_result.objective) * 0.01
    stress_col = min(2, fields.shape[1] - 1)
    max_stress = float(fields[:, stress_col].max())
    return {
        "points": sample.points.astype(np.float32),
        "fields": fields.astype(np.float32),
        "max_stress": np.array(max_stress, dtype=np.float32),
    }


def promote_verified_to_dataset(
    flywheel: DataFlywheel,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    min_score_rank: int | None = None,
) -> PromotionResult:
    """Export top verified flywheel runs into curated training shards.

    Only verified entries are eligible. Geometry artifacts must exist and parse.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = flywheel.promote_best(limit=limit or 10_000)
    if min_score_rank is not None:
        entries = entries[:min_score_rank]

    shard_paths: list[str] = []
    reasons: list[str] = []
    skipped = 0
    catalog: list[dict[str, Any]] = []

    for i, entry in enumerate(entries):
        if not entry.verified:
            skipped += 1
            reasons.append(f"{entry.manifest_fingerprint}: not verified")
            continue
        try:
            arrays = entry_to_shard_arrays(entry, num_points=num_points, num_fields=num_fields)
        except Exception as exc:  # noqa: BLE001 — record and continue
            skipped += 1
            reasons.append(f"{entry.manifest_fingerprint}: {exc}")
            continue

        name = f"curated_{i:06d}_{entry.manifest_fingerprint}.{fmt}"
        path = out_dir / name
        if fmt == "npz":
            np.savez_compressed(path, **arrays)
        else:
            import torch

            torch.save({k: torch.from_numpy(v) for k, v in arrays.items()}, path)
        shard_paths.append(str(path))
        catalog.append(
            {
                "shard": name,
                "manifest_fingerprint": entry.manifest_fingerprint,
                "objective": entry.solver_result.objective,
                "recorded_at": entry.recorded_at,
                "tags": list(entry.manifest.tags),
            }
        )

    manifest_path = out_dir / "curated_manifest.json"
    payload = {
        "num_points": num_points,
        "num_fields": num_fields,
        "format": fmt,
        "shards": [Path(p).name for p in shard_paths],
        "catalog": catalog,
        "source_flywheel": str(flywheel.path),
        "skipped": skipped,
        "reasons": reasons,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Compatibility with CADSimulationDataset / prepare_data layout
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_points": num_points,
                "num_fields": num_fields,
                "format": fmt,
                "shards": [Path(p).name for p in shard_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PromotionResult(
        promoted=len(shard_paths),
        skipped=skipped,
        shard_paths=tuple(shard_paths),
        manifest_path=str(manifest_path),
        reasons=tuple(reasons),
    )
