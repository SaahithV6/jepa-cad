"""Large-scale ingestion for CAD/CAE + verified flywheel data.

This module turns heterogeneous raw sources into a single curated shard
folder that the JEPA training stack can consume directly.

Supported sources:
  - Raw geometry / field files accepted by `data.parsers.parse_raw_file`
  - Verified flywheel entries via `cadflow.flywheel.DataFlywheel`

Outputs are written in the same on-disk layout expected by
`data.dataset.CADSimulationDataset`:
  - `manifest.json`
  - shard files (`.npz` or `.pt`)

The manifest is intentionally compatible with the existing training code,
while `ingestion_manifest.json` keeps richer provenance for downstream audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import numpy as np

from cadflow.flywheel import DataFlywheel
from cadflow.promotion import entry_to_shard_arrays
from data.parsers import CAD_SUFFIXES, FIELD_SUFFIXES, MESH_SUFFIXES, ParseError
from data.prepare_data import process_sample, save_shard

SUPPORTED_SUFFIXES = {".npz", ".pt", *CAD_SUFFIXES, *FIELD_SUFFIXES, *MESH_SUFFIXES}


@dataclass(frozen=True, slots=True)
class IngestionResult:
    ingested: int
    skipped: int
    shard_paths: tuple[str, ...]
    manifest_path: str
    reasons: tuple[str, ...]
    sources: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingested": self.ingested,
            "skipped": self.skipped,
            "shard_paths": list(self.shard_paths),
            "manifest_path": self.manifest_path,
            "reasons": list(self.reasons),
            "sources": list(self.sources),
        }


def _iter_input_files(raw_dirs: Sequence[str | Path], recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    for raw_dir in raw_dirs:
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"raw input directory not found: {root}")
        if recursive:
            candidates = (p for p in root.rglob("*") if p.is_file())
        else:
            candidates = (p for p in root.iterdir() if p.is_file())
        for path in candidates:
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
    return sorted(files)


def _save_arrays(
    arrays: dict[str, np.ndarray],
    out_path: Path,
    fmt: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        save_npz = cast(Any, np.savez_compressed)
        save_npz(str(out_path), **arrays)
    else:
        import torch

        torch.save({k: torch.from_numpy(v) for k, v in arrays.items()}, out_path)


def ingest_raw_sources(
    raw_dirs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    recursive: bool = True,
    limit: int | None = None,
    allow_synthetic_fallback: bool = False,
) -> IngestionResult:
    """Convert raw CAD/CFD/FEA files into a curated shard directory."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(raw_dirs, recursive=recursive)
    if limit is not None:
        files = files[: max(0, limit)]

    shard_paths: list[str] = []
    sources: list[dict[str, Any]] = []
    reasons: list[str] = []
    skipped = 0

    for index, raw_path in enumerate(files):
        try:
            sample = process_sample(
                raw_path,
                num_points=num_points,
                num_fields=num_fields,
                allow_synthetic_fallback=allow_synthetic_fallback,
            )
        except ParseError as exc:
            skipped += 1
            reasons.append(f"{raw_path}: {exc}")
            continue

        out_path = out_dir / f"raw_{index:06d}_{raw_path.stem}.{fmt}"
        _save_arrays(sample, out_path, fmt)
        shard_paths.append(str(out_path))
        sources.append(
            {
                "kind": "raw",
                "source_path": str(raw_path),
                "shard": out_path.name,
                "format": fmt,
            }
        )

    payload = {
        "num_points": num_points,
        "num_fields": num_fields,
        "format": fmt,
        "shards": [Path(p).name for p in shard_paths],
        "sources": sources,
        "skipped": skipped,
        "reasons": reasons,
        "raw_dirs": [str(Path(d)) for d in raw_dirs],
    }
    (out_dir / "ingestion_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    return IngestionResult(
        ingested=len(shard_paths),
        skipped=skipped,
        shard_paths=tuple(shard_paths),
        manifest_path=str(out_dir / "ingestion_manifest.json"),
        reasons=tuple(reasons),
        sources=tuple(sources),
    )


def ingest_verified_flywheel(
    flywheel: DataFlywheel,
    out_dir: str | Path,
    *,
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    limit: int | None = None,
) -> IngestionResult:
    """Convert verified flywheel entries into curated training shards."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [entry for entry in flywheel.load_entries() if entry.verification.passed and entry.solver_result.ok]
    if limit is not None:
        entries = entries[:limit]
    shard_paths: list[str] = []
    sources: list[dict[str, Any]] = []
    reasons: list[str] = []
    skipped = 0

    for index, entry in enumerate(entries):
        try:
            arrays = entry_to_shard_arrays(entry, num_points=num_points, num_fields=num_fields)
        except Exception as exc:  # noqa: BLE001 - record and continue
            skipped += 1
            reasons.append(f"{entry.manifest_fingerprint}: {exc}")
            continue

        out_path = out_dir / f"flywheel_{index:06d}_{entry.manifest_fingerprint}.{fmt}"
        _save_arrays(arrays, out_path, fmt)
        shard_paths.append(str(out_path))
        sources.append(
            {
                "kind": "flywheel",
                "manifest_fingerprint": entry.manifest_fingerprint,
                "recorded_at": entry.recorded_at,
                "shard": out_path.name,
                "format": fmt,
            }
        )

    payload = {
        "num_points": num_points,
        "num_fields": num_fields,
        "format": fmt,
        "shards": [Path(p).name for p in shard_paths],
        "sources": sources,
        "skipped": skipped,
        "reasons": reasons,
        "source_flywheel": str(flywheel.path),
    }
    (out_dir / "ingestion_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    return IngestionResult(
        ingested=len(shard_paths),
        skipped=skipped,
        shard_paths=tuple(shard_paths),
        manifest_path=str(out_dir / "ingestion_manifest.json"),
        reasons=tuple(reasons),
        sources=tuple(sources),
    )


def ingest_sources(
    raw_dirs: Sequence[str | Path] | None,
    out_dir: str | Path,
    *,
    flywheel_path: str | Path | None = None,
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    recursive: bool = True,
    limit: int | None = None,
    allow_synthetic_fallback: bool = False,
) -> IngestionResult:
    """Ingest raw directories and verified flywheel entries into one shard set."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    sources: list[dict[str, Any]] = []
    reasons: list[str] = []
    skipped = 0

    raw_result = None
    if raw_dirs:
        raw_result = ingest_raw_sources(
            raw_dirs,
            out_dir,
            num_points=num_points,
            num_fields=num_fields,
            fmt=fmt,
            recursive=recursive,
            limit=limit,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        shard_paths.extend(raw_result.shard_paths)
        sources.extend(raw_result.sources)
        reasons.extend(raw_result.reasons)
        skipped += raw_result.skipped

    flywheel_result = None
    if flywheel_path is not None:
        flywheel = DataFlywheel(flywheel_path)
        flywheel_result = ingest_verified_flywheel(
            flywheel,
            out_dir,
            num_points=num_points,
            num_fields=num_fields,
            fmt=fmt,
            limit=limit,
        )
        shard_paths.extend(flywheel_result.shard_paths)
        sources.extend(flywheel_result.sources)
        reasons.extend(flywheel_result.reasons)
        skipped += flywheel_result.skipped

    payload = {
        "num_points": num_points,
        "num_fields": num_fields,
        "format": fmt,
        "shards": [Path(p).name for p in shard_paths],
        "sources": sources,
        "raw_dirs": [str(Path(d)) for d in raw_dirs or []],
        "flywheel_path": str(flywheel_path) if flywheel_path is not None else None,
        "skipped": skipped,
        "reasons": reasons,
        "raw_result": raw_result.to_dict() if raw_result is not None else None,
        "flywheel_result": flywheel_result.to_dict() if flywheel_result is not None else None,
    }
    (out_dir / "ingestion_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    return IngestionResult(
        ingested=len(shard_paths),
        skipped=skipped,
        shard_paths=tuple(shard_paths),
        manifest_path=str(out_dir / "ingestion_manifest.json"),
        reasons=tuple(reasons),
        sources=tuple(sources),
    )
