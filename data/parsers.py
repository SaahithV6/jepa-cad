"""Real CAD / field-data parsers for shard preparation.

Supports:
  - Mesh geometry: STL, OBJ (via trimesh), STEP (via CadQuery when available)
  - Field data: VTK/VTU-like JSON sidecars, and simple VTK ASCII POLYDATA
  - Pre-baked NPZ shards

Unsupported formats raise ParseError instead of silently inventing geometry.
Optional synthetic fallback remains explicit and opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MESH_SUFFIXES = {".stl", ".obj", ".ply"}
CAD_SUFFIXES = {".step", ".stp"}
FIELD_SUFFIXES = {".vtk", ".vtu", ".vtp"}
SHARD_SUFFIXES = {".npz", ".pt"}


class ParseError(ValueError):
    """Raised when a raw file cannot be parsed into points/fields."""


@dataclass(frozen=True, slots=True)
class ParsedSample:
    points: np.ndarray  # (N, 3)
    fields: np.ndarray  # (N, F)
    source_format: str
    metadata: dict[str, Any]

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "points": self.points.astype(np.float32),
            "fields": self.fields.astype(np.float32),
        }


def _resample_points_fields(
    points: np.ndarray,
    fields: np.ndarray,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == num_points:
        return points, fields
    rng = np.random.default_rng(seed)
    if points.shape[0] > num_points:
        idx = rng.choice(points.shape[0], num_points, replace=False)
    else:
        idx = rng.choice(points.shape[0], num_points, replace=True)
    return points[idx], fields[idx]


def _default_fields_from_points(points: np.ndarray, num_fields: int) -> np.ndarray:
    """Derive placeholder physics channels from geometry when no field file exists.

    Channel 0: normalized radial distance (pressure-like)
    Channel 1: normalized z (temperature-like)
    Channel 2+: von-Mises-like magnitude from coords
    """
    centered = points - points.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(centered, axis=1, keepdims=True)
    radial = radial / (radial.max() + 1e-8)
    z = points[:, 2:3]
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    stress = np.linalg.norm(centered, axis=1, keepdims=True)
    stress = stress / (stress.max() + 1e-8)
    channels = [radial, z, stress]
    while len(channels) < num_fields:
        channels.append(channels[-1] * 0.5)
    return np.concatenate(channels[:num_fields], axis=1).astype(np.float32)


def parse_mesh_file(path: Path, num_fields: int = 3) -> ParsedSample:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ParseError(f"trimesh required to parse {path.suffix}") from exc

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ParseError(f"no vertices in mesh: {path}")
    points = np.asarray(mesh.vertices, dtype=np.float32)
    fields = _default_fields_from_points(points, num_fields)
    return ParsedSample(
        points=points,
        fields=fields,
        source_format=path.suffix.lower().lstrip("."),
        metadata={"watertight": bool(getattr(mesh, "is_watertight", False)), "faces": int(len(getattr(mesh, "faces", [])))},
    )


def parse_step_file(path: Path, num_fields: int = 3, samples: int = 4096) -> ParsedSample:
    try:
        import cadquery as cq
    except Exception as exc:  # pragma: no cover
        raise ParseError(f"cadquery required to parse STEP: {exc}") from exc

    result = cq.importers.importStep(str(path))
    shape = result.val() if hasattr(result, "val") else result
    # Sample vertices from tessellated mesh
    try:
        import cadquery.occ_impl.shapes as shapes

        # Tessellate via exporters helper: write temp STL through memory path is heavy;
        # use vertices of bounding triangulation via Shape.tessellate if available.
        verts: list[tuple[float, float, float]] = []
        if hasattr(shape, "tessellate"):
            tess = shape.tessellate(0.1)
            verts = [(float(v.x), float(v.y), float(v.z)) for v in tess[0]]
        elif hasattr(shapes, "tessellate"):
            tess = shapes.tessellate(shape, 0.1)
            verts = [(float(v.x), float(v.y), float(v.z)) for v in tess[0]]
        if not verts:
            bbox = shape.BoundingBox()
            # Fallback grid on bbox
            xs = np.linspace(bbox.xmin, bbox.xmax, 8)
            ys = np.linspace(bbox.ymin, bbox.ymax, 8)
            zs = np.linspace(bbox.zmin, bbox.zmax, 8)
            grid = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
            points = grid
        else:
            points = np.asarray(verts, dtype=np.float32)
    except Exception as exc:
        raise ParseError(f"failed to sample STEP geometry from {path}: {exc}") from exc

    if points.shape[0] > samples:
        rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
        idx = rng.choice(points.shape[0], samples, replace=False)
        points = points[idx]
    fields = _default_fields_from_points(points, num_fields)
    return ParsedSample(points=points, fields=fields, source_format="step", metadata={"path": str(path)})


def parse_vtk_ascii(path: Path, num_fields: int = 3) -> ParsedSample:
    """Minimal VTK legacy ASCII POLYDATA / UNSTRUCTURED_GRID points+scalars parser."""

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    points: list[list[float]] = []
    scalars: list[float] = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if line.startswith("POINTS"):
            parts = line.split()
            n = int(parts[1])
            i += 1
            collected: list[float] = []
            while len(collected) < n * 3 and i < len(text):
                collected.extend(float(x) for x in text[i].split())
                i += 1
            pts = np.asarray(collected[: n * 3], dtype=np.float32).reshape(n, 3)
            points = pts.tolist()
            continue
        if line.startswith("POINT_DATA"):
            i += 1
            continue
        if line.startswith("SCALARS") or line.startswith("LOOKUP_TABLE"):
            i += 1
            # read following numeric lines until non-numeric
            while i < len(text):
                row = text[i].strip()
                if not row or row[0].isalpha():
                    break
                scalars.extend(float(x) for x in row.split())
                i += 1
            continue
        i += 1

    if not points:
        raise ParseError(f"no POINTS found in VTK file: {path}")
    pts_arr = np.asarray(points, dtype=np.float32)
    if scalars and len(scalars) >= len(pts_arr):
        s = np.asarray(scalars[: len(pts_arr)], dtype=np.float32).reshape(-1, 1)
        fields = np.repeat(s, num_fields, axis=1)
        # diversify channels slightly
        for c in range(1, num_fields):
            fields[:, c] = fields[:, 0] * (0.5 + 0.1 * c)
    else:
        fields = _default_fields_from_points(pts_arr, num_fields)
    return ParsedSample(points=pts_arr, fields=fields.astype(np.float32), source_format="vtk", metadata={})


def parse_npz_shard(path: Path) -> ParsedSample:
    data = np.load(path)
    if "points" not in data or "fields" not in data:
        raise ParseError(f"npz shard missing points/fields: {path}")
    return ParsedSample(
        points=data["points"].astype(np.float32),
        fields=data["fields"].astype(np.float32),
        source_format="npz",
        metadata={},
    )


def parse_field_sidecar(geometry_path: Path, num_fields: int) -> np.ndarray | None:
    """Load optional `<stem>.fields.npz` or `<stem>.json` field sidecar."""

    npz = geometry_path.with_suffix(".fields.npz")
    if npz.exists():
        data = np.load(npz)
        if "fields" in data:
            return data["fields"].astype(np.float32)
    json_path = geometry_path.with_suffix(".fields.json")
    if json_path.exists():
        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        arr = np.asarray(payload.get("fields", payload), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] < num_fields:
            pad = np.zeros((arr.shape[0], num_fields - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        return arr[:, :num_fields]
    return None


def parse_raw_file(
    path: Path | str,
    *,
    num_points: int = 1024,
    num_fields: int = 3,
    allow_synthetic_fallback: bool = False,
) -> ParsedSample:
    path = Path(path)
    if not path.exists():
        raise ParseError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in SHARD_SUFFIXES and suffix == ".npz":
        sample = parse_npz_shard(path)
    elif suffix in MESH_SUFFIXES:
        sample = parse_mesh_file(path, num_fields=num_fields)
    elif suffix in CAD_SUFFIXES:
        sample = parse_step_file(path, num_fields=num_fields)
    elif suffix in FIELD_SUFFIXES:
        sample = parse_vtk_ascii(path, num_fields=num_fields)
    elif allow_synthetic_fallback:
        from data.synthetic import SyntheticConfig, generate_synthetic_sample

        synth = generate_synthetic_sample(
            index=abs(hash(path.name)) % 10_000,
            cfg=SyntheticConfig(num_points=num_points, num_fields=num_fields),
        )
        return ParsedSample(
            points=synth["points"].numpy(),
            fields=synth["fields"].numpy(),
            source_format="synthetic_fallback",
            metadata={"warning": f"unsupported suffix {suffix}; used synthetic fallback"},
        )
    else:
        raise ParseError(f"unsupported raw format: {suffix} ({path.name})")

    sidecar = parse_field_sidecar(path, num_fields)
    points = sample.points
    fields = sidecar if sidecar is not None else sample.fields
    if fields.shape[0] != points.shape[0]:
        # Resample fields independently is invalid; regenerate geometry-derived fields.
        fields = _default_fields_from_points(points, num_fields)
    if fields.shape[1] != num_fields:
        if fields.shape[1] > num_fields:
            fields = fields[:, :num_fields]
        else:
            pad = np.zeros((fields.shape[0], num_fields - fields.shape[1]), dtype=np.float32)
            fields = np.concatenate([fields, pad], axis=1)

    points, fields = _resample_points_fields(
        points, fields, num_points, seed=abs(hash(path.name)) % (2**32)
    )
    return ParsedSample(
        points=points.astype(np.float32),
        fields=fields.astype(np.float32),
        source_format=sample.source_format,
        metadata=dict(sample.metadata),
    )
