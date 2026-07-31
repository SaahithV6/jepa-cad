"""Analogue summaries for graph knowledge entries.

These helpers turn a raw asset, tensor shard, document, or parsed geometry into a
structured analogue profile with:
- a high-level summary
- extracted feature statistics
- parametric / dimension-effect statistics
- physical / geometry statistics

The graph builders use these profiles to create first-class Analogue nodes so
JEPA training and graph querying can work from the same semantics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
from typing import Any

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _safe_float(value: Any) -> float | None:
    try:
        value = float(np.asarray(value).reshape(()))
    except Exception:
        return None
    if math.isfinite(value):
        return value
    return None


def _corr(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    try:
        value = float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _axis_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or points.size == 0:
        return {"available": False}
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    extents = bounds_max - bounds_min
    centroid = np.mean(points, axis=0)
    radial = np.linalg.norm(points - centroid, axis=1)
    summary: dict[str, Any] = {
        "available": True,
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "centroid": centroid.tolist(),
        "extents": extents.tolist(),
        "extent_ratios": {
            "xy": float(extents[0] / (extents[1] + 1e-8)),
            "xz": float(extents[0] / (extents[2] + 1e-8)),
            "yz": float(extents[1] / (extents[2] + 1e-8)),
        },
        "radial": {
            "mean": float(np.mean(radial)),
            "std": float(np.std(radial)),
            "min": float(np.min(radial)),
            "max": float(np.max(radial)),
        },
        "volume_proxy": float(np.prod(np.maximum(extents, 0.0))),
    }
    return summary


def _field_stats(fields: np.ndarray | None) -> dict[str, Any]:
    if fields is None:
        return {"available": False}
    arr = np.asarray(fields, dtype=np.float64)
    if arr.size == 0:
        return {"available": False}
    stats: dict[str, Any] = {
        "available": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    if arr.ndim == 2:
        stats["per_channel_mean"] = np.mean(arr, axis=0).tolist()
        stats["per_channel_std"] = np.std(arr, axis=0).tolist()
    return stats


def _stress_channel(fields: np.ndarray | None, max_stress: np.ndarray | float | None) -> np.ndarray | None:
    if max_stress is not None:
        return np.asarray(max_stress, dtype=np.float64).reshape(-1)
    if fields is None:
        return None
    arr = np.asarray(fields, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] > 0:
        return arr[:, min(2, arr.shape[1] - 1)].reshape(-1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return None


def _top_terms(text: str, limit: int = 10) -> list[tuple[str, int]]:
    tokens = [tok.lower() for tok in _WORD_RE.findall(text)]
    stopwords = {
        "the",
        "and",
        "or",
        "for",
        "with",
        "that",
        "from",
        "this",
        "are",
        "was",
        "were",
        "into",
        "than",
        "then",
        "via",
        "has",
        "have",
        "not",
        "using",
        "used",
        "use",
        "data",
        "file",
    }
    counts = Counter(tok for tok in tokens if len(tok) > 2 and tok not in stopwords)
    return counts.most_common(limit)


def summarize_numeric_analogue(
    *,
    points: np.ndarray,
    fields: np.ndarray | None = None,
    max_stress: np.ndarray | float | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    point_stats = _axis_stats(points)
    field_stats = _field_stats(fields)
    stress = _stress_channel(fields, max_stress)

    parametric_summary: dict[str, Any] = {
        "has_geometry": bool(point_stats.get("available")),
        "point_count": int(points.shape[0]) if points.ndim >= 1 else 0,
        "feature_count": int(fields.shape[1]) if fields is not None and np.asarray(fields).ndim == 2 else int(np.asarray(fields).size) if fields is not None else 0,
        "axis_correlations": {},
    }
    if point_stats.get("available"):
        extents = np.asarray(point_stats["extents"], dtype=np.float64)
        centroid = np.asarray(point_stats["centroid"], dtype=np.float64)
        parametric_summary["axis_extents"] = point_stats["extents"]
        parametric_summary["extent_ratios"] = point_stats["extent_ratios"]
        parametric_summary["centroid_offsets"] = {
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]),
            "radial_to_x_ratio": float(point_stats["radial"]["mean"] / (abs(float(centroid[0])) + 1e-8)),
        }
        parametric_summary["shape_proxy"] = {
            "surface_proxy": float(2.0 * (extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2])),
            "volume_proxy": point_stats["volume_proxy"],
        }
        if stress is not None and stress.size == points.shape[0]:
            parametric_summary["axis_correlations"] = {
                "x_to_stress": _corr(points[:, 0], stress),
                "y_to_stress": _corr(points[:, 1], stress),
                "z_to_stress": _corr(points[:, 2], stress),
                "radial_to_stress": _corr(np.linalg.norm(points - np.mean(points, axis=0, keepdims=True), axis=1), stress),
            }
    else:
        parametric_summary["shape_proxy"] = {}

    physical_summary: dict[str, Any] = {
        "point_bounds": point_stats.get("bounds") if point_stats.get("available") else {"min": [], "max": []},
        "centroid": point_stats.get("centroid") if point_stats.get("available") else [],
        "radial": point_stats.get("radial") if point_stats.get("available") else {},
        "field_stats": field_stats,
    }
    if stress is not None and stress.size:
        physical_summary["stress"] = {
            "mean": float(np.mean(stress)),
            "std": float(np.std(stress)),
            "min": float(np.min(stress)),
            "max": float(np.max(stress)),
        }

    feature_summary: dict[str, Any] = {
        "point_count": int(points.shape[0]) if points.ndim >= 1 else 0,
        "field_count": int(fields.shape[1]) if fields is not None and np.asarray(fields).ndim == 2 else int(np.asarray(fields).size) if fields is not None else 0,
        "field_stats": field_stats,
        "physics_signal": {
            "has_stress": stress is not None,
            "stress_correlation_available": any(v is not None for v in parametric_summary.get("axis_correlations", {}).values()),
        },
    }

    summary = {
        "kind": "numeric",
        "point_count": int(points.shape[0]) if points.ndim >= 1 else 0,
        "field_count": int(fields.shape[1]) if fields is not None and np.asarray(fields).ndim == 2 else int(np.asarray(fields).size) if fields is not None else 0,
        "stats": {
            "point_bounds": physical_summary["point_bounds"],
            "field_stats": field_stats,
        },
    }
    return {
        "analogue_kind": "geometry" if point_stats.get("available") else "numeric",
        "summary": summary,
        "feature_summary": feature_summary,
        "parametric_summary": parametric_summary,
        "physical_summary": physical_summary,
    }


def summarize_text_analogue(
    *,
    text: str,
    source_path: str | None = None,
    name: str | None = None,
    kind: str = "document",
    size_bytes: int | None = None,
) -> dict[str, Any]:
    lines = text.splitlines()
    words = _WORD_RE.findall(text)
    top_terms = _top_terms(text)
    summary = {
        "kind": kind,
        "text_length": len(text),
        "line_count": len(lines),
        "word_count": len(words),
        "unique_word_count": len({w.lower() for w in words}),
        "top_terms": top_terms,
    }
    feature_summary = {
        "line_count": len(lines),
        "word_count": len(words),
        "unique_word_count": len({w.lower() for w in words}),
        "top_terms": top_terms,
    }
    parametric_summary = {
        "token_density": float(len(words) / max(len(lines), 1)),
        "term_diversity": float(len({w.lower() for w in words}) / max(len(words), 1)),
    }
    physical_summary = {
        "size_bytes": size_bytes,
        "source_path": source_path,
        "name": name,
    }
    return {
        "analogue_kind": kind,
        "summary": summary,
        "feature_summary": feature_summary,
        "parametric_summary": parametric_summary,
        "physical_summary": physical_summary,
    }


def summarize_asset_analogue(
    *,
    source_path: str,
    name: str,
    node_type: str,
    size_bytes: int | None = None,
    points: np.ndarray | None = None,
    fields: np.ndarray | None = None,
    max_stress: np.ndarray | float | None = None,
    text: str | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind_map = {
        "RawAsset": "geometry",
        "TensorShard": "tensor",
        "Shard": "tensor",
        "Sample": "sample",
        "DocumentAsset": "document",
        "CodeArtifact": "code",
        "ImageAsset": "image",
        "ArchiveAsset": "archive",
        "OtherArtifact": "file",
        "Corpus": "collection",
        "Dataset": "dataset",
    }
    analogue_kind = kind_map.get(node_type, "mixed")
    base_summary = {
        "name": name,
        "source_path": source_path,
        "node_type": node_type,
        "size_bytes": size_bytes,
    }

    if points is not None:
        numeric = summarize_numeric_analogue(points=points, fields=fields, max_stress=max_stress)
        summary = {**base_summary, **numeric["summary"]}
        feature_summary = numeric["feature_summary"]
        parametric_summary = numeric["parametric_summary"]
        physical_summary = numeric["physical_summary"]
    elif text is not None:
        text_summary = summarize_text_analogue(text=text, source_path=source_path, name=name, kind=analogue_kind, size_bytes=size_bytes)
        summary = {**base_summary, **text_summary["summary"]}
        feature_summary = text_summary["feature_summary"]
        parametric_summary = text_summary["parametric_summary"]
        physical_summary = text_summary["physical_summary"]
    else:
        summary = {**base_summary, "kind": analogue_kind}
        feature_summary = {"available": False}
        parametric_summary = {"available": False}
        physical_summary = {"available": False}

    if extra_summary:
        summary = {**summary, **extra_summary}

    return {
        "analogue_kind": analogue_kind,
        "summary": summary,
        "feature_summary": feature_summary,
        "parametric_summary": parametric_summary,
        "physical_summary": physical_summary,
    }
