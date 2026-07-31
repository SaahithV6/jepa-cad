"""Corpus graph construction for processed spaceflight datasets.

This module turns a processed dataset manifest plus the corresponding shard
files into a graph document that can be imported into Neo4j. Each shard now
gets a detailed Analogue node with feature, parametric, and physical summaries,
so the JEPA graph can reason about both training items and their breakdowns.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from cadflow.analogue import summarize_asset_analogue
from data.parsers import ParseError, parse_raw_file
from .datasets import DATASET_REGISTRY, DatasetSource
from .graph_schema import GraphDocument, GraphEdge, GraphNode


@dataclass(frozen=True, slots=True)
class CorpusGraphReport:
    name: str
    generated_at: str
    graph: GraphDocument
    manifest_path: str
    processed_dir: str
    source_key: str | None
    shard_count: int
    raw_asset_count: int
    sample_count: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "manifest_path": self.manifest_path,
            "processed_dir": self.processed_dir,
            "source_key": self.source_key,
            "shard_count": self.shard_count,
            "raw_asset_count": self.raw_asset_count,
            "sample_count": self.sample_count,
            "notes": list(self.notes),
            "graph": self.graph.to_dict(),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in text.strip()]
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "node"


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _first_registry_source(source_key: str | None) -> DatasetSource | None:
    if source_key is None:
        return None
    return DATASET_REGISTRY.get(source_key)


def _sample_stats(points: np.ndarray, fields: np.ndarray | None, max_stress: np.ndarray | None) -> dict[str, Any]:
    points = np.asarray(points)
    stats: dict[str, Any] = {
        "num_points": int(points.shape[0]) if points.ndim >= 1 else 0,
        "point_bounds": {
            "min": np.min(points, axis=0).tolist() if points.size else [],
            "max": np.max(points, axis=0).tolist() if points.size else [],
            "mean": np.mean(points, axis=0).tolist() if points.size else [],
        },
    }
    if fields is not None:
        stats["num_fields"] = int(fields.shape[1]) if fields.ndim == 2 else int(fields.size)
        stats["field_stats"] = {
            "shape": list(fields.shape),
            "dtype": str(fields.dtype),
            "min": float(np.min(fields)) if fields.size else None,
            "max": float(np.max(fields)) if fields.size else None,
            "mean": float(np.mean(fields)) if fields.size else None,
            "std": float(np.std(fields)) if fields.size else None,
        }
    else:
        stats["num_fields"] = 0
        stats["field_stats"] = {}
    if max_stress is not None:
        stats["max_stress"] = float(np.asarray(max_stress).reshape(()))
    return stats


def _load_npz_arrays(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=False)
    points = np.asarray(data["points"])
    fields = np.asarray(data["fields"]) if "fields" in data.files else None
    max_stress = data["max_stress"] if "max_stress" in data.files else None
    return points, fields, max_stress


def _analogue_payload(
    *,
    source_path: str,
    name: str,
    node_type: str,
    size_bytes: int | None,
    points: np.ndarray | None = None,
    fields: np.ndarray | None = None,
    max_stress: np.ndarray | float | None = None,
    text: str | None = None,
    role: str,
) -> dict[str, Any]:
    return summarize_asset_analogue(
        source_path=source_path,
        name=name,
        node_type=node_type,
        size_bytes=size_bytes,
        points=points,
        fields=fields,
        max_stress=max_stress,
        text=text,
        extra_summary={"path": source_path, "node_role": role},
    )


def _read_text(path: Path, max_chars: int = 20_000) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return None


def build_processed_corpus_graph(
    manifest_path: str | Path,
    processed_dir: str | Path,
    *,
    source_key: str | None = None,
    include_raw_assets: bool = True,
) -> CorpusGraphReport:
    manifest_path = Path(manifest_path)
    processed_dir = Path(processed_dir)
    payload = _load_manifest(manifest_path)
    shards: list[str] = list(payload.get("shards", []))
    sources: list[dict[str, Any]] = list(payload.get("sources", []))
    registry_source = _first_registry_source(source_key)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    feature_nodes = {
        "points": GraphNode(id="feature:points", type="Feature", label="points", properties={"feature_kind": "point_cloud", "impact": "geometry", "geometry_ref": "points"}),
        "fields": GraphNode(id="feature:fields", type="Feature", label="fields", properties={"feature_kind": "per_point_fields", "impact": "attributes", "geometry_ref": "fields"}),
        "max_stress": GraphNode(id="feature:max_stress", type="Feature", label="max_stress", properties={"feature_kind": "scalar_label", "impact": "verification", "geometry_ref": "max_stress"}),
    }
    dimension_nodes = {
        "num_points": GraphNode(id="dimension:num_points", type="Dimension", label="num_points", properties={"name": "num_points", "value": int(payload.get("num_points", 0)), "unit": "count"}),
        "num_fields": GraphNode(id="dimension:num_fields", type="Dimension", label="num_fields", properties={"name": "num_fields", "value": int(payload.get("num_fields", 0)), "unit": "count"}),
    }
    nodes.extend(feature_nodes.values())
    nodes.extend(dimension_nodes.values())

    dataset_id = f"dataset:{_slug(manifest_path.stem)}"
    dataset_node = GraphNode(
        id=dataset_id,
        type="Dataset",
        label=manifest_path.stem,
        properties={
            "name": manifest_path.stem,
            "manifest_path": str(manifest_path),
            "processed_dir": str(processed_dir),
            "source_count": len(sources),
            "shard_count": len(shards),
            "num_points": payload.get("num_points"),
            "num_fields": payload.get("num_fields"),
            "format": payload.get("format"),
        },
    )
    nodes.append(dataset_node)

    manifest_doc_id = f"document:{_slug(manifest_path.name)}"
    nodes.append(GraphNode(id=manifest_doc_id, type="Document", label=manifest_path.name, properties={"title": manifest_path.name, "doc_type": "dataset-manifest", "pages": None, "manifest_path": str(manifest_path)}))
    edges.append(GraphEdge(id=f"edge:{dataset_id}:described-by", type="SOURCE_OF_TRUTH_FOR", source=manifest_doc_id, target=dataset_id, properties={"role": "manifest"}))

    if registry_source is not None:
        source_node_id = f"source:{registry_source.key}"
        nodes.append(
            GraphNode(
                id=source_node_id,
                type="Source",
                label=registry_source.title,
                properties={
                    "key": registry_source.key,
                    "domain": registry_source.domain,
                    "url": registry_source.url,
                    "license": registry_source.license,
                    "use_cases": list(registry_source.use_cases),
                    "notes": registry_source.notes,
                    "size_hint": registry_source.size_hint,
                    "recommended_for": list(registry_source.recommended_for),
                    "status": "registry",
                },
            )
        )
        edges.append(GraphEdge(id=f"edge:{dataset_id}:source:{registry_source.key}", type="SOURCE_OF_TRUTH_FOR", source=source_node_id, target=dataset_id, properties={"role": "registry_source"}))

    edges.extend([GraphEdge(id=f"edge:{dataset_id}:feature:{feature_id}", type="HAS_FEATURE", source=dataset_id, target=node.id, properties={}) for feature_id, node in feature_nodes.items()])
    edges.extend([GraphEdge(id=f"edge:{dataset_id}:dimension:{dim_id}", type="HAS_DIMENSION", source=dataset_id, target=node.id, properties={}) for dim_id, node in dimension_nodes.items()])

    raw_asset_count = 0
    sample_count = 0
    analogue_count = 0
    analogue_kind_counts: Counter[str] = Counter()

    for index, shard_name in enumerate(shards):
        shard_path = processed_dir / shard_name
        source_info = sources[index] if index < len(sources) else {}
        raw_source_path = source_info.get("source_path", "")
        raw_path = None
        if raw_source_path:
            raw_candidate = Path(raw_source_path)
            repo_root = manifest_path.resolve().parents[3] if len(manifest_path.resolve().parents) > 3 else manifest_path.parent
            raw_path = (repo_root / raw_candidate).resolve() if not raw_candidate.is_absolute() else raw_candidate

        shard_id = f"shard:{_slug(shard_name)}"
        nodes.append(GraphNode(id=shard_id, type="Shard", label=shard_name, properties={"name": shard_name, "shard_path": str(shard_path), "source_path": raw_source_path, "format": source_info.get("format", payload.get("format")), "index": index}))
        edges.append(GraphEdge(id=f"edge:{dataset_id}:shard:{index}", type="HAS_SHARD", source=dataset_id, target=shard_id, properties={"index": index}))

        points, fields, max_stress = _load_npz_arrays(shard_path)
        stats = _sample_stats(points, fields, max_stress)
        sample_count += 1
        sample_id = f"sample:{_slug(shard_name)}"
        nodes.append(GraphNode(id=sample_id, type="Sample", label=shard_name, properties={"name": shard_name, "num_points": stats["num_points"], "num_fields": stats["num_fields"], "max_stress": stats.get("max_stress"), "point_bounds": stats["point_bounds"], "field_stats": stats["field_stats"], "shard_path": str(shard_path)}))
        edges.append(GraphEdge(id=f"edge:{shard_id}:sample:{index}", type="HAS_SAMPLE", source=shard_id, target=sample_id, properties={}))
        edges.append(GraphEdge(id=f"edge:{dataset_id}:sample:{index}", type="HAS_SAMPLE", source=dataset_id, target=sample_id, properties={"index": index}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:feature:points", type="HAS_FEATURE", source=sample_id, target="feature:points", properties={}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:feature:fields", type="HAS_FEATURE", source=sample_id, target="feature:fields", properties={}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:feature:max_stress", type="HAS_FEATURE", source=sample_id, target="feature:max_stress", properties={}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:dimension:num_points", type="HAS_DIMENSION", source=sample_id, target="dimension:num_points", properties={}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:dimension:num_fields", type="HAS_DIMENSION", source=sample_id, target="dimension:num_fields", properties={}))

        sample_analogue = _analogue_payload(source_path=str(shard_path.relative_to(processed_dir)), name=f"{shard_name} sample", node_type="Sample", size_bytes=shard_path.stat().st_size, points=points, fields=fields, max_stress=max_stress, role="training_sample")
        sample_analogue_id = f"analogue:sample:{_slug(shard_name)}"
        nodes.append(GraphNode(id=sample_analogue_id, type="Analogue", label=f"{shard_name} sample analogue", properties={"name": f"{shard_name} sample", "analogue_kind": sample_analogue["analogue_kind"], "source_path": str(shard_path.relative_to(processed_dir)), "source_kind": "Sample", "summary": sample_analogue["summary"], "feature_summary": sample_analogue["feature_summary"], "parametric_summary": sample_analogue["parametric_summary"], "physical_summary": sample_analogue["physical_summary"]}))
        edges.append(GraphEdge(id=f"edge:{sample_id}:analogue:{index}", type="HAS_ANALOGUE", source=sample_id, target=sample_analogue_id, properties={"role": "sample_analogue"}))
        edges.append(GraphEdge(id=f"edge:{sample_analogue_id}:of:{index}", type="ANALOGUE_OF", source=sample_analogue_id, target=sample_id, properties={"role": "sample_analogue"}))
        analogue_kind_counts[sample_analogue["analogue_kind"]] += 1
        analogue_count += 1

        shard_analogue = _analogue_payload(source_path=str(shard_path.relative_to(processed_dir)), name=shard_name, node_type="TensorShard", size_bytes=shard_path.stat().st_size, points=points, fields=fields, max_stress=max_stress, role="processed_shard")
        shard_analogue_id = f"analogue:shard:{_slug(shard_name)}"
        nodes.append(GraphNode(id=shard_analogue_id, type="Analogue", label=f"{shard_name} analogue", properties={"name": shard_name, "analogue_kind": shard_analogue["analogue_kind"], "source_path": str(shard_path.relative_to(processed_dir)), "source_kind": "TensorShard", "summary": shard_analogue["summary"], "feature_summary": shard_analogue["feature_summary"], "parametric_summary": shard_analogue["parametric_summary"], "physical_summary": shard_analogue["physical_summary"]}))
        edges.append(GraphEdge(id=f"edge:{shard_id}:analogue:{index}", type="HAS_ANALOGUE", source=shard_id, target=shard_analogue_id, properties={"role": "detailed_analogue"}))
        edges.append(GraphEdge(id=f"edge:{shard_analogue_id}:of:{index}", type="ANALOGUE_OF", source=shard_analogue_id, target=shard_id, properties={"role": "detailed_analogue"}))
        analogue_kind_counts[shard_analogue["analogue_kind"]] += 1
        analogue_count += 1

        if include_raw_assets and raw_path is not None:
            raw_asset_count += 1
            raw_id = f"rawasset:{_slug(raw_source_path)}"
            exists = raw_path.exists()
            size_bytes = raw_path.stat().st_size if exists and raw_path.is_file() else None
            nodes.append(GraphNode(id=raw_id, type="RawAsset", label=raw_path.name if raw_path.name else raw_source_path, properties={"path": str(raw_path), "exists": exists, "size_bytes": size_bytes, "extension": raw_path.suffix.lower()}))
            if registry_source is not None:
                edges.append(GraphEdge(id=f"edge:{raw_id}:derived-from:{registry_source.key}", type="DERIVED_FROM", source=raw_id, target=f"source:{registry_source.key}", properties={"role": "raw_download"}))
            edges.append(GraphEdge(id=f"edge:{shard_id}:derived-from:{index}", type="DERIVED_FROM", source=shard_id, target=raw_id, properties={"role": "processed_shard"}))
            edges.append(GraphEdge(id=f"edge:{raw_id}:stored-at:{index}", type="STORED_AT", source=raw_id, target=dataset_id, properties={"role": "corpus_storage"}))
            try:
                points_raw = fields_raw = max_stress_raw = None
                parsed = parse_raw_file(raw_path, num_points=1024, num_fields=3, allow_synthetic_fallback=False)
                points_raw = parsed.points
                fields_raw = parsed.fields
                max_stress_raw = float(fields_raw[:, min(2, fields_raw.shape[1] - 1)].max()) if fields_raw.size else None
            except Exception:
                points_raw = fields_raw = max_stress_raw = None
            raw_analogue = _analogue_payload(source_path=str(raw_path), name=raw_path.name, node_type="RawAsset", size_bytes=size_bytes, points=points_raw, fields=fields_raw, max_stress=max_stress_raw, text=_read_text(raw_path) if raw_path.suffix.lower() in {".md", ".txt", ".json", ".csv", ".xml", ".yaml", ".yml", ".py"} else None, role="raw_download")
            raw_analogue_id = f"analogue:raw:{_slug(raw_source_path)}"
            nodes.append(GraphNode(id=raw_analogue_id, type="Analogue", label=f"{raw_path.stem} analogue", properties={"name": raw_path.name, "analogue_kind": raw_analogue["analogue_kind"], "source_path": str(raw_path), "source_kind": "RawAsset", "summary": raw_analogue["summary"], "feature_summary": raw_analogue["feature_summary"], "parametric_summary": raw_analogue["parametric_summary"], "physical_summary": raw_analogue["physical_summary"]}))
            edges.append(GraphEdge(id=f"edge:{raw_id}:analogue:{index}", type="HAS_ANALOGUE", source=raw_id, target=raw_analogue_id, properties={"role": "raw_asset_analogue"}))
            edges.append(GraphEdge(id=f"edge:{raw_analogue_id}:of:{index}", type="ANALOGUE_OF", source=raw_analogue_id, target=raw_id, properties={"role": "raw_asset_analogue"}))
            analogue_kind_counts[raw_analogue["analogue_kind"]] += 1
            analogue_count += 1

    graph = GraphDocument(
        name=f"{manifest_path.stem}-corpus-graph",
        generated_at=_utc_now(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "manifest_path": str(manifest_path),
            "processed_dir": str(processed_dir),
            "source_key": source_key,
            "source_count": len(sources),
            "shard_count": len(shards),
            "raw_asset_count": raw_asset_count,
            "sample_count": sample_count,
            "analogue_count": analogue_count,
            "analogue_kind_counts": dict(analogue_kind_counts),
            "include_raw_assets": include_raw_assets,
        },
    )
    notes = (
        "Manifest and processed shards are materialized as first-class graph nodes.",
        "Each shard now has detailed Analogue nodes for the shard itself and its sample view.",
        "Raw source assets preserve provenance and are linked to the registry source when available.",
        "The graph carries feature, dimension, and physics summaries to support JEPA conditioning.",
    )
    return CorpusGraphReport(
        name=graph.name,
        generated_at=graph.generated_at,
        graph=graph,
        manifest_path=str(manifest_path),
        processed_dir=str(processed_dir),
        source_key=source_key,
        shard_count=len(shards),
        raw_asset_count=raw_asset_count,
        sample_count=sample_count,
        notes=notes,
    )


def render_corpus_graph_report(report: CorpusGraphReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2)
    lines = [
        f"name={report.name}",
        f"generated_at={report.generated_at}",
        f"manifest_path={report.manifest_path}",
        f"processed_dir={report.processed_dir}",
        f"source_key={report.source_key}",
        f"shard_count={report.shard_count}",
        f"raw_asset_count={report.raw_asset_count}",
        f"sample_count={report.sample_count}",
        "notes:",
    ]
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
