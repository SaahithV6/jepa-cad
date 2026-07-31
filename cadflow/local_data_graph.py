"""Build a graph for every local file and corpus under `data/`.

This graph complements the source registry graph by materializing the local
on-disk inventory and giving every downloaded item a detailed Analogue node.
Each analogue carries feature, parametric, and physical summaries so the JEPA
pipeline can reason over geometry, fields, and dimensional effects.
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
from data.parsers import CAD_SUFFIXES, FIELD_SUFFIXES, MESH_SUFFIXES, ParseError, parse_raw_file
from .graph_schema import GraphDocument, GraphEdge, GraphNode

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ARCHIVE_SUFFIXES = {".7z", ".zip", ".tar", ".gz", ".bz2", ".xz"}
DOCUMENT_SUFFIXES = {".md", ".txt", ".csv", ".xml", ".json", ".pdf", ".pptx", ".yaml", ".yml"}
CODE_SUFFIXES = {".py"}
TENSOR_SUFFIXES = {".npz", ".pt"}
IGNORE_DIRS = {".git", "__pycache__"}


@dataclass(frozen=True, slots=True)
class LocalDataGraphReport:
    name: str
    generated_at: str
    graph: GraphDocument
    data_root: str
    corpus_count: int
    directory_count: int
    file_count: int
    sample_count: int
    category_counts: dict[str, int]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "data_root": self.data_root,
            "corpus_count": self.corpus_count,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "sample_count": self.sample_count,
            "category_counts": dict(self.category_counts),
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


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORE_DIRS or part.startswith(".git") for part in path.parts)


def _file_category(path: Path, *, parent_is_raw: bool) -> str:
    suffix = path.suffix.lower()
    if suffix in TENSOR_SUFFIXES:
        return "TensorShard"
    if suffix in MESH_SUFFIXES or suffix in CAD_SUFFIXES:
        return "RawAsset"
    if suffix in IMAGE_SUFFIXES:
        return "ImageAsset"
    if suffix in ARCHIVE_SUFFIXES:
        return "ArchiveAsset"
    if suffix in CODE_SUFFIXES:
        return "CodeArtifact"
    if suffix in DOCUMENT_SUFFIXES:
        return "DocumentAsset"
    return "OtherArtifact" if parent_is_raw else "OtherArtifact"


def _directory_counts(root: Path) -> tuple[dict[Path, int], dict[Path, int]]:
    file_counts: dict[Path, int] = {}
    dir_counts: dict[Path, int] = {}
    directories = [p for p in root.rglob("*") if p.is_dir() and not _is_ignored(p)]
    directories.append(root)
    for directory in sorted(set(directories), key=lambda p: len(p.relative_to(root).parts)):
        child_files = 0
        child_dirs = 0
        for child in directory.iterdir():
            if _is_ignored(child):
                continue
            if child.is_file():
                child_files += 1
            elif child.is_dir():
                child_dirs += 1
        file_counts[directory] = child_files
        dir_counts[directory] = child_dirs
    return file_counts, dir_counts


def _load_json_if_possible(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _text_summary(path: Path, max_chars: int = 1800) -> tuple[int | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None
    lines = text.splitlines()
    preview = text[:max_chars].strip()
    return len(lines), preview or None


def _read_text(path: Path, max_chars: int = 20_000) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return None


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


def _corpus_kind(path: Path, data_root: Path) -> str:
    if path == data_root:
        return "repo_root"
    parts = path.relative_to(data_root).parts
    if len(parts) == 1 and parts[0] in {"raw_downloads", "processed"}:
        return parts[0]
    if len(parts) == 2 and parts[0] in {"raw_downloads", "processed"}:
        return "dataset_collection"
    return "collection"


def _corpus_path_set(data_root: Path) -> set[Path]:
    candidates = {data_root}
    for child in data_root.iterdir():
        if child.is_dir() and not _is_ignored(child):
            candidates.add(child)
            if child.name in {"raw_downloads", "processed"}:
                for sub in child.iterdir():
                    if sub.is_dir() and not _is_ignored(sub):
                        candidates.add(sub)
    return candidates


def _create_analogue(
    *,
    path_rel: str,
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
        source_path=path_rel,
        name=name,
        node_type=node_type,
        size_bytes=size_bytes,
        points=points,
        fields=fields,
        max_stress=max_stress,
        text=text,
        extra_summary={"path": path_rel, "node_role": role},
    )


def build_local_data_graph(data_root: str | Path = "data") -> LocalDataGraphReport:
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    all_dirs = [p for p in data_root.rglob("*") if p.is_dir() and not _is_ignored(p)]
    all_files = [p for p in data_root.rglob("*") if p.is_file() and not _is_ignored(p)]
    file_counts, dir_counts = _directory_counts(data_root)
    corpus_paths = _corpus_path_set(data_root)

    processed_manifest: dict[str, str] = {}
    for manifest_path in data_root.rglob("ingestion_manifest.json"):
        if _is_ignored(manifest_path):
            continue
        payload = _load_json_if_possible(manifest_path)
        if not payload:
            continue
        for shard_name, source in zip(payload.get("shards", []), payload.get("sources", []), strict=False):
            source_path = source.get("source_path")
            if source_path:
                processed_manifest[shard_name] = source_path

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    category_counts: Counter[str] = Counter()
    analogue_kind_counts: Counter[str] = Counter()
    node_by_path: dict[Path, str] = {}
    corpus_by_path: dict[Path, str] = {}
    sample_count = 0
    analogue_count = 0

    def add_node(path: Path, node_id: str, node_type: str, label: str, properties: dict[str, Any]) -> None:
        nodes.append(GraphNode(id=node_id, type=node_type, label=label, properties=properties))
        node_by_path[path] = node_id

    for corpus_path in sorted(corpus_paths, key=lambda p: (len(p.relative_to(data_root).parts), str(p))):
        rel = corpus_path.relative_to(data_root)
        node_id = "corpus:" + ("data" if corpus_path == data_root else _slug(str(rel)))
        corpus_by_path[corpus_path] = node_id
        add_node(
            corpus_path,
            node_id,
            "Corpus",
            corpus_path.name if corpus_path != data_root else "data",
            {
                "path": "." if corpus_path == data_root else str(rel),
                "corpus_kind": _corpus_kind(corpus_path, data_root),
                "file_count": file_counts.get(corpus_path, 0),
                "directory_count": dir_counts.get(corpus_path, 0),
            },
        )

    for directory in sorted(all_dirs, key=lambda p: (len(p.relative_to(data_root).parts), str(p))):
        if directory in corpus_paths:
            continue
        rel = directory.relative_to(data_root)
        node_id = f"dir:{_slug(str(rel))}"
        add_node(
            directory,
            node_id,
            "Directory",
            directory.name,
            {
                "path": str(rel),
                "depth": len(rel.parts),
                "file_count": file_counts.get(directory, 0),
                "directory_count": dir_counts.get(directory, 0),
            },
        )

    def parent_node_id(path: Path) -> str | None:
        current = path.parent
        while True:
            if current in node_by_path:
                return node_by_path[current]
            if current == data_root.parent or current == current.parent:
                return None
            current = current.parent

    for file_path in sorted(all_files, key=lambda p: str(p)):
        rel = file_path.relative_to(data_root)
        parent_is_raw = any(part == "raw_downloads" for part in rel.parts)
        node_type = _file_category(file_path, parent_is_raw=parent_is_raw)
        category_counts[node_type] += 1
        node_id = f"{node_type.lower()}:{_slug(str(rel))}"
        size_bytes = file_path.stat().st_size
        props: dict[str, Any] = {
            "path": str(rel),
            "extension": file_path.suffix.lower(),
            "size_bytes": size_bytes,
            "category": node_type,
        }
        if node_type == "RawAsset":
            props["asset_kind"] = "geometry"
        elif node_type == "ImageAsset":
            props["asset_kind"] = "image"
        elif node_type == "ArchiveAsset":
            props["asset_kind"] = "archive"
        elif node_type == "CodeArtifact":
            props["language"] = "python"
            line_count, preview = _text_summary(file_path)
            if line_count is not None:
                props["line_count"] = line_count
            if preview:
                props["preview"] = preview
        elif node_type == "DocumentAsset":
            document_kind = "document"
            name = file_path.name.lower()
            if "manifest" in name:
                document_kind = "manifest"
            elif name == "meta.json":
                document_kind = "metadata"
            elif file_path.suffix.lower() == ".json":
                document_kind = "json"
            elif file_path.suffix.lower() == ".pdf":
                document_kind = "pdf"
            elif file_path.suffix.lower() == ".pptx":
                document_kind = "presentation"
            elif file_path.suffix.lower() == ".md":
                document_kind = "markdown"
            elif file_path.suffix.lower() in {".txt", ".csv", ".xml", ".yaml", ".yml"}:
                document_kind = file_path.suffix.lower().lstrip(".")
            props["document_kind"] = document_kind
            line_count, preview = _text_summary(file_path)
            if line_count is not None:
                props["line_count"] = line_count
            if preview:
                props["preview"] = preview
            if file_path.suffix.lower() == ".json":
                payload = _load_json_if_possible(file_path)
                if payload is not None:
                    if isinstance(payload, dict):
                        props["json_keys"] = list(payload.keys())[:50]
                    else:
                        props["json_type"] = type(payload).__name__
        elif node_type == "TensorShard":
            props["format"] = file_path.suffix.lower().lstrip(".")
        else:
            props["artifact_kind"] = "other"

        sample_id: str | None = None
        points = fields = max_stress = None
        analogue_payload: dict[str, Any] | None = None

        if node_type == "TensorShard":
            source_path = processed_manifest.get(file_path.name)
            if source_path:
                props["source_path"] = source_path
            try:
                data = np.load(file_path, allow_pickle=False)
                points = np.asarray(data["points"])
                fields = np.asarray(data["fields"]) if "fields" in data.files else None
                max_stress = data["max_stress"] if "max_stress" in data.files else None
                stats = _sample_stats(points, fields, max_stress)
                props.update({
                    "num_points": stats["num_points"],
                    "num_fields": stats["num_fields"],
                    "max_stress": stats.get("max_stress"),
                    "point_bounds": stats["point_bounds"],
                    "field_stats": stats["field_stats"],
                })
                sample_id = f"sample:{_slug(str(rel))}"
                nodes.append(
                    GraphNode(
                        id=sample_id,
                        type="Sample",
                        label=file_path.stem,
                        properties={
                            "name": file_path.name,
                            "num_points": stats["num_points"],
                            "num_fields": stats["num_fields"],
                            "max_stress": stats.get("max_stress"),
                            "point_bounds": stats["point_bounds"],
                            "field_stats": stats["field_stats"],
                            "source_shard_path": str(rel),
                        },
                    )
                )
                sample_count += 1
            except Exception:
                props["parse_error"] = True

            analogue_payload = _create_analogue(
                path_rel=str(rel),
                name=file_path.name,
                node_type=node_type,
                size_bytes=size_bytes,
                points=points,
                fields=fields,
                max_stress=max_stress,
                role="tensor_shard",
            )
        elif node_type == "RawAsset":
            try:
                parsed = parse_raw_file(file_path, num_points=1024, num_fields=3, allow_synthetic_fallback=False)
                points = parsed.points
                fields = parsed.fields
                max_stress = float(fields[:, min(2, fields.shape[1] - 1)].max()) if fields.size else None
            except Exception:
                pass
            analogue_payload = _create_analogue(
                path_rel=str(rel),
                name=file_path.name,
                node_type=node_type,
                size_bytes=size_bytes,
                points=points,
                fields=fields,
                max_stress=max_stress,
                role="downloaded_geometry",
            )
        elif node_type in {"DocumentAsset", "CodeArtifact"}:
            analogue_payload = _create_analogue(
                path_rel=str(rel),
                name=file_path.name,
                node_type=node_type,
                size_bytes=size_bytes,
                text=_read_text(file_path),
                role="textual_reference",
            )
        elif node_type in {"ImageAsset", "ArchiveAsset", "OtherArtifact"}:
            analogue_payload = _create_analogue(
                path_rel=str(rel),
                name=file_path.name,
                node_type=node_type,
                size_bytes=size_bytes,
                role="auxiliary_artifact",
            )

        add_node(file_path, node_id, node_type, file_path.name, props)

        parent_id = parent_node_id(file_path)
        if parent_id is not None:
            edges.append(
                GraphEdge(
                    id=f"edge:{parent_id}:contains:{_slug(str(rel))}",
                    type="CONTAINS",
                    source=parent_id,
                    target=node_id,
                    properties={"path": str(rel)},
                )
            )

        if analogue_payload is not None:
            analogue_id = f"analogue:{_slug(str(rel))}"
            analogue_kind_counts[analogue_payload["analogue_kind"]] += 1
            analogue_count += 1
            nodes.append(
                GraphNode(
                    id=analogue_id,
                    type="Analogue",
                    label=f"{file_path.stem} analogue",
                    properties={
                        "name": file_path.name,
                        "analogue_kind": analogue_payload["analogue_kind"],
                        "source_path": str(rel),
                        "source_kind": node_type,
                        "summary": analogue_payload["summary"],
                        "feature_summary": analogue_payload["feature_summary"],
                        "parametric_summary": analogue_payload["parametric_summary"],
                        "physical_summary": analogue_payload["physical_summary"],
                    },
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{node_id}:analogue:{_slug(str(rel))}",
                    type="HAS_ANALOGUE",
                    source=node_id,
                    target=analogue_id,
                    properties={"role": "detailed_analogue"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{analogue_id}:of:{_slug(str(rel))}",
                    type="ANALOGUE_OF",
                    source=analogue_id,
                    target=node_id,
                    properties={"role": "detailed_analogue"},
                )
            )

        if sample_id is not None and points is not None and fields is not None:
            sample_analogue = _create_analogue(
                path_rel=str(rel),
                name=f"{file_path.stem} sample",
                node_type="Sample",
                size_bytes=size_bytes,
                points=points,
                fields=fields,
                max_stress=max_stress,
                role="training_sample",
            )
            sample_analogue_id = f"analogue:sample:{_slug(str(rel))}"
            analogue_kind_counts[sample_analogue["analogue_kind"]] += 1
            analogue_count += 1
            nodes.append(
                GraphNode(
                    id=sample_analogue_id,
                    type="Analogue",
                    label=f"{file_path.stem} sample analogue",
                    properties={
                        "name": f"{file_path.name} sample",
                        "analogue_kind": sample_analogue["analogue_kind"],
                        "source_path": str(rel),
                        "source_kind": "Sample",
                        "summary": sample_analogue["summary"],
                        "feature_summary": sample_analogue["feature_summary"],
                        "parametric_summary": sample_analogue["parametric_summary"],
                        "physical_summary": sample_analogue["physical_summary"],
                    },
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{sample_id}:analogue:{_slug(str(rel))}",
                    type="HAS_ANALOGUE",
                    source=sample_id,
                    target=sample_analogue_id,
                    properties={"role": "sample_analogue"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{sample_analogue_id}:of:{_slug(str(rel))}",
                    type="ANALOGUE_OF",
                    source=sample_analogue_id,
                    target=sample_id,
                    properties={"role": "sample_analogue"},
                )
            )

        if node_type == "TensorShard" and sample_id is not None:
            edges.append(
                GraphEdge(
                    id=f"edge:{node_id}:sample:{_slug(str(rel))}",
                    type="HAS_SAMPLE",
                    source=node_id,
                    target=sample_id,
                    properties={},
                )
            )
            source_path = processed_manifest.get(file_path.name)
            if source_path:
                raw_candidate = Path(source_path)
                raw_abs = raw_candidate if raw_candidate.is_absolute() else (data_root.parent / raw_candidate).resolve()
                raw_id = None
                for p, existing_id in node_by_path.items():
                    if p.resolve() == raw_abs:
                        raw_id = existing_id
                        break
                if raw_id is not None:
                    edges.append(
                        GraphEdge(
                            id=f"edge:{node_id}:derived-from:{_slug(str(raw_abs.relative_to(data_root)))}",
                            type="DERIVED_FROM",
                            source=node_id,
                            target=raw_id,
                            properties={"source_path": str(raw_abs.relative_to(data_root))},
                        )
                    )

        if node_type == "DocumentAsset":
            name_lower = file_path.name.lower()
            describe_target = None
            current = file_path.parent
            while current != data_root.parent:
                if current in corpus_paths:
                    describe_target = node_by_path.get(current)
                    break
                current = current.parent
                if current == current.parent:
                    break
            if describe_target is not None and ("manifest" in name_lower or name_lower == "meta.json"):
                edges.append(
                    GraphEdge(
                        id=f"edge:{node_id}:describes:{_slug(str(rel))}",
                        type="DESCRIBES",
                        source=node_id,
                        target=describe_target,
                        properties={"role": "manifest" if "manifest" in name_lower else "metadata"},
                    )
                )

    graph = GraphDocument(
        name="local-data-inventory",
        generated_at=_utc_now(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "data_root": str(data_root),
            "corpus_count": len(corpus_paths),
            "directory_count": len(all_dirs),
            "file_count": len(all_files),
            "sample_count": sample_count,
            "analogue_count": analogue_count,
            "category_counts": dict(category_counts),
            "analogue_kind_counts": dict(analogue_kind_counts),
            "ignore_dirs": sorted(IGNORE_DIRS),
        },
    )
    notes = (
        "Corpus nodes model the major local collections; directory nodes preserve the filesystem tree.",
        "Every downloadable file now gets a first-class Analogue node with feature, parametric, and physical summaries.",
        "Tensor shards are materialized as Graph nodes and mirrored as Sample plus Analogue entries for JEPA conditioning.",
        "Manifest and metadata documents are linked to the corpora they describe.",
    )
    return LocalDataGraphReport(
        name=graph.name,
        generated_at=graph.generated_at,
        graph=graph,
        data_root=str(data_root),
        corpus_count=len(corpus_paths),
        directory_count=len(all_dirs),
        file_count=len(all_files),
        sample_count=sample_count,
        category_counts=dict(category_counts),
        notes=notes,
    )


def render_local_data_graph_report(report: LocalDataGraphReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2)
    lines = [
        f"name={report.name}",
        f"generated_at={report.generated_at}",
        f"data_root={report.data_root}",
        f"corpus_count={report.corpus_count}",
        f"directory_count={report.directory_count}",
        f"file_count={report.file_count}",
        f"sample_count={report.sample_count}",
        f"category_counts={json.dumps(report.category_counts, sort_keys=True)}",
        "notes:",
    ]
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
