#!/usr/bin/env python3
"""Solidify TAO associations: account for every document, orphan Parts, features.

- Remaining Documents: retry pdftotext; else metadata/title fallback + explicit
  text_extract_status so nothing is silently unaccounted.
- Orphan Parts/RealParts: ensure Sample + REPRESENTS/HAS_SAMPLE via associate.
- Features: ensure HAS_FEATURE edges from owning Parts when props carry part_id.
- Writes artifacts/tao_solidify_report.json

Does not train. Graph-locked.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from cadflow.associate_training_data import associate_parts  # noqa: E402
from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic  # noqa: E402

GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
REPORT = ROOT / "artifacts/tao_solidify_report.json"
TEXT_MAX = 12_000


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s[:100] or "x"


def _resolve_doc_path(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    for base in (
        ROOT,
        ROOT / "artifacts/jepa-train-bundle",
        ROOT / "artifacts/jepa-train-bundle/files",
        ROOT / "data",
        ROOT / "data/raw_downloads",
    ):
        cand = base / raw
        if cand.is_file():
            return cand
        if raw.startswith("files/"):
            cand2 = (ROOT / "artifacts/jepa-train-bundle") / raw
            if cand2.is_file():
                return cand2
    return None


def _pdf_meta(path: Path) -> str:
    try:
        r = subprocess.run(
            ["pdfinfo", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return (r.stdout or "").strip()[:4000]
    except Exception:
        return ""


def extract_text(path: Path) -> tuple[str, str]:
    """Return (text, status). status in extracted|fallback_meta|missing_file|empty."""
    if not path.is_file():
        return "", "missing_file"
    suf = path.suffix.lower()
    text = ""
    try:
        if suf == ".pdf":
            r = subprocess.run(
                ["pdftotext", "-layout", "-q", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            text = re.sub(r"\s+", " ", (r.stdout or "")).strip()
            # form-feed only / scanned
            if len(re.sub(r"[\x0c\s]", "", text)) < 40:
                meta = _pdf_meta(path)
                title = path.stem
                fallback = f"[scanned_or_empty_pdf] title={title}; path={path.name}; {meta}"
                return fallback[:TEXT_MAX], "fallback_meta"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"\s+", " ", text).strip()
    except Exception as exc:
        return f"[extract_error] {path.name}: {exc}"[:TEXT_MAX], "fallback_meta"
    if text:
        return text[:TEXT_MAX], "extracted"
    return f"[empty] path={path.name}"[:TEXT_MAX], "empty"


def _ensure_edge(edge_index: dict, *, edge_type: str, source: str, target: str) -> bool:
    key = (edge_type, source, target)
    if key in edge_index:
        return False
    eid = (
        f"edge:{edge_type}:{_slug(source)}:{_slug(target)}:"
        f"{hashlib.md5(f'{source}|{target}|{edge_type}'.encode()).hexdigest()[:8]}"
    )
    edge_index[key] = {
        "id": eid,
        "type": edge_type,
        "source": source,
        "target": target,
        "properties": {"source": "tao_solidify"},
    }
    return True


def account_documents(by_id: dict[str, dict]) -> dict[str, int]:
    stats = {
        "seen": 0,
        "extracted": 0,
        "fallback_meta": 0,
        "empty": 0,
        "missing_file": 0,
        "already_ok": 0,
    }
    for node in by_id.values():
        if node.get("type") != "Document":
            continue
        stats["seen"] += 1
        props = dict(node.get("properties") or {})
        existing = props.get("text")
        status = props.get("text_extract_status")
        if isinstance(existing, str) and existing.strip() and status not in {None, "pending"}:
            # already accounted; skip unless previous empty without status
            if len(existing.strip()) >= 20 or status:
                stats["already_ok"] += 1
                continue
        raw = props.get("source_path") or props.get("path") or props.get("file_path")
        path = _resolve_doc_path(str(raw) if raw else "")
        if path is None:
            title = props.get("title") or node.get("label") or node.get("id")
            props["text"] = f"[unresolved_path] title={title}; raw={raw}"
            props["text_chars"] = len(props["text"])
            props["text_extract_status"] = "missing_file"
            props["text_source"] = "tao_solidify"
            node["properties"] = props
            stats["missing_file"] += 1
            continue
        text, st = extract_text(path)
        props["text"] = text
        props["text_chars"] = len(text)
        props["text_extract_status"] = st
        props["text_source"] = "tao_solidify"
        props["resolved_path"] = str(path)
        node["properties"] = props
        stats[st] = stats.get(st, 0) + 1
    return stats


def backfill_orphan_geometry(by_id: dict[str, dict], edge_index: dict) -> dict[str, int]:
    """RealParts often only have source_path=files/...; wire geometry_ref + Samples."""
    from cadflow.associate_training_data import _resolve_geometry_path, _ensure_node, _ensure_edge, _REP_EDGE

    stats = {"orphans_seen": 0, "geometry_resolved": 0, "samples": 0, "edges": 0, "still_missing": 0}
    touched = set()
    for (etype, src, tgt) in edge_index:
        if etype in {"HAS_SAMPLE", "REPRESENTS"}:
            touched.add(src)
            touched.add(tgt)

    for nid, node in list(by_id.items()):
        if node.get("type") not in {"Part", "RealPart"}:
            continue
        if nid in touched:
            continue
        stats["orphans_seen"] += 1
        props = dict(node.get("properties") or {})
        geom = _resolve_geometry_path(node)
        if geom is None:
            # Still account with a Sample pointing at declared source_path (even if missing)
            raw = props.get("source_path") or props.get("relative_path")
            sample_id = f"sample:accounted:{nid}"
            created = _ensure_node(
                by_id,
                {
                    "id": sample_id,
                    "type": "Sample",
                    "label": props.get("name") or node.get("label") or nid,
                    "properties": {
                        "path": str(raw) if raw else "",
                        "source_path": str(raw) if raw else "",
                        "family": props.get("family") or "generic",
                        "part_id": nid,
                        "file_format": props.get("file_format"),
                        "summary": {
                            "kind": "accounted_unresolved_geometry",
                            "file_format": props.get("file_format"),
                        },
                        "parseable": False,
                        "source": "tao_solidify",
                    },
                },
            )
            if created:
                stats["samples"] += 1
            if _ensure_edge(edge_index, edge_type=_REP_EDGE, source=sample_id, target=nid):
                stats["edges"] += 1
            if _ensure_edge(edge_index, edge_type="HAS_SAMPLE", source=nid, target=sample_id):
                stats["edges"] += 1
            props["geometry_accounted"] = True
            props["geometry_resolve_status"] = "unresolved"
            node["properties"] = props
            stats["still_missing"] += 1
            continue

        stats["geometry_resolved"] += 1
        props["geometry_ref"] = str(geom)
        try:
            props["geometry_ref_rel"] = str(geom.relative_to(ROOT))
        except ValueError:
            props["geometry_ref_rel"] = str(geom)
        props["geometry_accounted"] = True
        props["geometry_resolve_status"] = "resolved"
        # Mark parseability for training filters
        props["parseable_for_pointcloud"] = geom.suffix.lower() in {".stl", ".obj", ".ply", ".step", ".stp"}
        node["properties"] = props

        sample_id = f"sample:accounted:{nid}"
        rel = props.get("geometry_ref_rel") or str(geom)
        created = _ensure_node(
            by_id,
            {
                "id": sample_id,
                "type": "Sample",
                "label": props.get("name") or node.get("label") or nid,
                "properties": {
                    "path": rel,
                    "source_path": rel,
                    "geometry_ref": str(geom),
                    "family": props.get("family") or "generic",
                    "part_id": nid,
                    "file_format": props.get("file_format") or geom.suffix.lower().lstrip("."),
                    "params": props.get("params") if isinstance(props.get("params"), dict) else {},
                    "summary": {
                        "kind": "accounted_part_geometry",
                        "parseable_for_pointcloud": props.get("parseable_for_pointcloud"),
                    },
                    "parseable": bool(props.get("parseable_for_pointcloud")),
                    "source": "tao_solidify",
                },
            },
        )
        if created:
            stats["samples"] += 1
        if _ensure_edge(edge_index, edge_type=_REP_EDGE, source=sample_id, target=nid):
            stats["edges"] += 1
        if _ensure_edge(edge_index, edge_type="HAS_SAMPLE", source=nid, target=sample_id):
            stats["edges"] += 1
    return stats


def link_features(by_id: dict[str, dict], edge_index: dict) -> dict[str, int]:
    stats = {"features": 0, "linked": 0, "unlinked": 0}
    part_ids = {nid for nid, n in by_id.items() if n.get("type") in {"Part", "RealPart"}}
    for nid, node in by_id.items():
        if node.get("type") != "Feature":
            continue
        stats["features"] += 1
        props = node.get("properties") or {}
        candidates = [
            props.get("part_id"),
            props.get("owner_id"),
            props.get("parent_id"),
            props.get("source_part_id"),
        ]
        linked = False
        for cand in candidates:
            if not cand:
                continue
            pid = str(cand)
            if pid not in part_ids and not pid.startswith("part:"):
                pid = f"part:{pid}"
            if pid in part_ids:
                if _ensure_edge(edge_index, edge_type="HAS_FEATURE", source=pid, target=nid):
                    stats["linked"] += 1
                linked = True
                break
        if not linked:
            stats["unlinked"] += 1
    return stats


def main() -> int:
    t0 = time.time()
    report: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not GRAPH.exists():
        report["error"] = "missing graph"
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        return 1

    with graph_lock(GRAPH):
        graph = read_graph(GRAPH)
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        by_id = {str(n["id"]): n for n in nodes if n.get("id")}
        edge_index = {
            (str(e.get("type")), str(e.get("source")), str(e.get("target"))): e
            for e in edges
            if e.get("type") and e.get("source") and e.get("target")
        }

        report["documents"] = account_documents(by_id)
        report["orphan_geometry"] = backfill_orphan_geometry(by_id, edge_index)
        report["features"] = link_features(by_id, edge_index)

        graph["nodes"] = list(by_id.values())
        graph["edges"] = list(edge_index.values())

        # Associate orphan geometry Parts → Samples / PhysicsTargets / metrics
        report["associate"] = associate_parts(graph, only_missing=True)

        write_graph_atomic(GRAPH, graph)

    # Post counts
    g2 = json.loads(GRAPH.read_text(encoding="utf-8"))
    docs = [n for n in g2["nodes"] if n.get("type") == "Document"]
    accounted = 0
    by_status: dict[str, int] = {}
    for d in docs:
        p = d.get("properties") or {}
        t = p.get("text")
        st = p.get("text_extract_status") or ("ok" if isinstance(t, str) and t.strip() else "missing")
        by_status[st] = by_status.get(st, 0) + 1
        if isinstance(t, str) and t.strip():
            accounted += 1
    part_ids = {n["id"] for n in g2["nodes"] if n.get("type") in {"Part", "RealPart"}}
    touched = set()
    for e in g2["edges"]:
        if e.get("type") in {"HAS_SAMPLE", "REPRESENTS"}:
            touched.add(e.get("source"))
            touched.add(e.get("target"))
    orphans = sum(1 for pid in part_ids if pid not in touched)
    report["post"] = {
        "nodes": len(g2["nodes"]),
        "edges": len(g2["edges"]),
        "documents": len(docs),
        "documents_accounted": accounted,
        "document_status": by_status,
        "parts": len(part_ids),
        "parts_without_sample_edge": orphans,
    }
    report["elapsed_s"] = round(time.time() - t0, 2)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
