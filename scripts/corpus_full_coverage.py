#!/usr/bin/env python3
"""Full-corpus densify: pull leftover mesh/text into the TAO train graph.

Does NOT launch training. Graph-locked, idempotent-ish.

Covers:
  1. data/generated_spaceflight_cad/*.stl → Parts + Samples
  2. extracted_geometries / spaceflight_components STEP/STL → Parts + Samples
  3. raw_downloads mesh (stl/step/obj/ply) as RealPart/Sample when missing
  4. Document text extract for nodes still missing text (+ new docs)
  5. ORK manifest / .ork design notes → Document text linked to Parts
  6. Rewrite nasa3d TensorShard paths → data/processed/nasa3d/...
  7. Register FEA/CFD shard manifests
  8. Re-associate Samples / PhysicsTargets / Dimensions
  9. Refresh spec_prompt on newly added Parts
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from cadflow.associate_training_data import associate_parts  # noqa: E402
from cadflow.build_physics_shards import (  # noqa: E402
    CFD_MANIFEST,
    FEA_MANIFEST,
    register_manifest_to_graph,
)
from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic  # noqa: E402

GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
REPORT = ROOT / "artifacts/corpus_full_coverage_report.json"

MESH_EXTS = {".stl", ".step", ".stp", ".obj", ".ply"}
DOC_EXTS = {".pdf", ".md", ".txt", ".rst", ".csv", ".html"}
TEXT_MAX = 12_000
RAW_MESH_LIMIT = 4000  # cap first-pass raw_downloads mesh ingest


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s[:100] or "x"


def extract_text(path: Path) -> str:
    if not path.is_file():
        return ""
    suf = path.suffix.lower()
    try:
        if suf == ".pdf":
            r = subprocess.run(
                ["pdftotext", "-layout", "-q", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            text = r.stdout or ""
        elif suf == ".ork":
            text = _ork_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:TEXT_MAX]


def _ork_text(path: Path) -> str:
    """OpenRocket designs are zip+XML; pull name/designer/description strings."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            xml_name = next((n for n in names if n.endswith(".ork") or n.endswith(".xml") or "rocket" in n.lower()), None)
            if xml_name is None and names:
                xml_name = names[0]
            raw = zf.read(xml_name).decode("utf-8", errors="ignore") if xml_name else ""
    except Exception:
        return path.stem
    bits = re.findall(
        r"<(?:name|designer|description|comment|manufacturer)[^>]*>([^<]{2,200})</",
        raw,
        flags=re.I,
    )
    joined = " | ".join(dict.fromkeys(b.strip() for b in bits if b.strip()))
    return f"openrocket {path.stem}: {joined}"[:TEXT_MAX]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _ensure_node(by_id: dict[str, dict], node: dict[str, Any]) -> bool:
    nid = str(node["id"])
    if nid in by_id:
        return False
    by_id[nid] = node
    return True


def _ensure_edge(edge_index: dict, *, edge_type: str, source: str, target: str, props: dict | None = None) -> bool:
    key = (edge_type, source, target)
    if key in edge_index:
        return False
    eid = f"edge:{edge_type}:{_slug(source)}:{_slug(target)}:{hashlib.md5(f'{source}|{target}'.encode()).hexdigest()[:8]}"
    edge_index[key] = {
        "id": eid,
        "type": edge_type,
        "source": source,
        "target": target,
        "properties": props or {},
    }
    return True


def _family_from_name(name: str) -> str:
    low = name.lower()
    for fam in (
        "nose_cone",
        "body_tube",
        "fin",
        "fairing",
        "tank",
        "nozzle",
        "transition",
        "bulkhead",
        "ring_frame",
        "engine_mount",
        "tps_tile",
        "blanket",
        "solar_panel",
        "antenna",
        "strut",
    ):
        if fam.replace("_", "") in low.replace("_", "") or fam in low:
            return fam
    return "structure"


def _spec_prompt(props: dict[str, Any]) -> str:
    fam = props.get("family") or props.get("part_class") or "part"
    params = props.get("params") if isinstance(props.get("params"), dict) else {}
    bits = [f"family={fam}"]
    for k, v in list(params.items())[:12]:
        bits.append(f"{k}={v}")
    for k in ("material_name", "material_id", "Cd", "mass_kg"):
        if props.get(k) is not None:
            bits.append(f"{k}={props[k]}")
    tags = props.get("tags")
    if isinstance(tags, list) and tags:
        bits.append("tags=" + ",".join(str(t) for t in tags[:8]))
    return "; ".join(bits)


def ingest_mesh_tree(
    *,
    by_id: dict[str, dict],
    edge_index: dict,
    root: Path,
    source_corpus: str,
    part_prefix: str,
    limit: int | None = None,
    as_real_part: bool = False,
) -> dict[str, int]:
    stats = {"seen": 0, "parts_added": 0, "samples_added": 0, "edges_added": 0}
    if not root.exists():
        return stats
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MESH_EXTS and "/.git/" not in str(p)
    )
    if limit is not None:
        files = files[:limit]

    existing_geom: set[str] = set()
    for existing in by_id.values():
        if existing.get("type") not in {"Part", "RealPart"}:
            continue
        gref = str((existing.get("properties") or {}).get("geometry_ref") or "")
        if gref:
            existing_geom.add(gref)
            try:
                existing_geom.add(str(Path(gref).resolve()))
            except OSError:
                pass

    for path in files:
        stats["seen"] += 1
        resolved = str(path.resolve())
        rel = _rel(path)
        if resolved in existing_geom or rel in existing_geom:
            # Still ensure a Sample for training reachability when Part exists.
            part_id = None
            for existing in by_id.values():
                if existing.get("type") not in {"Part", "RealPart"}:
                    continue
                gref = str((existing.get("properties") or {}).get("geometry_ref") or "")
                if gref in {resolved, rel} or Path(gref).name == path.name and resolved in gref:
                    part_id = str(existing["id"])
                    break
            if part_id is None:
                continue
        else:
            stem = path.stem
            fam = _family_from_name(stem)
            part_id = f"part:{part_prefix}:{_slug(stem)}"
            # disambiguate collisions
            if part_id in by_id:
                part_id = f"part:{part_prefix}:{_slug(stem)}-{hashlib.md5(resolved.encode()).hexdigest()[:8]}"
            ntype = "RealPart" if as_real_part else "Part"
            props = {
                "name": stem,
                "family": fam,
                "part_class": fam,
                "geometry_ref": resolved,
                "params": {},
                "tags": [source_corpus, "full_coverage_ingest", path.suffix.lower().lstrip(".")],
                "source_corpus": source_corpus,
                "raw_geometry": {
                    "format": path.suffix.lower().lstrip("."),
                    "path": resolved,
                    "path_rel": rel,
                    "family": fam,
                },
            }
            if path.suffix.lower() == ".stl":
                props["stl"] = rel
            else:
                props["cad_ref"] = rel
            props["spec_prompt"] = _spec_prompt(props)
            if _ensure_node(
                by_id,
                {
                    "id": part_id,
                    "label": stem,
                    "type": ntype,
                    "properties": props,
                    "has_fea": False,
                    "has_cfd": False,
                    "physics_verified": False,
                },
            ):
                stats["parts_added"] += 1
                existing_geom.add(resolved)
                existing_geom.add(rel)

        sample_id = f"sample:coverage:{part_id}"
        if _ensure_node(
            by_id,
            {
                "id": sample_id,
                "type": "Sample",
                "label": Path(path).stem,
                "properties": {
                    "path": rel,
                    "source_path": rel,
                    "geometry_ref": resolved,
                    "family": _family_from_name(path.stem),
                    "part_id": part_id,
                    "params": {},
                    "summary": {"kind": "full_coverage_mesh", "source": source_corpus},
                    "source": "corpus_full_coverage",
                },
            },
        ):
            stats["samples_added"] += 1
        if _ensure_edge(edge_index, edge_type="REPRESENTS", source=sample_id, target=part_id):
            stats["edges_added"] += 1
        if _ensure_edge(edge_index, edge_type="HAS_SAMPLE", source=part_id, target=sample_id):
            stats["edges_added"] += 1
    return stats


def densify_documents(by_id: dict[str, dict], edge_index: dict) -> dict[str, int]:
    stats = {"filled_text": 0, "new_docs": 0, "links": 0, "scanned_files": 0}
    # Fill existing Document nodes missing text
    for node in list(by_id.values()):
        if node.get("type") != "Document":
            continue
        props = dict(node.get("properties") or {})
        if isinstance(props.get("text"), str) and props["text"].strip():
            continue
        for key in ("source_path", "path", "file_path", "local_path"):
            raw = props.get(key)
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                for base in (ROOT, ROOT / "artifacts/jepa-train-bundle", ROOT / "data"):
                    cand = base / path
                    if cand.is_file():
                        path = cand
                        break
            if path.is_file():
                text = extract_text(path)
                if text:
                    props["text"] = text
                    props["text_chars"] = len(text)
                    props["text_source"] = "corpus_full_coverage"
                    node["properties"] = props
                    stats["filled_text"] += 1
                break

    # Discover new docs under key trees
    search_roots = [
        ROOT / "data/raw_downloads",
        ROOT / "data/openrocket_hardware_8k",
        ROOT / "artifacts/jepa-train-bundle/files",
        ROOT / "docs",
    ]
    existing_paths = set()
    for node in by_id.values():
        if node.get("type") != "Document":
            continue
        props = node.get("properties") or {}
        for key in ("source_path", "path"):
            if props.get(key):
                existing_paths.add(str(props[key]))

    part_names = []
    for node in by_id.values():
        if node.get("type") not in {"Part", "RealPart"}:
            continue
        props = node.get("properties") or {}
        part_names.append((str(node["id"]), str(props.get("name") or node.get("label") or "")))

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "/.git/" in str(path):
                continue
            if path.suffix.lower() not in DOC_EXTS | {".ork"}:
                continue
            stats["scanned_files"] += 1
            rel = _rel(path)
            if rel in existing_paths or str(path) in existing_paths:
                continue
            text = extract_text(path)
            if not text and path.suffix.lower() not in {".ork"}:
                # still register path for linking later
                text = path.stem
            doc_id = f"document:coverage:{_slug(rel)}"
            if _ensure_node(
                by_id,
                {
                    "id": doc_id,
                    "type": "Document",
                    "label": path.name,
                    "properties": {
                        "title": path.name,
                        "source_path": rel,
                        "path": rel,
                        "text": text,
                        "text_chars": len(text),
                        "text_source": "corpus_full_coverage",
                        "tags": ["full_coverage_doc"],
                    },
                },
            ):
                stats["new_docs"] += 1
                existing_paths.add(rel)
            # cheap stem link to parts
            stem = path.stem.lower()
            for pid, pname in part_names:
                if not pname:
                    continue
                pn = pname.lower()
                if pn in stem or stem in pn or _slug(pn) in _slug(stem):
                    if _ensure_edge(edge_index, edge_type="DESCRIBES", source=doc_id, target=pid):
                        stats["links"] += 1
                    if _ensure_edge(edge_index, edge_type="MENTIONS", source=doc_id, target=pid):
                        stats["links"] += 1
                    break
    return stats


def rewrite_nasa_paths(by_id: dict[str, dict]) -> int:
    n = 0
    processed = ROOT / "data/processed"
    for node in by_id.values():
        if node.get("type") != "TensorShard":
            continue
        props = dict(node.get("properties") or {})
        path = props.get("path") or props.get("shard_path")
        if not isinstance(path, str):
            continue
        if path.startswith("nasa3d/") or path.startswith("nasa3d\\"):
            new_rel = f"data/processed/{path.replace(chr(92), '/')}"
            abs_path = ROOT / new_rel
            if abs_path.is_file():
                props["path"] = new_rel
                props["source_path"] = new_rel
                props["path_rewrite"] = "corpus_full_coverage_nasa3d"
                node["properties"] = props
                n += 1
            elif (processed / path).is_file():
                props["path"] = _rel(processed / path)
                props["source_path"] = props["path"]
                props["path_rewrite"] = "corpus_full_coverage_nasa3d"
                node["properties"] = props
                n += 1
    return n


def refresh_prompts(by_id: dict[str, dict]) -> int:
    n = 0
    for node in by_id.values():
        if node.get("type") not in {"Part", "RealPart"}:
            continue
        props = dict(node.get("properties") or {})
        if props.get("spec_prompt"):
            continue
        props["spec_prompt"] = _spec_prompt(props)
        node["properties"] = props
        n += 1
    return n


def main() -> int:
    t0 = time.time()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graph": str(GRAPH),
    }
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

        mesh_stats = {}
        mesh_stats["generated"] = ingest_mesh_tree(
            by_id=by_id,
            edge_index=edge_index,
            root=ROOT / "data/generated_spaceflight_cad",
            source_corpus="generated_spaceflight_cad",
            part_prefix="generated",
        )
        mesh_stats["extracted"] = ingest_mesh_tree(
            by_id=by_id,
            edge_index=edge_index,
            root=ROOT / "data/extracted_geometries",
            source_corpus="extracted_geometries",
            part_prefix="extracted",
        )
        mesh_stats["spaceflight_components"] = ingest_mesh_tree(
            by_id=by_id,
            edge_index=edge_index,
            root=ROOT / "data/spaceflight_components",
            source_corpus="spaceflight_components",
            part_prefix="sfc",
        )
        mesh_stats["raw_downloads"] = ingest_mesh_tree(
            by_id=by_id,
            edge_index=edge_index,
            root=ROOT / "data/raw_downloads",
            source_corpus="raw_downloads",
            part_prefix="rawdl",
            limit=RAW_MESH_LIMIT,
            as_real_part=True,
        )
        report["mesh_ingest"] = mesh_stats

        report["documents"] = densify_documents(by_id, edge_index)
        report["nasa_path_rewrites"] = rewrite_nasa_paths(by_id)
        report["spec_prompts_added"] = refresh_prompts(by_id)

        # Rebuild node/edge lists
        graph["nodes"] = list(by_id.values())
        graph["edges"] = list(edge_index.values())

        assoc = associate_parts(graph, only_missing=True)
        report["associate"] = assoc

        write_graph_atomic(GRAPH, graph)

    # Shard register outside lock (it locks internally)
    shard_stats = {}
    for label, manifest in (("fea", FEA_MANIFEST), ("cfd", CFD_MANIFEST)):
        try:
            if Path(manifest).exists():
                shard_stats[label] = register_manifest_to_graph(GRAPH, manifest)
            else:
                shard_stats[label] = {"skipped": True, "reason": "missing_manifest"}
        except Exception as exc:  # noqa: BLE001
            shard_stats[label] = {"error": str(exc)}
    report["shard_register"] = shard_stats
    report["elapsed_s"] = round(time.time() - t0, 2)

    # Quick post counts
    g2 = json.loads(GRAPH.read_text(encoding="utf-8"))
    types = defaultdict(int)
    for n in g2.get("nodes") or []:
        types[str(n.get("type"))] += 1
    docs = [n for n in g2["nodes"] if n.get("type") == "Document"]
    with_text = sum(1 for d in docs if isinstance((d.get("properties") or {}).get("text"), str) and (d.get("properties") or {}).get("text").strip())
    report["post"] = {
        "nodes": len(g2.get("nodes") or []),
        "edges": len(g2.get("edges") or []),
        "by_type": dict(types),
        "documents_with_text": with_text,
        "documents": len(docs),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mesh_ingest", "documents", "nasa_path_rewrites", "spec_prompts_added", "post", "elapsed_s") if k in report}, indent=2))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
