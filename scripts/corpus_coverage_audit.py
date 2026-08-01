#!/usr/bin/env python3
"""Read-only audit: on-disk corpus vs TAO graph vs training resolvability.

Writes artifacts/corpus_coverage_audit.json. Does not mutate the graph.
Does not train.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
OUT = ROOT / "artifacts/corpus_coverage_audit.json"

MESH_EXTS = {".stl", ".step", ".stp", ".obj", ".ply"}
DOC_EXTS = {".pdf", ".md", ".txt", ".rst", ".csv", ".html", ".ork"}


def _count_glob(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _dir_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    n_files = 0
    bytes_ = 0
    by_ext: Counter[str] = Counter()
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        # skip nested git objects
        if "/.git/" in str(p):
            continue
        n_files += 1
        try:
            bytes_ += p.stat().st_size
        except OSError:
            pass
        by_ext[p.suffix.lower() or "<none>"] += 1
    return {
        "exists": True,
        "files": n_files,
        "bytes": bytes_,
        "gb": round(bytes_ / (1024**3), 3),
        "ext_top": by_ext.most_common(12),
    }


def main() -> int:
    t0 = time.time()
    report: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    disk_roots = {
        "openrocket_hardware_8k": ROOT / "data/openrocket_hardware_8k",
        "generated_spaceflight_cad": ROOT / "data/generated_spaceflight_cad",
        "extracted_geometries": ROOT / "data/extracted_geometries",
        "spaceflight_components": ROOT / "data/spaceflight_components",
        "raw_downloads": ROOT / "data/raw_downloads",
        "processed_nasa3d": ROOT / "data/processed/nasa3d",
        "physics_shards": ROOT / "artifacts/physics_shards",
        "rocket_fea_8k": ROOT / "artifacts/rocket_fea_8k",
        "rocket_cfd_bodyfit": ROOT / "artifacts/rocket_cfd_bodyfit",
        "jepa_train_bundle": ROOT / "artifacts/jepa-train-bundle",
    }
    report["disk"] = {k: _dir_stats(v) for k, v in disk_roots.items()}
    report["disk"]["physics_shards_fea_npz"] = _count_glob(ROOT / "artifacts/physics_shards/fea", "*.npz")
    report["disk"]["physics_shards_cfd_npz"] = _count_glob(ROOT / "artifacts/physics_shards/cfd", "*.npz")
    report["disk"]["generated_stl"] = _count_glob(ROOT / "data/generated_spaceflight_cad", "*.stl")
    report["disk"]["openrocket_stl"] = _count_glob(ROOT / "data/openrocket_hardware_8k", "*.stl")
    report["disk"]["openrocket_ork"] = _count_glob(ROOT / "data/openrocket_hardware_8k", "*.ork")
    report["disk"]["raw_mesh"] = sum(
        _count_glob(ROOT / "data/raw_downloads", f"*{ext}") for ext in MESH_EXTS
    )
    report["disk"]["raw_docs"] = sum(
        _count_glob(ROOT / "data/raw_downloads", f"*{ext}") for ext in (".pdf", ".md", ".txt")
    )

    if not GRAPH.exists():
        report["error"] = f"missing graph: {GRAPH}"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    by_type = Counter(str(n.get("type")) for n in nodes)
    report["graph"] = {
        "path": str(GRAPH.relative_to(ROOT)),
        "nodes": len(nodes),
        "edges": len(edges),
        "by_type": dict(by_type.most_common()),
        "bytes": GRAPH.stat().st_size,
    }

    parts = [n for n in nodes if n.get("type") in {"Part", "RealPart"}]
    docs = [n for n in nodes if n.get("type") == "Document"]
    shards = [n for n in nodes if n.get("type") == "TensorShard"]
    samples = [n for n in nodes if n.get("type") == "Sample"]

    docs_with_text = 0
    for d in docs:
        p = d.get("properties") or {}
        if isinstance(p.get("text"), str) and p["text"].strip():
            docs_with_text += 1

    rocket = legacy = 0
    with_fea = with_cfd = with_prompt = with_cd = 0
    geom_refs_generated = 0
    for part in parts:
        props = part.get("properties") or {}
        src = str(props.get("source_corpus") or "")
        gid = str(part.get("id") or "")
        if "rocket" in gid or src.startswith("openrocket"):
            rocket += 1
        elif "legacy" in gid or src.startswith("legacy"):
            legacy += 1
        if part.get("has_fea") or props.get("has_fea"):
            with_fea += 1
        if part.get("has_cfd") or props.get("has_cfd"):
            with_cfd += 1
        if props.get("spec_prompt"):
            with_prompt += 1
        cfd = part.get("simulation_results_cfd") if isinstance(part.get("simulation_results_cfd"), dict) else {}
        if props.get("Cd") is not None or cfd.get("Cd") is not None or cfd.get("Cd_proxy") is not None:
            with_cd += 1
        gref = str(props.get("geometry_ref") or "")
        if "generated_spaceflight_cad" in gref:
            geom_refs_generated += 1

    physics_shards = 0
    nasa_shards = 0
    for s in shards:
        props = s.get("properties") or {}
        path = str(props.get("shard_path") or props.get("path") or "")
        if "physics_shards" in path:
            physics_shards += 1
        elif "nasa" in path:
            nasa_shards += 1

    report["coverage"] = {
        "parts_total": len(parts),
        "parts_rocket_guess": rocket,
        "parts_legacy_guess": legacy,
        "parts_has_fea": with_fea,
        "parts_has_cfd": with_cfd,
        "parts_spec_prompt": with_prompt,
        "parts_with_cd_signal": with_cd,
        "parts_geometry_generated_cad": geom_refs_generated,
        "documents": len(docs),
        "documents_with_text": docs_with_text,
        "samples": len(samples),
        "tensor_shards": len(shards),
        "tensor_shards_physics_path": physics_shards,
        "tensor_shards_nasa_path": nasa_shards,
    }

    # Gaps vs disk
    report["gaps"] = {
        "generated_stl_not_as_parts": max(
            0, int(report["disk"]["generated_stl"]) - geom_refs_generated
        ),
        "docs_missing_text": max(0, len(docs) - docs_with_text),
        "cfd_npz_vs_bodyfit_metas": {
            "cfd_npz": report["disk"]["physics_shards_cfd_npz"],
            "note": "bodyfit metas often lack volume NPZ after field cleanup",
        },
        "modal_staging_risk": "jepa-train-bundle alone misses artifacts/physics_shards unless packaged",
    }

    report["elapsed_s"] = round(time.time() - t0, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({"coverage": report["coverage"], "gaps": report["gaps"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
