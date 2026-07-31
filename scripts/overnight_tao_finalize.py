#!/usr/bin/env python3
"""Finalize TAO after solvers drain: ingest, associate, enrich docs/CAD, validate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def ingest_physics() -> dict:
    from cadflow.enrich_tao_mass_properties import apply_sidecar_to_graph
    from cadflow.rocket_physics_suite import (
        DEFAULT_FEA_ROOT,
        ingest_cfd_to_graph,
        ingest_fea_to_graph,
    )
    from cadflow.build_physics_shards import register_manifest_to_graph

    fea_n = ingest_fea_to_graph(GRAPH, DEFAULT_FEA_ROOT)
    _log(f"fea_ingest linked={fea_n}")

    results = []
    cfd_root = ROOT / "artifacts/rocket_cfd_bodyfit"
    if cfd_root.is_dir():
        for meta in cfd_root.glob("*/meta.json"):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            metrics = m.get("metrics") or {}
            if float(metrics.get("U_mag_max") or 0) <= 1e-6:
                continue
            results.append(
                {
                    "part_id": m.get("part_id") or meta.parent.name,
                    "success": True,
                    "metrics": metrics,
                }
            )
    cfd_n = ingest_cfd_to_graph(GRAPH, results) if results else 0
    _log(f"cfd_ingest linked={cfd_n} from={len(results)}")

    # Legacy internal metas
    try:
        from cadflow.legacy_cfd_internal import ingest_internal, CFD_ROOT

        leg_results = []
        for meta in CFD_ROOT.glob("*/meta.json"):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            metrics = m.get("metrics") or {}
            if float(metrics.get("U_mag_max") or 0) <= 1e-6:
                continue
            leg_results.append(
                {
                    "part_id": m["part_id"],
                    "case_id": m.get("case_id") or meta.parent.name,
                    "recipe_id": m.get("recipe_id"),
                    "family": m.get("family"),
                    "success": True,
                    "metrics": metrics,
                }
            )
        leg_n = ingest_internal(GRAPH, leg_results) if leg_results else 0
        _log(f"legacy_internal_ingest linked={leg_n} from={len(leg_results)}")
    except Exception as exc:  # noqa: BLE001
        _log(f"legacy_internal_ingest_error {exc}")
        leg_n = 0

    try:
        from cadflow.legacy_cfd_bodyfit import ingest_legacy_bodyfit_from_disk

        body_stats = ingest_legacy_bodyfit_from_disk(GRAPH)
        body_n = int(body_stats.get("linked") or body_stats.get("ok") or 0)
        _log(f"legacy_bodyfit_ingest {body_stats}")
    except Exception as exc:  # noqa: BLE001
        _log(f"legacy_bodyfit_ingest_error {exc}")
        body_n = 0

    mass_n = apply_sidecar_to_graph()
    _log(f"mass_reapplied {mass_n}")
    try:
        reg = register_manifest_to_graph(GRAPH)
        _log(f"shard_register {reg}")
    except Exception as exc:  # noqa: BLE001
        _log(f"shard_register_error {exc}")
        reg = {"error": str(exc)}
    return {
        "fea": fea_n,
        "cfd": cfd_n,
        "legacy_internal": leg_n,
        "legacy_bodyfit": body_n,
        "mass": mass_n,
        "shards": reg,
    }


def associate() -> dict:
    from cadflow.associate_training_data import associate_graph_file

    stats = associate_graph_file(GRAPH)
    _log(f"associate {stats}")
    return stats


def enrich_documents() -> dict:
    """Merge Document/Source/geometry enrichment from raw spaceflight trees into TAO."""
    from cadflow.graph_enrichment import build_enrichment_graph, merge_graphs
    from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic
    from cadflow.graph_schema import GraphDocument, GraphNode, GraphEdge

    raw_dirs: list[Path] = []
    for base in (
        ROOT / "data/raw_downloads/external",
        ROOT / "data/raw_downloads/nasa3d",
        ROOT / "data/spaceflight_components",
        ROOT / "data/openrocket_hardware_8k",
        ROOT / "data/generated_spaceflight_cad",
    ):
        if base.is_dir():
            # Prefer leaf source dirs under external; otherwise the base itself.
            kids = [d for d in base.iterdir() if d.is_dir()]
            raw_dirs.extend(kids if kids and base.name in {"external", "nasa3d", "spaceflight_components"} else [base])

    with graph_lock(GRAPH):
        existing_data = read_graph(GRAPH)
        existing_doc = GraphDocument(
            name=existing_data.get("name") or "jepa-train-bundle",
            generated_at=existing_data.get("generated_at") or "",
            nodes=tuple(
                GraphNode(
                    id=n["id"],
                    type=n["type"],
                    label=n.get("label") or n["id"],
                    properties=n.get("properties") or {},
                )
                for n in existing_data["nodes"]
            ),
            edges=tuple(
                GraphEdge(
                    id=e["id"],
                    type=e["type"],
                    source=e["source"],
                    target=e["target"],
                    properties=e.get("properties") or {},
                )
                for e in existing_data["edges"]
            ),
            metadata=existing_data.get("metadata") or {},
        )
        enrichment_doc, report = build_enrichment_graph(raw_dirs, existing_graph=existing_doc)
        merged = merge_graphs(existing_doc, enrichment_doc, name="jepa-train-bundle")
        out = merged.to_dict()
        # Preserve training-critical metadata keys from live graph
        meta = dict(existing_data.get("metadata") or {})
        meta.update(out.get("metadata") or {})
        meta["overnight_enrich"] = {
            "document_nodes": report.document_nodes,
            "source_nodes": report.source_nodes,
            "realpart_nodes": report.realpart_nodes,
            "material_nodes": report.material_nodes,
        }
        out["metadata"] = meta
        write_graph_atomic(GRAPH, out)
    summary = {
        "document_nodes": report.document_nodes,
        "document_edges": report.document_edges,
        "source_nodes": report.source_nodes,
        "realpart_nodes": report.realpart_nodes,
        "material_nodes": report.material_nodes,
        "assembly_nodes": report.assembly_nodes,
        "total_new_nodes": report.total_new_nodes,
        "total_new_edges": report.total_new_edges,
        "raw_dirs": [str(p) for p in raw_dirs[:40]],
    }
    _log(f"enrich {summary}")
    return summary


def validate() -> dict:
    from collections import Counter

    g = json.loads(GRAPH.read_text())
    types = Counter(n.get("type") for n in g["nodes"])
    edges = Counter(e.get("type") for e in g["edges"])
    rocket = legacy = r_fea = r_cfd = l_fea = l_cfd = both = 0
    orphan_parts = 0
    part_ids = {n["id"] for n in g["nodes"] if n.get("type") == "Part"}
    linked = set()
    for e in g["edges"]:
        if e.get("source") in part_ids:
            linked.add(e["source"])
        if e.get("target") in part_ids:
            linked.add(e["target"])
    for n in g["nodes"]:
        if n.get("type") != "Part":
            continue
        pid = str(n.get("id") or "")
        is_r = pid.startswith("part:rocket:")
        hf = bool(n.get("has_fea") or n.get("simulation_results_fea"))
        hc = bool(n.get("has_cfd") or n.get("simulation_results_cfd"))
        if is_r:
            rocket += 1
            r_fea += int(hf)
            r_cfd += int(hc)
        else:
            legacy += 1
            l_fea += int(hf)
            l_cfd += int(hc)
        if hf and hc:
            both += 1
        if pid not in linked:
            orphan_parts += 1
    report = {
        "node_types": dict(types.most_common(30)),
        "edge_types": dict(edges.most_common(25)),
        "rocket_parts": rocket,
        "legacy_parts": legacy,
        "rocket_fea": r_fea,
        "rocket_cfd": r_cfd,
        "legacy_fea": l_fea,
        "legacy_cfd": l_cfd,
        "parts_with_both_sims": both,
        "orphan_parts": orphan_parts,
        "graph_mb": round(GRAPH.stat().st_size / 1e6, 1),
        "ok": rocket >= 10500
        and legacy >= 2150
        and r_fea >= 5500
        and r_cfd >= 5000
        and types.get("Document", 0) >= 500
        and types.get("TensorShard", 0) >= 3000,
    }
    _log(f"validate {json.dumps(report)}")
    return report


def main() -> int:
    report: dict = {"phases": {}}
    report["phases"]["ingest"] = ingest_physics()
    report["phases"]["associate"] = associate()
    try:
        report["phases"]["enrich"] = enrich_documents()
    except Exception as exc:  # noqa: BLE001
        _log(f"enrich_failed {exc}")
        report["phases"]["enrich"] = {"error": str(exc)}
    # Re-associate after enrich so new docs/materials wire into Samples
    report["phases"]["associate_pass2"] = associate()
    report["phases"]["validate"] = validate()
    out = ROOT / "artifacts/overnight_tao_finalize.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _log(f"wrote {out}")
    return 0 if report["phases"]["validate"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
