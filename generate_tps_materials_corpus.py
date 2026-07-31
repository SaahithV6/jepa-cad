#!/usr/bin/env python3.12
"""Add TPS tiles / spacecraft parts + space materials for LatticeZero.

Extends data/openrocket_hardware_8k with tiles/blankets/panels/antennas/frames,
writes a materials catalog, annotates every part with a space material, and
injects Material + RealPart nodes into the TAO training graph.

Does not run FEA or Modal training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cadflow.rocket_hardware_generator import (  # noqa: E402
    build_mesh,
    iter_tps_spacecraft_specs,
    write_stl,
)
from cadflow.space_materials import (  # noqa: E402
    FAMILY_MATERIAL_PRESETS,
    MATERIALS_BY_ID,
    assign_material_for_family,
    catalog_as_dicts,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_tps_extension(out_dir: Path, target: int) -> list[dict]:
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    specs = iter_tps_spacecraft_specs(target=target)
    records: list[dict] = []
    ok = failed = 0
    for i, spec in enumerate(specs):
        try:
            mesh = build_mesh(spec)
            if mesh.is_empty or len(mesh.faces) == 0:
                raise ValueError("empty mesh")
            # prefix to avoid colliding with existing part_id filenames from rocket set
            part_id = f"sc_{spec.part_id}"
            stl_path = parts_dir / f"{part_id}.stl"
            write_stl(mesh, stl_path)
            rec = {
                "part_id": part_id,
                "family": spec.family,
                "params": spec.params,
                "tags": list(spec.tags),
                "stl": str(stl_path.relative_to(out_dir)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "extents_mm": mesh.extents.tolist(),
            }
            records.append(rec)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            records.append({"part_id": f"sc_{spec.part_id}", "family": spec.family, "error": str(exc)})
        if (i + 1) % 400 == 0:
            print(f"  tps/sc [{i+1}/{len(specs)}] ok={ok} failed={failed}", flush=True)
    print(f"TPS/spacecraft extension: ok={ok} failed={failed}", flush=True)
    return records


def annotate_materials(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, rec in enumerate(records):
        if "error" in rec or "family" not in rec:
            out.append(rec)
            continue
        mat = assign_material_for_family(rec["family"], index=i)
        enriched = dict(rec)
        enriched["material_id"] = mat.material_id
        enriched["material_name"] = mat.name
        enriched["material_category"] = mat.category
        enriched["material"] = {
            "id": mat.material_id,
            "name": mat.name,
            "category": mat.category,
            "density_kg_m3": mat.density_kg_m3,
            "youngs_modulus_gpa": mat.youngs_modulus_gpa,
            "yield_mpa": mat.yield_mpa,
            "ultimate_mpa": mat.ultimate_mpa,
            "max_service_temp_k": mat.max_service_temp_k,
            "cte_1e6_k": mat.cte_1e6_k,
            "thermal_conductivity_w_mk": mat.thermal_conductivity_w_mk,
        }
        out.append(enriched)
    return out


def enrich_tao_graph(
    graph_path: Path,
    *,
    corpus_dir: Path,
    annotated: list[dict],
) -> dict:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])

    existing_ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    mat_added = part_added = edge_added = 0

    # Material catalog nodes
    for mat in catalog_as_dicts():
        nid = f"material:{mat['material_id']}"
        if nid in existing_ids:
            # refresh properties
            for n in nodes:
                if n.get("id") == nid:
                    props = dict(n.get("properties") or {})
                    props.update(mat)
                    props["source"] = "space_materials_catalog"
                    n["properties"] = props
                    break
            continue
        nodes.append(
            {
                "id": nid,
                "type": "Material",
                "label": mat["name"],
                "properties": {**mat, "source": "space_materials_catalog"},
            }
        )
        existing_ids.add(nid)
        mat_added += 1

    # RealPart nodes for annotated corpus geometries
    for rec in annotated:
        if "error" in rec or not rec.get("stl"):
            continue
        pid = f"realpart:or8k:{rec['part_id']}"
        mat_id = rec.get("material_id")
        props = {
            "name": rec["part_id"],
            "family": rec["family"],
            "source_path": str((corpus_dir / rec["stl"]).resolve()),
            "relative_path": rec["stl"],
            "corpus": "openrocket_hardware_8k",
            "params": rec.get("params"),
            "tags": rec.get("tags"),
            "material_id": mat_id,
            "material_name": rec.get("material_name"),
            "material_category": rec.get("material_category"),
            "extents_mm": rec.get("extents_mm"),
            "faces": rec.get("faces"),
        }
        if pid not in existing_ids:
            nodes.append({"id": pid, "type": "RealPart", "label": rec["part_id"], "properties": props})
            existing_ids.add(pid)
            part_added += 1
        else:
            for n in nodes:
                if n.get("id") == pid:
                    n["properties"] = {**(n.get("properties") or {}), **props}
                    break

        if mat_id:
            eid = f"edge:made_of:{rec['part_id']}:{mat_id}"
            if eid not in existing_ids:
                edges.append(
                    {
                        "id": eid,
                        "type": "MADE_OF",
                        "source": pid,
                        "target": f"material:{mat_id}",
                        "properties": {"corpus": "openrocket_hardware_8k"},
                    }
                )
                existing_ids.add(eid)
                edge_added += 1

    # Also attach materials to existing Part nodes by family heuristic when missing
    part_mat_linked = 0
    for n in nodes:
        if n.get("type") != "Part":
            continue
        props = n.setdefault("properties", {})
        if props.get("material_id"):
            continue
        family = props.get("family") or props.get("part_class") or "structure"
        # map generic box/sweep classes loosely
        if family in {"box", "generic", "generic-structure"}:
            family = "structure" if "structure" in FAMILY_MATERIAL_PRESETS else "body_tube"
        # structure not in presets — use body_tube metals
        if family not in FAMILY_MATERIAL_PRESETS:
            family = {
                "structure": "body_tube",
                "spacecraft_bus": "ring_frame",
                "feed_system": "tank",
                "fastener": "engine_mount",
                "injectors": "nozzle",
                "injector": "nozzle",
                "valve": "tank",
                "deployable": "solar_panel",
                "cubesat": "ring_frame",
                "combustion_chamber": "nozzle",
                "turbopump": "nozzle",
                "sensor": "bulkhead",
            }.get(str(family), "body_tube")
        mat = assign_material_for_family(str(family), index=hash(n.get("id", "")) % 97)
        props["material_id"] = mat.material_id
        props["material_name"] = mat.name
        props["material_category"] = mat.category
        eid = f"edge:made_of:part:{n['id'].split(':')[-1]}:{mat.material_id}"
        if eid not in existing_ids:
            edges.append(
                {
                    "id": eid,
                    "type": "MADE_OF",
                    "source": n["id"],
                    "target": f"material:{mat.material_id}",
                    "properties": {"inferred": True},
                }
            )
            existing_ids.add(eid)
            part_mat_linked += 1

    meta = graph.setdefault("metadata", {})
    meta["space_materials_enriched_at"] = _utc()
    meta["space_materials_catalog_size"] = len(MATERIALS_BY_ID)

    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    return {
        "materials_added": mat_added,
        "realparts_added": part_added,
        "made_of_edges_added": edge_added,
        "existing_parts_materialized": part_mat_linked,
        "nodes": len(nodes),
        "edges": len(edges),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/openrocket_hardware_8k"))
    parser.add_argument("--tps-parts", type=int, default=2500)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("artifacts/jepa-train-bundle/graph.json"),
    )
    parser.add_argument("--skip-generate", action="store_true", help="Only annotate + graph enrich")
    args = parser.parse_args()

    corpus = args.corpus
    corpus.mkdir(parents=True, exist_ok=True)

    # materials catalog (always)
    materials_path = corpus / "materials_catalog.json"
    materials_path.write_text(json.dumps(catalog_as_dicts(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {materials_path} ({len(MATERIALS_BY_ID)} materials)")

    existing_manifest_path = corpus / "manifest.json"
    existing: list[dict] = []
    if existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))

    new_recs: list[dict] = []
    if not args.skip_generate:
        print(f"Generating {args.tps_parts} TPS/spacecraft parts into {corpus} ...")
        new_recs = generate_tps_extension(corpus, args.tps_parts)
    else:
        # reload any sc_ entries already present
        new_recs = [r for r in existing if str(r.get("part_id", "")).startswith("sc_")]
        existing = [r for r in existing if not str(r.get("part_id", "")).startswith("sc_")]

    # merge rocket + new (drop prior sc_ if regenerating)
    base = [r for r in existing if not str(r.get("part_id", "")).startswith("sc_")]
    merged = annotate_materials(base + new_recs)

    existing_manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    fam = Counter(r.get("family") for r in merged if "error" not in r)
    mat_cat = Counter(r.get("material_category") for r in merged if r.get("material_category"))
    summary = {
        "parts_total": sum(1 for r in merged if "error" not in r),
        "family_counts": dict(fam),
        "material_category_counts": dict(mat_cat),
        "materials_catalog": len(MATERIALS_BY_ID),
        "updated_at": _utc(),
        "note": "OpenRocket + TPS/spacecraft hardware with space materials for LatticeZero",
    }
    (corpus / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("corpus summary:", summary)

    if args.graph.exists():
        print(f"Enriching TAO graph {args.graph} ...")
        stats = enrich_tao_graph(args.graph, corpus_dir=corpus, annotated=merged)
        (corpus / "graph_enrichment_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print("graph stats:", stats)
    else:
        print(f"graph missing: {args.graph}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
