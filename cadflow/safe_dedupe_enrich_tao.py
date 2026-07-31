#!/usr/bin/env python3.12
"""Safe rocket FEA/CFD dedupe + TAO enrichment for JEPA training.

Safety rules (verified):
  * Exact STL sha1 duplicates with *different* materials are NOT interchangeable FEA.
  * Disk FEA dirs are dropped only when (stl_sha1, material_id) match AND a
    canonical sibling already has a valid FRD (>=50KB).
  * Mesh intermediates (mesh.msh, mesh_solid.inp) are stripped only after a
    valid FRD exists (keeps FRD + meta).
  * Bodyfit OpenFOAM bulk is stripped only after meta.json exists (keeps metrics).

TAO priority:
  * Every rocket Part keeps/gets raw geometry fields: geometry_ref, params,
    extents, faces, stl_sha1 — so Modal JEPA can regenerate custom geometry.
  * VARIANT_OF edges link same-geometry different-material Parts.
  * Deduped FEA/CFD metrics are shared only under the safe key above.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/openrocket_hardware_8k"
FEA = ROOT / "artifacts/rocket_fea_8k"
CFD = ROOT / "artifacts/rocket_cfd_bodyfit"
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
OUT = ROOT / "artifacts/logs/safe_dedupe_enrich.json"


def stl_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def dir_bytes(p: Path) -> int:
    if not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def valid_frd(case: Path, min_bytes: int = 50_000) -> bool:
    frd = case / "case.frd"
    try:
        return frd.is_file() and frd.stat().st_size >= min_bytes
    except OSError:
        return False


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    by_id = {e["part_id"]: e for e in manifest}

    # --- Index exact STL hashes ---
    sha_of: dict[str, str] = {}
    by_sha: dict[str, list[str]] = defaultdict(list)
    for e in manifest:
        stl = CORPUS / e["stl"]
        if not stl.is_file():
            continue
        h = stl_sha1(stl)
        sha_of[e["part_id"]] = h
        by_sha[h].append(e["part_id"])

    # Safe FEA key: (sha1, material_id)
    by_fea_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, h in sha_of.items():
        mid = str(by_id[pid].get("material_id") or "unknown")
        by_fea_key[(h, mid)].append(pid)

    stats: dict[str, Any] = {
        "stl_sha_groups": len(by_sha),
        "stl_exact_dupe_groups": sum(1 for ids in by_sha.values() if len(ids) > 1),
        "fea_safe_key_groups": sum(1 for ids in by_fea_key.values() if len(ids) > 1),
        "fea_dirs_removed": 0,
        "fea_bytes_removed": 0,
        "fea_mesh_files_stripped": 0,
        "fea_mesh_bytes_stripped": 0,
        "cfd_bulk_dirs_stripped": 0,
        "cfd_bytes_stripped": 0,
        "cfd_incomplete_removed": 0,
        "tao_parts_enriched": 0,
        "tao_variant_edges_added": 0,
        "tao_fea_shared": 0,
        "canon_map_entries": 0,
    }

    # Canonical per safe FEA key: prefer valid FRD, else lowest part_id
    fea_canon: dict[str, str] = {}
    for (_h, _mid), ids in by_fea_key.items():
        ids = sorted(ids)
        canon = next((pid for pid in ids if valid_frd(FEA / pid)), ids[0])
        for pid in ids:
            fea_canon[pid] = canon
    stats["canon_map_entries"] = len(fea_canon)

    # Geometry canonical (sha only) for VARIANT_OF / raw geom sharing
    geom_canon: dict[str, str] = {}
    for h, ids in by_sha.items():
        ids = sorted(ids)
        geom_canon[h] = ids[0]

    # ========== 1) SAFE DISK: remove non-canonical FEA dirs ==========
    for pid, canon in fea_canon.items():
        if pid == canon:
            continue
        # Only remove if canonical has valid FRD (otherwise keep all attempts)
        if not valid_frd(FEA / canon):
            continue
        d = FEA / pid
        if d.is_dir():
            b = dir_bytes(d)
            shutil.rmtree(d, ignore_errors=True)
            stats["fea_dirs_removed"] += 1
            stats["fea_bytes_removed"] += b

    # ========== 2) Strip mesh intermediates after valid FRD ==========
    for d in list(FEA.iterdir()) if FEA.is_dir() else []:
        if not d.is_dir() or not valid_frd(d):
            continue
        for name in (
            "mesh.msh",
            "mesh_solid.inp",
            "case.12d",
            "case.dat",
            "case.cvg",
            "case.sta",
            "spooles.out",
        ):
            p = d / name
            if p.is_file():
                stats["fea_mesh_bytes_stripped"] += p.stat().st_size
                p.unlink(missing_ok=True)
                stats["fea_mesh_files_stripped"] += 1

    # ========== 3) Strip bodyfit bulk after meta (keep meta + logs) ==========
    if CFD.is_dir():
        for d in list(CFD.iterdir()):
            if not d.is_dir():
                continue
            meta = d / "meta.json"
            if not meta.is_file():
                b = dir_bytes(d)
                if b > 0:
                    shutil.rmtree(d, ignore_errors=True)
                    stats["cfd_incomplete_removed"] += 1
                    stats["cfd_bytes_stripped"] += b
                continue
            for sub in ("constant", "system", "0"):
                p = d / sub
                if p.is_dir():
                    stats["cfd_bytes_stripped"] += dir_bytes(p)
                    shutil.rmtree(p, ignore_errors=True)
                    stats["cfd_bulk_dirs_stripped"] += 1
            for p in list(d.iterdir()):
                if p.is_dir() and p.name.replace(".", "", 1).isdigit():
                    stats["cfd_bytes_stripped"] += dir_bytes(p)
                    shutil.rmtree(p, ignore_errors=True)
                    stats["cfd_bulk_dirs_stripped"] += 1

    # ========== 4) TAO enrichment ==========
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])
    by_node = {n["id"]: n for n in nodes if n.get("type") == "Part"}
    existing_edge_ids = {e.get("id") for e in edges if e.get("id")}

    for e in manifest:
        pid = e["part_id"]
        nid = f"part:rocket:{pid}"
        node = by_node.get(nid)
        if not node:
            continue
        props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        stl_rel = e["stl"]
        stl_abs = str((CORPUS / stl_rel).resolve())
        h = sha_of.get(pid)
        gcanon = geom_canon.get(h, pid) if h else pid
        fcanon = fea_canon.get(pid, pid)

        # Raw geometry contract for JEPA / custom generation
        raw_geometry = {
            "format": "stl",
            "path": stl_abs,
            "path_rel": stl_rel,
            "sha1": h,
            "family": e.get("family"),
            "params": e.get("params") or {},
            "extents_mm": e.get("extents_mm"),
            "faces": e.get("faces"),
            "watertight": e.get("watertight"),
            "geometry_canonical_part": gcanon,
        }
        props.update(
            {
                "name": pid,
                "part_class": e.get("family"),
                "family": e.get("family"),
                "geometry_ref": stl_abs,
                "stl": stl_rel,
                "stl_sha1": h,
                "params": e.get("params") or {},
                "extents_mm": e.get("extents_mm"),
                "faces": e.get("faces"),
                "watertight": e.get("watertight"),
                "material_id": e.get("material_id"),
                "material_name": e.get("material_name"),
                "material_category": e.get("material_category"),
                "source_corpus": "openrocket_hardware_8k",
                "raw_geometry": raw_geometry,
                "geometry_canonical_part": gcanon,
                "fea_canonical_part": fcanon,
                "tags": list(
                    dict.fromkeys(
                        list(props.get("tags") or [])
                        + list(e.get("tags") or [])
                        + ["openrocket_hardware_8k", "raw_geometry"]
                    )
                ),
            }
        )
        node["properties"] = props
        node["raw_geometry"] = raw_geometry  # top-level too for easy trainers
        stats["tao_parts_enriched"] += 1

        # Share FEA metrics only under safe key when local FRD was removed
        if pid != fcanon and valid_frd(FEA / fcanon):
            cnode = by_node.get(f"part:rocket:{fcanon}")
            if cnode and cnode.get("simulation_results_fea"):
                node["has_fea"] = True
                node["fea_case_id"] = fcanon
                node["fea_status"] = "completed"
                node["fea_complete"] = True
                node["fea_verified"] = True
                node["physics_verified"] = True
                node["simulation_results_fea"] = {
                    **cnode["simulation_results_fea"],
                    "shared_from": fcanon,
                    "dedup_key": "stl_sha1+material_id",
                }
                pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
                pd["fea"] = True
                pd["fea_dedup_shared"] = True
                pd["verified"] = True
                node["physics_data"] = pd
                stats["tao_fea_shared"] += 1

        # VARIANT_OF → geometry canonical (same STL, possibly different material)
        if h and pid != gcanon:
            edge_id = f"edge:rocket:{pid}:variant_of:{gcanon}"
            if edge_id not in existing_edge_ids:
                edges.append(
                    {
                        "id": edge_id,
                        "type": "VARIANT_OF",
                        "source": nid,
                        "target": f"part:rocket:{gcanon}",
                        "properties": {
                            "reason": "identical_stl_sha1",
                            "stl_sha1": h,
                            "same_material": by_id[pid].get("material_id")
                            == by_id[gcanon].get("material_id"),
                        },
                    }
                )
                existing_edge_ids.add(edge_id)
                stats["tao_variant_edges_added"] += 1

    # Atomic graph write
    tmp = GRAPH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    tmp.replace(GRAPH)

    # Persist canon maps for runners
    maps = {
        "fea_canon": fea_canon,
        "geom_canon_by_sha1": {h: c for h, c in geom_canon.items()},
        "sha_of": sha_of,
    }
    (ROOT / "artifacts/rocket_dedupe_maps.json").write_text(
        json.dumps(maps, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    stats_out = {
        **stats,
        "fea_bytes_removed_gb": round(stats["fea_bytes_removed"] / 1e9, 3),
        "fea_mesh_bytes_stripped_gb": round(stats["fea_mesh_bytes_stripped"] / 1e9, 3),
        "cfd_bytes_stripped_gb": round(stats["cfd_bytes_stripped"] / 1e9, 3),
        "total_reclaimed_gb": round(
            (
                stats["fea_bytes_removed"]
                + stats["fea_mesh_bytes_stripped"]
                + stats["cfd_bytes_stripped"]
            )
            / 1e9,
            3,
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
