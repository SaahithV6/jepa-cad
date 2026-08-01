"""Associate Part params/labels/physics with JEPA training file nodes.

Wires the TAO graph so GraphBackedCADDataset can condition on:
  - generation ``params`` (via Part properties + Dimension nodes)
  - family / part_class labels
  - Material catalog props (MADE_OF)
  - PhysicsTarget windows + measured FEA/CFD overlays
  - GeometricMetric from extents/faces

Creates Sample nodes that point at resolvable Part geometry (STL) with
``REPRESENTS`` edges when missing, so rocket Parts enter the train set.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cadflow.part_family import classify_part
from cadflow.physics_targets import physics_targets_for, resolve_family
from cadflow.space_materials import MATERIALS_BY_ID

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"

_GEOM_EDGE = "HAS_GEOMETRIC_METRIC"
_PHYS_EDGE = "HAS_PHYSICS_TARGET"
_DIM_EDGE = "HAS_DIMENSION"
_REP_EDGE = "REPRESENTS"
_MADE_EDGE = "MADE_OF"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_geometry_path(part: dict[str, Any]) -> Path | None:
    props = part.get("properties") or {}
    candidates: list[Path] = []
    for key in ("geometry_ref", "stl", "cad_ref", "source_path", "path", "file_path", "local_path"):
        raw = props.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(ROOT / p)
            candidates.append(ROOT / "data/openrocket_hardware_8k" / p)
            candidates.append(ROOT / "artifacts/jepa-train-bundle" / p)
            candidates.append(ROOT / "artifacts/jepa-train-bundle/files" / Path(str(raw)).name)
            if str(raw).startswith("files/"):
                candidates.append(ROOT / "artifacts/jepa-train-bundle" / raw)
    raw_geom = props.get("raw_geometry")
    if isinstance(raw_geom, dict):
        for key in ("path", "path_rel"):
            raw = raw_geom.get(key)
            if not raw:
                continue
            p = Path(str(raw))
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(ROOT / p)
                candidates.append(ROOT / "data/openrocket_hardware_8k" / p)
                candidates.append(ROOT / "artifacts/jepa-train-bundle" / p)
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand.resolve()
    return None


def _fea_overlay(part: dict[str, Any]) -> dict[str, float]:
    """Pull measured solver scalars into conditioning target keys."""
    out: dict[str, float] = {}
    fea = part.get("simulation_results_fea")
    if isinstance(fea, dict):
        for src, dst in (
            ("max_stress_mpa", "max_stress_mpa"),
            ("mean_stress_mpa", "mean_stress_mpa"),
            ("max_displacement_mm", "max_displacement_mm"),
            ("safety_factor", "safety_factor"),
        ):
            val = _as_float(fea.get(src))
            if val is not None:
                out[dst] = val
        parse = fea.get("parse")
        if isinstance(parse, dict):
            for src, dst in (("max_stress_mpa", "max_stress_mpa"), ("safety_factor", "safety_factor")):
                if dst in out:
                    continue
                val = _as_float(parse.get(src))
                if val is not None:
                    out[dst] = val
    cfd = part.get("simulation_results_cfd")
    if isinstance(cfd, dict):
        for src, dst in (("Cd", "Cd"), ("cd", "Cd"), ("CL_alpha_per_rad", "CL_alpha_per_rad")):
            val = _as_float(cfd.get(src))
            if val is not None and dst not in out:
                out[dst] = val
    return out


def _geometry_metric_props(part: dict[str, Any], family: str) -> dict[str, Any]:
    props = part.get("properties") or {}
    extents = props.get("extents_mm")
    bbox = [0.0, 0.0, 0.0]
    if isinstance(extents, (list, tuple)) and len(extents) >= 3:
        bbox = [float(extents[0]), float(extents[1]), float(extents[2])]
    vol = max(bbox[0] * bbox[1] * bbox[2], 1e-9)
    face_count = int(props.get("faces") or 0)
    aspect_xy = bbox[0] / max(bbox[1], 1e-9)
    aspect_xz = bbox[0] / max(bbox[2], 1e-9)
    return {
        "family": family,
        "volume": vol,
        "log_volume": math.log10(vol),
        "aspect_ratio_xy": aspect_xy,
        "aspect_ratio_xz": aspect_xz,
        "compactness": min(1.0, face_count / 5000.0) if face_count else 0.5,
        "bbox_x": bbox[0] / 1000.0,
        "bbox_y": bbox[1] / 1000.0,
        "bbox_z": bbox[2] / 1000.0,
        "face_count": face_count,
        "source": "associate_training_data",
    }


def _fingerprint(part: dict[str, Any]) -> str:
    props = part.get("properties") or {}
    return str(
        props.get("manifest_fingerprint")
        or props.get("stl_sha1")
        or part.get("id")
        or part.get("label")
        or "part"
    )


def _ensure_node(by_id: dict[str, dict[str, Any]], node: dict[str, Any]) -> bool:
    nid = str(node["id"])
    if nid in by_id:
        existing = by_id[nid]
        # Merge properties; prefer richer new values.
        props = dict(existing.get("properties") or {})
        props.update(node.get("properties") or {})
        existing["properties"] = props
        for key, value in node.items():
            if key in {"id", "type", "properties"}:
                continue
            if key not in existing or existing.get(key) in (None, "", {}, []):
                existing[key] = value
        return False
    by_id[nid] = node
    return True


def _ensure_edge(
    edge_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    edge_type: str,
    source: str,
    target: str,
    properties: dict[str, Any] | None = None,
) -> bool:
    key = (edge_type, source, target)
    if key in edge_index:
        return False
    edge = {
        "id": f"edge:{edge_type.lower()}:{source}:{target}",
        "type": edge_type,
        "source": source,
        "target": target,
        "properties": properties or {"source": "associate_training_data"},
    }
    edge_index[key] = edge
    return True


def _enrich_material_node(by_id: dict[str, dict[str, Any]], material_id: str) -> str | None:
    mid = str(material_id).strip()
    if not mid:
        return None
    nid = mid if mid.startswith("material:") else f"material:{mid}"
    catalog_key = mid.removeprefix("material:")
    catalog = MATERIALS_BY_ID.get(catalog_key)
    props: dict[str, Any] = {"material_id": catalog_key, "source": "associate_training_data"}
    if catalog is not None:
        props.update(catalog.to_dict())
    _ensure_node(
        by_id,
        {
            "id": nid,
            "type": "Material",
            "label": props.get("name") or catalog_key,
            "properties": props,
        },
    )
    return nid


def associate_parts(
    graph: dict[str, Any],
    *,
    limit: int | None = None,
    only_missing: bool = True,
) -> dict[str, int]:
    """Mutate graph in-place; return stats."""
    nodes: list[dict[str, Any]] = list(graph.get("nodes") or [])
    edges: list[dict[str, Any]] = list(graph.get("edges") or [])
    by_id = {str(n.get("id", "")): n for n in nodes if n.get("id")}
    edge_index = {
        (str(e.get("type")), str(e.get("source")), str(e.get("target"))): e
        for e in edges
        if e.get("type") and e.get("source") and e.get("target")
    }

    parts = [n for n in nodes if n.get("type") in {"Part", "RealPart"}]
    if limit is not None:
        parts = parts[: max(0, limit)]

    stats = {
        "parts_seen": 0,
        "parts_with_geometry": 0,
        "samples_created": 0,
        "represents_edges": 0,
        "physics_targets_upserted": 0,
        "physics_edges": 0,
        "geometry_metrics_upserted": 0,
        "geometry_edges": 0,
        "dimensions_upserted": 0,
        "dimension_edges": 0,
        "materials_linked": 0,
        "families_set": 0,
        "fea_overlays": 0,
        "skipped_no_geometry": 0,
    }

    for part in parts:
        stats["parts_seen"] += 1
        part_id = str(part.get("id") or "")
        if not part_id:
            continue
        props = dict(part.get("properties") or {})
        raw_family = str(props.get("family") or props.get("part_class") or "").lower().strip()
        if not raw_family:
            raw_family = classify_part(part)
            props["family"] = raw_family
            part["properties"] = props
            stats["families_set"] += 1
        family = raw_family  # keep generation label for one-hot / params
        physics_family = resolve_family(family)

        geom_path = _resolve_geometry_path(part)
        if geom_path is None:
            stats["skipped_no_geometry"] += 1
            # Still attach physics/material labels when Part already has Samples.
        else:
            stats["parts_with_geometry"] += 1
            sample_id = f"sample:assoc:{part_id}"
            rel_path = str(geom_path)
            try:
                rel_path = str(geom_path.relative_to(ROOT))
            except ValueError:
                pass
            created = _ensure_node(
                by_id,
                {
                    "id": sample_id,
                    "type": "Sample",
                    "label": props.get("name") or part.get("label") or part_id,
                    "properties": {
                        "path": rel_path,
                        "source_path": rel_path,
                        "geometry_ref": str(geom_path),
                        "family": family,
                        "part_id": part_id,
                        "material_id": props.get("material_id"),
                        "params": props.get("params") if isinstance(props.get("params"), dict) else {},
                        "index": abs(hash(part_id)) % 4096,
                        "summary": {"kind": "associated_part_geometry", "family": family},
                        "parametric_summary": props.get("params") if isinstance(props.get("params"), dict) else {},
                        "physical_summary": {
                            "extents_mm": props.get("extents_mm"),
                            "faces": props.get("faces"),
                            "watertight": props.get("watertight"),
                        },
                        "source": "associate_training_data",
                    },
                },
            )
            if created:
                stats["samples_created"] += 1
            if _ensure_edge(edge_index, edge_type=_REP_EDGE, source=sample_id, target=part_id):
                stats["represents_edges"] += 1

        # Physics targets: design window + measured FEA/CFD overlay.
        fp = _fingerprint(part)
        payload = physics_targets_for(fp, physics_family)
        overlay = _fea_overlay(part)
        if overlay:
            stats["fea_overlays"] += 1
            targets = dict(payload.get("targets") or {})
            targets.update(overlay)
            payload["targets"] = targets
            payload["measured_overlay"] = sorted(overlay.keys())
            payload["physics_verified"] = bool(part.get("physics_verified"))
        phys_id = f"physics_target:assoc:{part_id}"
        if only_missing and phys_id in by_id and not overlay:
            pass
        else:
            _ensure_node(
                by_id,
                {
                    "id": phys_id,
                    "type": "PhysicsTarget",
                    "label": f"{family} targets",
                    "properties": {
                        **payload,
                        "fingerprint": fp,
                        "part_id": part_id,
                        "source": "associate_training_data",
                    },
                },
            )
            stats["physics_targets_upserted"] += 1
        if _ensure_edge(edge_index, edge_type=_PHYS_EDGE, source=part_id, target=phys_id):
            stats["physics_edges"] += 1

        # Geometric metrics from extents / faces.
        geom_id = f"geometric_metric:assoc:{part_id}"
        _ensure_node(
            by_id,
            {
                "id": geom_id,
                "type": "GeometricMetric",
                "label": f"{family} geometry",
                "properties": _geometry_metric_props(part, family),
            },
        )
        stats["geometry_metrics_upserted"] += 1
        if _ensure_edge(edge_index, edge_type=_GEOM_EDGE, source=part_id, target=geom_id):
            stats["geometry_edges"] += 1

        # Dimension nodes from generation params (labels for text/param conditioning).
        params = props.get("params")
        if isinstance(params, dict):
            for name, value in params.items():
                if isinstance(value, bool):
                    continue
                if not isinstance(value, (int, float, str)):
                    continue
                dim_id = f"dimension:assoc:{part_id}:{name}"
                unit = "mm" if str(name).endswith("_mm") else ("-" if isinstance(value, str) else "1")
                created = _ensure_node(
                    by_id,
                    {
                        "id": dim_id,
                        "type": "Dimension",
                        "label": str(name),
                        "properties": {
                            "name": str(name),
                            "value": value,
                            "unit": unit,
                            "family": family,
                            "part_id": part_id,
                            "source": "associate_training_data",
                        },
                    },
                )
                if created:
                    stats["dimensions_upserted"] += 1
                if _ensure_edge(edge_index, edge_type=_DIM_EDGE, source=part_id, target=dim_id):
                    stats["dimension_edges"] += 1

        # Material link + catalog enrichment.
        material_id = props.get("material_id")
        if material_id:
            mat_nid = _enrich_material_node(by_id, str(material_id))
            if mat_nid and _ensure_edge(edge_index, edge_type=_MADE_EDGE, source=part_id, target=mat_nid):
                stats["materials_linked"] += 1

    graph["nodes"] = list(by_id.values())
    graph["edges"] = list(edge_index.values())
    meta = dict(graph.get("metadata") or {})
    meta["associate_training_data"] = {"updated_at": _utc_now(), "stats": stats}
    graph["metadata"] = meta
    return stats


def associate_graph_file(
    graph_path: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic

    if dry_run:
        graph = read_graph(graph_path)
        return associate_parts(graph, limit=limit)
    with graph_lock(graph_path):
        graph = read_graph(graph_path)
        stats = associate_parts(graph, limit=limit)
        write_graph_atomic(graph_path, graph)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    stats = associate_graph_file(args.graph, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
