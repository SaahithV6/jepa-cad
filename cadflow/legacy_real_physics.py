"""Promote class-conditioned FEA and demote empty-channel CFD on legacy Parts.

Does not touch rocket 8k artifact roots. Graph writes are atomic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "artifacts/jepa-train-bundle/graph.json"
FEA_ALT = ROOT / "artifacts/fea_alt"


def _atomic_write_graph(graph_path: Path, graph: dict[str, Any]) -> None:
    tmp = graph_path.with_suffix(graph_path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph), encoding="utf-8")
    tmp.replace(graph_path)


def _is_rocket_part(node: dict[str, Any]) -> bool:
    return str(node.get("id") or "").startswith("part:rocket:")


def _is_channel_proxy(cfd: Any) -> bool:
    if not isinstance(cfd, dict):
        return False
    mesh = str(cfd.get("mesh") or "")
    if mesh in {"blockMesh_channel", "channel_proxy"}:
        return True
    # Legacy ingest often omitted mesh but used empty-channel pipeline
    if cfd.get("replaced_channel_proxy"):
        return False
    if str(cfd.get("solver") or "") == "simpleFoam" and not cfd.get("geometry"):
        # bodyfit always sets geometry=stl_body_wall or mesh snappy
        if "snappy" in mesh or cfd.get("geometry") == "stl_body_wall":
            return False
        if mesh == "snappyHexMesh_external":
            return False
        # treat unmarked simpleFoam without bodyfit flag as proxy
        return not bool(cfd.get("cfd_bodyfit"))
    return False


def demote_channel_cfd(graph_path: Path = GRAPH_PATH) -> dict[str, int]:
    """Move empty-channel CFD off primary fields for non-rocket Parts."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    demoted = 0
    kept_bodyfit = 0
    skipped_rocket = 0
    for node in graph["nodes"]:
        if node.get("type") != "Part":
            continue
        if _is_rocket_part(node):
            skipped_rocket += 1
            continue
        cfd = node.get("simulation_results_cfd")
        if isinstance(cfd, dict) and (
            str(cfd.get("mesh") or "") == "snappyHexMesh_external"
            or cfd.get("geometry") == "stl_body_wall"
            or (node.get("physics_data") or {}).get("cfd_bodyfit")
        ):
            kept_bodyfit += 1
            continue
        # Also demote cfd_alt channel proxies
        for key in ("simulation_results_cfd", "simulation_results_cfd_alt"):
            blob = node.get(key)
            if not isinstance(blob, dict):
                continue
            if key == "simulation_results_cfd" and not _is_channel_proxy(blob):
                # if mesh missing but has_cfd from old pipeline — still demote legacy
                if blob.get("mesh") == "snappyHexMesh_external":
                    continue
            if key == "simulation_results_cfd_alt":
                # alts were always channel proxies
                node["simulation_results_cfd_alt_channel_proxy"] = blob
                node.pop(key, None)
                node.pop("has_cfd_alt", None)
                demoted += 1
                continue
            if _is_channel_proxy(blob) or blob.get("mesh") in ("", None) or "channel" in str(blob.get("mesh") or ""):
                node["simulation_results_cfd_channel_proxy"] = blob
                node.pop("simulation_results_cfd", None)
                node["has_cfd"] = False
                pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
                pd["cfd"] = False
                pd["cfd_channel_proxy_demoted"] = True
                pd.pop("cfd_bodyfit", None)
                node["physics_data"] = pd
                demoted += 1
            elif not blob.get("mesh") and blob.get("solver") == "simpleFoam":
                # unmarked simpleFoam on legacy → demote
                node["simulation_results_cfd_channel_proxy"] = blob
                node.pop("simulation_results_cfd", None)
                node["has_cfd"] = False
                pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
                pd["cfd"] = False
                pd["cfd_channel_proxy_demoted"] = True
                node["physics_data"] = pd
                demoted += 1

        # Force-clear has_cfd if primary gone
        if not node.get("simulation_results_cfd"):
            node["has_cfd"] = False

    _atomic_write_graph(graph_path, graph)
    return {
        "demoted": demoted,
        "kept_bodyfit": kept_bodyfit,
        "skipped_rocket_parts_seen": skipped_rocket,
    }


def _fea_alt_quality_ok(alt: dict[str, Any]) -> bool:
    stress = alt.get("max_stress_mpa")
    if stress is None:
        return False
    try:
        s = float(stress)
    except (TypeError, ValueError):
        return False
    if s <= 0.0:
        return False
    if not alt.get("family"):
        return False
    # Prefer explicit load metadata, but accept FRD-backed family solves.
    if any(alt.get(k) is not None for k in ("pressure_bar", "g_load", "aero_force_n", "total_load_n", "load_case")):
        return True
    return str(alt.get("source") or "") in {"case_alt.frd", "case.frd"} or bool(alt.get("sim_kind"))


def promote_fea_alts(graph_path: Path = GRAPH_PATH) -> dict[str, int]:
    """Promote quality-gated fea_alt → canonical simulation_results_fea."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    promoted = 0
    rejected = 0
    skipped_rocket = 0
    for node in graph["nodes"]:
        if node.get("type") != "Part":
            continue
        if _is_rocket_part(node):
            skipped_rocket += 1
            continue
        alt = node.get("simulation_results_fea_alt")
        if not isinstance(alt, dict):
            continue
        if not _fea_alt_quality_ok(alt):
            rejected += 1
            continue
        prev = node.get("simulation_results_fea")
        if isinstance(prev, dict) and not prev.get("family"):
            # axial baseline without family metadata
            node["simulation_results_fea_axial_baseline"] = prev
        node["simulation_results_fea"] = {
            **alt,
            "promoted_from": "fea_alt",
            "status": "completed",
            "solver": alt.get("solver") or "CalculiX",
        }
        node["has_fea"] = True
        node["physics_verified"] = True
        node["physics_family"] = alt.get("family") or node.get("physics_family")
        node["physics_sim_kind"] = alt.get("sim_kind") or node.get("physics_sim_kind")
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["fea"] = True
        pd["fea_family_load"] = True
        pd["verified"] = True
        node["physics_data"] = pd
        promoted += 1
    _atomic_write_graph(graph_path, graph)
    return {"promoted": promoted, "rejected_quality": rejected, "skipped_rocket": skipped_rocket}


def demote_synthetic_fea(graph_path: Path = GRAPH_PATH) -> dict[str, int]:
    """Archive fake FEA placeholders (no family load, template-style fields)."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    demoted = 0
    for node in graph["nodes"]:
        if node.get("type") != "Part" or _is_rocket_part(node):
            continue
        fea = node.get("simulation_results_fea")
        if not isinstance(fea, dict):
            continue
        # Real family loads always have family + promoted_from or pressure/g/aero meta
        if fea.get("family") and (
            fea.get("promoted_from") == "fea_alt"
            or any(fea.get(k) is not None for k in ("pressure_bar", "g_load", "aero_force_n", "sim_kind"))
        ):
            continue
        # Synthetic scripts used calculix lowercase + total_strain_energy_j without FRD source
        looks_fake = (
            str(fea.get("solver") or "").lower() == "calculix"
            and "total_strain_energy_j" in fea
            and not fea.get("source")
            and not fea.get("family")
        ) or (not fea.get("family") and not fea.get("source") and fea.get("status") == "completed")
        if not looks_fake:
            continue
        node["simulation_results_fea_synthetic_placeholder"] = fea
        node.pop("simulation_results_fea", None)
        node["has_fea"] = False
        node["physics_verified"] = False
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["fea"] = False
        pd["fea_synthetic_demoted"] = True
        pd["verified"] = bool(pd.get("cfd_bodyfit"))
        node["physics_data"] = pd
        demoted += 1
    _atomic_write_graph(graph_path, graph)
    return {"demoted_synthetic_fea": demoted}


def demote_and_promote(graph_path: Path = GRAPH_PATH) -> dict[str, Any]:
    d = demote_channel_cfd(graph_path)
    s = demote_synthetic_fea(graph_path)
    p = promote_fea_alts(graph_path)
    return {"demote_cfd": d, "demote_synthetic_fea": s, "promote": p}
