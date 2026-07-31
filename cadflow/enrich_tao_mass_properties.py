#!/usr/bin/env python3.12
"""Enrich TAO Parts (rocket + legacy) with mass / COM / inertia.

Computes unique STL geometry in parallel, scales by material density, writes:
  - Part.mass_kg / mass_properties / properties.* on graph.json
  - artifacts/mass_properties_sidecar.json  (survives FEA/CFD ingest races)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic  # noqa: E402
from cadflow.mass_properties import mass_properties_from_stl  # noqa: E402
from cadflow.physics_alternates import _material_e  # noqa: E402
from cadflow.legacy_gap_fix import resolve_stl  # noqa: E402

CORPUS = ROOT / "data/openrocket_hardware_8k"
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
SIDECAR = ROOT / "artifacts/mass_properties_sidecar.json"
DEFAULT_DENSITY = 2700.0


def _unit_geom_worker(payload: dict) -> tuple[str, dict | None]:
    key = payload["key"]
    mp = mass_properties_from_stl(
        payload["stl"],
        1.0,
        extents_mm=payload.get("extents_mm"),
    )
    if mp is None:
        return key, None
    return key, {
        "volume_m3": mp.volume_m3,
        "center_of_mass_m": list(mp.center_of_mass_m),
        "center_of_mass_mm": list(mp.center_of_mass_mm),
        "inertia_per_density": list(mp.inertia_kg_m2),
        "principal_per_density": list(mp.principal_inertia_kg_m2),
        "watertight": mp.watertight,
        "method": mp.method,
    }


def _scale(unit: dict, density: float, material_id: str | None) -> dict[str, Any]:
    vol = float(unit["volume_m3"])
    mass = density * vol
    I = [density * float(x) for x in unit["inertia_per_density"]]
    Ip = [density * float(x) for x in unit["principal_per_density"]]
    return {
        "mass_kg": round(mass, 6),
        "volume_m3": vol,
        "density_kg_m3": density,
        "center_of_mass_m": unit["center_of_mass_m"],
        "center_of_mass_mm": unit["center_of_mass_mm"],
        "inertia_kg_m2": {
            "Ixx": I[0],
            "Iyy": I[1],
            "Izz": I[2],
            "Ixy": I[3],
            "Ixz": I[4],
            "Iyz": I[5],
        },
        "principal_inertia_kg_m2": Ip,
        "watertight": unit["watertight"],
        "method": unit["method"],
        "material_id": material_id,
    }


def apply_mass_dict(node: dict[str, Any], mp: dict[str, Any], extra_mat: dict[str, Any] | None = None) -> None:
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    props.update(
        {
            "density_kg_m3": mp["density_kg_m3"],
            "mass_kg": mp["mass_kg"],
            "volume_m3": mp["volume_m3"],
            "center_of_mass_m": mp["center_of_mass_m"],
            "center_of_mass_mm": mp["center_of_mass_mm"],
            "inertia_kg_m2": mp["inertia_kg_m2"],
            "principal_inertia_kg_m2": mp["principal_inertia_kg_m2"],
            "mass_properties": mp,
        }
    )
    if extra_mat:
        for k in (
            "youngs_modulus_gpa",
            "yield_mpa",
            "ultimate_mpa",
            "max_service_temp_k",
            "cte_1e6_k",
            "thermal_conductivity_w_mk",
            "material_id",
            "material_name",
            "material_category",
        ):
            if extra_mat.get(k) is not None:
                props[k] = extra_mat[k]
    node["properties"] = props
    node["mass_properties"] = mp
    node["mass_kg"] = mp["mass_kg"]


def apply_sidecar_to_graph(graph_path: Path = GRAPH, sidecar_path: Path = SIDECAR) -> int:
    """Fast re-merge after FEA/CFD ingest races (no STL recompute).

    Serialized against the other ingest pipelines (see cadflow.graph_lock).
    """
    if not sidecar_path.is_file():
        return 0
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    by_id = side.get("by_part_id") or {}
    with graph_lock(graph_path):
        graph = read_graph(graph_path)
        n = 0
        for node in graph["nodes"]:
            if node.get("type") != "Part":
                continue
            mp = by_id.get(node.get("id"))
            if not isinstance(mp, dict):
                continue
            apply_mass_dict(node, mp)
            n += 1
        write_graph_atomic(graph_path, graph)
    return n


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    by_pid = {e["part_id"]: e for e in manifest}

    print("loading graph…", flush=True)
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))

    # Build unique STL jobs for rocket + legacy
    jobs: dict[str, dict] = {}
    rocket_plan: list[tuple[dict, str, float, dict]] = []  # node, key, density, mat_extra
    legacy_plan: list[tuple[dict, str, float, dict]] = []

    for node in graph["nodes"]:
        if node.get("type") != "Part":
            continue
        nid = str(node.get("id") or "")
        if nid.startswith("part:rocket:"):
            pid = nid.split("part:rocket:", 1)[-1]
            entry = by_pid.get(pid)
            if not entry:
                continue
            mat = entry.get("material") if isinstance(entry.get("material"), dict) else {}
            density = float(mat.get("density_kg_m3") or DEFAULT_DENSITY)
            key = str(entry.get("stl_sha1") or entry["stl"])
            if key not in jobs:
                jobs[key] = {
                    "key": key,
                    "stl": str(CORPUS / entry["stl"]),
                    "extents_mm": entry.get("extents_mm"),
                }
            rocket_plan.append(
                (
                    node,
                    key,
                    density,
                    {
                        "youngs_modulus_gpa": mat.get("youngs_modulus_gpa"),
                        "yield_mpa": mat.get("yield_mpa"),
                        "ultimate_mpa": mat.get("ultimate_mpa"),
                        "max_service_temp_k": mat.get("max_service_temp_k"),
                        "cte_1e6_k": mat.get("cte_1e6_k"),
                        "thermal_conductivity_w_mk": mat.get("thermal_conductivity_w_mk"),
                        "material_id": entry.get("material_id"),
                        "material_name": entry.get("material_name"),
                        "material_category": entry.get("material_category"),
                    },
                )
            )
        else:
            stl = resolve_stl(node)
            if not stl:
                continue
            _e, _nu, density, _name = _material_e(node)
            props = node.get("properties") or {}
            key = f"legacy:{stl.resolve()}"
            if key not in jobs:
                jobs[key] = {"key": key, "stl": str(stl), "extents_mm": props.get("extents_mm")}
            legacy_plan.append(
                (
                    node,
                    key,
                    float(density),
                    {
                        "material_id": props.get("material_id"),
                        "material_name": props.get("material_name"),
                        "material_category": props.get("material_category"),
                    },
                )
            )

    print(
        f"jobs={len(jobs)} rocket_parts={len(rocket_plan)} legacy_parts={len(legacy_plan)}",
        flush=True,
    )

    geom_cache: dict[str, dict | None] = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_unit_geom_worker, j) for j in jobs.values()]
        done = 0
        for fut in as_completed(futs):
            key, payload = fut.result()
            geom_cache[key] = payload
            done += 1
            if done % 200 == 0 or done == len(jobs):
                ok = sum(1 for v in geom_cache.values() if v)
                print(f"  geom {done}/{len(jobs)} ok={ok}", flush=True)

    sidecar: dict[str, Any] = {"by_part_id": {}}
    enriched_r = enriched_l = failed = 0

    for plan, counter_name in ((rocket_plan, "r"), (legacy_plan, "l")):
        for node, key, density, extra in plan:
            unit = geom_cache.get(key)
            if not unit:
                failed += 1
                continue
            mp = _scale(unit, density, extra.get("material_id"))
            apply_mass_dict(node, mp, extra)
            sidecar["by_part_id"][node["id"]] = mp
            if counter_name == "r":
                enriched_r += 1
            else:
                enriched_l += 1

    print(
        f"writing graph+sidecar rocket={enriched_r} legacy={enriched_l} failed={failed}",
        flush=True,
    )
    SIDECAR.write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")
    tmp = GRAPH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    tmp.replace(GRAPH)

    summary = {
        "rocket_enriched": enriched_r,
        "legacy_enriched": enriched_l,
        "failed": failed,
        "unique_geom_ok": sum(1 for v in geom_cache.values() if v),
        "sidecar": str(SIDECAR),
        "graph": str(GRAPH),
    }
    (ROOT / "artifacts/mass_properties_enrichment.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0 if (enriched_r + enriched_l) else 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--apply-sidecar":
        n = apply_sidecar_to_graph()
        print(json.dumps({"reapplied": n}), flush=True)
        raise SystemExit(0 if n else 2)
    raise SystemExit(main())
