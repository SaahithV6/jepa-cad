"""Fix legacy Parts that still carry synthetic FEA and lack real geometry runs.

Resolves STL from ``properties.source_path``, volume-meshes with gmsh, runs
family-conditioned CalculiX, and bodyfit CFD for aero families.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadflow.legacy_cfd_bodyfit import (
    AERO_FAMILIES,
    CFD_ROOT,
    ingest_legacy_bodyfit,
    run_legacy_bodyfit_case,
)
from cadflow.legacy_real_physics import demote_synthetic_fea, promote_fea_alts
from cadflow.msh_to_calculix import parse_frd_summary, parse_msh2_solid, run_calculix_case
from cadflow.part_family import classify_part
from cadflow.physics_alternates import (
    FEA_ALT,
    FEA_BASE,
    GRAPH_PATH,
    _alt_frd_valid,
    _material_e,
    write_alternate_fea_deck,
)
from cadflow.physics_targets import physics_targets_for
from cadflow.rocket_physics_suite import mesh_stl_volume

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/physics_gap_fix_summary.json"


def _atomic_write_graph(graph_path: Path, graph: dict[str, Any]) -> None:
    tmp = graph_path.with_suffix(graph_path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph), encoding="utf-8")
    tmp.replace(graph_path)


def resolve_stl(part: dict[str, Any]) -> Path | None:
    props = part.get("properties") or {}
    gref = props.get("geometry_ref")
    if gref:
        stl = Path(str(gref)).with_suffix(".stl")
        if stl.exists():
            return stl
        p = Path(str(gref))
        if p.suffix.lower() == ".stl" and p.exists():
            return p
    sp = props.get("source_path")
    if not sp:
        return None
    path = Path(str(sp))
    if path.suffix.lower() == ".stl" and path.exists():
        return path
    cand = path.with_suffix(".stl")
    if cand.exists():
        return cand
    # N3 layout: STEP/... → STL/...
    if "STEP" in path.parts:
        parts = list(path.parts)
        parts[parts.index("STEP")] = "STL"
        alt = Path(*parts).with_suffix(".stl")
        if alt.exists():
            return alt
    if path.parent.exists():
        hits = list(path.parent.glob(path.stem + "*.stl"))
        if hits:
            return hits[0]
    return None


def case_id_for_gap(part: dict[str, Any]) -> str:
    props = part.get("properties") or {}
    for key in ("fea_case_id", "cfd_case_id"):
        v = part.get(key) or props.get(key)
        if v:
            return str(v)
    fp = str(props.get("manifest_fingerprint") or part["id"])
    return hashlib.sha1(fp.encode()).hexdigest()[:16]


def gap_parts(graph: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        if str(n.get("id") or "").startswith("part:rocket:"):
            continue
        fea = n.get("simulation_results_fea") or {}
        if fea.get("family") and fea.get("promoted_from") == "fea_alt":
            continue
        # Needs real family FEA
        if not fea.get("family"):
            out.append(n)
    return out


def _mesh_worker(stl_s: str, msh_s: str, q: Any) -> None:
    try:
        mr = mesh_stl_volume(Path(stl_s), Path(msh_s), cl_max_mm=6.0, cl_min_mm=1.5)
        q.put((mr.success, mr.error))
    except Exception as exc:  # noqa: BLE001
        q.put((False, f"exception:{type(exc).__name__}"))


def _mesh_with_timeout(stl: Path, msh: Path, timeout_s: int = 120) -> tuple[bool, str | None]:
    """Run gmsh volume mesh in a child process so hung meshing can be killed."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    proc = ctx.Process(target=_mesh_worker, args=(str(stl), str(msh), q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return False, "mesh_timeout"
    if q.empty():
        return False, "mesh_no_result"
    ok, err = q.get()
    return bool(ok), err


def run_gap_fea(part: dict[str, Any], *, force: bool = False, timeout: int = 300) -> dict[str, Any]:
    result: dict[str, Any] = {"part_id": part["id"], "success": False, "modality": "fea"}
    stl = resolve_stl(part)
    if not stl:
        result["error"] = "no_stl"
        return result
    # Skip huge STLs that routinely hang gmsh
    try:
        if stl.stat().st_size > 25_000_000:
            result["error"] = "stl_too_large"
            return result
    except OSError:
        result["error"] = "stl_stat"
        return result
    cid = case_id_for_gap(part)
    result["case_id"] = cid
    case_dir = FEA_BASE / cid
    alt_dir = FEA_ALT / cid
    result["stl"] = str(stl)

    if not force and _alt_frd_valid(alt_dir):
        summary = parse_frd_summary(alt_dir / "case_alt.frd", min_bytes=20_000)
        meta: dict[str, Any] = {}
        if (alt_dir / "meta.json").exists():
            meta = json.loads((alt_dir / "meta.json").read_text(encoding="utf-8"))
        if summary and summary.max_von_mises_mpa > 0:
            family = meta.get("family") or classify_part(part)
            result.update(
                {
                    "success": True,
                    "cached": True,
                    "family": family,
                    "sim_kind": meta.get("sim_kind"),
                    "meta": meta,
                    "metrics": {
                        "max_stress_mpa": summary.max_von_mises_mpa,
                        "mean_stress_mpa": summary.mean_von_mises_mpa,
                        "max_displacement_mm": summary.max_displacement_mm,
                        "frd_bytes": summary.frd_bytes,
                    },
                }
            )
            return result

    case_dir.mkdir(parents=True, exist_ok=True)
    msh = case_dir / "mesh.msh"
    need_mesh = force or not msh.exists()
    if msh.exists() and not need_mesh:
        try:
            if not parse_msh2_solid(msh).elements:
                need_mesh = True
        except Exception:  # noqa: BLE001
            need_mesh = True
    if need_mesh:
        ok, err = _mesh_with_timeout(stl, msh, timeout_s=150)
        if not ok:
            result["error"] = f"mesh:{err}"
            return result

    # Attach geometry_ref for later bodyfit selection
    props = part.setdefault("properties", {})
    props["geometry_ref"] = str(stl)
    props["fea_case_id"] = cid

    family = classify_part(part)
    fp = str(props.get("manifest_fingerprint") or part["id"])
    targets = physics_targets_for(fp, family)
    youngs, poisson, density, mat = _material_e(part)
    mesh = parse_msh2_solid(msh)
    if not mesh.elements:
        result["error"] = "no_tets"
        return result

    alt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(msh, alt_dir / "mesh.msh")
    meta = write_alternate_fea_deck(
        mesh,
        alt_dir,
        family,
        targets,
        youngs=youngs,
        poisson=poisson,
        density=density,
        material_name=mat,
    )
    meta["part_id"] = part["id"]
    meta["stl"] = str(stl)
    (alt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run = run_calculix_case(alt_dir, job_name="case_alt", timeout=timeout)
    if not run.converged or not _alt_frd_valid(alt_dir):
        result["error"] = "calculix"
        result["frd_bytes"] = run.frd_bytes
        return result
    summary = parse_frd_summary(alt_dir / "case_alt.frd", min_bytes=20_000)
    if not summary or summary.max_von_mises_mpa <= 0:
        result["error"] = "empty_stress"
        return result
    result.update(
        {
            "success": True,
            "family": family,
            "sim_kind": targets["sim_kind"],
            "meta": meta,
            "metrics": {
                "max_stress_mpa": summary.max_von_mises_mpa,
                "mean_stress_mpa": summary.mean_von_mises_mpa,
                "max_displacement_mm": summary.max_displacement_mm,
                "frd_bytes": summary.frd_bytes,
            },
            "targets": targets["targets"],
        }
    )
    return result


def ingest_gap_fea(graph_path: Path, results: list[dict[str, Any]]) -> int:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(r["part_id"])
        if not node:
            continue
        props = node.setdefault("properties", {})
        if r.get("stl"):
            props["geometry_ref"] = r["stl"]
        if r.get("case_id"):
            props["fea_case_id"] = r["case_id"]
            node["fea_case_id"] = r["case_id"]
        meta = r.get("meta") or {}
        metrics = r.get("metrics") or {}
        family = r.get("family") or classify_part(node)
        node["simulation_results_fea_alt"] = {
            "solver": "CalculiX",
            "status": "completed",
            "source": "case_alt.frd",
            "case_id": r.get("case_id"),
            "family": family,
            "sim_kind": r.get("sim_kind"),
            "load_case": meta.get("load_case"),
            "pressure_bar": meta.get("pressure_bar"),
            "g_load": meta.get("g_load"),
            "aero_force_n": meta.get("aero_force_n"),
            "total_load_n": meta.get("total_load_n"),
            **metrics,
            "targets": r.get("targets"),
        }
        node["has_fea_alt"] = True
        linked += 1
    _atomic_write_graph(graph_path, graph)
    return linked


def run_gap_fix(*, workers: int = 4, force: bool = False, fea_only: bool = False) -> dict[str, Any]:
    dem = demote_synthetic_fea(GRAPH_PATH)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    parts = gap_parts(graph)
    # Prefer parts with STL
    with_stl = [p for p in parts if resolve_stl(p)]
    print(f"gap_parts={len(parts)} with_stl={len(with_stl)} demote={dem}", flush=True)

    fea_results: list[dict[str, Any]] = []
    # gmsh cannot initialize in worker threads — run FEA serially (mesh is the bottleneck).
    for i, p in enumerate(with_stl, 1):
        r = run_gap_fea(p, force=force)
        fea_results.append(r)
        status = "OK" if r.get("success") else f"FAIL:{r.get('error')}"
        if i % 5 == 0 or not r.get("success") or i == len(with_stl):
            print(f"  [gap-fea {i}/{len(with_stl)}] {r.get('case_id')} {status}", flush=True)

    fea_ok = [r for r in fea_results if r.get("success")]
    linked_fea = ingest_gap_fea(GRAPH_PATH, fea_ok)
    promoted = promote_fea_alts(GRAPH_PATH)

    cfd_results: list[dict[str, Any]] = []
    if not fea_only:
        # Reload graph after geometry_ref updates
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
        aero = []
        for r in fea_ok:
            node = by_id.get(r["part_id"])
            if not node:
                continue
            if classify_part(node) in AERO_FAMILIES:
                aero.append(node)
        print(f"gap aero for bodyfit={len(aero)}", flush=True)
        for i, p in enumerate(aero, 1):
            r = run_legacy_bodyfit_case(p, CFD_ROOT, force=force)
            cfd_results.append(r)
            print(
                f"  [gap-cfd {i}/{len(aero)}] {r.get('case_id')} "
                f"{'OK' if r.get('success') else r.get('error')}",
                flush=True,
            )
        ingest_legacy_bodyfit(GRAPH_PATH, [r for r in cfd_results if r.get("success")])

    summary = {
        "demote_synthetic": dem,
        "gap_total": len(parts),
        "gap_with_stl": len(with_stl),
        "fea_ok": len(fea_ok),
        "fea_linked": linked_fea,
        "promote": promoted,
        "cfd_ok": sum(1 for r in cfd_results if r.get("success")),
        "fea_failures": [
            {"part_id": r["part_id"], "error": r.get("error")}
            for r in fea_results
            if not r.get("success")
        ][:30],
        "cfd_failures": [
            {"part_id": r["part_id"], "error": r.get("error")}
            for r in cfd_results
            if not r.get("success")
        ][:20],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run_gap_fix(workers=4)
