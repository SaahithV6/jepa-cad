"""Body-fitted external CFD for legacy TAO Parts (not the rocket 8k corpus).

Reuses snappyHexMesh case construction from ``cadflow.rocket_cfd_bodyfit``.
Artifacts go to ``artifacts/cfd_bodyfit/`` — never ``rocket_cfd_bodyfit/``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadflow.part_family import classify_part
from cadflow.physics_targets import physics_targets_for
from cadflow.rocket_cfd_bodyfit import (
    BodyfitResult,
    _domain_m,
    bodyfit_env,
    write_bodyfit_case,
)
from run_cfd_5k_proper import run_cmd, summarize_fields

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "artifacts/jepa-train-bundle/graph.json"
CFD_ROOT = ROOT / "artifacts/cfd_bodyfit"
SUMMARY_PATH = ROOT / "data/legacy_cfd_bodyfit_summary.json"
AERO_FAMILIES = frozenset({"fin", "nose_cone", "fairing"})
_RUN_RE = re.compile(r"/runs/([0-9a-f]{8,})/")


def _fix_location_in_mesh(case_dir: Path, dom: dict[str, float]) -> None:
    """Keep locationInMesh in fluid: off centerline (bodies often fill y=z=0)."""
    snappy = case_dir / "system/snappyHexMeshDict"
    if not snappy.exists():
        return
    lx = dom["xmin"] + 0.02 * (dom["xmax"] - dom["xmin"])
    ly = dom["ymin"] + 0.18 * (dom["ymax"] - dom["ymin"])
    lz = dom["zmin"] + 0.18 * (dom["zmax"] - dom["zmin"])
    text = snappy.read_text(encoding="utf-8")
    text2 = re.sub(
        r"locationInMesh\s*\([^)]+\);",
        f"locationInMesh ({lx} {ly} {lz});",
        text,
    )
    if text2 != text:
        snappy.write_text(text2, encoding="utf-8")


def _atomic_write_graph(graph_path: Path, graph: dict[str, Any]) -> None:
    tmp = graph_path.with_suffix(graph_path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph), encoding="utf-8")
    tmp.replace(graph_path)


def case_id_for_part(part: dict[str, Any]) -> str | None:
    props = part.get("properties") or {}
    for key in ("cfd_case_id", "fea_case_id"):
        v = part.get(key) or props.get(key)
        if v:
            return str(v)
    gref = str(props.get("geometry_ref") or "")
    m = _RUN_RE.search(gref)
    if m:
        return m.group(1)
    fp = str(props.get("manifest_fingerprint") or "")
    if len(fp) >= 12:
        return fp[:16]
    return None


def stl_for_part(part: dict[str, Any]) -> Path | None:
    props = part.get("properties") or {}
    gref = props.get("geometry_ref")
    if not gref:
        return None
    p = Path(str(gref))
    # Keep original .STL casing (Linux is case-sensitive); with_suffix(".stl") breaks.
    if p.suffix.lower() == ".stl" and p.exists():
        return p
    stl = p.with_suffix(".stl")
    return stl if stl.exists() else None


def select_aero_parts(graph: dict[str, Any], limit: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        if str(n.get("id") or "").startswith("part:rocket:"):
            continue
        fam = classify_part(n)
        if fam not in AERO_FAMILIES:
            continue
        if not stl_for_part(n) or not case_id_for_part(n):
            continue
        out.append(n)
        if limit > 0 and len(out) >= limit:
            break
    return out


def run_legacy_bodyfit_case(
    part: dict[str, Any],
    cfd_root: Path = CFD_ROOT,
    *,
    force: bool = False,
    timeout_mesh: int = 420,
    timeout_solve: int = 300,
) -> dict[str, Any]:
    cid = case_id_for_part(part)
    stl = stl_for_part(part)
    result: dict[str, Any] = {
        "part_id": part["id"],
        "case_id": cid,
        "success": False,
    }
    if not cid or not stl:
        result["error"] = "missing_stl_or_case_id"
        return result

    case_dir = (cfd_root / cid).resolve()
    family = classify_part(part)
    props = part.get("properties") or {}
    fp = str(props.get("manifest_fingerprint") or part["id"])
    targets = physics_targets_for(fp, family)
    bcs = targets["boundary_conditions"]

    if not force:
        existing = summarize_fields(case_dir)
        boundary = case_dir / "constant/polyMesh/boundary"
        body_stl = case_dir / "constant/triSurface/body.stl"
        if (
            existing
            and existing.get("U_mag_max", 0) > 1e-6
            and body_stl.exists()
            and boundary.exists()
            and b"body" in boundary.read_bytes()
        ):
            result.update(
                {
                    "success": True,
                    "cached": True,
                    "family": family,
                    "metrics": {
                        **existing,
                        "mesh": "snappyHexMesh_external",
                        "geometry": "stl_body_wall",
                    },
                }
            )
            return result

    env = bodyfit_env()
    try:
        if case_dir.exists() and force:
            shutil.rmtree(case_dir)
        dom = _domain_m(stl)
        # Family freestream overrides Re-scaled default when available
        U_fs = bcs.get("freestream_velocity_mps")
        if isinstance(U_fs, (int, float)):
            dom = {**dom, "U": float(min(80.0, max(0.5, U_fs)))}
        write_bodyfit_case(case_dir, stl, dom)
        _fix_location_in_mesh(case_dir, dom)

        bm = run_cmd(["blockMesh", "-case", str(case_dir)], case_dir, env, 120)
        (case_dir / "log.blockMesh").write_text((bm.stdout or "") + "\n" + (bm.stderr or ""))
        if bm.returncode != 0:
            result["error"] = "blockMesh"
            return result

        sn = run_cmd(
            ["snappyHexMesh", "-overwrite", "-case", str(case_dir)],
            case_dir,
            env,
            timeout_mesh,
        )
        (case_dir / "log.snappyHexMesh").write_text((sn.stdout or "") + "\n" + (sn.stderr or ""))
        if sn.returncode != 0:
            result["error"] = "snappyHexMesh"
            return result

        boundary = case_dir / "constant/polyMesh/boundary"
        if not boundary.exists() or b"body" not in boundary.read_bytes():
            result["error"] = "no_body_patch"
            return result

        sf = run_cmd(["simpleFoam", "-case", str(case_dir)], case_dir, env, timeout_solve)
        (case_dir / "log.simpleFoam").write_text((sf.stdout or "") + "\n" + (sf.stderr or ""))
        if sf.returncode != 0:
            result["error"] = "simpleFoam"
            return result

        metrics = summarize_fields(case_dir)
        if not metrics or metrics.get("U_mag_max", 0) < 1e-6:
            result["error"] = "empty_fields"
            return result

        metrics = {
            **metrics,
            "solver": "simpleFoam",
            "mesh": "snappyHexMesh_external",
            "U_inlet": dom["U"],
            "nu": float(bcs.get("kinematic_viscosity_m2s") or 1e-5),
            "geometry": "stl_body_wall",
            "family": family,
            "sim_kind": targets["sim_kind"],
            "turbulence_model": bcs.get("turbulence_model", "laminar_proxy"),
        }
        (case_dir / "meta.json").write_text(
            json.dumps(
                {
                    "part_id": part["id"],
                    "case_id": cid,
                    "family": family,
                    "mesh": "snappyHexMesh_external",
                    "U_inlet": dom["U"],
                    "metrics": metrics,
                    "targets": targets["targets"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result.update(
            {
                "success": True,
                "family": family,
                "sim_kind": targets["sim_kind"],
                "metrics": metrics,
                "targets": targets["targets"],
            }
        )
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"exception:{type(exc).__name__}:{exc}"[:180]
        return result


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    return run_legacy_bodyfit_case(
        payload["part"],
        Path(payload["cfd_root"]),
        force=payload["force"],
        timeout_mesh=payload["timeout_mesh"],
        timeout_solve=payload["timeout_solve"],
    )


def run_batch_legacy_bodyfit(
    parts: list[dict[str, Any]],
    cfd_root: Path = CFD_ROOT,
    *,
    workers: int = 2,
    force: bool = False,
    timeout_mesh: int = 420,
    timeout_solve: int = 300,
) -> list[dict[str, Any]]:
    import multiprocessing as mp

    cfd_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    payloads = [
        {
            "part": p,
            "cfd_root": str(cfd_root),
            "force": force,
            "timeout_mesh": timeout_mesh,
            "timeout_solve": timeout_solve,
        }
        for p in parts
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_worker, pl): pl["part"]["id"] for pl in payloads}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "part_id": futs[fut],
                        "success": False,
                        "error": f"future:{type(exc).__name__}:{exc}"[:120],
                    }
                )
            if i % 5 == 0 or i == len(futs) or not results[-1].get("success"):
                ok = sum(1 for r in results if r.get("success"))
                print(
                    f"  [legacy-bodyfit {i}/{len(futs)}] ok={ok} "
                    f"last={results[-1].get('case_id')} "
                    f"{'OK' if results[-1].get('success') else results[-1].get('error')}",
                    flush=True,
                )
    return results


def ingest_legacy_bodyfit(graph_path: Path, results: list[dict[str, Any]]) -> int:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(r["part_id"])
        if not node:
            continue
        metrics = r.get("metrics") or {}
        prev = node.get("simulation_results_cfd")
        if isinstance(prev, dict) and prev.get("mesh") != "snappyHexMesh_external":
            node["simulation_results_cfd_channel_proxy"] = prev
        # Keep demoted proxy if present
        node["has_cfd"] = True
        node["cfd_case_id"] = r.get("case_id")
        node["cfd_mesh"] = "snappyHexMesh_external"
        node["simulation_results_cfd"] = {
            "solver": "simpleFoam",
            "status": "completed",
            "source": "U,p fields",
            "case_id": r.get("case_id"),
            "family": r.get("family") or metrics.get("family"),
            "sim_kind": r.get("sim_kind") or metrics.get("sim_kind"),
            "replaced_channel_proxy": True,
            "cfd_bodyfit": True,
            **{k: v for k, v in metrics.items() if k not in {"family", "sim_kind"}},
        }
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["cfd"] = True
        pd["cfd_bodyfit"] = True
        pd["fea"] = bool(node.get("has_fea"))
        pd["verified"] = bool(pd.get("fea") or True)
        node["physics_data"] = pd
        linked += 1
    _atomic_write_graph(graph_path, graph)
    return linked


def ingest_legacy_bodyfit_from_disk(graph_path: Path = GRAPH_PATH, cfd_root: Path = CFD_ROOT) -> dict[str, int]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    # case_id → part
    case_to_part: dict[str, str] = {}
    for n in by_id.values():
        cid = case_id_for_part(n)
        if cid:
            case_to_part[cid] = n["id"]

    results: list[dict[str, Any]] = []
    if not cfd_root.exists():
        return {"disk": 0, "linked": 0}
    for case_dir in sorted(cfd_root.iterdir()):
        if not case_dir.is_dir():
            continue
        metrics = summarize_fields(case_dir)
        boundary = case_dir / "constant/polyMesh/boundary"
        if (
            not metrics
            or metrics.get("U_mag_max", 0) < 1e-6
            or not boundary.exists()
            or b"body" not in boundary.read_bytes()
        ):
            continue
        meta: dict[str, Any] = {}
        if (case_dir / "meta.json").exists():
            meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        part_id = meta.get("part_id") or case_to_part.get(case_dir.name)
        if not part_id or part_id not in by_id:
            continue
        m = meta.get("metrics") or metrics
        results.append(
            {
                "part_id": part_id,
                "case_id": case_dir.name,
                "success": True,
                "family": meta.get("family"),
                "sim_kind": meta.get("sim_kind") or (m.get("sim_kind") if isinstance(m, dict) else None),
                "metrics": {
                    **metrics,
                    "mesh": "snappyHexMesh_external",
                    "geometry": "stl_body_wall",
                    **({k: v for k, v in m.items()} if isinstance(m, dict) else {}),
                },
            }
        )
    linked = ingest_legacy_bodyfit(graph_path, results)
    return {"disk": len(results), "linked": linked}
