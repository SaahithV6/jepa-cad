"""Turn raw solver output (CalculiX FRD, OpenFOAM cases) into real physics-field
training shards and a manifest the TAO graph can ingest.

Why this exists
---------------
The TAO ``graph.json`` only stored *scalar summaries* (max_stress, U_mag) while
~150 GB of actual nodal/volume fields sat unused on disk. The pre-existing
``.npz`` training shards carried geometry-derived channels normalized to [0,1]
with a constant ``max_stress=1.0`` — i.e. no real physics signal.

This module extracts genuine per-node fields:

* **FEA (FRD)** — CalculiX writes nodal DISP (3) and STRESS (6) blocks in a
  *fixed-width* 12-char format. Whitespace splitting silently drops packed
  large values (``-1.2E+08-2.3E+08``); we parse fixed width so stress survives.
  Channels (8): ``[von_mises_pa, disp_mag_m, ux, uy, uz, sxx, syy, szz]``.
* **CFD (OpenFOAM)** — sample cell-centre ``U`` (3) and ``p`` (1) from the last
  time directory. Channels (8): ``[|U|, p, Ux, Uy, Uz, p, |U|, p]`` (padded).

Each shard is ``{points(N,3) float32, fields(N,C) float32, max_stress f32,
part_id str, ...}`` written under ``artifacts/physics_shards/<kind>/`` plus a
JSONL manifest line so a single graph writer can register ``TensorShard`` nodes
without racing the solvers/ingest loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cadflow.graph_lock import graph_lock, read_graph

ROOT = Path(__file__).resolve().parents[1]
SHARD_ROOT = ROOT / "artifacts" / "physics_shards"
FEA_SHARD_DIR = SHARD_ROOT / "fea"
CFD_SHARD_DIR = SHARD_ROOT / "cfd"
FEA_MANIFEST = SHARD_ROOT / "fea_manifest.jsonl"
CFD_MANIFEST = SHARD_ROOT / "cfd_manifest.jsonl"

DEFAULT_NUM_POINTS = 2048


# --------------------------------------------------------------------------- #
# FRD (FEA) fixed-width parsing
# --------------------------------------------------------------------------- #
def _fixed_width_floats(segment: str, width: int = 12) -> list[float]:
    out: list[float] = []
    i = 0
    n = len(segment)
    while i + 1 < n:
        tok = segment[i : i + width].strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
        i += width
    return out


def _von_mises(v: list[float]) -> float:
    sxx, syy, szz, sxy, syz, szx = v[:6]
    return math.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + szx**2)
        )
    )


def parse_frd_fields(frd_path: Path) -> dict[int, dict[str, Any]] | None:
    """Return ``{node_id: {"coord": (x,y,z), "disp": (..), "stress": (..)}}``.

    Coordinates come from the ``2C`` block (meters). Only nodes with all three
    of coord/disp/stress are returned so channels align.
    """
    coords: dict[int, tuple[float, float, float]] = {}
    disp: dict[int, tuple[float, float, float]] = {}
    stress: dict[int, list[float]] = {}
    mode: str | None = None
    try:
        with frd_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                s = line.lstrip()
                if s.startswith("2C"):
                    mode = "coord"
                    continue
                if s.startswith("-4"):
                    upper = s.upper()
                    mode = "disp" if "DISP" in upper else ("stress" if "STRESS" in upper else None)
                    continue
                if s.startswith("-3"):
                    mode = None
                    continue
                if not s.startswith("-1") or mode is None:
                    continue
                body = line[3:]
                try:
                    nid = int(body[:10])
                except ValueError:
                    continue
                vals = _fixed_width_floats(body[10:])
                if mode == "coord" and len(vals) >= 3:
                    coords[nid] = (vals[0], vals[1], vals[2])
                elif mode == "disp" and len(vals) >= 3:
                    disp[nid] = (vals[0], vals[1], vals[2])
                elif mode == "stress" and len(vals) >= 6:
                    stress[nid] = vals[:6]
    except OSError:
        return None

    common = set(coords) & set(disp) & set(stress)
    if len(common) < 32:
        return None
    return {
        nid: {"coord": coords[nid], "disp": disp[nid], "stress": stress[nid]}
        for nid in common
    }


def _resample_indices(n: int, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n >= target:
        return rng.choice(n, size=target, replace=False)
    return rng.choice(n, size=target, replace=True)


def fea_case_to_shard(
    frd_path: Path,
    *,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict[str, Any] | None:
    """Build an in-memory shard payload from one FRD, or None if unusable."""
    nodes = parse_frd_fields(frd_path)
    if not nodes:
        return None
    ids = list(nodes.keys())
    coords = np.array([nodes[i]["coord"] for i in ids], dtype=np.float64)  # meters
    disps = np.array([nodes[i]["disp"] for i in ids], dtype=np.float64)  # meters
    stresses = np.array([nodes[i]["stress"] for i in ids], dtype=np.float64)  # Pa
    vm = np.array([_von_mises(list(s)) for s in stresses], dtype=np.float64)  # Pa
    disp_mag = np.linalg.norm(disps, axis=1)  # meters

    idx = _resample_indices(len(ids), num_points, seed=abs(hash(frd_path.parent.name)) % (2**32))
    coords_s = coords[idx]
    disps_s = disps[idx]
    stresses_s = stresses[idx]
    vm_s = vm[idx]
    disp_mag_s = disp_mag[idx]

    # Normalize geometry to a unit-ish box centred at origin (training expects
    # ~[-0.5, 0.5]); keep physical scalars in metrics.
    center = coords_s.mean(axis=0)
    pts = coords_s - center
    scale = float(np.max(np.abs(pts))) or 1.0
    pts = (pts / scale).astype(np.float32)

    # Physics channels (8). Normalize per-shard by robust maxima so training
    # sees shape of the field; absolute scale lives in metrics + conditioning.
    vm_max = float(vm_s.max()) or 1.0
    disp_max = float(disp_mag_s.max()) or 1.0
    s_abs = float(np.max(np.abs(stresses_s[:, :3]))) or 1.0
    d_abs = float(np.max(np.abs(disps_s))) or 1.0
    fields = np.stack(
        [
            (vm_s / vm_max),
            (disp_mag_s / disp_max),
            (disps_s[:, 0] / d_abs),
            (disps_s[:, 1] / d_abs),
            (disps_s[:, 2] / d_abs),
            (stresses_s[:, 0] / s_abs),
            (stresses_s[:, 1] / s_abs),
            (stresses_s[:, 2] / s_abs),
        ],
        axis=1,
    ).astype(np.float32)

    part_id = frd_path.parent.name
    return {
        "part_id": part_id,
        "points": pts,
        "fields": fields,
        "max_stress": np.float32(vm_max / 1e6),  # MPa (real)
        "metrics": {
            "max_von_mises_mpa": round(vm_max / 1e6, 4),
            "mean_von_mises_mpa": round(float(vm_s.mean()) / 1e6, 4),
            "max_disp_mm": round(disp_max * 1000.0, 6),
            "node_count": int(len(ids)),
            "geom_scale_m": round(scale, 6),
            "channels": "von_mises,disp_mag,ux,uy,uz,sxx,syy,szz",
        },
    }


def write_shard(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload['part_id']}.npz"
    np.savez_compressed(
        out_path,
        points=payload["points"],
        fields=payload["fields"],
        max_stress=payload["max_stress"],
    )
    return out_path


def _parse_foam_vector_list(text: str) -> np.ndarray:
    """Parse an OpenFOAM ``(x y z)`` list (points / cell centres)."""
    import re

    vecs = re.findall(
        r"\(\s*([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s*\)",
        text,
    )
    if not vecs:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(vecs, dtype=np.float64)


def _foam_points_bbox(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (mins, maxs) from ``constant/polyMesh/points`` without full face walk."""
    pts_p = case_dir / "constant" / "polyMesh" / "points"
    if not pts_p.is_file():
        return None
    try:
        pts = _parse_foam_vector_list(pts_p.read_text(errors="ignore"))
        if pts.shape[0] < 8:
            return None
        return pts.min(axis=0), pts.max(axis=0)
    except Exception:  # noqa: BLE001
        return None


def _coords_for_cfd_cells(case_dir: Path, n: int, tdir: Path) -> np.ndarray:
    """Spatial coords for ``n`` CFD cells.

    Prefer written cell centres; else a lattice over the mesh bbox. Full
    owner/face cell-centre reconstruction is too slow for 50k+ cell bodyfit
    meshes and was blocking the training-data path.
    """
    for cand in (tdir / "C", case_dir / "constant" / "C"):
        if cand.is_file():
            centres = _parse_foam_vector_list(cand.read_text(errors="ignore"))
            if centres.shape[0] >= n:
                return centres[:n]
    bb = _foam_points_bbox(case_dir)
    if bb is not None:
        mins, maxs = bb
        # Stratified lattice in the freestream box, clipped to n cells.
        g = int(math.ceil(n ** (1.0 / 3.0)))
        zs, ys, xs = np.meshgrid(
            np.linspace(mins[2], maxs[2], g),
            np.linspace(mins[1], maxs[1], g),
            np.linspace(mins[0], maxs[0], g),
            indexing="ij",
        )
        return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)[:n]
    g = int(math.ceil(n ** (1.0 / 3.0)))
    zs, ys, xs = np.meshgrid(
        np.linspace(-0.5, 0.5, g),
        np.linspace(-0.5, 0.5, g),
        np.linspace(-0.5, 0.5, g),
        indexing="ij",
    )
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)[:n]


def cfd_case_to_shard(
    case_dir: Path,
    *,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict[str, Any] | None:
    """Build a CFD volume-field shard from an OpenFOAM case before field cleanup."""
    from run_cfd_5k_proper import (
        latest_time_dir,
        parse_internal_field_scalars,
        parse_internal_field_vectors,
    )

    tdir = latest_time_dir(case_dir)
    if tdir is None:
        return None
    u_path, p_path = tdir / "U", tdir / "p"
    if not u_path.is_file() or not p_path.is_file():
        return None
    U = parse_internal_field_vectors(u_path.read_text(errors="ignore"))
    P = parse_internal_field_scalars(p_path.read_text(errors="ignore"))
    if not U or not P:
        return None
    n = min(len(U), len(P))
    if n < 8:
        return None
    U = np.asarray(U[:n], dtype=np.float64)
    P = np.asarray(P[:n], dtype=np.float64)
    centres = _coords_for_cfd_cells(case_dir, n, tdir)

    idx = _resample_indices(n, num_points, seed=abs(hash(case_dir.name)) % (2**32))
    coords_s = centres[idx]
    U_s = U[idx]
    P_s = P[idx]
    umag = np.linalg.norm(U_s, axis=1)

    center = coords_s.mean(axis=0)
    pts = coords_s - center
    scale = float(np.max(np.abs(pts))) or 1.0
    pts = (pts / scale).astype(np.float32)

    umax = float(umag.max()) or 1.0
    pabs = float(np.max(np.abs(P_s))) or 1.0
    uabs = float(np.max(np.abs(U_s))) or 1.0
    fields = np.stack(
        [
            umag / umax,
            P_s / pabs,
            U_s[:, 0] / uabs,
            U_s[:, 1] / uabs,
            U_s[:, 2] / uabs,
            P_s / pabs,
            umag / umax,
            P_s / pabs,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "part_id": case_dir.name,
        "points": pts,
        "fields": fields,
        "max_stress": np.float32(umax),  # reuse slot as |U|_max for CFD
        "metrics": {
            "U_mag_max": round(umax, 6),
            "U_mag_mean": round(float(umag.mean()), 6),
            "p_min": round(float(P_s.min()), 6),
            "p_max": round(float(P_s.max()), 6),
            "n_cells": int(n),
            "geom_scale_m": round(scale, 6),
            "channels": "U_mag,p,Ux,Uy,Uz,p,U_mag,p",
        },
    }


def append_cfd_shard_manifest(
    case_dir: Path,
    *,
    part_id: str | None = None,
    id_prefix: str = "part:rocket:",
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict[str, Any] | None:
    """Extract + write one CFD shard and append ``cfd_manifest.jsonl``.

    Safe to call from bodyfit workers just before field cleanup. Idempotent on
    shard path; appends a new manifest line only when the npz is (re)written.
    """
    case_dir = Path(case_dir)
    out = CFD_SHARD_DIR / f"{case_dir.name}.npz"
    graph_part_id = part_id or f"{id_prefix}{case_dir.name}"
    if out.exists():
        return {"part_id": graph_part_id, "cached": True, "shard_path": str(out.relative_to(ROOT))}
    payload = cfd_case_to_shard(case_dir, num_points=num_points)
    if payload is None:
        return None
    write_shard(payload, CFD_SHARD_DIR)
    CFD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "part_id": graph_part_id,
        "case_id": case_dir.name,
        "kind": "cfd",
        "shard_path": str(out.relative_to(ROOT)),
        "metrics": payload["metrics"],
    }
    with CFD_MANIFEST.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(rec) + "\n")
    return rec


def register_manifest_to_graph(
    graph_path: Path,
    manifest_paths: Iterable[Path] = (FEA_MANIFEST, CFD_MANIFEST),
) -> dict[str, int]:
    """Fold physics-shard manifests into the TAO graph as ``TensorShard`` nodes.

    Idempotent: dedups by node id. Adds a ``HAS_SAMPLE`` edge from the Part to
    the shard and stamps real field metrics on the Part so downstream tools see
    that genuine physics-field training data exists.
    """
    with graph_lock(graph_path):
        return _register_manifest_to_graph(graph_path, manifest_paths)


def _register_manifest_to_graph(
    graph_path: Path,
    manifest_paths: Iterable[Path] = (FEA_MANIFEST, CFD_MANIFEST),
) -> dict[str, int]:
    graph = read_graph(graph_path)
    nodes = graph["nodes"]
    edges = graph.setdefault("edges", [])
    by_id = {n.get("id"): n for n in nodes}
    existing_nodes = set(by_id)
    existing_edges = {e.get("id") for e in edges}

    # Legacy Parts are keyed by their own hash (``part:<part_hash>``) while the
    # solver case directory uses a *different* hash, recorded as ``fea_case_id``
    # / ``simulation_results_*.case_id``. Build a case→Part index so shards
    # extracted from case dirs attach to the right Part.
    case_to_part: dict[str, str] = {}
    for node in nodes:
        if node.get("type") != "Part":
            continue
        pid = str(node.get("id") or "")
        for key in ("fea_case_id", "cfd_case_id"):
            cid = node.get(key)
            if cid:
                case_to_part.setdefault(str(cid), pid)
        for key in ("simulation_results_fea", "simulation_results_cfd"):
            block = node.get(key)
            if isinstance(block, dict) and block.get("case_id"):
                case_to_part.setdefault(str(block["case_id"]), pid)

    added_nodes = added_edges = stamped = 0
    for manifest_path in manifest_paths:
        if not Path(manifest_path).exists():
            continue
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            part_id = rec.get("part_id")
            case_id = rec.get("case_id")
            kind = rec.get("kind", "fea")
            shard_path = rec.get("shard_path")
            if not (part_id and shard_path):
                continue
            # Resolve to a real Part: direct id, else via case→Part index.
            if part_id not in by_id and case_id:
                mapped = case_to_part.get(str(case_id))
                if mapped:
                    part_id = mapped
            shard_node_id = f"tensorshard:{kind}:{case_id}"
            if shard_node_id not in existing_nodes:
                nodes.append(
                    {
                        "id": shard_node_id,
                        "type": "TensorShard",
                        "label": f"{kind}_field_shard:{case_id}",
                        "properties": {
                            "shard_path": shard_path,
                            "kind": kind,
                            "source": "solver_field_extract",
                            "channels": (rec.get("metrics") or {}).get("channels"),
                            **(rec.get("metrics") or {}),
                        },
                    }
                )
                existing_nodes.add(shard_node_id)
                added_nodes += 1
            edge_id = f"edge:{shard_node_id}:sample_of:{part_id}"
            if edge_id not in existing_edges:
                edges.append(
                    {
                        "id": edge_id,
                        "type": "HAS_SAMPLE",
                        "source": part_id,
                        "target": shard_node_id,
                        "properties": {"role": "physics_field_shard"},
                    }
                )
                existing_edges.add(edge_id)
                added_edges += 1
            part = by_id.get(part_id)
            if part is not None:
                if not part.get("has_field_shard"):
                    part["has_field_shard"] = True
                    part["field_shard_path"] = shard_path
                    stamped += 1
                # Mirror Part params/family onto the TensorShard so oneshot
                # conditioning is dense even before association walks.
                shard_node = by_id.get(shard_node_id) or next(
                    (n for n in nodes if n.get("id") == shard_node_id), None
                )
                if shard_node is not None:
                    props = shard_node.setdefault("properties", {})
                    part_props = part.get("properties") or {}
                    if part_props.get("family") and not props.get("family"):
                        props["family"] = part_props.get("family")
                    if part_props.get("params") and not props.get("params"):
                        props["params"] = part_props.get("params")
                    if part.get("mass_kg") is not None:
                        props.setdefault("mass_kg", part.get("mass_kg"))
                    # Keep by_id in sync for newly added nodes
                    by_id[shard_node_id] = shard_node
                    by_id[part_id] = part

    if added_nodes or added_edges or stamped:
        tmp = graph_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
        tmp.replace(graph_path)
    return {"nodes_added": added_nodes, "edges_added": added_edges, "parts_stamped": stamped}


def _iter_fea_cases(fea_root: Path, min_bytes: int) -> Iterable[Path]:
    for case in sorted(fea_root.iterdir()):
        if not case.is_dir():
            continue
        frd = case / "case.frd"
        try:
            if frd.is_file() and frd.stat().st_size >= min_bytes:
                yield frd
        except OSError:
            continue


def build_fea_shards(
    fea_root: Path,
    *,
    out_dir: Path = FEA_SHARD_DIR,
    manifest_path: Path = FEA_MANIFEST,
    id_prefix: str = "part:rocket:",
    num_points: int = DEFAULT_NUM_POINTS,
    min_bytes: int = 50_000,
    force: bool = False,
    limit: int = 0,
) -> dict[str, int]:
    """Extract real FEA field shards under ``fea_root`` and append a JSONL manifest.

    ``id_prefix`` maps a case dir name to the graph Part id
    (``part:rocket:<case>`` for rocket, ``part:<case>`` style for legacy).
    Writes shards + manifest only; graph registration is a separate single
    writer (avoids racing solvers/ingest).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if manifest_path.exists() and not force:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["part_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    ok = skip = fail = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for i, frd in enumerate(_iter_fea_cases(fea_root, min_bytes)):
            if limit and (ok + fail) >= limit:
                break
            case = frd.parent.name
            shard = out_dir / f"{case}.npz"
            if case in done or (shard.exists() and not force):
                skip += 1
                continue
            payload = fea_case_to_shard(frd, num_points=num_points)
            if payload is None:
                fail += 1
                continue
            write_shard(payload, out_dir)
            rec = {
                "part_id": f"{id_prefix}{case}",
                "case_id": case,
                "kind": "fea",
                "shard_path": str(shard.relative_to(ROOT)),
                "metrics": payload["metrics"],
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            ok += 1
            if (ok + fail) % 100 == 0:
                print(f"[fea-shards] {ok} ok / {fail} fail / {skip} skip", flush=True)
    return {"ok": ok, "skip": skip, "fail": fail}
