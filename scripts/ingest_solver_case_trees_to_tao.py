#!/usr/bin/env python3
"""Ingest FEA/CFD case directories into TAO as SimulationCase + SolverRun provenance.

The large artifacts/{rocket_fea_8k,fea_final,fea_alt,cfd_*} trees are the
process of running sims. Training currently mostly sees physics_shards NPZs.
This script records every case dir (success or fail) with paths, sizes, status,
and edges to Part / TensorShard so the workflow is first-class in the graph.

Does not delete case files. Does not launch Modal/train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = (
    "artifacts/rocket_fea_8k",
    "artifacts/fea_final",
    "artifacts/fea_alt",
    "artifacts/cfd_internal",
    "artifacts/cfd_bodyfit",
    "artifacts/rocket_cfd_bodyfit",
    "artifacts/cfd_final",
    "artifacts/cfd_alt",
)


def _sha16(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return h[:16]


def _file_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return {"path": str(path.as_posix()), "bytes": int(st.st_size), "mtime": int(st.st_mtime)}


def _detect_kind(case_dir: Path, corpus: str) -> str:
    if "cfd" in corpus.lower():
        return "cfd"
    if (case_dir / "case.frd").exists() or (case_dir / "case.inp").exists():
        return "fea"
    if (case_dir / "system").is_dir() or (case_dir / "constant").is_dir():
        return "cfd"
    return "fea" if "fea" in corpus.lower() else "unknown"


def _case_status(case_dir: Path, kind: str) -> str:
    if kind == "fea":
        frd = case_dir / "case.frd"
        if frd.is_file() and frd.stat().st_size > 50_000:
            return "completed"
        if (case_dir / "case.inp").exists():
            return "failed_or_incomplete"
        return "empty"
    # CFD heuristic
    if (case_dir / "meta.json").exists():
        try:
            meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            if meta.get("success") is True or meta.get("status") == "completed":
                return "completed"
            if meta.get("success") is False:
                return "failed"
        except Exception:
            pass
    if any(case_dir.rglob("U")) or (case_dir / "postProcessing").exists():
        return "completed_or_partial"
    return "unknown"


def _load_meta(case_dir: Path) -> dict[str, Any]:
    p = case_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _list_artifacts(case_dir: Path, kind: str) -> dict[str, Any]:
    names = [
        "case.inp",
        "case.frd",
        "case.dat",
        "case.sta",
        "case.cvg",
        "meta.json",
        "mesh.msh",
        "domain.msh",
        "geometry.stl",
        "part.stl",
    ]
    arts: dict[str, Any] = {}
    for name in names:
        meta = _file_meta(case_dir / name)
        if meta:
            arts[name] = meta
    # logs
    logs = sorted(case_dir.glob("*.log"))[:5]
    if logs:
        arts["logs"] = [m for p in logs if (m := _file_meta(p))]
    if kind == "cfd":
        for sub in ("system", "constant", "0", "postProcessing"):
            if (case_dir / sub).exists():
                arts[f"dir:{sub}"] = {"path": str((case_dir / sub).as_posix()), "exists": True}
    return arts


def inventory_cases(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        corpus = root.name
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # skip cache/utility dirs
            if case_dir.name.startswith("_") or case_dir.name in {"cache", "logs"}:
                continue
            kind = _detect_kind(case_dir, corpus)
            status = _case_status(case_dir, kind)
            meta = _load_meta(case_dir)
            arts = _list_artifacts(case_dir, kind)
            rows.append(
                {
                    "corpus": corpus,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir.as_posix()),
                    "kind": kind,
                    "status": status,
                    "artifacts": arts,
                    "meta": {
                        k: meta[k]
                        for k in (
                            "part_id",
                            "material_id",
                            "load_n",
                            "metrics",
                            "success",
                            "solver",
                            "geometry_ref",
                        )
                        if k in meta
                    },
                    "total_artifact_bytes": sum(
                        int(v.get("bytes") or 0)
                        for v in arts.values()
                        if isinstance(v, dict) and "bytes" in v
                    ),
                }
            )
    return rows


def _find_part_id(by_part: dict[str, dict], case_id: str, meta: dict[str, Any]) -> str | None:
    candidates = [
        f"part:rocket:{case_id}",
        f"part:{case_id}",
        str(meta.get("part_id") or ""),
    ]
    # body_tube_02600 style → try suffix match index
    for c in candidates:
        if c and c in by_part:
            return c
    # fuzzy: any part id ending with case_id
    suffix = f":{case_id}"
    for pid in by_part:
        if pid.endswith(suffix) or pid.endswith(case_id):
            return pid
    return None


def _find_shard_id(by_shard_path: dict[str, str], case_id: str, kind: str) -> str | None:
    for key in (
        f"artifacts/physics_shards/{kind}/{case_id}.npz",
        f"artifacts/physics_shards/fea/{case_id}.npz",
        f"artifacts/physics_shards/cfd/{case_id}.npz",
    ):
        if key in by_shard_path:
            return by_shard_path[key]
    # basename match
    for path, sid in by_shard_path.items():
        if path.endswith(f"/{case_id}.npz"):
            return sid
    return None


def upsert_into_graph(graph: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])
    by_id = {n["id"]: n for n in nodes if isinstance(n.get("id"), str)}
    by_part = {n["id"]: n for n in nodes if n.get("type") in {"Part", "RealPart"}}
    by_shard_path: dict[str, str] = {}
    for n in nodes:
        if n.get("type") != "TensorShard":
            continue
        props = n.get("properties") or {}
        for k in ("shard_path", "path", "source_path"):
            v = props.get(k)
            if isinstance(v, str) and v:
                by_shard_path[v.replace("\\", "/")] = n["id"]

    edge_keys = {(e.get("type"), e.get("source"), e.get("target")) for e in edges}
    added_nodes = 0
    updated_nodes = 0
    added_edges = 0
    linked_part = 0
    linked_shard = 0
    status_counts: dict[str, int] = {}

    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        sim_id = f"simulationcase:disk:{row['corpus']}:{row['case_id']}"
        run_id = f"solverrun:disk:{row['corpus']}:{row['case_id']}"
        props = {
            "case_id": row["case_id"],
            "case_dir": row["case_dir"],
            "corpus": row["corpus"],
            "kind": row["kind"],
            "status": row["status"],
            "solver": "calculix" if row["kind"] == "fea" else "openfoam",
            "artifacts": row["artifacts"],
            "meta": row["meta"],
            "total_artifact_bytes": row["total_artifact_bytes"],
            "source": "ingest_solver_case_trees_to_tao",
            "process_provenance": True,
        }
        if sim_id in by_id:
            by_id[sim_id]["properties"] = props
            by_id[sim_id]["label"] = f"{row['corpus']}/{row['case_id']} ({row['status']})"
            updated_nodes += 1
        else:
            node = {
                "id": sim_id,
                "type": "SimulationCase",
                "label": f"{row['corpus']}/{row['case_id']} ({row['status']})",
                "properties": props,
            }
            nodes.append(node)
            by_id[sim_id] = node
            added_nodes += 1

        run_props = {
            "case_id": row["case_id"],
            "case_dir": row["case_dir"],
            "corpus": row["corpus"],
            "kind": row["kind"],
            "status": row["status"],
            "solver": props["solver"],
            "outcomes": {"status": row["status"], "bytes": row["total_artifact_bytes"]},
            "parameters": row["meta"],
            "source": "ingest_solver_case_trees_to_tao",
        }
        if run_id in by_id:
            by_id[run_id]["properties"] = run_props
            updated_nodes += 1
        else:
            node = {
                "id": run_id,
                "type": "SolverRun",
                "label": f"run {row['case_id']}",
                "properties": run_props,
            }
            nodes.append(node)
            by_id[run_id] = node
            added_nodes += 1

        def _edge(etype: str, src: str, tgt: str) -> None:
            nonlocal added_edges
            key = (etype, src, tgt)
            if key in edge_keys:
                return
            edges.append(
                {
                    "id": f"edge:{etype}:{_sha16(src, tgt, etype)}",
                    "type": etype,
                    "source": src,
                    "target": tgt,
                    "properties": {"source": "ingest_solver_case_trees_to_tao"},
                }
            )
            edge_keys.add(key)
            added_edges += 1

        _edge("HAS_SOLVER_RUN", sim_id, run_id)

        part_id = _find_part_id(by_part, row["case_id"], row["meta"])
        if part_id:
            _edge("HAS_SIMULATION", part_id, sim_id)
            _edge("VERIFIED_BY", part_id, run_id)
            linked_part += 1

        shard_id = _find_shard_id(by_shard_path, row["case_id"], row["kind"])
        if shard_id:
            _edge("HAS_SHARD", sim_id, shard_id)
            _edge("PRODUCED_SHARD", run_id, shard_id)
            linked_shard += 1

    return {
        "cases_seen": len(rows),
        "nodes_added": added_nodes,
        "nodes_updated": updated_nodes,
        "edges_added": added_edges,
        "linked_part": linked_part,
        "linked_shard": linked_shard,
        "status_counts": status_counts,
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--report", type=Path, default=ROOT / "artifacts/solver_case_tao_ingest_report.json")
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    roots = [(ROOT / r).resolve() if not Path(r).is_absolute() else Path(r) for r in args.roots]
    rows = inventory_cases(roots)
    print(json.dumps({"inventory": len(rows), "roots": [str(r) for r in roots if r.exists()]}, indent=2))

    if args.dry_run:
        from collections import Counter

        c = Counter(r["status"] for r in rows)
        corp = Counter(r["corpus"] for r in rows)
        args.report.write_text(
            json.dumps({"dry_run": True, "cases": len(rows), "status": dict(c), "corpus": dict(corp)}, indent=2)
            + "\n"
        )
        print(args.report)
        return 0

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    summary = upsert_into_graph(graph, rows)
    args.graph.write_text(json.dumps(graph, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["graph_path"] = str(args.graph)
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
