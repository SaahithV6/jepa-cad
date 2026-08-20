#!/usr/bin/env python3
"""Coverage report: Part↔SimulationCase↔TensorShard for solver case trees."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
OUT = ROOT / "artifacts/solver_case_tao_coverage.json"


def main() -> int:
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = g["nodes"]
    edges = g["edges"]
    by_id = {n["id"]: n for n in nodes}
    sim = [n for n in nodes if n.get("type") == "SimulationCase" and (n.get("properties") or {}).get("process_provenance")]
    runs = [n for n in nodes if n.get("type") == "SolverRun" and str(n.get("id", "")).startswith("solverrun:disk:")]
    parts = [n for n in nodes if n.get("type") in {"Part", "RealPart"}]
    shards = [n for n in nodes if n.get("type") == "TensorShard"]

    has_sim = set()
    has_shard = set()
    for e in edges:
        if e.get("type") == "HAS_SIMULATION":
            has_sim.add(e.get("source"))
        if e.get("type") == "HAS_SHARD" and str(e.get("source", "")).startswith("simulationcase:disk:"):
            has_shard.add(e.get("source"))

    corpus = Counter((n.get("properties") or {}).get("corpus") for n in sim)
    status = Counter((n.get("properties") or {}).get("status") for n in sim)
    report = {
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "process_simulation_cases": len(sim),
        "process_solver_runs": len(runs),
        "parts": len(parts),
        "parts_with_has_simulation": len(has_sim),
        "process_sims_with_shard": len(has_shard),
        "tensor_shards": len(shards),
        "corpus_counts": dict(corpus),
        "status_counts": dict(status),
        "graph_metadata_dim_expected": None,
    }
    try:
        from data.graph_dataset import GRAPH_METADATA_DIM

        report["graph_metadata_dim_expected"] = int(GRAPH_METADATA_DIM)
    except Exception as exc:  # noqa: BLE001
        report["graph_metadata_dim_error"] = str(exc)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
