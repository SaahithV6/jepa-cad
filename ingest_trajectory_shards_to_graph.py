#!/usr/bin/env python3
"""Ingest propulsion/trajectory shards into the TAO graph as TensorShard nodes.

Shards written by generate_propulsion_trajectory_corpus.py are invisible to
training until the graph references them -- data/graph_dataset.py builds its
record list by walking graph nodes, not by globbing directories.

Node contract (matches what GraphBackedDataset expects):
  type            "TensorShard"          -- in _FILE_NODE_TYPES
  properties.shard_path                  -- first entry of _PATH_KEYS
  properties.kind "traj"
  properties.source "solver_field_extract"  -- marks it physics for ranking

The path also contains "physics_shards", so the is_physics test in the dataset
passes on three independent grounds.

Writes atomically (temp file then rename) because graph.json is ~409 MB and a
partial write would destroy the corpus index -- there is already a
graph.json.corrupt in the artifacts directory from a previous incident.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=Path("artifacts/jepa-train-bundle/graph.json"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("artifacts/physics_shards/traj_manifest.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}")
        return 1

    entries = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    print(f"manifest entries : {len(entries)}")

    print(f"loading graph    : {args.graph} ({args.graph.stat().st_size/1e6:.0f} MB)")
    payload = json.loads(args.graph.read_text())
    nodes = payload.setdefault("nodes", [])
    edges = payload.setdefault("edges", [])
    print(f"existing nodes   : {len(nodes)}  edges: {len(edges)}")

    existing_ids = {str(n.get("id", "")) for n in nodes}
    added = skipped = 0

    for e in entries:
        node_id = f"tensorshard:traj_{e['case_id']}"
        if node_id in existing_ids:
            skipped += 1
            continue
        m = e.get("metrics", {})
        nodes.append({
            "id": node_id,
            "type": "TensorShard",
            "label": f"traj_{e['case_id']}",
            "properties": {
                "shard_path": e["shard_path"],
                "kind": "traj",
                "source": "solver_field_extract",
                "family": "nozzle",
                "is_synthetic": 0,
                "channels": m.get("channels", ""),
                # Property names and units below MUST match
                # cadflow.physics_targets.CONDITIONING_QUANTITIES exactly --
                # the dataset looks slots up by name, so "thrust_sl_n" would be
                # silently ignored where "thrust_kN" is conditioned on. The
                # shard itself is scaled to [-1,1] per channel, so absolute
                # magnitudes survive only here.
                "isp_vac_s": m.get("isp_vac_s"),
                "thrust_kN": (m.get("thrust_sl_n") or 0.0) / 1000.0,
                "burn_time_s": m.get("burn_time_s"),
                "delta_v_ms": m.get("delta_v_ideal_m_s"),
                "apogee_km": m.get("apogee_km"),
                "downrange_km": m.get("downrange_km"),
                "max_dynamic_pressure_kpa": (m.get("max_q_pa") or 0.0) / 1000.0,
                "payload_kg": m.get("payload_kg"),
                "chamber_pressure_bar": (m.get("chamber_pressure_pa") or 0.0) / 1e5,
                "expansion_ratio": m.get("expansion_ratio"),
                "mass_flow_kgps": m.get("mdot_kg_s"),
                "mass_fraction": m.get("mass_fraction"),
                "Cd": m.get("cd"),
            },
        })
        existing_ids.add(node_id)
        added += 1

    print(f"nodes to add     : {added} (skipped {skipped} already present)")
    if args.dry_run:
        print("dry run -- graph not written")
        return 0
    if added == 0:
        print("nothing to do")
        return 0

    tmp = args.graph.with_suffix(".json.tmp")
    print(f"writing          : {tmp}")
    tmp.write_text(json.dumps(payload))
    tmp.replace(args.graph)
    print(f"done             : {len(nodes)} nodes total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
