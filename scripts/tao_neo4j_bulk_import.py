#!/usr/bin/env python3
"""Bulk-load the densified TAO JSON graph into local Neo4j via neo4j-admin CSV import.

Per-node MERGE Cypher is not viable at ~250k+ nodes. This path:

  1. Exports nodes.csv + relationships.csv under artifacts/neo4j-tao-bulk/
  2. Stops Neo4j if running
  3. neo4j-admin database import full --overwrite-destination
  4. Starts Neo4j
  5. Cypher parity counts vs JSON → artifacts/neo4j_tao_parity.json

Auth: expects dbms.security.auth_enabled=false (current local config).
Does not train.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
OUT = ROOT / "artifacts/neo4j-tao-bulk"
PARITY = ROOT / "artifacts/neo4j_tao_parity.json"
NEO4J_HOME = Path.home() / ".local/neo4j/current"
JAVA_HOME = Path.home() / ".local/java/temurin-21"
DATABASE = "neo4j"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["NEO4J_HOME"] = str(NEO4J_HOME)
    env["PATH"] = f"{JAVA_HOME}/bin:{NEO4J_HOME}/bin:" + env.get("PATH", "")
    return env


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
        timeout=timeout,
    )


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _scalar_props(properties: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            # Cap huge strings for CSV top-level props; full copy stays in properties_json
            if isinstance(value, str) and len(value) > 8000:
                out[key] = value[:8000]
            else:
                out[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) or x is None for x in value
        ):
            out[key] = ";".join("" if x is None else str(x) for x in value)
    return out


def export_csv(graph: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.csv"
    rels_path = out_dir / "relationships.csv"
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    # Collect a stable set of scalar property columns (bounded).
    prop_keys: set[str] = set()
    for node in nodes:
        props = node.get("properties") or {}
        for k, v in _scalar_props(props).items():
            if k in {"id", "type", "label"}:
                continue
            if isinstance(v, str) and len(v) > 500:
                continue  # keep wide text only inside properties_json
            prop_keys.add(k)
    # Prefer useful conditioning keys; cap columns for import speed
    preferred = [
        "name",
        "family",
        "part_class",
        "source_corpus",
        "geometry_ref",
        "shard_path",
        "path",
        "source_path",
        "text_extract_status",
        "text_chars",
        "mass_kg",
        "has_fea",
        "has_cfd",
        "spec_prompt",
    ]
    columns = []
    for k in preferred:
        if k in prop_keys:
            columns.append(k)
            prop_keys.discard(k)
    columns.extend(sorted(prop_keys)[:40])

    with nodes_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        header = [":ID", ":LABEL", "node_label", "node_type", "properties_json", *columns]
        writer.writerow(header)
        for node in nodes:
            nid = str(node.get("id") or "")
            if not nid:
                continue
            ntype = str(node.get("type") or "Entity").replace("`", "")
            props = node.get("properties") or {}
            # Include top-level flags commonly stored beside properties
            merged = dict(props)
            for k in ("has_fea", "has_cfd", "physics_verified"):
                if k in node and k not in merged:
                    merged[k] = node[k]
            scalars = _scalar_props(merged)
            # Keep a compact properties_json: drop ultra-long text duplicates when status exists
            props_for_json = dict(merged)
            text = props_for_json.get("text")
            if isinstance(text, str) and len(text) > 12000:
                props_for_json["text"] = text[:12000]
            row = [
                nid,
                f"{ntype};Entity",
                str(node.get("label") or ""),
                ntype,
                _json_text(props_for_json),
            ]
            for col in columns:
                val = scalars.get(col)
                row.append("" if val is None else val)
            writer.writerow(row)

    with rels_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([":START_ID", ":END_ID", ":TYPE", "id", "properties_json"])
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            et = str(edge.get("type") or "RELATED").replace("`", "")
            if not src or not tgt:
                continue
            eid = str(edge.get("id") or f"{et}:{src}->{tgt}")
            writer.writerow(
                [
                    str(src),
                    str(tgt),
                    et,
                    eid,
                    _json_text(edge.get("properties") or {}),
                ]
            )

    return {
        "nodes_csv": str(nodes_path),
        "relationships_csv": str(rels_path),
        "node_rows": len(nodes),
        "edge_rows": len(edges),
        "property_columns": columns,
    }


def neo4j_status() -> str:
    proc = _run([str(NEO4J_HOME / "bin/neo4j"), "status"])
    return (proc.stdout or proc.stderr or "").strip()


def stop_neo4j() -> None:
    _run([str(NEO4J_HOME / "bin/neo4j"), "stop"], timeout=120)


def start_neo4j() -> None:
    proc = _run([str(NEO4J_HOME / "bin/neo4j"), "start"], timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"neo4j start failed: {proc.stderr or proc.stdout}")
    # wait for bolt
    for _ in range(60):
        check = _run(
            [
                str(NEO4J_HOME / "bin/cypher-shell"),
                "-a",
                "bolt://localhost:7687",
                "--non-interactive",
                "RETURN 1 AS ok;",
            ],
            timeout=30,
        )
        if check.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("neo4j started but bolt not ready")


def run_import(nodes_csv: Path, rels_csv: Path) -> dict[str, Any]:
    report_file = OUT / "import.report"
    cmd = [
        str(NEO4J_HOME / "bin/neo4j-admin"),
        "database",
        "import",
        "full",
        f"--nodes={nodes_csv}",
        f"--relationships={rels_csv}",
        "--id-type=string",
        "--overwrite-destination=true",
        "--skip-bad-relationships=true",
        "--bad-tolerance=1000000",
        f"--report-file={report_file}",
        "--multiline-fields=true",
        DATABASE,
    ]
    proc = _run(cmd, timeout=60 * 60 * 2)
    (OUT / "import.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (OUT / "import.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    return {
        "exit_code": proc.returncode,
        "cmd": cmd,
        "report_file": str(report_file),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def cypher_count(query: str) -> int | None:
    proc = _run(
        [
            str(NEO4J_HOME / "bin/cypher-shell"),
            "-a",
            "bolt://localhost:7687",
            "-d",
            DATABASE,
            "--non-interactive",
            "--format",
            "plain",
            query,
        ],
        timeout=120,
    )
    if proc.returncode != 0:
        return None
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    # plain format: header then value
    for ln in reversed(lines):
        if ln.isdigit():
            return int(ln)
        try:
            return int(ln.split()[-1])
        except Exception:
            continue
    return None


def parity(graph: dict[str, Any]) -> dict[str, Any]:
    json_nodes = len(graph.get("nodes") or [])
    json_edges = len(graph.get("edges") or [])
    by_type = Counter(str(n.get("type")) for n in graph.get("nodes") or [])
    neo_nodes = cypher_count("MATCH (n) RETURN count(n);")
    neo_edges = cypher_count("MATCH ()-[r]->() RETURN count(r);")
    neo_parts = cypher_count("MATCH (n:Part) RETURN count(n);")
    neo_docs = cypher_count("MATCH (n:Document) RETURN count(n);")
    neo_shards = cypher_count("MATCH (n:TensorShard) RETURN count(n);")
    neo_features = cypher_count("MATCH (n:Feature) RETURN count(n);")
    neo_real = cypher_count("MATCH (n:RealPart) RETURN count(n);")
    result = {
        "json_nodes": json_nodes,
        "json_edges": json_edges,
        "json_parts": by_type.get("Part", 0),
        "json_realparts": by_type.get("RealPart", 0),
        "json_documents": by_type.get("Document", 0),
        "json_tensorshards": by_type.get("TensorShard", 0),
        "json_features": by_type.get("Feature", 0),
        "neo_nodes": neo_nodes,
        "neo_edges": neo_edges,
        "neo_parts": neo_parts,
        "neo_realparts": neo_real,
        "neo_documents": neo_docs,
        "neo_tensorshards": neo_shards,
        "neo_features": neo_features,
    }
    result["nodes_match"] = neo_nodes == json_nodes
    result["edges_match"] = neo_edges == json_edges
    result["ok"] = bool(result["nodes_match"] and result["edges_match"])
    return result


def main() -> int:
    t0 = time.time()
    if not GRAPH.exists():
        print("missing graph")
        return 1
    if not NEO4J_HOME.exists():
        print(f"missing neo4j home: {NEO4J_HOME}")
        return 1

    print("loading graph…")
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    print(f"exporting CSV for {len(graph.get('nodes') or [])} nodes / {len(graph.get('edges') or [])} edges…")
    export_info = export_csv(graph, OUT)
    print(json.dumps(export_info, indent=2))

    print("stopping neo4j (if running)…")
    stop_neo4j()
    time.sleep(2)

    print("neo4j-admin import full…")
    import_info = run_import(Path(export_info["nodes_csv"]), Path(export_info["relationships_csv"]))
    print(json.dumps({k: import_info[k] for k in ("exit_code", "stderr_tail")}, indent=2))
    if import_info["exit_code"] != 0:
        PARITY.write_text(
            json.dumps({"ok": False, "import": import_info, "export": export_info}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1

    print("starting neo4j…")
    start_neo4j()
    print("parity counts…")
    # Ensure indexes helpful for id lookups
    _run(
        [
            str(NEO4J_HOME / "bin/cypher-shell"),
            "-a",
            "bolt://localhost:7687",
            "-d",
            DATABASE,
            "--non-interactive",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
        ],
        timeout=120,
    )
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export": export_info,
        "import": {"exit_code": import_info["exit_code"]},
        "parity": parity(graph),
        "elapsed_s": round(time.time() - t0, 2),
        "neo4j_status": neo4j_status(),
    }
    PARITY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["parity"].get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
