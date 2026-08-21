"""Neo4j export/import helpers for the spaceflight graph.

Neo4j cannot store nested property maps on nodes or relationships, so this module
preserves the full graph payload in a JSON string while also promoting simple
scalar fields to queryable top-level properties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .graph_schema import GraphDocument, GraphEdge, GraphNode, build_source_registry_graph


@dataclass(frozen=True, slots=True)
class Neo4jExportBundle:
    """Files generated for a Neo4j load."""

    graph_path: str
    cypher_path: str
    manifest_path: str
    log_path: str
    node_count: int
    edge_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Neo4jImportReport:
    """Result of loading a graph into a Neo4j database."""

    bundle: Neo4jExportBundle
    database: str
    cypher_shell: str
    exit_code: int
    stdout_path: str
    stderr_path: str
    node_count: int
    edge_count: int
    status: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bundle"] = self.bundle.to_dict()
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _is_scalar_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return all(_is_scalar(item) for item in value)


def _cypher_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _cypher_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _cypher_string(value)
    if isinstance(value, dict):
        inner = ", ".join(f"{key}: {_cypher_literal(val)}" for key, val in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cypher_literal(item) for item in value) + "]"
    return _cypher_string(str(value))


def _importable_properties(properties: dict[str, Any]) -> dict[str, Any]:
    importable: dict[str, Any] = {}
    for key, value in properties.items():
        if _is_scalar(value):
            importable[key] = value
        elif _is_scalar_sequence(value):
            importable[key] = list(value)
    return importable


def _node_payload(node: GraphNode) -> dict[str, Any]:
    payload = {
        "id": node.id,
        "type": node.type,
        "label": node.label,
        "properties_json": _json_text(node.properties),
    }
    payload.update(_importable_properties(node.properties))
    return payload


def _edge_payload(edge: GraphEdge) -> dict[str, Any]:
    payload = {
        "id": edge.id,
        "type": edge.type,
        "source": edge.source,
        "target": edge.target,
        "properties_json": _json_text(edge.properties),
    }
    payload.update(_importable_properties(edge.properties))
    return payload


def render_neo4j_cypher(graph: GraphDocument, *, database: str = "neo4j") -> str:
    """Render a Cypher load script for a graph document."""

    lines: list[str] = [
        f"// graph={graph.name}",
        f"// generated_at={graph.generated_at}",
        f"// database={database}",
        "",
        "CREATE CONSTRAINT source_node_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT entity_node_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
        "",
    ]

    # Nodes
    for node in graph.nodes:
        payload = _node_payload(node)
        label = node.type.replace("`", "")
        lines.append(f"MERGE (n:`{label}` {{id: {_cypher_literal(node.id)}}})")
        lines.append("SET n += " + _cypher_literal(payload) + ";")
        lines.append("")

    # Relationships
    for edge in graph.edges:
        payload = _edge_payload(edge)
        rel_type = edge.type.replace("`", "")
        lines.append(
            f"MATCH (s {{id: {_cypher_literal(edge.source)}}}), (t {{id: {_cypher_literal(edge.target)}}})"
        )
        lines.append(f"MERGE (s)-[r:`{rel_type}` {{id: {_cypher_literal(edge.id)}}}]->(t)")
        lines.append("SET r += " + _cypher_literal(payload) + ";")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_neo4j_bundle(
    graph: GraphDocument,
    out_dir: str | Path,
    *,
    database: str = "neo4j",
) -> Neo4jExportBundle:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cypher_path = out_path / "spaceflight.cypher"
    graph_path = out_path / "spaceflight-graph.json"
    manifest_path = out_path / "neo4j-import-manifest.json"
    log_path = out_path / "neo4j-import.log"

    cypher_path.write_text(render_neo4j_cypher(graph, database=database), encoding="utf-8")
    graph_path.write_text(_json_text(graph.to_dict()), encoding="utf-8")
    bundle = Neo4jExportBundle(
        graph_path=str(graph_path),
        cypher_path=str(cypher_path),
        manifest_path=str(manifest_path),
        log_path=str(log_path),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        created_at=_utc_now(),
    )
    manifest_path.write_text(_json_text(bundle.to_dict()), encoding="utf-8")
    return bundle


def import_graph_to_neo4j(
    graph: GraphDocument | None = None,
    *,
    out_dir: str | Path = "artifacts/neo4j-import",
    database: str = "neo4j",
    neo4j_home: str | Path | None = None,
    java_home: str | Path | None = None,
    required: bool = True,
) -> Neo4jImportReport:
    """Write a Cypher bundle and load it into the local Neo4j instance.

    With ``required=False`` a missing Neo4j is reported rather than raised. The
    Cypher bundle is written either way, so the import can be replayed later on
    a machine that has one. That matters for running corpus sweeps in a
    throwaway environment -- a Colab runtime has no Neo4j, and the graph export
    is a side artifact of a sweep, not the thing the sweep is for.
    """

    graph = graph or build_source_registry_graph()
    bundle = write_neo4j_bundle(graph, out_dir, database=database)

    neo4j_home_path = Path(neo4j_home) if neo4j_home is not None else Path.home() / ".local/neo4j/current"
    cypher_shell = neo4j_home_path / "bin" / "cypher-shell"
    if not cypher_shell.exists():
        if required:
            raise FileNotFoundError(f"cypher-shell not found: {cypher_shell}")
        return Neo4jImportReport(
            bundle=bundle,
            database=database,
            cypher_shell=str(cypher_shell),
            exit_code=0,
            stdout_path="",
            stderr_path="",
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            status="skipped_no_neo4j",
            notes=(
                f"cypher-shell not found at {cypher_shell}; the Cypher bundle "
                f"was written to {bundle.cypher_path} and can be imported later.",
            ),
        )

    env = os.environ.copy()
    if java_home is not None:
        env["JAVA_HOME"] = str(java_home)
    else:
        env.setdefault("JAVA_HOME", str(Path.home() / ".local/java/temurin-21"))
    env.setdefault("NEO4J_HOME", str(neo4j_home_path))

    stdout_path = Path(bundle.log_path).with_suffix(".stdout.log")
    stderr_path = Path(bundle.log_path).with_suffix(".stderr.log")
    cmd = [
        str(cypher_shell),
        "-a",
        "bolt://localhost:7687",
        "-d",
        database,
        "--non-interactive",
        "--format",
        "plain",
        "-f",
        bundle.cypher_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    notes = [f"loaded from {bundle.cypher_path}"]
    if proc.returncode != 0:
        status = "failed"
        notes.append("cypher-shell returned non-zero exit code")
    else:
        status = "loaded"

    report = Neo4jImportReport(
        bundle=bundle,
        database=database,
        cypher_shell=str(cypher_shell),
        exit_code=proc.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        node_count=bundle.node_count,
        edge_count=bundle.edge_count,
        status=status,
        notes=tuple(notes),
    )
    Path(bundle.log_path).write_text(_json_text(report.to_dict()), encoding="utf-8")
    return report


def render_neo4j_import_report(report: Neo4jImportReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2)
    lines = [
        f"status={report.status}",
        f"database={report.database}",
        f"exit_code={report.exit_code}",
        f"node_count={report.node_count}",
        f"edge_count={report.edge_count}",
        f"cypher_shell={report.cypher_shell}",
        f"stdout_path={report.stdout_path}",
        f"stderr_path={report.stderr_path}",
        f"bundle={report.bundle.cypher_path}",
        "notes:",
    ]
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
