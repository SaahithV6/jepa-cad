from __future__ import annotations

import json
from pathlib import Path

from cadflow.cli import main as cadflow_main
from cadflow.graph_schema import GraphDocument, GraphEdge, GraphNode, build_source_registry_graph
from cadflow.neo4j_store import import_graph_to_neo4j, render_neo4j_cypher, write_neo4j_bundle


def test_render_neo4j_cypher_handles_nested_properties(tmp_path: Path) -> None:
    graph = GraphDocument(
        name="mini",
        generated_at="2026-07-22T00:00:00+00:00",
        nodes=(
            GraphNode(
                id="source:one",
                type="Source",
                label="One",
                properties={"url": "https://example.com", "domain": "space", "meta": {"a": 1}, "tags": ["x", "y"]},
            ),
            GraphNode(
                id="part:two",
                type="Part",
                label="Two",
                properties={"name": "Two", "part_class": "nozzle", "dims": [1, 2, 3]},
            ),
        ),
        edges=(
            GraphEdge(
                id="edge:1",
                type="PART_OF",
                source="source:one",
                target="part:two",
                properties={"weight": 0.5, "meta": {"reason": "demo"}},
            ),
        ),
    )

    cypher = render_neo4j_cypher(graph)
    assert "MERGE (n:`Source`" in cypher
    assert "MERGE (n:`Part`" in cypher
    assert "properties_json" in cypher
    assert "payload_json" not in cypher  # renderer uses properties_json for graph payloads
    assert "meta" in cypher

    bundle = write_neo4j_bundle(graph, tmp_path)
    assert Path(bundle.cypher_path).exists()
    assert Path(bundle.graph_path).exists()
    manifest = json.loads(Path(bundle.manifest_path).read_text())
    assert manifest["node_count"] == 2
    assert manifest["edge_count"] == 1


def test_neo4j_import_cli_bundle_only(capsys) -> None:
    exit_code = cadflow_main(["neo4j-import", "--json", "--out-dir", "artifacts/test-neo4j-import"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["bundle"]["node_count"] == len(build_source_registry_graph().nodes)
    assert payload["bundle"]["edge_count"] == len(build_source_registry_graph().edges)
