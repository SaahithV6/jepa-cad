from __future__ import annotations

import json

from cadflow.graph_schema import (
    build_source_registry_graph,
    build_spaceflight_graph_schema,
    render_graph_document,
    render_graph_schema,
)


def test_graph_schema_includes_open_world_nozzle_extension() -> None:
    catalog = build_spaceflight_graph_schema()
    node_types = {node_type.name: node_type for node_type in catalog.node_types}

    assert node_types["Entity"].open_world is True
    assert node_types["RocketNozzle"].extends == "Part"
    nozzle_props = {prop.name for prop in node_types["RocketNozzle"].properties}
    assert {"chamber_pressure", "expansion_ratio", "contraction_ratio", "construction_ratio"}.issubset(nozzle_props)
    assert {"Tank", "Valve", "StructurePart", "Assembly", "TuningGuidance"}.issubset(node_types)

    edge_types = {edge_type.name: edge_type for edge_type in catalog.edge_types}
    assert "HAS_FEATURE" in edge_types
    assert "SOURCE_OF_TRUTH_FOR" in edge_types
    assert "VARIANT_OF" in edge_types
    assert "GUIDES" in edge_types


def test_graph_schema_renderer_and_json_roundtrip() -> None:
    catalog = build_spaceflight_graph_schema()
    text = render_graph_schema(catalog, as_json=False)
    data = json.loads(render_graph_schema(catalog, as_json=True))

    assert "RocketNozzle" in text
    assert data["name"] == "spaceflight-graph"
    assert data["version"] == "1.0"
    assert any(node_type["name"] == "RocketNozzle" for node_type in data["node_types"])


def test_source_registry_graph_exports_current_registry() -> None:
    graph = build_source_registry_graph()
    node_ids = {node.id for node in graph.nodes}
    node_types = {node.type for node in graph.nodes}
    edge_types = {edge.type for edge in graph.edges}

    assert graph.metadata["source_count"] >= 100
    assert graph.metadata["information_mode_counts"]["cad-model"] >= 1
    assert graph.metadata["training_eligible_count"] >= 1
    assert graph.metadata["status_counts"]["usable"] >= 1
    assert graph.metadata["source_kind_counts"]
    assert "corpus:source-registry" in node_ids
    assert "source:nasa_3d_resources" in node_ids
    assert "domain:space" in node_ids
    assert "Statistic" in node_types
    assert "Analogue" in node_types
    assert "PART_OF" in edge_types
    assert "CONTAINS" in edge_types
    assert "HAS_STATISTIC" in edge_types
    assert "HAS_ANALOGUE" in edge_types
    assert "ANALOGUE_OF" in edge_types
    assert "MENTIONS" in edge_types
    assert "RECOMMENDED_FOR" in edge_types
    assert any(node.type == "Source" for node in graph.nodes)
    assert any(node.type == "Feature" for node in graph.nodes)
    assert any(node.type == "Statistic" for node in graph.nodes)
    assert any(node.type == "Analogue" for node in graph.nodes)


def test_graph_export_cli_json(capsys) -> None:
    from cadflow.cli import main as cadflow_main

    exit_code = cadflow_main(["graph-export", "--json"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(captured)
    assert payload["name"] == "source-registry-graph"
    assert payload["metadata"]["source_count"] >= 100


def test_graph_schema_cli_json(capsys) -> None:
    from cadflow.cli import main as cadflow_main

    exit_code = cadflow_main(["graph-schema", "--json"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(captured)
    assert payload["name"] == "spaceflight-graph"
    assert any(node_type["name"] == "RocketNozzle" for node_type in payload["node_types"])
