from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadflow.cli import main as cadflow_main
from cadflow.corpus_graph import build_processed_corpus_graph, render_corpus_graph_report


def test_build_processed_corpus_graph_summarizes_shards(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    np.savez(
        processed / "raw_000000_demo.npz",
        points=np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        max_stress=np.array(0.75, dtype=np.float32),
    )
    manifest = {
        "num_points": 2,
        "num_fields": 3,
        "format": "npz",
        "shards": ["raw_000000_demo.npz"],
        "sources": [
            {
                "kind": "raw",
                "source_path": "raw/demo.stl",
                "shard": "raw_000000_demo.npz",
                "format": "npz",
            }
        ],
    }
    manifest_path = tmp_path / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_processed_corpus_graph(manifest_path, processed, include_raw_assets=False)
    assert report.shard_count == 1
    assert report.sample_count == 1
    assert report.raw_asset_count == 0
    assert report.graph.metadata["source_count"] == 1
    assert report.graph.metadata["shard_count"] == 1
    assert report.graph.metadata["analogue_count"] >= 2
    assert report.graph.metadata["analogue_kind_counts"]
    sample_nodes = [node for node in report.graph.nodes if node.type == "Sample"]
    assert len(sample_nodes) == 1
    assert sample_nodes[0].properties["num_points"] == 2
    assert sample_nodes[0].properties["max_stress"] == 0.75
    assert any(node.type == "Analogue" for node in report.graph.nodes)
    assert any(edge.type == "HAS_ANALOGUE" for edge in report.graph.edges)
    assert any(edge.type == "ANALOGUE_OF" for edge in report.graph.edges)
    text = render_corpus_graph_report(report, as_json=False)
    assert "sample_count=1" in text


def test_neo4j_import_corpus_dump_only_cli(tmp_path: Path, capsys) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    np.savez(
        processed / "raw_000000_demo.npz",
        points=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        max_stress=np.array(0.25, dtype=np.float32),
    )
    manifest = {
        "num_points": 1,
        "num_fields": 3,
        "format": "npz",
        "shards": ["raw_000000_demo.npz"],
        "sources": [],
    }
    manifest_path = tmp_path / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = cadflow_main(
        [
            "neo4j-import-corpus",
            "--dump-only",
            "--json",
            "--manifest",
            str(manifest_path),
            "--processed-dir",
            str(processed),
            "--no-raw-assets",
        ]
    )
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["sample_count"] == 1
    assert payload["graph"]["metadata"]["shard_count"] == 1
