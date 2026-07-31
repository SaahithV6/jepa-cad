from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadflow.cli import main as cadflow_main
from cadflow.local_data_graph import build_local_data_graph, render_local_data_graph_report


def test_build_local_data_graph_scans_tree_and_summarizes_npz(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw_downloads" / "nasa3d"
    processed_dir = data_root / "processed" / "nasa3d"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    (data_root / "script.py").write_text('"""demo"""\nprint("hi")\n', encoding="utf-8")
    (raw_dir / "demo.stl").write_text("solid demo\nendsolid demo\n", encoding="utf-8")
    (raw_dir / "meta.json").write_text(json.dumps({"title": "demo", "source": "local"}), encoding="utf-8")
    np.savez(
        processed_dir / "sample_000.npz",
        points=np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        max_stress=np.array(0.75, dtype=np.float32),
    )
    (processed_dir / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "num_points": 2,
                "num_fields": 3,
                "format": "npz",
                "shards": ["sample_000.npz"],
                "sources": [{"kind": "raw", "source_path": "data/raw_downloads/nasa3d/demo.stl", "shard": "sample_000.npz", "format": "npz"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_local_data_graph(data_root)
    assert report.corpus_count >= 3
    assert report.file_count == 5
    assert report.sample_count == 1
    assert report.graph.metadata["analogue_count"] >= 2
    assert report.graph.metadata["analogue_kind_counts"]
    assert report.category_counts["RawAsset"] == 1
    assert report.category_counts["TensorShard"] == 1
    assert report.category_counts["DocumentAsset"] >= 1
    text = render_local_data_graph_report(report, as_json=False)
    assert "sample_count=1" in text
    assert any(node.type == "Sample" for node in report.graph.nodes)
    assert any(node.type == "TensorShard" for node in report.graph.nodes)
    assert any(node.type == "Analogue" for node in report.graph.nodes)
    assert any(edge.type == "HAS_ANALOGUE" for edge in report.graph.edges)
    assert any(edge.type == "ANALOGUE_OF" for edge in report.graph.edges)


def test_neo4j_import_local_data_dump_only_cli(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw_downloads" / "nasa3d"
    processed_dir = data_root / "processed" / "nasa3d"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (data_root / "script.py").write_text("print('hi')\n", encoding="utf-8")
    (raw_dir / "demo.stl").write_text("solid demo\nendsolid demo\n", encoding="utf-8")
    np.savez(
        processed_dir / "sample_000.npz",
        points=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        max_stress=np.array(0.25, dtype=np.float32),
    )
    (processed_dir / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "num_points": 1,
                "num_fields": 3,
                "format": "npz",
                "shards": ["sample_000.npz"],
                "sources": [{"kind": "raw", "source_path": "data/raw_downloads/nasa3d/demo.stl", "shard": "sample_000.npz", "format": "npz"}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cadflow_main([
        "neo4j-import-local-data",
        "--dump-only",
        "--json",
        "--data-root",
        str(data_root),
    ])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["sample_count"] == 1
    assert payload["graph"]["metadata"]["file_count"] == 4
