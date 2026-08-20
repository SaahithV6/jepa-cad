from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cadflow.associate_training_data import associate_parts
from data.graph_dataset import GRAPH_METADATA_DIM, GraphBackedCADDataset
from data.transforms import MaskingConfig, sample_jepa_masks
from models.jepa import JEPAModel


def _write_mesh_stl(path: Path) -> None:
    # Minimal ASCII STL triangle
    path.write_text(
        "\n".join(
            [
                "solid demo",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                "endsolid demo",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_graph_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    shard_path = processed / "sample_000.npz"
    np.savez_compressed(
        shard_path,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.25, 0.75]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
        max_stress=np.array(0.9, dtype=np.float32),
    )
    graph_path = tmp_path / "spaceflight-graph.json"
    graph_payload = {
        "name": "mini-graph",
        "generated_at": "2026-07-21T00:00:00Z",
        "metadata": {
            "data_root": str(data_root),
        },
        "nodes": [
            {
                "id": "analogue:sample-000",
                "type": "Analogue",
                "label": "sample_000 analogue",
                "properties": {
                    "path": "processed/sample_000.npz",
                    "source_path": "processed/sample_000.npz",
                    "size_bytes": shard_path.stat().st_size,
                    "index": 0,
                    "summary": {"kind": "numeric", "point_count": 3},
                    "feature_summary": {"point_count": 3, "field_count": 3},
                    "parametric_summary": {"axis_extents": [1.0, 1.0, 1.0]},
                    "physical_summary": {"point_bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
                },
            },
            {
                "id": "shard:sample-000",
                "type": "TensorShard",
                "label": "sample_000.npz",
                "properties": {
                    "path": "processed/sample_000.npz",
                    "size_bytes": shard_path.stat().st_size,
                    "index": 0,
                    "source_path": "raw_downloads/nasa3d/demo.stl",
                },
            }
        ],
        "edges": [
            {"id": "edge:shard:sample-000:analogue", "type": "HAS_ANALOGUE", "source": "shard:sample-000", "target": "analogue:sample-000", "properties": {}},
        ],
    }
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    return graph_path, data_root


def test_graph_backed_dataset_loads_samples_and_metadata(tmp_path: Path) -> None:
    graph_path, data_root = _write_graph_fixture(tmp_path)
    dataset = GraphBackedCADDataset(graph_path, data_root=data_root, num_points=8, num_fields=3)

    sample = dataset[0]
    assert sample["points"].shape == (8, 3)
    assert sample["fields"].shape == (8, 3)
    assert sample["graph_metadata"].shape == (GRAPH_METADATA_DIM,)
    assert dataset.records[0].node_type in {"Analogue", "TensorShard"}
    # legacy slot 5 = analogue_flag
    expect_analogue = 1.0 if dataset.records[0].node_type == "Analogue" else 0.0
    assert sample["graph_metadata"][5].item() == expect_analogue
    assert sample["max_stress"].ndim == 0
    # process provenance slots are the trailing PROCESS_META_DIM entries
    assert sample["graph_metadata"].shape[0] >= 12
    assert sample["graph_metadata"][-12:].numel() == 12


def test_graph_metadata_conditions_jepa_forward(tmp_path: Path) -> None:
    graph_path, data_root = _write_graph_fixture(tmp_path)
    dataset = GraphBackedCADDataset(graph_path, data_root=data_root, num_points=32, num_fields=3)
    sample = dataset[0]
    points = sample["points"].unsqueeze(0)
    fields = sample["fields"].unsqueeze(0)
    metadata = sample["graph_metadata"].unsqueeze(0)
    masks = sample_jepa_masks(points, MaskingConfig(grid_size=(2, 2, 2), num_target_blocks=2), generator=torch.Generator().manual_seed(0))

    cfg = {
        "data": {"num_fields": 3, "graph_metadata_dim": GRAPH_METADATA_DIM},
        "model": {
            "embed_dim": 32,
            "encoder": {"num_layers": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0, "use_field_features": True},
            "predictor": {"num_layers": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0},
            "ema_decay": 0.99,
            "loss_type": "smooth_l1",
        },
        "masking": {"grid_size": [2, 2, 2]},
    }
    model = JEPAModel.from_config(cfg)
    out = model(points, fields, masks["context_mask"], masks["target_masks"], masks["target_block_ids"], graph_metadata=metadata)
    assert out["loss"].ndim == 0
    assert out["predicted"].shape == out["target"].shape


def test_associate_training_data_wires_params_material_physics(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    stl = parts_dir / "nose_cone_00000.stl"
    _write_mesh_stl(stl)

    graph = {
        "name": "assoc-test",
        "nodes": [
            {
                "id": "part:rocket:nose_cone_00000",
                "type": "Part",
                "label": "nose_cone_00000",
                "physics_verified": True,
                "simulation_results_fea": {
                    "solver": "calculix",
                    "status": "ok",
                    "max_stress_mpa": 123.4,
                    "mean_stress_mpa": 40.0,
                },
                "properties": {
                    "name": "nose_cone_00000",
                    "family": "nose_cone",
                    "part_class": "nose_cone",
                    "geometry_ref": str(stl),
                    "stl": str(stl),
                    "params": {"diameter_mm": 35.0, "length_mm": 60.0, "shape": "ogive", "power": 0.5},
                    "extents_mm": [35.0, 35.0, 60.0],
                    "faces": 100,
                    "material_id": "al-6061-t6",
                    "material_name": "Al 6061-T6",
                    "material_category": "aluminum",
                    "tags": ["openrocket"],
                },
            }
        ],
        "edges": [],
    }
    stats = associate_parts(graph)
    assert stats["samples_created"] == 1
    assert stats["represents_edges"] == 1
    assert stats["physics_edges"] == 1
    assert stats["geometry_edges"] == 1
    assert stats["fea_overlays"] == 1

    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    dataset = GraphBackedCADDataset(graph_path, num_points=16, num_fields=3)
    assert len(dataset) >= 1
    sample = dataset[0]
    assert sample["graph_metadata"].shape == (GRAPH_METADATA_DIM,)
    assoc = dataset._walk_associations(dataset.records[0].node_id)
    assert assoc["family"] == "nose_cone"
    assert assoc["params"].get("diameter_mm") == 35.0
    assert assoc["material"].get("density_kg_m3") == 2700
    assert abs(assoc["physics"].get("max_stress_mpa", 0.0) - 123.4) < 1e-3
    assert abs(sample["max_stress"].item() - 123.4) < 1e-3
