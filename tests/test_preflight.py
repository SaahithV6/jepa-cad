"""Tests for the executable pre-training preflight."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadflow.cli import main
from cadflow.e2e import EndToEndResult
from cadflow.preflight import run_pretraining_preflight
from data.ingest import IngestionResult


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "assembly.step").write_text("STEP DATA", encoding="utf-8")
    return project


def _make_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savez_compressed(
        data_root / "sample.npz",
        points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )
    return data_root


def test_preflight_report_without_smoke(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    data_root = _make_data_root(tmp_path)
    out_dir = tmp_path / "out"

    report = run_pretraining_preflight(
        project_root=project,
        goal="reduce stress in an existing spacecraft bracket",
        family="space",
        material="Al 6061-T6",
        out_dir=out_dir,
        data_root=data_root,
        run_smoke=False,
    )

    assert report.ok is True
    assert report.cloud_plan is not None
    assert report.cloud_plan["family"] == "space"
    assert report.sections[0].name == "training-goal"
    assert report.sections[0].passed is True
    assert report.sections[1].passed is True
    assert report.sections[2].passed is True
    assert report.sections[3].passed is True
    assert report.sections[4].required is False


def test_preflight_cli_runs_smoke_and_graph_backed_dataset(tmp_path: Path, monkeypatch, capsys) -> None:
    project = _make_project(tmp_path)
    data_root = _make_data_root(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "part.stl").write_text(
        "\n".join(
            [
                "solid part",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                "endsolid part",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    smoke_dataset = out_dir / "smoke_dataset"
    smoke_dataset.mkdir(parents=True)
    shard = smoke_dataset / "raw_000000_part.npz"
    np.savez_compressed(
        shard,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        fields=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )
    ingestion = IngestionResult(
        ingested=1,
        skipped=0,
        shard_paths=(str(shard),),
        manifest_path=str(smoke_dataset / "ingestion_manifest.json"),
        reasons=(),
        sources=(({"kind": "raw", "source_path": str(raw_dir / "part.stl"), "shard": shard.name, "format": "npz"}),),
    )
    (smoke_dataset / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "num_points": 2,
                "num_fields": 3,
                "format": "npz",
                "shards": [shard.name],
                "sources": [
                    {
                        "kind": "raw",
                        "source_path": str(raw_dir / "part.stl"),
                        "shard": shard.name,
                        "format": "npz",
                    }
                ],
                "raw_dirs": [str(raw_dir)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fake_result = EndToEndResult(
        ingestion=ingestion,
        train_returncode=0,
        train_command=("python", "train.py"),
        train_stdout="trained\n",
        train_stderr="",
    )

    monkeypatch.setattr("cadflow.preflight.run_end_to_end", lambda *args, **kwargs: fake_result)

    code = main(
        [
            "preflight",
            "--project-root",
            str(project),
            "--goal",
            "reduce stress in an existing spacecraft bracket",
            "--family",
            "space",
            "--material",
            "Al 6061-T6",
            "--out-dir",
            str(out_dir),
            "--data-root",
            str(data_root),
            "--raw-dir",
            str(raw_dir),
            "--num-points",
            "2",
            "--num-fields",
            "3",
            "--run-smoke",
            "--json",
        ]
    )
    captured = capsys.readouterr().out
    assert code == 0
    payload = json.loads(captured)
    assert payload["ok"] is True
    assert payload["smoke_result"]["ok"] is True
    assert payload["graph_dataset"]["records"] >= 1
    assert payload["graph_dataset"]["points_shape"] == [2, 3]
    assert payload["sections"][4]["passed"] is True
    assert payload["sections"][5]["passed"] is True
