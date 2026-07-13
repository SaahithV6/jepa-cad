"""CLI and distributed helper smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from cadflow.cli import main as cli_main
from utils.distributed import DistInfo, all_reduce_mean, init_distributed, wrap_ddp
import torch
import torch.nn as nn


def test_cli_run_and_promote(tmp_path: Path) -> None:
    manifest = {
        "name": "cli-box",
        "inputs": {"geometry": {"kind": "box", "width": 1.5, "height": 1.0, "depth": 0.5}},
        "parameters": {"solver": "fea", "objective": 0.7, "max_stress_mpa": 0.7},
        "tags": ["cli"],
    }
    man_path = tmp_path / "job.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    flywheel = tmp_path / "fw.jsonl"
    outdir = tmp_path / "out"
    code = cli_main(
        [
            "run",
            "--manifest",
            str(man_path),
            "--workdir",
            str(tmp_path / "work"),
            "--outdir",
            str(outdir),
            "--flywheel",
            str(flywheel),
            "--mock-cad",
            "--promote-to",
            str(tmp_path / "curated"),
            "--promote-limit",
            "3",
        ]
    )
    assert code == 0
    assert (outdir / "result.json").exists()
    assert flywheel.exists()


def test_distributed_helpers_single_process() -> None:
    info = init_distributed("cpu")
    assert isinstance(info, DistInfo)
    assert info.enabled is False
    assert info.is_primary is True
    model = wrap_ddp(nn.Linear(3, 2), info)
    assert isinstance(model, nn.Linear)
    t = torch.tensor(4.0)
    assert float(all_reduce_mean(t, info)) == 4.0
