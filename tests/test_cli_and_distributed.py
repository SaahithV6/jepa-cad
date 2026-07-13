"""CLI and distributed helper smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from cadflow.cli import main as cli_main
from utils.distributed import DistInfo, all_reduce_mean, init_distributed, wrap_ddp


def _make_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


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


def test_cli_module_execution_has_no_runpy_warning(tmp_path: Path) -> None:
    manifest = {
        "name": "cli-box-module",
        "inputs": {"geometry": {"kind": "box", "width": 1.0, "height": 1.0, "depth": 1.0}},
        "parameters": {"solver": "fea", "objective": 0.5},
        "tags": ["cli"],
    }
    man_path = tmp_path / "job.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cadflow.cli",
            "run",
            "--manifest",
            str(man_path),
            "--mock-cad",
            "--workdir",
            str(tmp_path / "work"),
            "--outdir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RuntimeWarning" not in proc.stderr


def test_distributed_helpers_single_process() -> None:
    info = init_distributed("cpu")
    assert isinstance(info, DistInfo)
    assert info.enabled is False
    assert info.is_primary is True
    model = wrap_ddp(nn.Linear(3, 2), info)
    assert isinstance(model, nn.Linear)
    t = torch.tensor(4.0)
    assert float(all_reduce_mean(t, info)) == 4.0


def test_cli_doctor_reports_native_ready_with_fake_tools(tmp_path: Path, monkeypatch, capsys) -> None:
    bin_dir = tmp_path / "bin"
    lib_dir = tmp_path / "lib"
    bin_dir.mkdir()
    lib_dir.mkdir()
    _make_executable(bin_dir / "simpleFoam", "#!/usr/bin/env bash\necho simpleFoam\n")
    _make_executable(bin_dir / "ccx", "#!/usr/bin/env bash\necho ccx\n")

    monkeypatch.setenv("CADFLOW_SOLVER_BIN_DIRS", str(bin_dir))
    monkeypatch.setenv("CADFLOW_SOLVER_LIB_DIRS", str(lib_dir))

    code = cli_main(["doctor", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert '"native_ready": true' in captured.out
    assert str(bin_dir) in captured.out
