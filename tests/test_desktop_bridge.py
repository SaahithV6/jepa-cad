"""Smoke tests for the LatticeZero desktop Python bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _bridge_requests(tmp_path: Path, requests: list[dict]) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    bridge = root / "desktop" / "python" / "bridge.py"
    env = {
        **os.environ,
        "LATTICEZERO_DATA_DIR": str(tmp_path / "appdata"),
        "LATTICEZERO_REPO_ROOT": str(root),
        "PYTHONPATH": str(root),
    }
    proc = subprocess.run(
        [sys.executable, str(bridge)],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=root,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def test_desktop_bridge_bootstrap_and_atlas(tmp_path: Path) -> None:
    messages = _bridge_requests(
        tmp_path,
        [
            {"id": 1, "method": "bootstrap", "params": {}},
            {"id": 2, "method": "latent_atlas", "params": {"seed": 4}},
        ],
    )
    replies = {message["id"]: message["result"] for message in messages if "id" in message}
    assert replies[1]["stats"]["modelVersion"].startswith("JEPA")
    assert set(replies[1]["doctor"]["probes"]) == {"openfoam", "fea", "mbd"}
    assert len(replies[2]["points"]) == 72


def test_desktop_bridge_verified_pipeline(tmp_path: Path) -> None:
    messages = _bridge_requests(
        tmp_path,
        [
            {
                "id": 1,
                "method": "run_pipeline",
                "params": {
                    "name": "Desktop smoke",
                    "geometry": {"kind": "box", "width": 2, "height": 1, "depth": 1},
                    "solver": "fea",
                    "mockCad": True,
                },
            }
        ],
    )
    result = next(message["result"] for message in messages if message.get("id") == 1)
    assert result["ok"] is True
    assert result["verification"]["passed"] is True
    assert result["metrics"]["watertight"] is True
    assert len(result["ghosts"]) == 3
    assert all(Path(path).exists() for path in result["artifacts"][:3])
