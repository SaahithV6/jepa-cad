from __future__ import annotations

import json
from pathlib import Path

from cadflow.cli import main


def test_project_cli_intake_and_cloud_plan(tmp_path: Path, capsys) -> None:
    project = tmp_path / "existing"
    project.mkdir()
    (project / "assembly.step").write_text("STEP", encoding="utf-8")
    out_dir = tmp_path / "out"

    assert main(
        [
            "project",
            "--project-root",
            str(project),
            "--goal",
            "reduce stress in existing spacecraft bracket",
            "--family",
            "space",
            "--material",
            "Al 6061-T6",
            "--out-dir",
            str(out_dir),
            "--json",
        ]
    ) == 0


    manifest_path = out_dir / "project_manifest.json"
    assert manifest_path.exists()

    capsys.readouterr()
    assert main(["cloud-plan", "--manifest", str(manifest_path), "--json"]) == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["primary_provider"] == "Modal"
    assert payload["family"] == "space"
