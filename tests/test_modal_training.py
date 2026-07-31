"""Tests for the Modal-backed cloud training path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import contextlib

from cadflow.cli import main
from cadflow.manifest import JobManifest
from cadflow import modal_training


class _FakeVolumeUpload:
    def __init__(self, recorder: list[tuple[str, str]]):
        self.recorder = recorder

    def __enter__(self) -> "_FakeVolumeUpload":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def put_directory(self, local: str, remote: str) -> None:
        self.recorder.append((local, remote))


class _FakeVolume:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def batch_upload(self) -> _FakeVolumeUpload:
        return _FakeVolumeUpload(self.uploads)


class _FakeRemoteResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.payload


def test_launch_modal_training_uploads_and_syncs(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "assembly.step").write_text("STEP DATA", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "part.stl").write_text("solid part\nendsolid part\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    manifest = JobManifest(
        name="space bracket",
        inputs={"project_root": str(project_root)},
        parameters={"family": "space"},
        tags=("space",),
        notes="reduce stress",
    )
    monkeypatch.setattr(
        modal_training,
        "intake_project",
        lambda *args, **kwargs: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(modal_training, "_utc_run_id", lambda prefix="modal": "modal_20260722_120000")
    preflight_report = SimpleNamespace(ok=True, checked_at="2026-07-22T12:00:00Z", to_dict=lambda: {"ok": True})
    preflight_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        modal_training,
        "run_pretraining_preflight",
        lambda **kwargs: preflight_calls.append(kwargs) or preflight_report,
    )
    remote = _FakeRemoteResult(
        {
            "ok": True,
            "decision": "promoted",
            "train_stdout": "trained\n",
            "train_stderr": "",
            "remote_run_root": "/vol/jepa/runs/modal_20260722_120000",
            "remote_registry_dir": "/vol/jepa/runs/modal_20260722_120000/registry",
            "remote_flywheel_path": "/vol/jepa/flywheel.jsonl",
        }
    )
    monkeypatch.setattr(
        modal_training,
        "_build_modal_runner",
        lambda raw_dirs, run_id, graph_path=None: (remote, ("/root/staged_inputs/modal_20260722_120000/raw_00",), None),
    )
    monkeypatch.setattr(modal_training.modal, "enable_output", lambda: contextlib.nullcontext())
    monkeypatch.setattr(modal_training.app, "run", lambda: contextlib.nullcontext())
    sync_calls: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        modal_training,
        "sync_latticezero_artifacts",
        lambda **kwargs: sync_calls.append(
            (kwargs["remote_registry_dir"], kwargs["remote_flywheel_path"], kwargs["local_root"])
        )
        or (str(tmp_path / "lz" / "flywheel" / "registry"), str(tmp_path / "lz" / "flywheel.jsonl")),
    )

    result = modal_training.launch_modal_training(
        project_root=project_root,
        goal="reduce stress in an existing spacecraft bracket",
        raw_dirs=[raw_dir],
        out_dir=out_dir,
        family="space",
        material="Al 6061-T6",
        sync_to_latticezero=True,
        latticezero_root=tmp_path / "lz",
    )

    assert result.ok is True
    assert result.run_id == "modal_20260722_120000"
    assert preflight_calls and preflight_calls[0]["family"] == "space"
    assert preflight_calls[0]["run_smoke"] is True
    assert remote.calls[0][1]["family"] == "space"
    assert remote.calls[0][1]["num_fields"] == 6
    assert remote.calls[0][0][1] == ("/root/staged_inputs/modal_20260722_120000/raw_00",)
    assert sync_calls == [
        (
            "/vol/jepa/runs/modal_20260722_120000/registry",
            "/vol/jepa/flywheel.jsonl",
            tmp_path / "lz",
        )
    ]
    assert (out_dir / "modal_training_result.json").exists()
    assert result.remote_result["decision"] == "promoted"
    assert result.cloud_plan["primary_provider"] == "Modal"


def test_modal_train_cli_dispatches_to_cloud_trainer(tmp_path: Path, monkeypatch, capsys) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "assembly.step").write_text("STEP DATA", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "part.stl").write_text("solid part\nendsolid part\n", encoding="utf-8")

    fake_result = SimpleNamespace(
        ok=True,
        remote_result={"train_stdout": "trained\n", "train_stderr": ""},
        to_dict=lambda: {"ok": True, "run_id": "modal_test", "remote_result": {"decision": "promoted"}},
    )
    monkeypatch.setattr("cadflow.modal_training.launch_modal_training", lambda **kwargs: fake_result)

    code = main(
        [
            "modal-train",
            "--project-root",
            str(project_root),
            "--goal",
            "reduce stress in an existing spacecraft bracket",
            "--family",
            "space",
            "--material",
            "Al 6061-T6",
            "--out-dir",
            str(tmp_path / "out"),
            "--raw-dir",
            str(raw_dir),
            "--json",
        ]
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert "modal_test" in captured
