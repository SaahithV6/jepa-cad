"""Tests for the autonomous maintenance supervisor."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
import sys

from cadflow.autopilot import run_autopilot
from cadflow.flywheel_loop import FlywheelLoopResult
from cadflow.promotion import PromotionResult
from data.ingest import IngestionResult
from eval.probe import ProbeResult


def _ingestion_result(tmp_path: Path) -> IngestionResult:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    shard = staging / "sample.npz"
    shard.write_bytes(b"npz")
    return IngestionResult(
        ingested=1,
        skipped=0,
        shard_paths=(str(shard),),
        manifest_path=str(staging / "manifest.json"),
        reasons=(),
        sources=(),
    )


def _promotion_result(tmp_path: Path) -> PromotionResult:
    curated = tmp_path / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    shard = curated / "curated_000000_deadbeef.npz"
    shard.write_bytes(b"npz")
    return PromotionResult(
        promoted=1,
        skipped=0,
        shard_paths=(str(shard),),
        manifest_path=str(curated / "curated_manifest.json"),
        reasons=(),
    )


def _probe_result(checkpoint: str | Path, score: float) -> ProbeResult:
    return ProbeResult(
        checkpoint=str(checkpoint),
        data_source="real",
        score_name="val_mse",
        score=score,
        train_mse=score + 0.05,
        val_mse=score,
        train_samples=8,
        val_samples=2,
        epochs=1,
        seed=42,
    )


def _loop_result(tmp_path: Path) -> FlywheelLoopResult:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    latest.write_text("candidate", encoding="utf-8")
    return FlywheelLoopResult(
        cycle_id="20260713_130000",
        workdir=str(tmp_path),
        staging_dir=str(tmp_path / "staging"),
        curated_dir=str(tmp_path / "curated"),
        checkpoint_dir=str(checkpoint_dir),
        baseline_checkpoint=None,
        dataset_dir=str(tmp_path / "curated"),
        ingestion=_ingestion_result(tmp_path),
        promotion=_promotion_result(tmp_path),
        train_returncode=0,
        train_command=("python", "train.py"),
        train_stdout="trained\n",
        train_stderr="",
        candidate_probe=_probe_result(latest, 0.4),
        baseline_probe=None,
        promoted_checkpoint=str(tmp_path / "registry" / "best.pt"),
        decision="promoted",
        summary_path=str(tmp_path / "flywheel_cycle.json"),
    )


def test_run_autopilot_runs_pytest_then_loop_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "autopilot"

    monkeypatch.setattr("cadflow.autopilot._check_imports", lambda required: ())

    def fake_run(cmd, cwd, capture_output, text, check):
        if cmd[:3] == ["git", "rev-parse", "--short"]:
            return CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return CompletedProcess(cmd, 0, stdout=" M cadflow/cli.py\n", stderr="")
        if cmd[:3] == [sys.executable, "-m", "pytest"]:
            return CompletedProcess(cmd, 0, stdout="56 passed\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("cadflow.autopilot.subprocess.run", fake_run)
    monkeypatch.setattr("cadflow.autopilot.run_flywheel_loop", lambda *args, **kwargs: _loop_result(tmp_path))

    result = run_autopilot([tmp_path / "raw"], out_dir, flywheel_path=tmp_path / "fw.jsonl")

    assert result.ok
    assert result.decision == "promoted"
    assert result.pytest_returncode == 0
    assert result.loop is not None
    assert result.loop.decision == "promoted"
    assert Path(result.summary_path).exists()
    report = Path(result.summary_path).read_text(encoding="utf-8")
    assert '"git_commit": "abc123"' in report
    assert '"pytest_returncode": 0' in report


def test_run_autopilot_repairs_env_and_skips_loop_when_imports_missing(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "autopilot"
    calls: list[list[str]] = []

    monkeypatch.setattr("cadflow.autopilot._check_imports", lambda required: ("cadquery",))

    def fake_run(cmd, cwd, capture_output, text, check):
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "--short"]:
            return CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            return CompletedProcess(cmd, 0, stdout="installed\n", stderr="")
        if cmd[:3] == [sys.executable, "-m", "pytest"]:
            return CompletedProcess(cmd, 0, stdout="56 passed\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("cadflow.autopilot.subprocess.run", fake_run)

    result = run_autopilot([], out_dir, repair_env=True)

    assert result.env_repaired is True
    assert result.missing_imports == ("cadquery",)
    assert result.loop is None
    assert result.decision == "env_incomplete"
    assert result.loop_skipped_reason is not None
    assert Path(result.summary_path).exists()
    assert any(cmd[:4] == ["/home/best/jepa-cad/.venv/bin/python", "-m", "pip", "install"] or cmd[1:4] == ["-m", "pip", "install"] for cmd in calls)
