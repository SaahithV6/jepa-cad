"""Tests for the repeated loop controller."""

from __future__ import annotations

from pathlib import Path

from cadflow.loop_controller import run_loop_controller
from cadflow.promotion import PromotionResult
from data.ingest import IngestionResult
from eval.probe import ProbeResult
from cadflow.flywheel_loop import FlywheelLoopResult


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


def _loop_result(tmp_path: Path, cycle_id: str, score: float = 0.4) -> FlywheelLoopResult:
    workdir = tmp_path / cycle_id
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    latest.write_text("candidate", encoding="utf-8")
    return FlywheelLoopResult(
        cycle_id=cycle_id,
        workdir=str(workdir),
        staging_dir=str(workdir / "staging"),
        curated_dir=str(workdir / "curated"),
        checkpoint_dir=str(checkpoint_dir),
        baseline_checkpoint=None,
        dataset_dir=str(workdir / "curated"),
        ingestion=_ingestion_result(workdir),
        promotion=_promotion_result(workdir),
        train_returncode=0,
        train_command=("python", "train.py"),
        train_stdout="trained\n",
        train_stderr="",
        candidate_probe=_probe_result(latest, score),
        baseline_probe=None,
        promoted_checkpoint=str(workdir / "registry" / "best.pt"),
        decision="promoted",
        summary_path=str(workdir / "flywheel_cycle.json"),
    )


def test_run_loop_controller_repeats_and_persists_history(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "loop"
    calls: list[str] = []

    def fake_loop(raw_dirs, out_dir, **kwargs):
        cycle_id = f"cycle-{len(calls) + 1}"
        calls.append(cycle_id)
        return _loop_result(tmp_path, cycle_id)

    monkeypatch.setattr("cadflow.loop_controller.run_flywheel_loop", fake_loop)

    result = run_loop_controller([tmp_path / "raw"], out_dir, repeat=2, interval_seconds=0)

    assert result.ok
    assert result.iterations == 2
    assert result.stop_reason == "repeat-exhausted"
    assert len(result.results) == 2
    assert calls == ["cycle-1", "cycle-2"]
    assert Path(result.history_path).exists()
    assert Path(result.latest_path).exists()
    history = Path(result.history_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 2
    assert '"cycle_id": "cycle-2"' in Path(result.latest_path).read_text(encoding="utf-8")
    assert (out_dir / "cycles" / "0001_cycle-1.json").exists()
    assert (out_dir / "cycles" / "0002_cycle-2.json").exists()


def test_run_loop_controller_honors_preexisting_stop_file(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "loop"
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop", encoding="utf-8")

    def fail_loop(*args, **kwargs):
        raise AssertionError("loop should not run when the stop file already exists")

    monkeypatch.setattr("cadflow.loop_controller.run_flywheel_loop", fail_loop)

    result = run_loop_controller([tmp_path / "raw"], out_dir, repeat=0, interval_seconds=0, stop_file=stop_file)

    assert result.ok
    assert result.iterations == 0
    assert result.stop_reason == "stop-file"
    assert result.results == ()
    assert Path(result.history_path).exists() is False
    assert Path(result.latest_path).exists() is False
