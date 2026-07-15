"""Tests for the recursive verified-data flywheel loop."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from cadflow.flywheel_loop import run_flywheel_loop
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


def _promotion_result(tmp_path: Path, promoted: int) -> PromotionResult:
    curated = tmp_path / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    return PromotionResult(
        promoted=promoted,
        skipped=0,
        shard_paths=(str(curated / "curated_000000_deadbeef.npz"),) if promoted else (),
        manifest_path=str(curated / "curated_manifest.json"),
        reasons=(),
    )


def _probe_result(checkpoint: str | Path, score: float) -> ProbeResult:
    checkpoint = str(checkpoint)
    return ProbeResult(
        checkpoint=checkpoint,
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


def test_run_flywheel_loop_promotes_when_candidate_beats_baseline(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "loop"
    flywheel_path = tmp_path / "flywheel.jsonl"
    baseline = out_dir / "registry" / "best.pt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("baseline", encoding="utf-8")

    monkeypatch.setattr("cadflow.flywheel_loop._cycle_id", lambda: "20260713_120000")
    monkeypatch.setattr("cadflow.flywheel_loop.ingest_sources", lambda *args, **kwargs: _ingestion_result(tmp_path))
    monkeypatch.setattr(
        "cadflow.flywheel_loop.promote_verified_to_dataset",
        lambda *args, **kwargs: _promotion_result(tmp_path, promoted=1),
    )

    def fake_run(cmd, cwd, capture_output, text, check):
        checkpoint_dir = None
        for item in cmd:
            if isinstance(item, str) and item.startswith("checkpoint.checkpoint_dir="):
                checkpoint_dir = Path(item.split("=", 1)[1])
                break
        assert checkpoint_dir is not None
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "latest.pt").write_text("candidate", encoding="utf-8")
        return CompletedProcess(cmd, 0, stdout="trained\n", stderr="")

    monkeypatch.setattr("cadflow.flywheel_loop.subprocess.run", fake_run)

    def fake_probe(cfg, checkpoint, data_source, device, verbose=False):
        if Path(checkpoint).name == "best.pt":
            return _probe_result(checkpoint, 0.8)
        return _probe_result(checkpoint, 0.5)

    monkeypatch.setattr("cadflow.flywheel_loop.probe_checkpoint", fake_probe)

    result = run_flywheel_loop([tmp_path / "raw"], out_dir, flywheel_path=flywheel_path, max_steps=1)

    assert result.decision == "promoted"
    assert result.train_returncode == 0
    assert result.candidate_probe is not None and result.baseline_probe is not None
    assert result.candidate_probe.score < result.baseline_probe.score
    assert result.promoted_checkpoint is not None
    assert Path(result.promoted_checkpoint).exists()
    assert Path(result.summary_path).exists()
    assert (out_dir / "registry" / "checkpoints" / "20260713_120000.pt").exists()
    assert (out_dir / "registry" / "best.json").exists()
    history = (out_dir / "registry" / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 1
    assert '"decision": "promoted"' in history[0]


def test_run_flywheel_loop_rejects_when_candidate_is_worse(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "loop"
    flywheel_path = tmp_path / "flywheel.jsonl"
    baseline = out_dir / "registry" / "best.pt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("baseline", encoding="utf-8")

    monkeypatch.setattr("cadflow.flywheel_loop._cycle_id", lambda: "20260713_120001")
    monkeypatch.setattr("cadflow.flywheel_loop.ingest_sources", lambda *args, **kwargs: _ingestion_result(tmp_path))
    monkeypatch.setattr(
        "cadflow.flywheel_loop.promote_verified_to_dataset",
        lambda *args, **kwargs: _promotion_result(tmp_path, promoted=1),
    )

    def fake_run(cmd, cwd, capture_output, text, check):
        checkpoint_dir = None
        for item in cmd:
            if isinstance(item, str) and item.startswith("checkpoint.checkpoint_dir="):
                checkpoint_dir = Path(item.split("=", 1)[1])
                break
        assert checkpoint_dir is not None
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "latest.pt").write_text("candidate", encoding="utf-8")
        return CompletedProcess(cmd, 0, stdout="trained\n", stderr="")

    monkeypatch.setattr("cadflow.flywheel_loop.subprocess.run", fake_run)

    def fake_probe(cfg, checkpoint, data_source, device, verbose=False):
        if Path(checkpoint).name == "best.pt":
            return _probe_result(checkpoint, 0.4)
        return _probe_result(checkpoint, 0.6)

    monkeypatch.setattr("cadflow.flywheel_loop.probe_checkpoint", fake_probe)

    def fail_copy(*args, **kwargs):
        raise AssertionError("copy2 should not run when the candidate is worse")

    monkeypatch.setattr("cadflow.flywheel_loop.shutil.copy2", fail_copy)

    result = run_flywheel_loop([tmp_path / "raw"], out_dir, flywheel_path=flywheel_path, max_steps=1)

    assert result.decision == "rejected"
    assert result.promoted_checkpoint is None
    assert result.candidate_probe is not None and result.baseline_probe is not None
    assert result.candidate_probe.score > result.baseline_probe.score
    assert Path(result.summary_path).exists()
    history = (out_dir / "registry" / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 1
    assert '"decision": "rejected"' in history[0]
