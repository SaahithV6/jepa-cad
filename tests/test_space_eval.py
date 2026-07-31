"""Tests for space-family checkpoint comparison and CLI wiring."""

from __future__ import annotations

from pathlib import Path

from cadflow.cli import main as cadflow_main
from eval.probe import ProbeResult
from eval.space_eval import compare_checkpoints


def _probe(checkpoint: str | Path, score: float, *, score_name: str = "val_mse") -> ProbeResult:
    return ProbeResult(
        checkpoint=str(checkpoint),
        data_source="real",
        score_name=score_name,
        score=score,
        train_mse=score + 0.1,
        val_mse=score,
        train_samples=8,
        val_samples=2,
        epochs=1,
        seed=42,
    )


def test_compare_checkpoints_reports_improvement(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    candidate.write_text("candidate", encoding="utf-8")
    baseline.write_text("baseline", encoding="utf-8")

    def fake_probe(cfg, checkpoint, data_source, device, seed=None, verbose=False):
        if Path(checkpoint).name == "baseline.pt":
            return _probe(checkpoint, 0.8)
        return _probe(checkpoint, 0.5)

    monkeypatch.setattr("eval.space_eval.probe_checkpoint", fake_probe)

    result = compare_checkpoints({}, candidate, baseline, "real", threshold=0.1)
    assert result.improved is True
    assert result.score_name == "val_mse"
    assert result.candidate_score == 0.5
    assert result.baseline_score == 0.8
    assert result.improvement is not None and result.improvement > 0.3


def test_cli_space_eval_json(tmp_path: Path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    candidate.write_text("candidate", encoding="utf-8")
    baseline.write_text("baseline", encoding="utf-8")

    def fake_probe(cfg, checkpoint, data_source, device, seed=None, verbose=False):
        if Path(checkpoint).name == "baseline.pt":
            return _probe(checkpoint, 0.9)
        return _probe(checkpoint, 0.6)

    monkeypatch.setattr("eval.space_eval.probe_checkpoint", fake_probe)

    code = cadflow_main([
        "space-eval",
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--json",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert '"improved": true' in out
    assert '"candidate_score": 0.6' in out
    assert '"baseline_score": 0.9' in out
