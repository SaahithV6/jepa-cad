"""Recursive verified-data flywheel orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from data.ingest import IngestionResult, ingest_sources
from eval.probe import ProbeResult, load_config, probe_checkpoint
from utils.config import apply_overrides

from .flywheel import DataFlywheel
from .promotion import PromotionResult, promote_verified_to_dataset


@dataclass(frozen=True, slots=True)
class FlywheelLoopResult:
    cycle_id: str
    workdir: str
    staging_dir: str
    curated_dir: str
    checkpoint_dir: str
    baseline_checkpoint: str | None
    dataset_dir: str
    ingestion: IngestionResult
    promotion: PromotionResult
    train_returncode: int | None
    train_command: tuple[str, ...]
    train_stdout: str
    train_stderr: str
    candidate_probe: ProbeResult | None
    baseline_probe: ProbeResult | None
    promoted_checkpoint: str | None
    decision: str
    summary_path: str

    @property
    def ok(self) -> bool:
        return self.train_returncode == 0 and self.decision != "train_failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "workdir": self.workdir,
            "staging_dir": self.staging_dir,
            "curated_dir": self.curated_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "baseline_checkpoint": self.baseline_checkpoint,
            "dataset_dir": self.dataset_dir,
            "ingestion": self.ingestion.to_dict(),
            "promotion": self.promotion.to_dict(),
            "train_returncode": self.train_returncode,
            "train_command": list(self.train_command),
            "train_stdout": self.train_stdout,
            "train_stderr": self.train_stderr,
            "candidate_probe": self.candidate_probe.to_dict() if self.candidate_probe is not None else None,
            "baseline_probe": self.baseline_probe.to_dict() if self.baseline_probe is not None else None,
            "promoted_checkpoint": self.promoted_checkpoint,
            "decision": self.decision,
            "summary_path": self.summary_path,
            "ok": self.ok,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cycle_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _best_checkpoint(registry_dir: Path, baseline_checkpoint: str | Path | None) -> Path | None:
    if baseline_checkpoint is not None:
        path = Path(baseline_checkpoint)
        return path if path.exists() else None
    candidate = registry_dir / "best.pt"
    return candidate if candidate.exists() else None


def _registry_history_path(registry_dir: Path) -> Path:
    return registry_dir / "history.jsonl"


def _registry_best_meta_path(registry_dir: Path) -> Path:
    return registry_dir / "best.json"


def _append_registry_history(registry_dir: Path, payload: dict[str, Any]) -> None:
    history = _registry_history_path(registry_dir)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_best_registry_meta(registry_dir: Path, payload: dict[str, Any]) -> None:
    meta_path = _registry_best_meta_path(registry_dir)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_better(candidate: ProbeResult, baseline: ProbeResult, threshold: float) -> bool:
    if baseline.score <= 0:
        return candidate.score < baseline.score
    return candidate.score <= baseline.score * (1.0 - threshold)


def run_flywheel_loop(
    raw_dirs: Sequence[str | Path] | None,
    out_dir: str | Path,
    *,
    flywheel_path: str | Path | None = None,
    config: str | Path = "configs/base.yaml",
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    recursive: bool = True,
    limit: int | None = None,
    allow_synthetic_fallback: bool = False,
    data_source: str = "real",
    probe_data_source: str = "real",
    max_steps: int | None = 1,
    grad_accum_steps: int | None = None,
    extra_overrides: Sequence[str] | None = None,
    promote_limit: int = 50,
    baseline_checkpoint: str | Path | None = None,
    improvement_threshold: float = 0.0,
) -> FlywheelLoopResult:
    """Run the recursive self-improvement loop.

    1. Ingest raw sources into a staging area.
    2. Promote verified flywheel runs into a curated dataset.
    3. Train a candidate checkpoint on the curated dataset.
    4. Probe the candidate against the prior best checkpoint.
    5. Promote the candidate only if it improves the probe score.
    """

    root = Path(out_dir)
    cycle_id = _cycle_id()
    workdir = root / cycle_id
    staging_dir = workdir / "staging"
    curated_dir = workdir / "curated"
    checkpoint_dir = workdir / "checkpoints"
    registry_dir = root / "registry"
    runs_dir = workdir / "runs"
    registry_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    ingestion = ingest_sources(
        raw_dirs,
        staging_dir,
        flywheel_path=flywheel_path,
        num_points=num_points,
        num_fields=num_fields,
        fmt=fmt,
        recursive=recursive,
        limit=limit,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    flywheel_store = DataFlywheel(flywheel_path or (workdir / "flywheel.jsonl"))
    promotion = promote_verified_to_dataset(
        flywheel_store,
        curated_dir,
        limit=promote_limit,
        num_points=num_points,
        num_fields=num_fields,
        fmt=fmt,
    )
    dataset_dir = curated_dir if promotion.promoted > 0 else staging_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    repo_root = _repo_root()
    train_py = repo_root / "train.py"
    cmd: list[str] = [
        sys.executable,
        str(train_py),
        "--config",
        str(config),
        "--data-source",
        data_source,
        "--set",
        f"data.data_dir={dataset_dir}",
        "--set",
        f"checkpoint.checkpoint_dir={checkpoint_dir}",
        "--set",
        f"logging.log_dir={runs_dir}",
        "--set",
        f"logging.experiment_name={cycle_id}",
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if grad_accum_steps is not None:
        cmd.extend(["--grad-accum-steps", str(grad_accum_steps)])
    for override in extra_overrides or ():
        cmd.extend(["--set", override])

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    candidate_probe: ProbeResult | None = None
    baseline_probe: ProbeResult | None = None
    promoted_checkpoint: Path | None = None
    decision = "train_failed" if proc.returncode != 0 else "rejected"
    checkpoint_path = checkpoint_dir / "latest.pt"
    baseline_path = _best_checkpoint(registry_dir, baseline_checkpoint)

    if proc.returncode == 0 and checkpoint_path.exists():
        cfg = apply_overrides(load_config(config), list(extra_overrides or ()))
        cfg["data"]["data_dir"] = str(dataset_dir)
        candidate_probe = probe_checkpoint(cfg, checkpoint_path, probe_data_source, device=_probe_device(), verbose=False)
        if baseline_path is not None:
            baseline_probe = probe_checkpoint(cfg, baseline_path, probe_data_source, device=_probe_device(), verbose=False)
        if baseline_probe is None or _is_better(candidate_probe, baseline_probe, improvement_threshold):
            versioned_checkpoint = registry_dir / "checkpoints" / f"{cycle_id}.pt"
            versioned_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, versioned_checkpoint)
            promoted_checkpoint = registry_dir / "best.pt"
            shutil.copy2(versioned_checkpoint, promoted_checkpoint)
            _write_best_registry_meta(
                registry_dir,
                {
                    "cycle_id": cycle_id,
                    "checkpoint": str(promoted_checkpoint),
                    "versioned_checkpoint": str(versioned_checkpoint),
                    "candidate_probe": candidate_probe.to_dict(),
                    "baseline_checkpoint": str(baseline_path) if baseline_path is not None else None,
                    "baseline_probe": baseline_probe.to_dict() if baseline_probe is not None else None,
                    "decision": "promoted",
                },
            )
            decision = "promoted"
        else:
            decision = "rejected"

    summary_path = workdir / "flywheel_cycle.json"
    result = FlywheelLoopResult(
        cycle_id=cycle_id,
        workdir=str(workdir),
        staging_dir=str(staging_dir),
        curated_dir=str(curated_dir),
        checkpoint_dir=str(checkpoint_dir),
        baseline_checkpoint=str(baseline_path) if baseline_path is not None else None,
        dataset_dir=str(dataset_dir),
        ingestion=ingestion,
        promotion=promotion,
        train_returncode=proc.returncode,
        train_command=tuple(cmd),
        train_stdout=proc.stdout,
        train_stderr=proc.stderr,
        candidate_probe=candidate_probe,
        baseline_probe=baseline_probe,
        promoted_checkpoint=str(promoted_checkpoint) if promoted_checkpoint is not None else None,
        decision=decision,
        summary_path=str(summary_path),
    )
    _append_registry_history(
        registry_dir,
        {
            "cycle_id": cycle_id,
            "workdir": result.workdir,
            "decision": result.decision,
            "train_returncode": result.train_returncode,
            "candidate_probe": result.candidate_probe.to_dict() if result.candidate_probe is not None else None,
            "baseline_probe": result.baseline_probe.to_dict() if result.baseline_probe is not None else None,
            "promoted_checkpoint": result.promoted_checkpoint,
            "summary_path": result.summary_path,
        },
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _probe_device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
