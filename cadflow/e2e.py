"""End-to-end CAD/CAE -> dataset -> JEPA training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from data.ingest import IngestionResult, ingest_sources


@dataclass(frozen=True, slots=True)
class EndToEndResult:
    ingestion: IngestionResult
    train_returncode: int | None
    train_command: tuple[str, ...]
    train_stdout: str
    train_stderr: str

    @property
    def ok(self) -> bool:
        return self.train_returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion": self.ingestion.to_dict(),
            "train_returncode": self.train_returncode,
            "train_command": list(self.train_command),
            "train_stdout": self.train_stdout,
            "train_stderr": self.train_stderr,
            "ok": self.ok,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_end_to_end(
    raw_dirs: Sequence[str | Path] | None,
    out_dir: str | Path,
    *,
    flywheel_path: str | Path | None = None,
    num_points: int = 1024,
    num_fields: int = 3,
    fmt: str = "npz",
    recursive: bool = True,
    limit: int | None = None,
    allow_synthetic_fallback: bool = False,
    config: str | Path = "configs/base.yaml",
    data_source: str = "real",
    max_steps: int | None = 1,
    grad_accum_steps: int | None = None,
    extra_overrides: Sequence[str] | None = None,
) -> EndToEndResult:
    """Ingest data and run a short JEPA training job against it."""

    ingestion = ingest_sources(
        raw_dirs,
        out_dir,
        flywheel_path=flywheel_path,
        num_points=num_points,
        num_fields=num_fields,
        fmt=fmt,
        recursive=recursive,
        limit=limit,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

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
        f"data.data_dir={Path(out_dir)}",
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
    return EndToEndResult(
        ingestion=ingestion,
        train_returncode=proc.returncode,
        train_command=tuple(cmd),
        train_stdout=proc.stdout,
        train_stderr=proc.stderr,
    )
