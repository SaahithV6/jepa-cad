"""Autonomous maintenance and improvement supervisor for JEPA-CAD."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .flywheel_loop import FlywheelLoopResult, run_flywheel_loop


@dataclass(frozen=True, slots=True)
class AutopilotResult:
    repo_root: str
    git_commit: str | None
    git_status: tuple[str, ...]
    required_imports: tuple[str, ...]
    missing_imports: tuple[str, ...]
    env_repaired: bool
    pytest_returncode: int | None
    pytest_command: tuple[str, ...]
    pytest_stdout: str
    pytest_stderr: str
    loop: FlywheelLoopResult | None
    loop_skipped_reason: str | None
    decision: str
    summary_path: str

    @property
    def ok(self) -> bool:
        if self.pytest_returncode not in (0, None):
            return False
        if self.loop is not None:
            return self.loop.ok
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "git_commit": self.git_commit,
            "git_status": list(self.git_status),
            "required_imports": list(self.required_imports),
            "missing_imports": list(self.missing_imports),
            "env_repaired": self.env_repaired,
            "pytest_returncode": self.pytest_returncode,
            "pytest_command": list(self.pytest_command),
            "pytest_stdout": self.pytest_stdout,
            "pytest_stderr": self.pytest_stderr,
            "loop": self.loop.to_dict() if self.loop is not None else None,
            "loop_skipped_reason": self.loop_skipped_reason,
            "decision": self.decision,
            "summary_path": self.summary_path,
            "ok": self.ok,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(args: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _check_imports(required: Sequence[str]) -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - report missing environment deps, don't crash autopilot
            missing.append(module_name)
    return tuple(missing)


def _repair_environment(repo_root: Path) -> bool:
    requirements = repo_root / "requirements.txt"
    if not requirements.exists():
        return False
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _run_pytest(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def run_autopilot(
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
    skip_tests: bool = False,
    required_imports: Sequence[str] = ("numpy", "torch", "yaml", "trimesh", "scipy", "shapely", "cadquery"),
    repair_env: bool = True,
) -> AutopilotResult:
    """Run the maintenance gate and, if it passes, the recursive improvement loop."""

    repo_root = _repo_root()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    git_commit = _git_output(["rev-parse", "--short", "HEAD"], repo_root) or None
    git_status_lines = tuple(line for line in _git_output(["status", "--short"], repo_root).splitlines() if line)

    missing_imports = _check_imports(required_imports)
    env_repaired = False
    if missing_imports and repair_env:
        env_repaired = _repair_environment(repo_root)
        if env_repaired:
            missing_imports = _check_imports(required_imports)

    pytest_returncode: int | None = None
    pytest_stdout = ""
    pytest_stderr = ""
    pytest_command: tuple[str, ...] = ()
    if not skip_tests:
        pytest_proc = _run_pytest(repo_root)
        pytest_returncode = pytest_proc.returncode
        pytest_stdout = pytest_proc.stdout
        pytest_stderr = pytest_proc.stderr
        pytest_command = (sys.executable, "-m", "pytest", "tests/", "-q")

    loop: FlywheelLoopResult | None = None
    loop_skipped_reason: str | None = None
    decision = "tests_skipped" if skip_tests else "tests_passed"
    if missing_imports:
        decision = "env_incomplete"
        loop_skipped_reason = f"missing imports: {', '.join(missing_imports)}"
    elif pytest_returncode not in (0, None):
        decision = "tests_failed"
        loop_skipped_reason = "pytest gate failed"
    elif raw_dirs or flywheel_path is not None:
        loop = run_flywheel_loop(
            raw_dirs,
            out_root / "autopilot",
            flywheel_path=flywheel_path,
            config=config,
            num_points=num_points,
            num_fields=num_fields,
            fmt=fmt,
            recursive=recursive,
            limit=limit,
            allow_synthetic_fallback=allow_synthetic_fallback,
            data_source=data_source,
            probe_data_source=probe_data_source,
            max_steps=max_steps,
            grad_accum_steps=grad_accum_steps,
            extra_overrides=extra_overrides,
            promote_limit=promote_limit,
            baseline_checkpoint=baseline_checkpoint,
            improvement_threshold=improvement_threshold,
        )
        decision = loop.decision
    else:
        loop_skipped_reason = "no raw_dir or flywheel_path provided"

    summary_path = out_root / "autopilot_report.json"
    result = AutopilotResult(
        repo_root=str(repo_root),
        git_commit=git_commit,
        git_status=git_status_lines,
        required_imports=tuple(required_imports),
        missing_imports=missing_imports,
        env_repaired=env_repaired,
        pytest_returncode=pytest_returncode,
        pytest_command=pytest_command,
        pytest_stdout=pytest_stdout,
        pytest_stderr=pytest_stderr,
        loop=loop,
        loop_skipped_reason=loop_skipped_reason,
        decision=decision,
        summary_path=str(summary_path),
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
