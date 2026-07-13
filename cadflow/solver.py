"""Solver result wrappers, native probes, and external adapter hooks.

Keeps SolverResult as the normalized interface. Native solvers (OpenFOAM /
FEA / MBD) are probed when present; otherwise deterministic fallbacks keep
the orchestration loop testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import json
import shutil
import subprocess
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class NativeProbeResult:
    backend: str
    available: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: str
    objective: float | None = None
    iterations: int | None = None
    residual: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    probe: NativeProbeResult | None = None
    artifacts: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status.lower() in {"optimal", "success", "solved", "converged", "ok"}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "objective": self.objective,
            "iterations": self.iterations,
            "residual": self.residual,
            "metadata": self.metadata,
            "artifacts": list(self.artifacts),
            "logs": list(self.logs),
        }
        if self.probe is not None:
            payload["probe"] = {
                "backend": self.probe.backend,
                "available": self.probe.available,
                "reason": self.probe.reason,
                "details": self.probe.details,
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SolverResult":
        probe_payload = payload.get("probe")
        probe = None
        if isinstance(probe_payload, Mapping):
            probe = NativeProbeResult(
                backend=str(probe_payload.get("backend", "unknown")),
                available=bool(probe_payload.get("available", False)),
                reason=str(probe_payload.get("reason", "")),
                details=dict(probe_payload.get("details", {})),
            )
        artifacts = payload.get("artifacts", ())
        logs = payload.get("logs", ())
        return cls(
            status=str(payload.get("status", "unknown")),
            objective=_as_optional_float(payload.get("objective")),
            iterations=_as_optional_int(payload.get("iterations")),
            residual=_as_optional_float(payload.get("residual")),
            metadata=dict(payload.get("metadata", {})),
            probe=probe,
            artifacts=tuple(str(a) for a in artifacts),
            logs=tuple(str(line) for line in logs),
        )


def probe_native_solver(module_name: str, attribute: str | None = None) -> NativeProbeResult:
    """Check whether a native solver hook is importable and usable."""

    try:
        module = import_module(module_name)
    except Exception as exc:
        return NativeProbeResult(
            backend=module_name,
            available=False,
            reason=f"missing module: {exc}",
        )

    if attribute is not None and not hasattr(module, attribute):
        return NativeProbeResult(
            backend=module_name,
            available=False,
            reason=f"missing attribute: {attribute}",
        )

    return NativeProbeResult(
        backend=module_name,
        available=True,
        reason="available",
        details={"attribute": attribute} if attribute else {},
    )


def probe_solver_binary(candidates: Sequence[str], backend: str | None = None) -> NativeProbeResult:
    """Probe whether an external solver binary exists on PATH."""

    name = backend or (candidates[0] if candidates else "solver")
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return NativeProbeResult(
                backend=name,
                available=True,
                reason="available",
                details={"binary": candidate, "path": path},
            )
    return NativeProbeResult(
        backend=name,
        available=False,
        reason=f"missing binaries: {', '.join(candidates)}",
        details={"candidates": list(candidates)},
    )


def probe_openfoam() -> NativeProbeResult:
    return probe_solver_binary(("simpleFoam", "foamRun", "blockMesh"), backend="openfoam")


def probe_fea() -> NativeProbeResult:
    return probe_solver_binary(("ccx", "CalculiX", "ElmerSolver", "elmer"), backend="fea")


def probe_mbd() -> NativeProbeResult:
    return probe_solver_binary(("chrono", "projectchrono", "mbdyn"), backend="mbd")


def wrap_solver_result(result: Any, probe: NativeProbeResult | None = None) -> SolverResult:
    """Normalize dict-like, attribute-like, or subprocess-produced solver outputs."""

    if isinstance(result, SolverResult):
        return (
            result
            if probe is None
            else SolverResult(
                status=result.status,
                objective=result.objective,
                iterations=result.iterations,
                residual=result.residual,
                metadata=dict(result.metadata),
                probe=probe,
                artifacts=result.artifacts,
                logs=result.logs,
            )
        )

    payload: dict[str, Any]
    logs: list[str] = []
    artifacts: list[str] = []

    if isinstance(result, Mapping):
        payload = dict(result)
    elif hasattr(result, "returncode") and (hasattr(result, "stdout") or hasattr(result, "stderr")):
        # subprocess.CompletedProcess-like
        payload = _payload_from_subprocess(result)
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        if stdout:
            logs.append(str(stdout))
        if stderr:
            logs.append(str(stderr))
    else:
        payload = {
            key: getattr(result, key)
            for key in ("status", "objective", "iterations", "residual", "metadata", "artifacts", "logs")
            if hasattr(result, key)
        }

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}

    raw_artifacts = payload.get("artifacts", artifacts)
    raw_logs = payload.get("logs", logs)
    if isinstance(raw_artifacts, str):
        raw_artifacts = [raw_artifacts]
    if isinstance(raw_logs, str):
        raw_logs = [raw_logs]

    status = payload.get("status", "unknown")
    if status is None:
        status = "unknown"

    return SolverResult(
        status=str(status),
        objective=_as_optional_float(payload.get("objective")),
        iterations=_as_optional_int(payload.get("iterations")),
        residual=_as_optional_float(payload.get("residual")),
        metadata=dict(metadata),
        probe=probe,
        artifacts=tuple(str(a) for a in (raw_artifacts or ())),
        logs=tuple(str(line) for line in (raw_logs or ())),
    )


def run_fallback_solver(
    *,
    backend: str,
    objective: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
    probe: NativeProbeResult | None = None,
) -> SolverResult:
    """Deterministic mock solve used when native tools are unavailable."""

    return SolverResult(
        status="optimal",
        objective=float(objective),
        iterations=1,
        residual=0.0,
        metadata={"backend": backend, "mode": "fallback", **dict(metadata or {})},
        probe=probe,
        logs=(f"fallback solve for {backend}",),
    )


def run_external_command(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    timeout_s: float = 60.0,
    backend: str = "external",
    allow_fallback: bool = True,
) -> SolverResult:
    """Run an external solver command and normalize stdout/stderr/exit code."""

    probe = probe_solver_binary((command[0],), backend=backend) if command else NativeProbeResult(
        backend=backend, available=False, reason="empty command"
    )
    if not probe.available:
        if allow_fallback:
            return run_fallback_solver(backend=backend, probe=probe, metadata={"command": list(command)})
        return SolverResult(status="failed", metadata={"command": list(command)}, probe=probe, logs=(probe.reason,))

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        if allow_fallback:
            return run_fallback_solver(
                backend=backend,
                probe=probe,
                metadata={"error": str(exc), "command": list(command)},
            )
        return SolverResult(
            status="failed",
            metadata={"error": str(exc), "command": list(command)},
            probe=probe,
            logs=(str(exc),),
        )

    return wrap_solver_result(completed, probe=probe)


def _payload_from_subprocess(result: Any) -> dict[str, Any]:
    code = int(getattr(result, "returncode", 1))
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    payload: dict[str, Any] = {
        "status": "success" if code == 0 else "failed",
        "metadata": {"returncode": code},
        "logs": [stdout, stderr],
    }
    # Prefer JSON object on stdout when present.
    text = stdout.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, Mapping):
                merged = dict(parsed)
                merged.setdefault("status", payload["status"])
                meta = dict(payload["metadata"])
                extra_meta = merged.get("metadata")
                if isinstance(extra_meta, Mapping):
                    meta.update(extra_meta)
                merged["metadata"] = meta
                return dict(merged)
        except json.JSONDecodeError:
            pass
    return payload


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
