"""Solver result wrappers and native-probe helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping


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

    @property
    def ok(self) -> bool:
        return self.status.lower() in {"optimal", "success", "solved", "converged", "ok"}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "objective": self.objective,
            "iterations": self.iterations,
            "residual": self.residual,
            "metadata": self.metadata,
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
        return cls(
            status=str(payload.get("status", "unknown")),
            objective=payload.get("objective"),
            iterations=payload.get("iterations"),
            residual=payload.get("residual"),
            metadata=dict(payload.get("metadata", {})),
            probe=probe,
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


def wrap_solver_result(result: Any, probe: NativeProbeResult | None = None) -> SolverResult:
    """Normalize dict-like or attribute-like solver outputs."""

    if isinstance(result, SolverResult):
        return result if probe is None else SolverResult(
            status=result.status,
            objective=result.objective,
            iterations=result.iterations,
            residual=result.residual,
            metadata=dict(result.metadata),
            probe=probe,
        )

    if isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {
            key: getattr(result, key)
            for key in ("status", "objective", "iterations", "residual", "metadata")
            if hasattr(result, key)
        }

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}

    return SolverResult(
        status=str(payload.get("status", "unknown")),
        objective=payload.get("objective"),
        iterations=payload.get("iterations"),
        residual=payload.get("residual"),
        metadata=dict(metadata),
        probe=probe,
    )
