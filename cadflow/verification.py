"""Verification and reporting helpers for CAD/CAE artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .backends import CadBackend, get_backend


@dataclass(frozen=True, slots=True)
class VerificationReport:
    name: str
    passed: bool
    findings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    backend: str = "unknown"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "findings": list(self.findings),
            "metrics": self.metrics,
            "backend": self.backend,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerificationReport":
        return cls(
            name=str(payload["name"]),
            passed=bool(payload["passed"]),
            findings=tuple(payload.get("findings", ())),
            metrics=dict(payload.get("metrics", {})),
            backend=str(payload.get("backend", "unknown")),
            notes=payload.get("notes"),
        )


def verify_solid(
    shape: Any,
    backend: CadBackend | None = None,
    expected_volume: float | None = None,
    volume_tol: float = 1e-6,
) -> VerificationReport:
    """Perform a tiny but real verification pass against a solid."""

    backend = backend or get_backend(prefer_real=True)
    volume = float(backend.volume(shape))
    findings: list[str] = []
    passed = True

    if expected_volume is not None and abs(volume - expected_volume) > volume_tol:
        passed = False
        findings.append(f"volume mismatch: expected {expected_volume:.6g}, got {volume:.6g}")

    if volume <= 0:
        passed = False
        findings.append("non-positive volume")

    metrics = {
        "volume": volume,
        "backend": backend.name,
        "shape_type": type(shape).__name__,
    }
    return VerificationReport(
        name="solid_verification",
        passed=passed,
        findings=tuple(findings),
        metrics=metrics,
        backend=backend.name,
    )


def render_verification_report(report: VerificationReport) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"{report.name}: {status}", f"backend={report.backend}"]
    for key in sorted(report.metrics):
        lines.append(f"{key}={report.metrics[key]}")
    if report.findings:
        lines.append("findings:")
        lines.extend(f"- {finding}" for finding in report.findings)
    return "\n".join(lines)
