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
    require_watertight: bool = True,
    min_bbox_span: float = 1e-9,
) -> VerificationReport:
    """Perform an auditable verification pass against a solid."""

    backend = backend or get_backend(prefer_real=True)
    findings: list[str] = []
    passed = True
    notes: list[str] = []

    try:
        volume = float(backend.volume(shape))
        bbox = backend.bounding_box(shape)
        faces = int(backend.face_count(shape))
        valid = bool(backend.is_valid(shape))
        watertight = bool(backend.is_watertight(shape))
    except Exception as exc:
        return VerificationReport(
            name="solid_verification",
            passed=False,
            findings=(f"invalid geometry: {exc}",),
            metrics={"backend": backend.name, "shape_type": type(shape).__name__},
            backend=backend.name,
            notes="geometry interrogation failed",
        )

    if expected_volume is not None and abs(volume - expected_volume) > volume_tol:
        passed = False
        findings.append(f"volume mismatch: expected {expected_volume:.6g}, got {volume:.6g}")

    if volume <= 0:
        passed = False
        findings.append("non-positive volume")

    spans = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    if any(span <= min_bbox_span for span in spans):
        passed = False
        findings.append(f"degenerate bounding box spans={spans}")

    if faces <= 0:
        passed = False
        findings.append("no faces detected")

    if not valid:
        passed = False
        findings.append("invalid geometry flag")

    if require_watertight and not watertight:
        passed = False
        findings.append("solid is not closed/watertight")
    elif not watertight:
        notes.append("watertight check skipped or soft-failed")

    metrics = {
        "volume": volume,
        "bounding_box": list(bbox),
        "bbox_spans": list(spans),
        "face_count": faces,
        "valid": valid,
        "watertight": watertight,
        "backend": backend.name,
        "shape_type": type(shape).__name__,
    }
    return VerificationReport(
        name="solid_verification",
        passed=passed,
        findings=tuple(findings),
        metrics=metrics,
        backend=backend.name,
        notes="; ".join(notes) if notes else None,
    )


def render_verification_report(report: VerificationReport) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        f"{report.name}: {status}",
        f"backend={report.backend}",
    ]
    if report.notes:
        lines.append(f"notes={report.notes}")
    for key in sorted(report.metrics):
        lines.append(f"{key}={report.metrics[key]}")
    if report.findings:
        lines.append("findings:")
        lines.extend(f"- {finding}" for finding in report.findings)
    else:
        lines.append("findings: none")
    return "\n".join(lines)
