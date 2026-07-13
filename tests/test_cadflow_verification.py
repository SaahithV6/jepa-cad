from __future__ import annotations

from cadflow.backends import CadQueryBackend
from cadflow.verification import VerificationReport, render_verification_report, verify_solid


def test_verify_solid_reports_volume_and_passes() -> None:
    backend = CadQueryBackend()
    solid = backend.box(1.0, 2.0, 3.0)

    report = verify_solid(solid, backend=backend, expected_volume=6.0)

    assert isinstance(report, VerificationReport)
    assert report.passed is True
    assert report.metrics["volume"] == 6.0
    assert report.findings == ()


def test_render_verification_report_includes_summary() -> None:
    report = VerificationReport(
        name="example",
        passed=False,
        findings=("volume mismatch",),
        metrics={"volume": 5.5},
    )

    text = render_verification_report(report)

    assert "example" in text
    assert "FAILED" in text
    assert "volume mismatch" in text
