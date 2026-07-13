from __future__ import annotations

from cadflow.backends import CadQueryBackend, MockCadBackend
from cadflow.verification import VerificationReport, render_verification_report, verify_solid


def test_verify_solid_reports_volume_and_passes() -> None:
    backend = CadQueryBackend()
    solid = backend.box(1.0, 2.0, 3.0)

    report = verify_solid(solid, backend=backend, expected_volume=6.0)

    assert isinstance(report, VerificationReport)
    assert report.passed is True
    assert report.metrics["volume"] == 6.0
    assert report.metrics["face_count"] >= 6
    assert report.metrics["watertight"] is True
    assert report.findings == ()


def test_verify_solid_fails_on_volume_mismatch() -> None:
    backend = MockCadBackend()
    solid = backend.box(1.0, 1.0, 1.0)
    report = verify_solid(solid, backend=backend, expected_volume=99.0)
    assert report.passed is False
    assert any("volume mismatch" in f for f in report.findings)


def test_render_verification_report_includes_summary() -> None:
    report = VerificationReport(
        name="example",
        passed=False,
        findings=("volume mismatch",),
        metrics={"volume": 5.5},
        notes="check mesh",
    )

    text = render_verification_report(report)

    assert "example" in text
    assert "FAILED" in text
    assert "volume mismatch" in text
    assert "notes=check mesh" in text
