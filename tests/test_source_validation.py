from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cadflow.cli import main as cadflow_main
from cadflow.datasets import DatasetSource
from cadflow.source_validation import (
    BLOCKED,
    MANUAL_REVIEW,
    REFERENCE_ONLY,
    USABLE,
    validate_source,
    validate_source_registry,
)


class _Handler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler hook
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", "12")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler hook
        body = b"hello world\n"
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - silence test server
        return


def _serve_once() -> tuple[str, ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = server.server_address[0]
    port = server.server_address[1]
    return f"http://{host}:{port}/report.pdf", server


def test_validate_source_classifies_usable_reference_blocked_and_manual_review() -> None:
    url, server = _serve_once()
    try:
        usable = DatasetSource(
            key="local-report",
            title="Local PDF report",
            domain="space",
            url=url,
            license="public release",
            use_cases=("engine reports",),
        )
        reference_only = DatasetSource(
            key="grabcad-spacecraft",
            title="GrabCAD spacecraft",
            domain="space",
            url="https://grabcad.com/library/tag/spacecraft",
            license="third-party user uploads / GrabCAD terms",
            use_cases=("reference geometry",),
        )
        blocked = DatasetSource(
            key="bad-scheme",
            title="Bad URL",
            domain="space",
            url="ftp://example.com/file.pdf",
            license="public release",
            use_cases=("reference geometry",),
        )
        manual = DatasetSource(
            key="unclear-license",
            title="Unclear license source",
            domain="space",
            url="https://example.com/catalog.pdf",
            license="unclear rights",
            use_cases=("reference geometry",),
        )

        usable_result = validate_source(usable, live_check=True, timeout=2.0)
        assert usable_result.status == USABLE
        assert usable_result.training_eligible is True
        assert usable_result.reachable is True

        reference_result = validate_source(reference_only, live_check=False)
        assert reference_result.status == REFERENCE_ONLY
        assert reference_result.reference_only is True

        blocked_result = validate_source(blocked, live_check=False)
        assert blocked_result.status == BLOCKED
        assert blocked_result.blocked is True

        manual_result = validate_source(manual, live_check=False)
        assert manual_result.status == MANUAL_REVIEW
        assert manual_result.manual_review is True
    finally:
        server.shutdown()
        server.server_close()


def test_validate_source_registry_subset_reports_expected_statuses() -> None:
    report = validate_source_registry(keys=["nasa_3d_resources", "grabcad_spacecraft"], live_check=False)
    assert report.counts[USABLE] == 1
    assert report.counts[REFERENCE_ONLY] == 1
    assert report.counts[BLOCKED] == 0
    assert report.counts[MANUAL_REVIEW] == 0
    assert report.information_mode_counts["cad-model"] == 1
    assert report.information_mode_counts["reference-only-geometry"] == 1
    statuses = {result.source.key: result.status for result in report.results}
    assert statuses["nasa_3d_resources"] == USABLE
    assert statuses["grabcad_spacecraft"] == REFERENCE_ONLY


def test_validate_source_registry_keeps_numeric_physics_sources_usable() -> None:
    report = validate_source_registry(
        keys=[
            "esa_anomaly_dataset",
            "nasa_cmapss_turbofan",
            "blastnet_premixed_h2_air_dns",
            "fem_simulations",
        ],
        live_check=False,
    )
    assert report.counts[USABLE] == 4
    assert report.counts[REFERENCE_ONLY] == 0
    assert report.counts[BLOCKED] == 0
    assert report.counts[MANUAL_REVIEW] == 0
    assert report.information_mode_counts["database-catalog"] == 4
    assert all(result.training_eligible for result in report.results)


def test_validate_source_registry_registers_new_space_sweep_sources() -> None:
    report = validate_source_registry(
        keys=[
            "nasa_cassini_3d_model",
            "nasa_parker_solar_probe_3d_model",
            "nasa_europa_clipper_scale_model",
            "nasa_tess_3d_model",
            "esa_euclid_payload_module",
            "esa_cassini_huygens_3d_model",
            "jaxa_m_v_configuration",
            "jaxa_m_v_design_guidelines",
            "jaxa_jedi_numerical_simulation",
            "isro_lvm3_m6_bluebird",
            "isro_pslv_c61_eos09",
            "isro_sslv_d3_eos08",
        ],
        live_check=False,
    )
    assert report.counts[BLOCKED] == 0
    assert report.counts[USABLE] >= 4
    assert report.counts[MANUAL_REVIEW] >= 4
    assert report.information_mode_counts["cad-model"] >= 4
    assert report.information_mode_counts["technical-report"] >= 4
    assert {result.source.key for result in report.results} == {
        "nasa_cassini_3d_model",
        "nasa_parker_solar_probe_3d_model",
        "nasa_europa_clipper_scale_model",
        "nasa_tess_3d_model",
        "esa_euclid_payload_module",
        "esa_cassini_huygens_3d_model",
        "jaxa_m_v_configuration",
        "jaxa_m_v_design_guidelines",
        "jaxa_jedi_numerical_simulation",
        "isro_lvm3_m6_bluebird",
        "isro_pslv_c61_eos09",
        "isro_sslv_d3_eos08",
    }


def test_validate_source_exposes_information_modes() -> None:
    model = validate_source(DatasetSource(
        key="nasa-3d",
        title="NASA 3D Model",
        domain="space",
        url="https://science.nasa.gov/3d-resources/",
        license="NASA open data",
        use_cases=("spacecraft geometry",),
    ))
    report = validate_source(DatasetSource(
        key="esa-diagram",
        title="ESA spacecraft schematic",
        domain="space",
        url="https://example.com/diagram.pdf",
        license="public release",
        use_cases=("schematic diagram",),
    ))
    assert model.information_mode == "cad-model"
    assert report.information_mode == "diagram-blueprint"


def test_validate_sources_cli_emits_json(tmp_path: Path, capsys) -> None:
    exit_code = cadflow_main(["validate-sources", "--key", "nasa_3d_resources", "--json"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(captured)
    assert payload["counts"][USABLE] == 1
    assert payload["results"][0]["source"]["key"] == "nasa_3d_resources"
