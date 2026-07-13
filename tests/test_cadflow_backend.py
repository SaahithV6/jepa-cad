from __future__ import annotations

import math

from cadflow.backends import CadQueryBackend, MockCadBackend, get_backend


def test_cadquery_backend_builds_real_solid() -> None:
    backend = CadQueryBackend()
    solid = backend.box(1.0, 2.0, 3.0)

    assert backend.name == "cadquery"
    assert math.isclose(backend.volume(solid), 6.0, rel_tol=1e-6)
    assert backend.describe(solid)["type"] == "cadquery"


def test_mock_backend_is_available_as_fallback() -> None:
    backend = get_backend(prefer_real=False)
    solid = backend.box(1.0, 2.0, 3.0)

    assert isinstance(backend, MockCadBackend)
    assert backend.describe(solid)["type"] == "mock"
    assert math.isclose(backend.volume(solid), 6.0, rel_tol=1e-6)
