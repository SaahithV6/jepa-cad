"""CAD backend abstraction with CadQuery and mock fallback."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Protocol

try:  # pragma: no cover - import availability is environment-dependent
    import cadquery as cq
except Exception:  # pragma: no cover
    cq = None  # type: ignore[assignment]


class CadBackend(Protocol):
    name: str

    def box(self, width: float, height: float, depth: float) -> Any: ...

    def volume(self, shape: Any) -> float: ...

    def describe(self, shape: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MockCadSolid:
    kind: str
    dimensions: tuple[float, float, float]


class MockCadBackend:
    """Deterministic fallback backend for tests and environments without CadQuery."""

    name = "mock"

    def box(self, width: float, height: float, depth: float) -> MockCadSolid:
        return MockCadSolid(kind="box", dimensions=(float(width), float(height), float(depth)))

    def volume(self, shape: MockCadSolid) -> float:
        return float(prod(shape.dimensions))

    def describe(self, shape: MockCadSolid) -> dict[str, Any]:
        return {
            "type": "mock",
            "kind": shape.kind,
            "dimensions": list(shape.dimensions),
            "volume": self.volume(shape),
        }


class CadQueryBackend:
    """Real backend powered by cadquery."""

    name = "cadquery"

    def __init__(self) -> None:
        if cq is None:  # pragma: no cover - exercised through fallback path
            raise RuntimeError("cadquery is not available")

    def box(self, width: float, height: float, depth: float) -> Any:
        cadquery = cq
        assert cadquery is not None  # for type checkers; guarded by __init__
        return cadquery.Workplane("XY").box(float(width), float(height), float(depth))

    def volume(self, shape: Any) -> float:
        solid = shape.val() if hasattr(shape, "val") else shape
        return float(solid.Volume())

    def describe(self, shape: Any) -> dict[str, Any]:
        return {
            "type": "cadquery",
            "volume": self.volume(shape),
            "class": type(shape).__name__,
        }


def get_backend(prefer_real: bool = True) -> CadBackend:
    """Return a real backend when available, otherwise a mock fallback."""

    if prefer_real:
        try:
            return CadQueryBackend()
        except Exception:
            pass
    return MockCadBackend()
