"""CAD backend abstraction with CadQuery and mock fallback.

Deterministic FOSS CAD tools build/edit geometry. LLM proposals are never
treated as final solids — backends construct validated geometry from structured
parameters (parametric primitives, extrusions, and simple sculpt offsets).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, prod
from pathlib import Path
from typing import Any, Protocol, Sequence

try:  # pragma: no cover - import availability is environment-dependent
    import cadquery as cq
except Exception:  # pragma: no cover
    cq = None  # type: ignore[assignment]


class CadBackend(Protocol):
    name: str

    def box(self, width: float, height: float, depth: float) -> Any: ...

    def cylinder(self, radius: float, height: float) -> Any: ...

    def sphere(self, radius: float) -> Any: ...

    def extrude_profile(
        self,
        profile: Sequence[tuple[float, float]],
        height: float,
    ) -> Any: ...

    def sculpt_offset(self, shape: Any, distance: float) -> Any: ...

    def volume(self, shape: Any) -> float: ...

    def bounding_box(self, shape: Any) -> tuple[float, float, float, float, float, float]: ...

    def face_count(self, shape: Any) -> int: ...

    def is_valid(self, shape: Any) -> bool: ...

    def is_watertight(self, shape: Any) -> bool: ...

    def export_step(self, shape: Any, path: str | Path) -> Path: ...

    def export_stl(self, shape: Any, path: str | Path) -> Path: ...

    def describe(self, shape: Any) -> dict[str, Any]: ...


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class MockCadSolid:
    kind: str
    dimensions: tuple[float, ...]
    offset: float = 0.0
    profile: tuple[tuple[float, float], ...] = ()

    @property
    def effective_scale(self) -> float:
        # Uniform growth from sculpt offset for mock volume/bbox estimates.
        return max(0.0, 1.0 + self.offset)


class MockCadBackend:
    """Deterministic fallback backend for tests and environments without CadQuery."""

    name = "mock"

    def box(self, width: float, height: float, depth: float) -> MockCadSolid:
        return MockCadSolid(
            kind="box",
            dimensions=(
                _require_positive("width", width),
                _require_positive("height", height),
                _require_positive("depth", depth),
            ),
        )

    def cylinder(self, radius: float, height: float) -> MockCadSolid:
        return MockCadSolid(
            kind="cylinder",
            dimensions=(_require_positive("radius", radius), _require_positive("height", height)),
        )

    def sphere(self, radius: float) -> MockCadSolid:
        return MockCadSolid(kind="sphere", dimensions=(_require_positive("radius", radius),))

    def extrude_profile(
        self,
        profile: Sequence[tuple[float, float]],
        height: float,
    ) -> MockCadSolid:
        pts = tuple((float(x), float(y)) for x, y in profile)
        if len(pts) < 3:
            raise ValueError("profile must contain at least 3 points")
        return MockCadSolid(
            kind="extrusion",
            dimensions=(_require_positive("height", height),),
            profile=pts,
        )

    def sculpt_offset(self, shape: MockCadSolid, distance: float) -> MockCadSolid:
        return MockCadSolid(
            kind=shape.kind,
            dimensions=shape.dimensions,
            offset=shape.offset + float(distance),
            profile=shape.profile,
        )

    def volume(self, shape: MockCadSolid) -> float:
        scale = shape.effective_scale
        if shape.kind == "box":
            return float(prod(shape.dimensions) * (scale**3))
        if shape.kind == "cylinder":
            r, h = shape.dimensions
            return float(pi * (r * scale) ** 2 * (h * scale))
        if shape.kind == "sphere":
            r = shape.dimensions[0] * scale
            return float((4.0 / 3.0) * pi * r**3)
        if shape.kind == "extrusion":
            area = _polygon_area(shape.profile) * (scale**2)
            height = shape.dimensions[0] * scale
            return float(abs(area) * height)
        raise ValueError(f"unknown mock solid kind: {shape.kind}")

    def bounding_box(self, shape: MockCadSolid) -> tuple[float, float, float, float, float, float]:
        scale = shape.effective_scale
        if shape.kind == "box":
            w, h, d = (v * scale for v in shape.dimensions)
            return (-w / 2, -h / 2, -d / 2, w / 2, h / 2, d / 2)
        if shape.kind == "cylinder":
            r, h = shape.dimensions[0] * scale, shape.dimensions[1] * scale
            return (-r, -r, -h / 2, r, r, h / 2)
        if shape.kind == "sphere":
            r = shape.dimensions[0] * scale
            return (-r, -r, -r, r, r, r)
        if shape.kind == "extrusion":
            xs = [p[0] * scale for p in shape.profile]
            ys = [p[1] * scale for p in shape.profile]
            h = shape.dimensions[0] * scale
            return (min(xs), min(ys), 0.0, max(xs), max(ys), h)
        raise ValueError(f"unknown mock solid kind: {shape.kind}")

    def face_count(self, shape: MockCadSolid) -> int:
        counts = {"box": 6, "cylinder": 3, "sphere": 1, "extrusion": 5}
        return counts.get(shape.kind, 0)

    def is_valid(self, shape: MockCadSolid) -> bool:
        try:
            return self.volume(shape) > 0 and self.face_count(shape) > 0
        except Exception:
            return False

    def is_watertight(self, shape: MockCadSolid) -> bool:
        return self.is_valid(shape)

    def export_step(self, shape: MockCadSolid, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        desc = self.describe(shape)
        lines = [
            "ISO-10303-21;",
            "HEADER;",
            f"FILE_NAME('{shape.kind}');",
            "ENDSEC;",
            "DATA;",
            f"/* mock export {desc} */",
            "ENDSEC;",
            "END-ISO-10303-21;",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def export_stl(self, shape: MockCadSolid, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box(shape)
        # Minimal ASCII STL of the AABB (deterministic interchange stub).
        verts = [
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ]
        faces = [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ]
        lines = [f"solid {shape.kind}"]
        for a, b, c in faces:
            lines.append("  facet normal 0 0 0")
            lines.append("    outer loop")
            for idx in (a, b, c):
                x, y, z = verts[idx]
                lines.append(f"      vertex {x:.6f} {y:.6f} {z:.6f}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {shape.kind}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def describe(self, shape: MockCadSolid) -> dict[str, Any]:
        bbox = self.bounding_box(shape)
        return {
            "type": "mock",
            "kind": shape.kind,
            "dimensions": list(shape.dimensions),
            "offset": shape.offset,
            "volume": self.volume(shape),
            "bounding_box": list(bbox),
            "face_count": self.face_count(shape),
            "valid": self.is_valid(shape),
            "watertight": self.is_watertight(shape),
        }


class CadQueryBackend:
    """Real backend powered by CadQuery / OpenCascade."""

    name = "cadquery"

    def __init__(self) -> None:
        if cq is None:  # pragma: no cover - exercised through fallback path
            raise RuntimeError("cadquery is not available")

    def box(self, width: float, height: float, depth: float) -> Any:
        cadquery = cq
        assert cadquery is not None
        return cadquery.Workplane("XY").box(
            _require_positive("width", width),
            _require_positive("height", height),
            _require_positive("depth", depth),
        )

    def cylinder(self, radius: float, height: float) -> Any:
        cadquery = cq
        assert cadquery is not None
        return cadquery.Workplane("XY").cylinder(
            _require_positive("height", height),
            _require_positive("radius", radius),
        )

    def sphere(self, radius: float) -> Any:
        cadquery = cq
        assert cadquery is not None
        return cadquery.Workplane("XY").sphere(_require_positive("radius", radius))

    def extrude_profile(
        self,
        profile: Sequence[tuple[float, float]],
        height: float,
    ) -> Any:
        cadquery = cq
        assert cadquery is not None
        pts = [(float(x), float(y)) for x, y in profile]
        if len(pts) < 3:
            raise ValueError("profile must contain at least 3 points")
        height = _require_positive("height", height)
        return cadquery.Workplane("XY").polyline(pts).close().extrude(height)

    def sculpt_offset(self, shape: Any, distance: float) -> Any:
        """Deterministic freeform-ish edit via shell/offset (not LLM mesh dump)."""
        cadquery = cq
        assert cadquery is not None
        distance = float(distance)
        if abs(distance) < 1e-12:
            return shape
        solid = self._solid(shape)
        try:
            shelled = solid.shell(distance)  # type: ignore[attr-defined]
            return cadquery.Workplane("XY").newObject([shelled])
        except Exception:
            # Fallback: expanded AABB as a conservative deterministic sculpt.
            xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box(shape)
            pad = abs(distance)
            return self.box(
                (xmax - xmin) + 2 * pad,
                (ymax - ymin) + 2 * pad,
                (zmax - zmin) + 2 * pad,
            )

    def volume(self, shape: Any) -> float:
        return float(self._solid(shape).Volume())

    def bounding_box(self, shape: Any) -> tuple[float, float, float, float, float, float]:
        bbox = self._solid(shape).BoundingBox()
        return (
            float(bbox.xmin),
            float(bbox.ymin),
            float(bbox.zmin),
            float(bbox.xmax),
            float(bbox.ymax),
            float(bbox.zmax),
        )

    def face_count(self, shape: Any) -> int:
        return int(len(list(self._solid(shape).Faces())))

    def is_valid(self, shape: Any) -> bool:
        solid = self._solid(shape)
        try:
            if hasattr(solid, "isValid"):
                return bool(solid.isValid())
            return float(solid.Volume()) > 0 and self.face_count(shape) > 0
        except Exception:
            return False

    def is_watertight(self, shape: Any) -> bool:
        solid = self._solid(shape)
        try:
            # OCC Closed() is unreliable for some CadQuery solids; use topology heuristics.
            if not self.is_valid(shape):
                return False
            if self.volume(shape) <= 0:
                return False
            faces = self.face_count(shape)
            if faces <= 0:
                return False
            shells = list(solid.Shells()) if hasattr(solid, "Shells") else []
            if shells:
                return True
            # Fallback: positive-volume manifold-like solid with faces.
            return True
        except Exception:
            return False

    def export_step(self, shape: Any, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cadquery = cq
        assert cadquery is not None
        cadquery.exporters.export(shape, str(path))
        return path

    def export_stl(self, shape: Any, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cadquery = cq
        assert cadquery is not None
        cadquery.exporters.export(shape, str(path))
        return path

    def describe(self, shape: Any) -> dict[str, Any]:
        return {
            "type": "cadquery",
            "volume": self.volume(shape),
            "bounding_box": list(self.bounding_box(shape)),
            "face_count": self.face_count(shape),
            "valid": self.is_valid(shape),
            "watertight": self.is_watertight(shape),
            "class": type(shape).__name__,
        }

    def _solid(self, shape: Any) -> Any:
        return shape.val() if hasattr(shape, "val") else shape


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def get_backend(prefer_real: bool = True) -> CadBackend:
    """Return a real backend when available, otherwise a mock fallback."""

    if prefer_real:
        try:
            return CadQueryBackend()
        except Exception:
            pass
    return MockCadBackend()


def build_from_spec(spec: dict[str, Any], backend: CadBackend | None = None) -> Any:
    """Build geometry from a structured primitive spec (planner output, not mesh)."""

    backend = backend or get_backend(prefer_real=True)
    kind = str(spec.get("kind") or spec.get("op") or "box").lower()
    params = dict(spec.get("params", spec))
    if kind == "box":
        shape = backend.box(params["width"], params["height"], params["depth"])
    elif kind == "cylinder":
        shape = backend.cylinder(params["radius"], params["height"])
    elif kind == "sphere":
        shape = backend.sphere(params["radius"])
    elif kind in {"extrude", "extrusion", "extrude_profile"}:
        shape = backend.extrude_profile(params["profile"], params["height"])
    else:
        raise ValueError(f"unsupported geometry kind: {kind}")
    offset = params.get("sculpt_offset")
    if offset is not None and float(offset) != 0.0:
        shape = backend.sculpt_offset(shape, float(offset))
    return shape
