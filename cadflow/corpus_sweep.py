"""Parallel parametric sweep over geometry-bearing source trees.

This module turns existing CAD/CAE source directories into a broad set of
family-aware, solver-backed verification runs. The goal is to generate more
useful training data from real source designs, not to synthesize generic noise.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

from cadflow.backends import CadBackend
from cadflow.corpus_graph import build_processed_corpus_graph
from cadflow.evaluation_graph import build_flywheel_evaluation_graph
from cadflow.flywheel import DataFlywheel
from cadflow.graph_schema import GraphDocument, build_source_registry_graph
from cadflow.local_data_graph import build_local_data_graph
from cadflow.manifest import JobManifest
from cadflow.neo4j_store import Neo4jImportReport, import_graph_to_neo4j
from cadflow.pipeline import PipelineResult, run_pipeline
from cadflow.promotion import PromotionResult, promote_verified_to_dataset

_GEOMETRY_SUFFIXES = {".step", ".stp", ".stl", ".obj", ".ply", ".igs", ".iges", ".glb", ".gltf", ".x_t", ".x_b"}

_ENGINEERING_KEYWORDS = (
    "nozzle",
    "chamber",
    "combustion",
    "injector",
    "thrust",
    "engine",
    "tank",
    "propellant",
    "feed",
    "fairing",
    "nose cone",
    "nosecone",
    "cone",
    "fin",
    "fin can",
    "airframe",
    "body tube",
    "tube",
    "bracket",
    "adapter",
    "coupler",
    "retainer",
    "mechanism",
    "hinge",
    "latch",
    "arm",
    "valve",
    "manifold",
    "thermal",
    "shield",
    "blanket",
    "insulation",
    "structure",
    "panel",
    "bus",
    "spacecraft",
    "satellite",
    "rocket",
    "stage",
    "booster",
    "payload",
    "antenna",
    "deploy",
)

_REFERENCE_KEYWORDS = (
    "moon",
    "mars",
    "earth",
    "vesta",
    "tycho",
    "copernicus",
    "gassendi",
    "aristarchus",
    "supernova",
    "nebula",
    "crater",
    "insignia",
    "emblem",
    "logo",
    "artifact",
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "source"


@dataclass(frozen=True, slots=True)
class SourceProfile:
    path: Path
    family: str
    priority: int
    solver_suite: tuple[tuple[str, str, dict[str, Any]], ...]
    extents: tuple[float, float, float]
    raw_kind: str


@dataclass(frozen=True, slots=True)
class SweepCase:
    source_path: str
    source_key: str
    family: str
    suite_name: str
    solver: str
    variant_index: int
    variant_count: int
    geometry: dict[str, Any]
    test_profile: dict[str, Any]
    manifest: JobManifest


@dataclass(frozen=True, slots=True)
class SweepResult:
    discovered_sources: int
    sweep_cases: int
    run_ok: int
    verified: int
    promoted: int
    skipped_promotion: int
    source_report_path: str
    sweep_report_path: str
    curated_manifest_path: str
    master_graph_path: str
    neo4j_report: Neo4jImportReport
    promotion: PromotionResult
    pipeline_results: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return self.run_ok == self.sweep_cases and self.verified >= 1 and self.neo4j_report.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered_sources": self.discovered_sources,
            "sweep_cases": self.sweep_cases,
            "run_ok": self.run_ok,
            "verified": self.verified,
            "promoted": self.promoted,
            "skipped_promotion": self.skipped_promotion,
            "source_report_path": self.source_report_path,
            "sweep_report_path": self.sweep_report_path,
            "curated_manifest_path": self.curated_manifest_path,
            "master_graph_path": self.master_graph_path,
            "neo4j_report": self.neo4j_report.to_dict(),
            "promotion": self.promotion.to_dict(),
            "ok": self.ok,
        }


class _ThreadSafeFlywheel:
    def __init__(self, flywheel: DataFlywheel):
        self._flywheel = flywheel
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._flywheel.path

    def record(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._flywheel.record(*args, **kwargs)

    def promote_best(self, limit: int = 1):
        with self._lock:
            return self._flywheel.promote_best(limit=limit)

    def verified_entries(self):
        with self._lock:
            return self._flywheel.verified_entries()


def _text_blob(path: Path) -> str:
    return " ".join((path.name, *path.parts)).lower()


def classify_geometry_family(path: str | Path) -> str:
    """Classify a geometry-bearing source path into an engineering family."""

    p = Path(path)
    blob = _text_blob(p)

    if any(token in blob for token in ("nose_cone", "nosecone", "nose cone")):
        return "nose_cone"
    if any(token in blob for token in ("combustion", "chamber", "injector", "nozzle", "thrust", "engine")):
        return "combustion_chamber"
    if any(token in blob for token in ("tank", "propellant", "feed", "pressur", "copv", "header")):
        return "tank"
    if any(token in blob for token in ("fin can", "fincan", "fin", "wing", "tail", "control surface")):
        return "fin"
    if any(token in blob for token in ("fairing", "shroud", "shell", "heat shield", "heatshield", "insulation", "blanket", "thermal", "tps")):
        return "fairing"
    if any(token in blob for token in ("mechanism", "hinge", "latch", "linkage", "arm", "actuator", "retainer", "holder", "coupler", "adapter")):
        return "mechanism"
    if any(token in blob for token in ("frame", "bracket", "structure", "panel", "bus", "airframe", "body tube", "tube", "stage", "booster", "pod", "module")):
        return "structure"
    if any(token in blob for token in ("antenna", "boom", "dish", "solar array", "solar panel", "mast", "deploy")):
        return "deployable"
    if any(token in blob for token in _REFERENCE_KEYWORDS):
        return "reference_shape"
    if any(token in blob for token in ("spacecraft", "satellite", "probe", "orbiter", "lander", "vehicle")):
        return "spacecraft_bus"
    return "generic"


def _priority_for_family(family: str) -> int:
    if family in {"combustion_chamber", "tank", "nose_cone", "fin", "fairing", "mechanism", "structure", "deployable", "spacecraft_bus"}:
        return 3
    if family == "generic":
        return 2
    if family == "reference_shape":
        return 1
    return 2


def _solver_suite_for_family(family: str) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    if family == "nose_cone":
        return (
            ("openfoam", "external-flow", {"targets": {"Cd": 0.18, "stagnation_pressure": 1.0}}),
            ("fea", "shell-stiffness", {"targets": {"max_stress_mpa": 180.0, "displacement_mm": 1.5}}),
        )
    if family == "combustion_chamber":
        return (
            ("openfoam", "internal-flow", {"targets": {"pressure_drop": 0.15, "heat_flux": 1.0}}),
            ("fea", "wall-stress", {"targets": {"max_stress_mpa": 220.0, "wall_temperature_c": 900.0}}),
        )
    if family == "tank":
        return (
            ("fea", "hoop-stress", {"targets": {"max_stress_mpa": 140.0, "buckling_margin": 1.5}}),
            ("openfoam", "feed-drop", {"targets": {"pressure_drop": 0.08, "flow_uniformity": 0.9}}),
        )
    if family == "fin":
        return (
            ("openfoam", "aero-load", {"targets": {"Cd": 0.22, "Cl": 0.5}}),
            ("fea", "root-stress", {"targets": {"max_stress_mpa": 160.0, "deflection_mm": 2.0}}),
        )
    if family == "fairing":
        return (
            ("openfoam", "aero-shape", {"targets": {"Cd": 0.16, "heat_flux": 0.7}}),
            ("fea", "thermal-shell", {"targets": {"max_stress_mpa": 130.0, "wall_temperature_c": 380.0}}),
        )
    if family == "mechanism":
        return (
            ("mbd", "motion", {"targets": {"peak_torque": 22.0, "clearance_mm": 0.25}}),
            ("fea", "load-path", {"targets": {"max_stress_mpa": 110.0, "displacement_mm": 1.0}}),
        )
    if family == "structure":
        return (
            ("fea", "primary-structure", {"targets": {"max_stress_mpa": 150.0, "modal_frequency_hz": 35.0}}),
        )
    if family == "deployable":
        return (
            ("mbd", "deployment", {"targets": {"peak_torque": 18.0, "clearance_mm": 0.3}}),
            ("fea", "stowed-load", {"targets": {"max_stress_mpa": 100.0}}),
        )
    if family == "spacecraft_bus":
        return (
            ("fea", "bus-structure", {"targets": {"max_stress_mpa": 120.0, "modal_frequency_hz": 40.0}}),
            ("openfoam", "external-aero", {"targets": {"Cd": 0.21, "heat_flux": 0.8}}),
        )
    if family == "reference_shape":
        return (("fea", "reference-shape", {"targets": {"max_stress_mpa": 200.0}}),)
    return (
        ("fea", "generic-structure", {"targets": {"max_stress_mpa": 180.0}}),
    )


@dataclass(frozen=True, slots=True)
class SourceGeometryProfile:
    path: Path
    family: str
    priority: int
    solver_suite: tuple[tuple[str, str, dict[str, Any]], ...]
    extents: tuple[float, float, float]
    notes: str


def _safe_extents(path: Path, family: str) -> tuple[float, float, float]:
    size_kb = max(path.stat().st_size / 1024.0, 1.0)
    seed = abs(hash(path.name)) % 997
    family_scale = {
        "nose_cone": 1.85,
        "combustion_chamber": 1.7,
        "tank": 1.55,
        "fin": 1.45,
        "fairing": 1.6,
        "mechanism": 1.2,
        "structure": 1.1,
        "deployable": 1.25,
        "spacecraft_bus": 1.5,
        "reference_shape": 1.0,
        "generic": 1.0,
    }.get(family, 1.0)
    scale = family_scale * (0.35 + min(size_kb / 2048.0, 2.5))
    return (
        round(scale * (0.8 + 0.01 * (seed % 17)), 6),
        round(scale * (0.6 + 0.01 * ((seed // 17) % 17)), 6),
        round(scale * (0.45 + 0.01 * ((seed // 289) % 17)), 6),
    )


def discover_geometry_sources(raw_roots: Sequence[str | Path], *, recursive: bool = True, limit: int | None = None) -> list[SourceGeometryProfile]:
    """Discover geometry-bearing files and classify them for targeted sweeps."""

    candidates: list[Path] = []
    for root in raw_roots:
        base = Path(root)
        if not base.exists():
            continue
        iterator: Iterable[Path] = base.rglob("*") if recursive else base.glob("*")
        for path in iterator:
            if limit is not None and len(candidates) >= limit:
                break
            if path.is_file() and path.suffix.lower() in _GEOMETRY_SUFFIXES:
                candidates.append(path)
        if limit is not None and len(candidates) >= limit:
            break

    def _profile_path(path: Path) -> SourceGeometryProfile:
        family = classify_geometry_family(path)
        priority = _priority_for_family(family)
        suite = _solver_suite_for_family(family)
        return SourceGeometryProfile(
            path=path,
            family=family,
            priority=priority,
            solver_suite=suite,
            extents=_safe_extents(path, family),
            notes=f"{family} sweep source from {path.name}",
        )

    if not candidates:
        return []

    max_workers = min(32, max(1, len(candidates)))
    profiles: list[SourceGeometryProfile] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_profile_path, path) for path in candidates]
        for future in as_completed(futures):
            profiles.append(future.result())
    profiles.sort(key=lambda profile: str(profile.path))
    return profiles


def _variant_factor(index: int, count: int) -> float:
    if count <= 1:
        return 0.5
    return (index + 1) / float(count + 1)


def _base_geometry_profile(profile: SourceGeometryProfile) -> dict[str, Any]:
    x, y, z = profile.extents
    return {
        "extents": {"x": x, "y": y, "z": z},
        "source_path": str(profile.path),
        "source_name": profile.path.name,
        "family": profile.family,
    }


def _nose_cone_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    length = max(x, y, z) * (0.95 + 0.35 * t)
    radius = max(y, z) * (0.32 + 0.22 * (1.0 - t))
    tip = radius * (0.03 + 0.12 * (1.0 - abs(0.5 - t) * 2.0))
    roundness = 0.25 + 0.75 * t
    profile_pts = [
        (0.0, tip),
        (length * 0.05, tip * (1.0 + roundness)),
        (length * 0.18, radius * (0.18 + 0.12 * roundness)),
        (length * 0.52, radius * (0.55 + 0.22 * roundness)),
        (length * 0.82, radius * (0.92 - 0.05 * roundness)),
        (length, 0.0),
    ]
    return {
        "kind": "extrude",
        "profile": profile_pts,
        "height": max(radius * 2.5, z * (0.7 + 0.2 * t)),
        "features": [
            {"op": "fillet", "radius": round(max(0.005, radius * (0.015 + 0.05 * t)), 5)},
            {"op": "sculpt", "distance": round(0.01 * (variant_index + 1), 5)},
        ],
    }


def _combustion_chamber_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    length = max(x, y, z) * (0.85 + 0.45 * t)
    chamber_radius = max(y, z) * (0.38 + 0.18 * t)
    throat_radius = chamber_radius * (0.2 + 0.12 * (1.0 - t))
    exit_radius = chamber_radius * (1.6 + 0.45 * t)
    profile_pts = [
        (-length * 0.45, chamber_radius * 0.92),
        (-length * 0.15, chamber_radius * 0.98),
        (0.0, chamber_radius),
        (length * 0.2, throat_radius * 1.04),
        (length * 0.55, throat_radius * (0.98 + 0.05 * t)),
        (length * 0.88, exit_radius * 0.92),
        (length, exit_radius),
    ]
    return {
        "kind": "extrude",
        "profile": profile_pts,
        "height": max(chamber_radius * 2.0, z * (0.85 + 0.15 * t)),
        "features": [
            {"op": "fillet", "radius": round(max(0.004, throat_radius * 0.08), 5)},
            {"op": "sculpt", "distance": round(0.008 * (variant_index + 1), 5)},
        ],
    }


def _tank_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    radius = max(y, z) * (0.35 + 0.18 * t)
    cylinder_height = max(x, y, z) * (0.65 + 0.25 * t)
    dome_ratio = 0.2 + 0.45 * t
    wall_thickness = max(0.003, radius * (0.015 + 0.015 * (1.0 - t)))
    return {
        "kind": "assembly",
        "parts": [
            {"kind": "cylinder", "radius": radius, "height": cylinder_height},
            {"kind": "sphere", "radius": radius * dome_ratio},
            {"kind": "sphere", "radius": radius * dome_ratio},
        ],
        "features": [
            {"op": "sculpt_offset", "distance": round(0.006 * (variant_index + 1), 5)},
            {"op": "fillet", "radius": round(wall_thickness, 5)},
        ],
    }


def _fin_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    chord = max(x, y, z) * (0.6 + 0.3 * t)
    span = max(y, z) * (0.4 + 0.35 * t)
    sweep = chord * (0.08 + 0.25 * t)
    taper = 0.22 + 0.55 * t
    thickness = max(0.004, span * (0.03 + 0.04 * (1.0 - t)))
    profile_pts = [
        (0.0, 0.0),
        (chord * 0.35, span * 0.08),
        (chord * 0.76, span * 0.62),
        (chord, 0.0),
        (chord * taper, -span * 0.05),
    ]
    return {
        "kind": "extrude",
        "profile": profile_pts,
        "height": max(thickness * 2.0, z * (0.25 + 0.15 * t)),
        "features": [
            {"op": "fillet", "radius": round(max(0.003, thickness * 0.55), 5)},
            {"op": "sculpt", "distance": round(0.005 * (variant_index + 1), 5)},
        ],
    }


def _fairing_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    length = max(x, y, z) * (0.9 + 0.3 * t)
    radius = max(y, z) * (0.34 + 0.18 * t)
    shell = max(0.004, radius * (0.018 + 0.02 * (1.0 - t)))
    profile_pts = [
        (0.0, 0.0),
        (length * 0.08, radius * (0.08 + 0.02 * t)),
        (length * 0.2, radius * (0.28 + 0.06 * t)),
        (length * 0.52, radius * (0.78 + 0.1 * t)),
        (length * 0.82, radius * (0.98 - 0.03 * t)),
        (length, 0.0),
    ]
    return {
        "kind": "extrude",
        "profile": profile_pts,
        "height": max(shell * 2.0, z * (0.6 + 0.15 * t)),
        "features": [
            {"op": "fillet", "radius": round(shell, 5)},
            {"op": "sculpt", "distance": round(0.007 * (variant_index + 1), 5)},
        ],
    }


def _mechanism_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    arm = max(x, y, z) * (0.5 + 0.25 * t)
    pin = max(0.003, min(y, z) * (0.06 + 0.04 * (1.0 - t)))
    boss = pin * (2.2 + 1.0 * t)
    return {
        "kind": "assembly",
        "parts": [
            {"kind": "box", "width": arm, "height": boss, "depth": boss * 0.7},
            {"kind": "cylinder", "radius": pin, "height": arm * 0.8},
            {"kind": "cylinder", "radius": pin * 0.85, "height": arm * 0.45},
        ],
        "features": [
            {"op": "sculpt_offset", "distance": round(0.005 * (variant_index + 1), 5)},
        ],
    }


def _structure_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    width = max(x, y, z) * (0.7 + 0.2 * t)
    height = max(y, z) * (0.45 + 0.2 * t)
    depth = max(0.004, min(x, y, z) * (0.1 + 0.08 * (1.0 - t)))
    return {
        "kind": "box",
        "width": width,
        "height": height,
        "depth": depth,
        "features": [
            {"op": "cut", "tool": {"kind": "cylinder", "radius": depth * (0.6 + 0.2 * t), "height": depth * 3.0}},
            {"op": "fillet", "radius": round(max(0.002, depth * 0.3), 5)},
        ],
    }


def _spacecraft_bus_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    width = max(x, y, z) * (0.55 + 0.25 * t)
    height = max(y, z) * (0.55 + 0.15 * t)
    depth = max(0.01, min(x, y, z) * (0.18 + 0.1 * (1.0 - t)))
    return {
        "kind": "box",
        "width": width,
        "height": height,
        "depth": depth,
        "features": [
            {"op": "sculpt", "distance": round(0.004 * (variant_index + 1), 5)},
            {"op": "fillet", "radius": round(max(0.002, depth * 0.18), 5)},
        ],
    }


def _reference_geometry(profile: SourceGeometryProfile, t: float, variant_index: int) -> dict[str, Any]:
    x, y, z = profile.extents
    width = max(x, y, z) * (0.45 + 0.15 * t)
    height = max(y, z) * (0.45 + 0.15 * t)
    depth = max(0.02, min(x, y, z) * (0.45 + 0.15 * (1.0 - t)))
    return {"kind": "box", "width": width, "height": height, "depth": depth}


def build_variant_geometry(profile: SourceGeometryProfile, variant_index: int, variant_count: int) -> dict[str, Any]:
    """Construct a family-aware geometry spec for a particular source variant."""

    t = _variant_factor(variant_index, variant_count)
    family = profile.family
    if family == "nose_cone":
        return _nose_cone_geometry(profile, t, variant_index)
    if family == "combustion_chamber":
        return _combustion_chamber_geometry(profile, t, variant_index)
    if family == "tank":
        return _tank_geometry(profile, t, variant_index)
    if family == "fin":
        return _fin_geometry(profile, t, variant_index)
    if family == "fairing":
        return _fairing_geometry(profile, t, variant_index)
    if family == "mechanism":
        return _mechanism_geometry(profile, t, variant_index)
    if family == "structure":
        return _structure_geometry(profile, t, variant_index)
    if family == "deployable":
        return _spacecraft_bus_geometry(profile, t, variant_index)
    if family == "spacecraft_bus":
        return _spacecraft_bus_geometry(profile, t, variant_index)
    if family == "reference_shape":
        return _reference_geometry(profile, t, variant_index)
    return _reference_geometry(profile, t, variant_index)


def _source_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(Path(path.anchor))
    except ValueError:
        rel = resolved.relative_to(resolved.anchor) if resolved.anchor else resolved
    return _slugify(str(rel))


def build_sweep_cases(
    profiles: Sequence[SourceGeometryProfile],
    *,
    variants_per_source: int = 2,
    include_reference: bool = False,
) -> list[SweepCase]:
    cases: list[SweepCase] = []
    for profile in profiles:
        if profile.family == "reference_shape" and not include_reference:
            continue
        variant_count = max(1, variants_per_source if profile.priority >= 3 else max(1, variants_per_source - 1))
        for solver, suite_name, suite_payload in profile.solver_suite:
            for variant_index in range(variant_count):
                geometry = build_variant_geometry(profile, variant_index, variant_count)
                source_key = _source_key(profile.path)
                variant_tag = f"v{variant_index + 1:02d}"
                manifest = JobManifest(
                    name=f"sweep-{profile.family}-{source_key}-{suite_name}-{variant_tag}",
                    inputs={
                        "geometry": geometry,
                        "source_path": str(profile.path),
                        "source_key": source_key,
                        "source_family": profile.family,
                        "source_priority": profile.priority,
                        "source_extents": {"x": profile.extents[0], "y": profile.extents[1], "z": profile.extents[2]},
                        "variant_index": variant_index,
                        "variant_count": variant_count,
                        "suite_name": suite_name,
                        "test_profile": suite_payload,
                    },
                    parameters={
                        "solver": solver,
                        "family": profile.family,
                        "variant_index": variant_index,
                        "variant_count": variant_count,
                        **suite_payload,
                    },
                    tags=("parametric-sweep", profile.family, solver, suite_name),
                    notes=profile.notes,
                )
                cases.append(
                    SweepCase(
                        source_path=str(profile.path),
                        source_key=source_key,
                        family=profile.family,
                        suite_name=suite_name,
                        solver=solver,
                        variant_index=variant_index,
                        variant_count=variant_count,
                        geometry=geometry,
                        test_profile=suite_payload,
                        manifest=manifest,
                    )
                )
    return cases


def _merge_graphs(graphs: Sequence[GraphDocument], *, name: str, generated_at: str, metadata: dict[str, Any]) -> GraphDocument:
    node_map: dict[str, Any] = {}
    edge_map: dict[str, Any] = {}
    for graph in graphs:
        for node in graph.nodes:
            node_map.setdefault(node.id, node)
        for edge in graph.edges:
            edge_map.setdefault(edge.id, edge)
    return GraphDocument(
        name=name,
        generated_at=generated_at,
        nodes=tuple(node_map.values()),
        edges=tuple(edge_map.values()),
        metadata=metadata,
    )


def _safe_result_dict(case: SweepCase, result: PipelineResult) -> dict[str, Any]:
    return {
        "source_path": case.source_path,
        "source_key": case.source_key,
        "family": case.family,
        "suite_name": case.suite_name,
        "solver": case.solver,
        "variant_index": case.variant_index,
        "variant_count": case.variant_count,
        "manifest": result.run.manifest.to_dict(),
        "ok": result.ok,
        "solver_ok": result.solver_result.ok,
        "verification_ok": result.verification.passed,
        "objective": result.solver_result.objective,
        "artifact_refs": list(result.artifacts),
        "report_text": result.report_text,
    }


def run_parametric_corpus_sweep(
    raw_roots: Sequence[str | Path],
    out_dir: str | Path,
    *,
    flywheel_path: str | Path | None = None,
    seed_flywheel: str | Path | None = None,
    data_root: str | Path = "data/processed",
    variants_per_source: int = 2,
    include_reference: bool = False,
    max_sources: int | None = None,
    recursive: bool = True,
    max_workers: int = 8,
    backend: CadBackend | None = None,
    prefer_real_cad: bool = True,
    allow_solver_fallback: bool = True,
    num_points: int = 1024,
    num_fields: int = 6,
    fmt: str = "npz",
    promote_limit: int = 10_000,
) -> SweepResult:
    """Run a broad, parallel, family-aware corpus sweep and materialize artifacts."""

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sweep_dir = out_root / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_root = sweep_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    source_report_path = sweep_dir / "source-report.jsonl"
    sweep_report_path = sweep_dir / "sweep-report.jsonl"
    curated_dir = sweep_dir / "curated-dataset"
    curated_dir.mkdir(parents=True, exist_ok=True)
    master_graph_path = sweep_dir / "master-spaceflight-graph.json"
    neo4j_dir = sweep_dir / "neo4j"

    source_profiles = discover_geometry_sources(raw_roots, recursive=recursive, limit=max_sources)
    source_report = [
        {
            "path": str(profile.path),
            "family": profile.family,
            "priority": profile.priority,
            "solver_suite": [list(item[:2]) + [item[2]] for item in profile.solver_suite],
            "extents": {"x": profile.extents[0], "y": profile.extents[1], "z": profile.extents[2]},
            "notes": profile.notes,
        }
        for profile in source_profiles
    ]
    source_report_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in source_report) + ("\n" if source_report else ""), encoding="utf-8")

    cases = build_sweep_cases(source_profiles, variants_per_source=variants_per_source, include_reference=include_reference)
    backend = backend or None
    flywheel = DataFlywheel(flywheel_path or sweep_dir / "flywheel.jsonl")
    if seed_flywheel is not None:
        seed_path = Path(seed_flywheel)
        if seed_path.exists() and seed_path != flywheel.path:
            flywheel.path.write_text(seed_path.read_text(encoding="utf-8"), encoding="utf-8")
    flywheel_wrapper: Any = cast(Any, _ThreadSafeFlywheel(flywheel))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_pipeline,
                case.manifest,
                backend=backend,
                workdir=run_root,
                flywheel=flywheel_wrapper,
                source="cadflow.parametric_sweep",
                solver_kind=case.solver,
                prefer_real_cad=prefer_real_cad,
                allow_solver_fallback=allow_solver_fallback,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            row = _safe_result_dict(case, result)
            results.append(row)
            with sweep_report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    promotion = promote_verified_to_dataset(
        flywheel,
        curated_dir,
        limit=promote_limit,
        num_points=num_points,
        num_fields=num_fields,
        fmt=fmt,
    )
    curated_manifest = curated_dir / "manifest.json"
    processed_graph = build_processed_corpus_graph(curated_manifest, curated_dir)
    source_graph = build_source_registry_graph()
    local_graph = build_local_data_graph(Path(data_root))
    eval_graph = build_flywheel_evaluation_graph(flywheel.path)
    merged = _merge_graphs(
        [source_graph, local_graph.graph, processed_graph.graph, eval_graph.graph],
        name="master-spaceflight-database-sweep",
        generated_at=eval_graph.generated_at,
        metadata={
            "source_graph_nodes": len(source_graph.nodes),
            "local_graph_nodes": len(local_graph.graph.nodes),
            "curated_graph_nodes": len(processed_graph.graph.nodes),
            "evaluation_graph_nodes": len(eval_graph.graph.nodes),
            "source_graph_edges": len(source_graph.edges),
            "local_graph_edges": len(local_graph.graph.edges),
            "curated_graph_edges": len(processed_graph.graph.edges),
            "evaluation_graph_edges": len(eval_graph.graph.edges),
            "discovered_sources": len(source_profiles),
            "sweep_cases": len(cases),
            "promotion": promotion.to_dict(),
            "flywheel_path": str(flywheel.path),
        },
    )
    master_graph_path.write_text(json.dumps(merged.to_dict(), indent=2), encoding="utf-8")
    neo4j_report = import_graph_to_neo4j(merged, out_dir=neo4j_dir, database="neo4j")

    return SweepResult(
        discovered_sources=len(source_profiles),
        sweep_cases=len(cases),
        run_ok=sum(1 for row in results if row["ok"]),
        verified=sum(1 for row in results if row["verification_ok"]),
        promoted=promotion.promoted,
        skipped_promotion=promotion.skipped,
        source_report_path=str(source_report_path),
        sweep_report_path=str(sweep_report_path),
        curated_manifest_path=str(curated_manifest),
        master_graph_path=str(master_graph_path),
        neo4j_report=neo4j_report,
        promotion=promotion,
        pipeline_results=tuple(results),
    )
