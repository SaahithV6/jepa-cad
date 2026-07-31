"""Evaluation graph construction from verified and near-verified flywheel runs.

This graph turns real solver + verification history into first-class Part,
Assembly, Dimension, Feature, SimulationCase, TestCase, and FailureMode nodes so
JEPA training can see the iteration loop, not just raw geometry and provenance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .flywheel import DataFlywheel, FlywheelEntry
from .graph_schema import GraphDocument, GraphEdge, GraphNode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in text.strip()]
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "node"


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _geometry_artifact(entry: FlywheelEntry) -> str | None:
    for ref in entry.run.artifact_refs:
        path = Path(ref)
        if path.suffix.lower() in {".stl", ".obj", ".ply", ".step", ".stp", ".npz"} and path.exists():
            return str(path)
    for ref in entry.manifest.artifacts:
        path = Path(ref)
        if path.suffix.lower() in {".stl", ".obj", ".ply", ".step", ".stp", ".npz"} and path.exists():
            return str(path)
    return None


def _geometry_inputs(entry: FlywheelEntry) -> dict[str, Any]:
    geometry = entry.manifest.inputs.get("geometry")
    return dict(geometry) if isinstance(geometry, Mapping) else {}


def _material_input(entry: FlywheelEntry) -> dict[str, Any] | None:
    for key in ("material", "materials"):
        value = entry.manifest.inputs.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            return {"name": value}
    value = entry.manifest.parameters.get("material")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {"name": value}
    return None


def _infer_unit(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("pressure", "stress", "modulus", "yield", "young")):
        return "Pa"
    if any(token in lowered for token in ("heat_flux", "heat-flux")):
        return "W/m^2"
    if any(token in lowered for token in ("mass",)):
        return "kg"
    if any(token in lowered for token in ("volume",)):
        return "m^3"
    if any(token in lowered for token in ("angle",)):
        return "deg"
    if any(token in lowered for token in ("ratio", "objective", "score", "loss", "drag", "lift", "thrust")):
        return "unitless"
    if any(token in lowered for token in ("width", "height", "depth", "radius", "diameter", "length", "thickness", "span", "chord", "gap", "offset", "clearance")):
        return "m"
    return "unitless"


def _flatten_numeric(prefix: str, payload: Mapping[str, Any]) -> list[tuple[str, float, str]]:
    items: list[tuple[str, float, str]] = []
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            items.append((name, float(value), _infer_unit(name)))
        elif isinstance(value, Mapping):
            items.extend(_flatten_numeric(name, value))
    return items


def _part_kind(geometry: Mapping[str, Any]) -> str:
    kind = str(geometry.get("kind") or geometry.get("type") or "part").lower()
    if kind in {"assembly", "union", "multi-body", "multibody"} or isinstance(geometry.get("parts"), list):
        return "Assembly"
    return "Part"


def _part_properties(entry: FlywheelEntry) -> dict[str, Any]:
    geometry = _geometry_inputs(entry)
    kind = _part_kind(geometry)
    props: dict[str, Any] = {
        "name": entry.manifest.name,
        "part_class": str(geometry.get("kind") or geometry.get("type") or kind.lower()),
        "geometry_ref": _geometry_artifact(entry) or entry.manifest_fingerprint,
        "manifest_fingerprint": entry.manifest_fingerprint,
        "tags": list(entry.manifest.tags),
    }
    material = _material_input(entry)
    if material is not None:
        props["material_ref"] = str(material.get("name") or material.get("material") or material.get("family") or "material")
    if kind == "Assembly":
        parts = geometry.get("parts")
        if isinstance(parts, list):
            props["children_count"] = len(parts)
    return props


def _simulation_inputs(entry: FlywheelEntry) -> dict[str, Any]:
    return {
        "geometry": _geometry_inputs(entry),
        "parameters": dict(entry.manifest.parameters),
        "tags": list(entry.manifest.tags),
        "notes": entry.manifest.notes,
    }


def _tuning_guidance(
    entry: FlywheelEntry,
    *,
    solver_name: str,
    target_parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    targets = dict(target_parameters) if isinstance(target_parameters, Mapping) else {}
    family = entry.manifest.parameters.get("family")
    if not isinstance(family, str) or not family:
        family = str(_geometry_inputs(entry).get("family") or "unknown")
    if targets:
        guidance_kind = "target_window"
        rationale = "Keep the geometry within the declared target window and compare the solver response against the recorded objective."
    elif not entry.solver_result.ok or not entry.verification.passed:
        guidance_kind = "failure_recovery"
        rationale = "Use the failure evidence to tighten geometry, material, or solver settings before retrying."
    else:
        guidance_kind = "family_heuristic"
        rationale = "Use the family-specific sweep heuristics to explore nearby admissible geometry and solver settings."
    payload: dict[str, Any] = {
        "name": f"{entry.manifest.name} tuning guidance",
        "guidance_kind": guidance_kind,
        "solver": solver_name,
        "family": family,
        "targets": targets,
        "rationale": rationale,
        "manifest_fingerprint": entry.manifest_fingerprint,
        "recorded_at": entry.recorded_at,
        "observed_objective": entry.solver_result.objective,
        "solver_status": entry.solver_result.status,
        "verification_passed": entry.verification.passed,
    }
    return payload


@dataclass(frozen=True, slots=True)
class EvaluationGraphReport:
    name: str
    generated_at: str
    graph: GraphDocument
    flywheel_path: str
    entry_count: int
    verified_count: int
    part_count: int
    simulation_count: int
    test_count: int
    failure_mode_count: int
    dimension_count: int
    feature_count: int
    material_count: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "flywheel_path": self.flywheel_path,
            "entry_count": self.entry_count,
            "verified_count": self.verified_count,
            "part_count": self.part_count,
            "simulation_count": self.simulation_count,
            "test_count": self.test_count,
            "failure_mode_count": self.failure_mode_count,
            "dimension_count": self.dimension_count,
            "feature_count": self.feature_count,
            "material_count": self.material_count,
            "notes": list(self.notes),
            "graph": self.graph.to_dict(),
        }


def build_flywheel_evaluation_graph(
    flywheel_path: str | Path = "artifacts/flywheel.jsonl",
    *,
    include_unverified: bool = True,
) -> EvaluationGraphReport:
    flywheel = DataFlywheel(flywheel_path)
    entries = list(flywheel.load_entries())
    if not include_unverified:
        entries = [entry for entry in entries if entry.verified]

    entries.sort(key=lambda entry: (entry.recorded_at, entry.manifest_fingerprint))

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    part_count = 0
    simulation_count = 0
    test_count = 0
    failure_mode_count = 0
    dimension_count = 0
    feature_count = 0
    material_count = 0
    verified_count = 0
    kind_counts: Counter[str] = Counter()

    corpus_id = "corpus:flywheel"
    nodes.append(
        GraphNode(
            id=corpus_id,
            type="Corpus",
            label="flywheel",
            properties={
                "path": str(flywheel.path),
                "corpus_kind": "dataset_collection",
                "file_count": 1,
                "directory_count": 0,
            },
        )
    )

    flywheel_doc_id = "documentasset:flywheel-jsonl"
    if flywheel.path.exists():
        nodes.append(
            GraphNode(
                id=flywheel_doc_id,
                type="DocumentAsset",
                label=flywheel.path.name,
                properties={
                    "path": str(flywheel.path),
                    "document_kind": "flywheel-jsonl",
                    "line_count": sum(1 for _ in flywheel.path.open("r", encoding="utf-8")),
                    "size_bytes": flywheel.path.stat().st_size,
                },
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{corpus_id}:contains:flywheel-jsonl",
                type="CONTAINS",
                source=corpus_id,
                target=flywheel_doc_id,
                properties={"role": "flywheel_store"},
            )
        )

    previous_by_name: dict[str, dict[str, Any]] = {}

    for entry in entries:
        geometry = _geometry_inputs(entry)
        part_kind = _part_kind(geometry)
        kind_counts[part_kind] += 1
        if entry.verified:
            verified_count += 1

        part_id = f"part:{entry.manifest_fingerprint}"
        simulation_id = f"simulationcase:{entry.manifest_fingerprint}"
        test_id = f"testcase:{entry.manifest_fingerprint}"

        part_props = _part_properties(entry)
        nodes.append(GraphNode(id=part_id, type=part_kind, label=entry.manifest.name, properties=part_props))
        part_count += 1

        solver_name = str(entry.manifest.parameters.get("solver") or entry.solver_result.metadata.get("backend") or "unknown")
        simulation_props = {
            "solver": solver_name,
            "inputs": _simulation_inputs(entry),
            "outputs": entry.solver_result.to_dict(),
            "status": entry.solver_result.status,
            "manifest_fingerprint": entry.manifest_fingerprint,
            "recorded_at": entry.recorded_at,
        }
        nodes.append(GraphNode(id=simulation_id, type="SimulationCase", label=f"{entry.manifest.name} simulation", properties=simulation_props))
        simulation_count += 1

        verification = entry.verification
        test_props = {
            "test_kind": "verification",
            "conditions": {
                "backend": verification.backend,
                "solver_status": entry.solver_result.status,
                "manifest_fingerprint": entry.manifest_fingerprint,
            },
            "results": {
                **verification.metrics,
                "findings": list(verification.findings),
                "passed": verification.passed,
                "notes": verification.notes,
            },
            "manifest_fingerprint": entry.manifest_fingerprint,
            "recorded_at": entry.recorded_at,
        }
        nodes.append(GraphNode(id=test_id, type="TestCase", label=f"{entry.manifest.name} verification", properties=test_props))
        test_count += 1

        edges.append(
            GraphEdge(
                id=f"edge:{part_id}:simulated-in",
                type="SIMULATED_IN",
                source=part_id,
                target=simulation_id,
                properties={"solver": solver_name},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{part_id}:tested-in",
                type="TESTED_IN",
                source=part_id,
                target=test_id,
                properties={"test_kind": "verification"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{simulation_id}:verified-by",
                type="VERIFIED_BY",
                source=simulation_id,
                target=test_id,
                properties={"role": "solver_to_verification"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{simulation_id}:source-of-truth:{entry.manifest_fingerprint}",
                type="SOURCE_OF_TRUTH_FOR",
                source=simulation_id,
                target=part_id,
                properties={"role": "simulation_evidence"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{test_id}:source-of-truth:{entry.manifest_fingerprint}",
                type="SOURCE_OF_TRUTH_FOR",
                source=test_id,
                target=part_id,
                properties={"role": "verification_evidence"},
            )
        )

        target_parameters = entry.manifest.parameters.get("targets")

        guidance_payload = _tuning_guidance(
            entry,
            solver_name=solver_name,
            target_parameters=target_parameters if isinstance(target_parameters, Mapping) else None,
        )
        guidance_id = f"tuningguidance:{entry.manifest_fingerprint}"
        nodes.append(
            GraphNode(
                id=guidance_id,
                type="TuningGuidance",
                label=f"{entry.manifest.name} guidance",
                properties=guidance_payload,
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{guidance_id}:guides:{entry.manifest_fingerprint}",
                type="GUIDES",
                source=guidance_id,
                target=part_id,
                properties={"role": "tuning_guidance"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{guidance_id}:guides-simulation:{entry.manifest_fingerprint}",
                type="GUIDES",
                source=guidance_id,
                target=simulation_id,
                properties={"role": "solver_guidance"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{guidance_id}:guides-test:{entry.manifest_fingerprint}",
                type="GUIDES",
                source=guidance_id,
                target=test_id,
                properties={"role": "verification_guidance"},
            )
        )

        # Geometry dimensions are the tunable parameters the iteration loop should act on.
        dimension_specs = _flatten_numeric("geometry", geometry)
        if isinstance(target_parameters, Mapping):
            dimension_specs.extend(_flatten_numeric("target", target_parameters))
        if isinstance(entry.manifest.parameters.get("objective"), (int, float)):
            objective_value = float(entry.manifest.parameters["objective"])
            dimension_specs.append(("target.objective", objective_value, "unitless"))

        seen_dimension_names: set[str] = set()
        for dim_name, dim_value, unit in dimension_specs:
            if dim_name in seen_dimension_names:
                continue
            seen_dimension_names.add(dim_name)
            dim_id = f"dimension:{entry.manifest_fingerprint}:{_slug(dim_name)}"
            nodes.append(
                GraphNode(
                    id=dim_id,
                    type="Dimension",
                    label=dim_name,
                    properties={
                        "name": dim_name,
                        "value": dim_value,
                        "unit": unit,
                        "min_value": None,
                        "max_value": None,
                        "manifest_fingerprint": entry.manifest_fingerprint,
                    },
                )
            )
            dimension_count += 1
            edges.append(
                GraphEdge(
                    id=f"edge:{part_id}:dimension:{_slug(dim_name)}",
                    type="HAS_DIMENSION",
                    source=part_id,
                    target=dim_id,
                    properties={"role": "tunable_geometry"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{dim_id}:affects:{entry.manifest_fingerprint}",
                    type="AFFECTS",
                    source=dim_id,
                    target=simulation_id,
                    properties={"role": "tunable_parameter"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{dim_id}:tested:{entry.manifest_fingerprint}",
                    type="AFFECTS",
                    source=dim_id,
                    target=test_id,
                    properties={"role": "verification_parameter"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{guidance_id}:guides-dimension:{_slug(dim_name)}",
                    type="GUIDES",
                    source=guidance_id,
                    target=dim_id,
                    properties={"role": "target_window", "dimension_name": dim_name},
                )
            )

        features = geometry.get("features")
        if isinstance(features, list):
            for index, feature in enumerate(features):
                if isinstance(feature, Mapping):
                    feature_name = str(feature.get("name") or feature.get("kind") or feature.get("type") or f"feature_{index}")
                    feature_props = {
                        "feature_kind": str(feature.get("kind") or feature.get("type") or "feature"),
                        "geometry_ref": str(feature.get("geometry_ref") or entry.manifest_fingerprint),
                        "impact": str(feature.get("impact") or feature.get("role") or "geometry"),
                    }
                else:
                    feature_name = str(feature)
                    feature_props = {
                        "feature_kind": feature_name,
                        "geometry_ref": entry.manifest_fingerprint,
                        "impact": "geometry",
                    }
                feature_id = f"feature:{entry.manifest_fingerprint}:{_slug(feature_name)}:{index}"
                nodes.append(GraphNode(id=feature_id, type="Feature", label=feature_name, properties=feature_props))
                feature_count += 1
                edges.append(
                    GraphEdge(
                        id=f"edge:{part_id}:feature:{index}",
                        type="HAS_FEATURE",
                        source=part_id,
                        target=feature_id,
                        properties={"role": "geometry_feature"},
                    )
                )
                edges.append(
                    GraphEdge(
                        id=f"edge:{feature_id}:affects:{entry.manifest_fingerprint}",
                        type="AFFECTS",
                        source=feature_id,
                        target=simulation_id,
                        properties={"role": "feature_behavior"},
                    )
                )
                edges.append(
                    GraphEdge(
                        id=f"edge:{feature_id}:tested:{entry.manifest_fingerprint}",
                        type="AFFECTS",
                        source=feature_id,
                        target=test_id,
                        properties={"role": "feature_verification"},
                    )
                )

        material = _material_input(entry)
        if material is not None:
            material_name = str(material.get("name") or material.get("family") or material.get("material") or "material")
            material_id = f"material:{entry.manifest_fingerprint}:{_slug(material_name)}"
            nodes.append(
                GraphNode(
                    id=material_id,
                    type="Material",
                    label=material_name,
                    properties={
                        "name": material_name,
                        "family": material.get("family") if isinstance(material.get("family"), str) else None,
                        "properties": material,
                    },
                )
            )
            material_count += 1
            edges.append(
                GraphEdge(
                    id=f"edge:{part_id}:made-of",
                    type="MADE_OF",
                    source=part_id,
                    target=material_id,
                    properties={"role": "material_assignment"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{material_id}:affects:{entry.manifest_fingerprint}",
                    type="AFFECTS",
                    source=material_id,
                    target=simulation_id,
                    properties={"role": "material_behavior"},
                )
            )

        if not entry.solver_result.ok or not verification.passed:
            failure_kind = "solver_failed" if not entry.solver_result.ok else "verification_failed"
            severity = "critical" if not entry.solver_result.ok else "high"
            failure_id = f"failuremode:{entry.manifest_fingerprint}"
            nodes.append(
                GraphNode(
                    id=failure_id,
                    type="FailureMode",
                    label=failure_kind,
                    properties={
                        "failure_kind": failure_kind,
                        "severity": severity,
                        "symptoms": list(verification.findings) or [entry.solver_result.status],
                        "manifest_fingerprint": entry.manifest_fingerprint,
                    },
                )
            )
            failure_mode_count += 1
            edges.append(
                GraphEdge(
                    id=f"edge:{simulation_id}:failure:{entry.manifest_fingerprint}",
                    type="AFFECTS",
                    source=simulation_id,
                    target=failure_id,
                    properties={"role": "solver_failure"},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{test_id}:failure:{entry.manifest_fingerprint}",
                    type="AFFECTS",
                    source=test_id,
                    target=failure_id,
                    properties={"role": "verification_failure"},
                )
            )

        if entry.manifest.name in previous_by_name:
            prev = previous_by_name[entry.manifest.name]
            prev_simulation = prev["simulation"]
            prev_part = prev["part"]
            prev_objective = prev.get("objective")
            current_objective = entry.solver_result.objective
            delta: dict[str, Any] = {}
            if prev_objective is not None and current_objective is not None:
                delta["objective_delta"] = float(current_objective) - float(prev_objective)
            edges.append(
                GraphEdge(
                    id=f"edge:{part_id}:variant-of:{prev_part}",
                    type="VARIANT_OF",
                    source=part_id,
                    target=prev_part,
                    properties={"role": "iteration_step", **delta},
                )
            )
            edges.append(
                GraphEdge(
                    id=f"edge:{simulation_id}:variant-of:{prev_simulation}",
                    type="VARIANT_OF",
                    source=simulation_id,
                    target=prev_simulation,
                    properties={"role": "solver_iteration", **delta},
                )
            )
        previous_by_name[entry.manifest.name] = {
            "part": part_id,
            "simulation": simulation_id,
            "objective": entry.solver_result.objective if isinstance(entry.solver_result.objective, (int, float)) else None,
        }

    graph = GraphDocument(
        name="flywheel-evaluation-graph",
        generated_at=_utc_now(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "flywheel_path": str(flywheel.path),
            "entry_count": len(entries),
            "verified_count": verified_count,
            "part_count": part_count,
            "simulation_count": simulation_count,
            "test_count": test_count,
            "failure_mode_count": failure_mode_count,
            "dimension_count": dimension_count,
            "feature_count": feature_count,
            "material_count": material_count,
            "kind_counts": dict(kind_counts),
        },
    )
    notes = (
        "Real solver and verification runs are promoted into Part / SimulationCase / TestCase / FailureMode nodes.",
        "Geometry dimensions are split into explicit Dimension nodes so the iteration loop has concrete tuning targets.",
        "Variant_of edges chain repeated runs, making objective deltas and iteration history queryable in the graph.",
        "Failed runs are preserved rather than discarded, because negative evidence is useful for JEPA tuning and failure prediction.",
    )
    return EvaluationGraphReport(
        name=graph.name,
        generated_at=graph.generated_at,
        graph=graph,
        flywheel_path=str(flywheel.path),
        entry_count=len(entries),
        verified_count=verified_count,
        part_count=part_count,
        simulation_count=simulation_count,
        test_count=test_count,
        failure_mode_count=failure_mode_count,
        dimension_count=dimension_count,
        feature_count=feature_count,
        material_count=material_count,
        notes=notes,
    )


def render_evaluation_graph_report(report: EvaluationGraphReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2)
    lines = [
        f"name={report.name}",
        f"generated_at={report.generated_at}",
        f"flywheel_path={report.flywheel_path}",
        f"entry_count={report.entry_count}",
        f"verified_count={report.verified_count}",
        f"part_count={report.part_count}",
        f"simulation_count={report.simulation_count}",
        f"test_count={report.test_count}",
        f"failure_mode_count={report.failure_mode_count}",
        f"dimension_count={report.dimension_count}",
        f"feature_count={report.feature_count}",
        f"material_count={report.material_count}",
        "notes:",
    ]
    for note in report.notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
