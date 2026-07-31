"""Executable pre-training preflight for the JEPA/CAD/spaceflight stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from cadflow.cloud import build_cloud_training_plan
from cadflow.corpus_graph import build_processed_corpus_graph
from cadflow.doctor import build_doctor_report
from cadflow.e2e import run_end_to_end
from cadflow.evaluation_graph import build_flywheel_evaluation_graph
from cadflow.graph_schema import build_source_registry_graph, build_spaceflight_graph_schema
from cadflow.local_data_graph import build_local_data_graph
from cadflow.project import ProjectIntakeResult, intake_project
from cadflow.source_validation import validate_source_registry
from data.graph_dataset import GraphBackedCADDataset


@dataclass(frozen=True, slots=True)
class PreflightSection:
    name: str
    passed: bool
    required: bool
    summary: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
            "summary": self.summary,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class PretrainingPreflightReport:
    checked_at: str
    project_intake: dict[str, Any] | None
    doctor: dict[str, Any]
    source_validation: dict[str, Any]
    graph_schema: dict[str, Any]
    source_registry_graph: dict[str, Any]
    local_data_graph: dict[str, Any]
    cloud_plan: dict[str, Any] | None
    smoke_result: dict[str, Any] | None
    corpus_graph: dict[str, Any] | None
    graph_dataset: dict[str, Any] | None
    evaluation_graph: dict[str, Any] | None
    sections: tuple[PreflightSection, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return all(section.passed or not section.required for section in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "project_intake": self.project_intake,
            "doctor": self.doctor,
            "source_validation": self.source_validation,
            "graph_schema": self.graph_schema,
            "source_registry_graph": self.source_registry_graph,
            "local_data_graph": self.local_data_graph,
            "cloud_plan": self.cloud_plan,
            "smoke_result": self.smoke_result,
            "corpus_graph": self.corpus_graph,
            "graph_dataset": self.graph_dataset,
            "evaluation_graph": self.evaluation_graph,
            "sections": [section.to_dict() for section in self.sections],
            "notes": list(self.notes),
            "ok": self.ok,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_intake_payload(result: ProjectIntakeResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return result.to_dict()


def _graph_summary(graph: Any) -> dict[str, Any]:
    payload = graph.to_dict() if hasattr(graph, "to_dict") else dict(graph)
    return {
        "name": payload.get("name"),
        "generated_at": payload.get("generated_at"),
        "metadata": payload.get("metadata", {}),
        "node_count": len(payload.get("nodes", [])),
        "edge_count": len(payload.get("edges", [])),
    }


def _section(name: str, passed: bool, required: bool, summary: str, **evidence: Any) -> PreflightSection:
    return PreflightSection(name=name, passed=passed, required=required, summary=summary, evidence=evidence)


def run_pretraining_preflight(
    *,
    project_root: str | Path,
    goal: str,
    family: str = "space",
    material: str | None = None,
    out_dir: str | Path = "artifacts/preflight",
    data_root: str | Path = "data",
    raw_dirs: Sequence[str | Path] | None = None,
    config: str | Path = "configs/base.yaml",
    data_source: str = "real",
    probe_data_source: str = "real",
    max_steps: int = 1,
    run_smoke: bool = False,
    smoke_num_points: int = 1024,
    smoke_num_fields: int = 6,
    smoke_format: str = "npz",
) -> PretrainingPreflightReport:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    intake = intake_project(
        project_root,
        goal=goal,
        family=family,
        material=material,
        out_dir=out_root / "project_intake",
    )
    manifest = intake.manifest

    # For graph-backed training, skip expensive graph builds
    skip_graph_validation = False  # Always validate graph integrity regardless of data source
    
    doctor = build_doctor_report()
    source_validation = validate_source_registry()
    graph_schema = build_spaceflight_graph_schema()
    source_registry_graph = build_source_registry_graph()
    # Skip expensive local data scan for graph-backed training
    if data_source != "graph":
        local_data_graph = build_local_data_graph(data_root)
    else:
        # Minimal stub for graph-backed training
        class _GraphStub:
            file_count = 0
            sample_count = 0
            corpus_count = 0
            category_counts = {}
            def to_dict(self): return {}
        local_data_graph = _GraphStub()
    cloud_plan = build_cloud_training_plan(manifest, family=family)

    evaluation_graph_report = None if skip_graph_validation else build_flywheel_evaluation_graph()
    evaluation_graph_payload = (
        {} if evaluation_graph_report is None else {
            **_graph_summary(evaluation_graph_report.graph),
            "flywheel_path": evaluation_graph_report.flywheel_path,
            "entry_count": evaluation_graph_report.entry_count,
            "verified_count": evaluation_graph_report.verified_count,
            "part_count": evaluation_graph_report.part_count,
            "simulation_count": evaluation_graph_report.simulation_count,
            "test_count": evaluation_graph_report.test_count,
            "failure_mode_count": evaluation_graph_report.failure_mode_count,
            "dimension_count": evaluation_graph_report.dimension_count,
            "feature_count": evaluation_graph_report.feature_count,
            "material_count": evaluation_graph_report.material_count,
            "notes": list(evaluation_graph_report.notes),
        }
    )
    if evaluation_graph_report is not None:
        evaluation_graph_path = out_root / "evaluation-graph.json"
        evaluation_graph_path.write_text(json.dumps(evaluation_graph_report.graph.to_dict(), indent=2), encoding="utf-8")

    smoke_result = None
    corpus_graph_payload: dict[str, Any] | None = None
    graph_dataset_payload: dict[str, Any] | None = None
    notes: list[str] = []

    if run_smoke:
        if not raw_dirs:
            raise ValueError("run_smoke=True requires at least one raw_dir")
        smoke_result = run_end_to_end(
            raw_dirs,
            out_root / "smoke_dataset",
            num_points=smoke_num_points,
            num_fields=smoke_num_fields,
            fmt=smoke_format,
            config=config,
            family=family,
            data_source=data_source,
            max_steps=max_steps,
        )
        smoke_payload = smoke_result.to_dict()
        if smoke_result.ok:
            processed_dir = Path(smoke_result.ingestion.manifest_path).parent
            corpus_report = build_processed_corpus_graph(smoke_result.ingestion.manifest_path, processed_dir)
            corpus_graph_payload = {
                **_graph_summary(corpus_report.graph),
                "manifest_path": corpus_report.manifest_path,
                "processed_dir": corpus_report.processed_dir,
                "shard_count": corpus_report.shard_count,
                "raw_asset_count": corpus_report.raw_asset_count,
                "sample_count": corpus_report.sample_count,
                "notes": list(corpus_report.notes),
            }
            corpus_graph_path = out_root / "corpus-graph.json"
            corpus_graph_path.write_text(json.dumps(corpus_report.graph.to_dict(), indent=2), encoding="utf-8")
            graph_dataset = GraphBackedCADDataset(corpus_graph_path, data_root=processed_dir, num_points=smoke_num_points, num_fields=smoke_num_fields)
            first_sample = graph_dataset[0]
            graph_dataset_payload = {
                "graph_path": str(corpus_graph_path),
                "data_root": str(processed_dir),
                "records": len(graph_dataset),
                "first_record_type": graph_dataset.records[0].node_type,
                "sample_keys": sorted(first_sample.keys()),
                "points_shape": list(first_sample["points"].shape),
                "fields_shape": list(first_sample["fields"].shape),
                "graph_metadata_shape": list(first_sample["graph_metadata"].shape),
            }
        else:
            notes.append("smoke run completed but training did not succeed")
    else:
        smoke_payload = None

    sections = (
        _section(
            "training-goal",
            passed=True,
            required=True,
            summary="Project goal and family were ingested into a manifest.",
            manifest_fingerprint=manifest.fingerprint,
            manifest_path=str(out_root / "project_intake" / "project_manifest.json"),
            goal=goal,
            family=family,
        ),
        _section(
            "corpus-plan",
            passed=bool(cloud_plan.dataset_sources),
            required=True,
            summary="Cloud plan selected space-relevant sources and staged preprocessing steps.",
            dataset_source_count=len(cloud_plan.dataset_sources),
            primary_provider=cloud_plan.primary_provider,
            secondary_provider=cloud_plan.secondary_provider,
        ),
        _section(
            "graph-wiring",
            passed=bool(source_registry_graph.nodes) and bool(source_registry_graph.edges) and bool(graph_schema.node_types) and bool(graph_schema.edge_types),
            required=True,
            summary="Schema and registry graphs are materializable with provenance-bearing nodes and edges.",
            graph_schema_nodes=0 if isinstance(graph_schema, dict) else len(graph_schema.node_types),
            graph_schema_edges=0 if isinstance(graph_schema, dict) else len(graph_schema.edge_types),
            registry_graph_nodes=0 if isinstance(source_registry_graph, dict) else len(source_registry_graph.nodes),
            registry_graph_edges=0 if isinstance(source_registry_graph, dict) else len(source_registry_graph.edges),
        ),
        _section(
            "data-cleaning",
            passed=data_source == "graph" or (local_data_graph.file_count > 0 and local_data_graph.sample_count >= 0),
            required=data_source != "graph",
            summary="Local data tree is scannable with sample and analogue summaries." if data_source != "graph" else "Graph-backed training (data-cleaning skipped).",
            data_root=str(Path(data_root).resolve()),
            corpus_count=local_data_graph.corpus_count,
            file_count=local_data_graph.file_count,
            sample_count=local_data_graph.sample_count,
            category_counts=local_data_graph.category_counts,
        ),
        _section(
            "training-representation",
            passed=graph_dataset_payload is not None or not run_smoke,
            required=run_smoke,
            summary="Graph-backed dataset metadata can be materialized for the JEPA input path.",
            graph_dataset=graph_dataset_payload,
        ),
        _section(
            "evaluation-graph",
            passed=bool(evaluation_graph_payload and evaluation_graph_payload.get("node_count", 0) > 0),
            required=True,
            summary="Flywheel solver + verification history can be materialized into a closed-loop evaluation graph.",
            evaluation_graph=evaluation_graph_payload,
        ),
        _section(
            "smoke-training",
            passed=bool(smoke_payload and smoke_payload.get("ok")),
            required=run_smoke,
            summary="Tiny ingest -> train smoke pass completed successfully.",
            smoke_result=smoke_payload,
        ),
        _section(
            "infrastructure",
            passed=bool(doctor.get("native_ready")),
            required=True,
            summary="Native solver runtime is usable enough for the preflight stack.",
            ready_backends=doctor.get("ready_backends", []),
            missing_backends=doctor.get("missing_backends", []),
        ),
        _section(
            "tracking-and-provenance",
            passed=bool(intake.manifest.fingerprint) and bool(manifest.artifacts == ()),
            required=True,
            summary="Manifest/provenance tracking is deterministic and artifact-free before execution.",
            fingerprint=manifest.fingerprint,
            provenance=intake.manifest.to_dict(),
        ),
    )

    return PretrainingPreflightReport(
        checked_at=_utc_now(),
        project_intake=_project_intake_payload(intake),
        doctor=doctor,
        source_validation=source_validation.to_dict(),
        graph_schema=graph_schema.to_dict(),
        source_registry_graph=source_registry_graph.to_dict(),
        local_data_graph=local_data_graph.to_dict(),
        cloud_plan=cloud_plan.to_dict(),
        smoke_result=smoke_payload,
        corpus_graph=corpus_graph_payload,
        graph_dataset=graph_dataset_payload,
        evaluation_graph=evaluation_graph_payload,
        sections=sections,
        notes=tuple(notes),
    )


def render_pretraining_preflight(report: PretrainingPreflightReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)

    lines = [
        "JEPA/CAD pre-training preflight",
        f"ok={report.ok}",
        f"checked_at={report.checked_at}",
    ]
    if report.project_intake:
        lines.append(f"manifest={report.project_intake['manifest']['fingerprint']}")
    lines.append(f"native_ready={report.doctor.get('native_ready', False)}")
    lines.append(f"usable_sources={report.source_validation['counts'].get('usable', 0)}")
    lines.append(f"graph_nodes={len(report.source_registry_graph.get('nodes', []))}")
    lines.append(f"graph_edges={len(report.source_registry_graph.get('edges', []))}")
    if report.cloud_plan:
        lines.append(f"cloud_sources={len(report.cloud_plan['dataset_sources'])}")
    if report.smoke_result:
        lines.append(f"smoke_ok={report.smoke_result.get('ok', False)}")
    for section in report.sections:
        mark = "PASS" if section.passed or not section.required else "FAIL"
        req = "required" if section.required else "optional"
        lines.append(f"{mark} [{req}] {section.name}: {section.summary}")
    if report.notes:
        lines.append("notes:")
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)
