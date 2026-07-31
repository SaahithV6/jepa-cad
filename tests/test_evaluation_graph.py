from __future__ import annotations

from pathlib import Path

from cadflow.backends import MockCadBackend
from cadflow.evaluation_graph import build_flywheel_evaluation_graph
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest, ProvenanceRecord, RunRecord
from cadflow.solver import SolverResult
from cadflow.verification import VerificationReport


def test_evaluation_graph_emits_tuning_guidance(tmp_path: Path) -> None:
    backend = MockCadBackend()
    solid = backend.box(1.0, 2.0, 3.0)
    stl_path = backend.export_stl(solid, tmp_path / "geom.stl")

    flywheel = DataFlywheel(tmp_path / "flywheel.jsonl")
    manifest = JobManifest(
        name="nozzle-iteration",
        inputs={"geometry": {"kind": "box", "width": 1.0, "height": 2.0, "depth": 3.0}},
        parameters={"solver": "fea", "targets": {"max_stress_mpa": 150.0, "buckling_margin": 1.5}, "objective": 0.25},
        tags=("jepa", "tuning"),
        artifacts=(str(stl_path),),
    )
    run = RunRecord(
        manifest=manifest,
        provenance=ProvenanceRecord.for_manifest(manifest, source="test"),
        status="verified",
        solver_result=SolverResult(status="optimal", objective=0.25).to_dict(),
        verification=VerificationReport(name="solid_verification", passed=True, metrics={"volume": 6.0}).to_dict(),
        artifact_refs=(str(stl_path),),
    )
    flywheel.record(
        run,
        SolverResult(status="optimal", objective=0.25),
        VerificationReport(name="solid_verification", passed=True, metrics={"volume": 6.0}),
    )

    report = build_flywheel_evaluation_graph(flywheel.path)
    node_types = {node.type for node in report.graph.nodes}
    edge_types = {edge.type for edge in report.graph.edges}

    assert "TuningGuidance" in node_types
    assert "GUIDES" in edge_types
    guidance = next(node for node in report.graph.nodes if node.type == "TuningGuidance")
    assert guidance.properties["guidance_kind"] == "target_window"
    assert guidance.properties["targets"]["max_stress_mpa"] == 150.0
