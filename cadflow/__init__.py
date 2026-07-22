"""CAD/CAE orchestration package with lazy public exports.

Lazy loading keeps lightweight geometry/doctor clients independent from the
PyTorch training stack while preserving the package's original public API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AutopilotResult": ("autopilot", "AutopilotResult"),
    "run_autopilot": ("autopilot", "run_autopilot"),
    "CadQueryBackend": ("backends", "CadQueryBackend"),
    "MockCadBackend": ("backends", "MockCadBackend"),
    "build_from_spec": ("backends", "build_from_spec"),
    "get_backend": ("backends", "get_backend"),
    "FEAAdapter": ("adapters", "FEAAdapter"),
    "MBDAdapter": ("adapters", "MBDAdapter"),
    "OpenFOAMAdapter": ("adapters", "OpenFOAMAdapter"),
    "get_adapter": ("adapters", "get_adapter"),
    "run_solver": ("adapters", "run_solver"),
    "EndToEndResult": ("e2e", "EndToEndResult"),
    "run_end_to_end": ("e2e", "run_end_to_end"),
    "DataFlywheel": ("flywheel", "DataFlywheel"),
    "FlywheelEntry": ("flywheel", "FlywheelEntry"),
    "FlywheelLoopResult": ("flywheel_loop", "FlywheelLoopResult"),
    "run_flywheel_loop": ("flywheel_loop", "run_flywheel_loop"),
    "LoopControllerResult": ("loop_controller", "LoopControllerResult"),
    "run_loop_controller": ("loop_controller", "run_loop_controller"),
    "IngestionResult": ("ingest", "IngestionResult"),
    "ingest_sources": ("ingest", "ingest_sources"),
    "ProjectIntakeResult": ("project", "ProjectIntakeResult"),
    "intake_project": ("project", "intake_project"),
    "CloudTrainingPlan": ("cloud", "CloudTrainingPlan"),
    "build_cloud_training_plan": ("cloud", "build_cloud_training_plan"),
    "DesignCycleResult": ("design_loop", "DesignCycleResult"),
    "DesignLoopResult": ("design_loop", "DesignLoopResult"),
    "run_design_loop": ("design_loop", "run_design_loop"),
    "JobManifest": ("manifest", "JobManifest"),
    "ProvenanceRecord": ("manifest", "ProvenanceRecord"),
    "RunRecord": ("manifest", "RunRecord"),
    "PipelineResult": ("pipeline", "PipelineResult"),
    "run_pipeline": ("pipeline", "run_pipeline"),
    "PromotionResult": ("promotion", "PromotionResult"),
    "promote_verified_to_dataset": ("promotion", "promote_verified_to_dataset"),
    "SolverRuntime": ("runtime", "SolverRuntime"),
    "resolve_solver_runtime": ("runtime", "resolve_solver_runtime"),
    "NativeProbeResult": ("solver", "NativeProbeResult"),
    "SolverResult": ("solver", "SolverResult"),
    "probe_fea": ("solver", "probe_fea"),
    "probe_mbd": ("solver", "probe_mbd"),
    "probe_native_solver": ("solver", "probe_native_solver"),
    "probe_openfoam": ("solver", "probe_openfoam"),
    "probe_solver_binary": ("solver", "probe_solver_binary"),
    "run_external_command": ("solver", "run_external_command"),
    "run_fallback_solver": ("solver", "run_fallback_solver"),
    "wrap_solver_result": ("solver", "wrap_solver_result"),
    "VerificationReport": ("verification", "VerificationReport"),
    "render_verification_report": ("verification", "render_verification_report"),
    "verify_solid": ("verification", "verify_solid"),
}

__all__ = [*_EXPORTS, "cli_main"]


def __getattr__(name: str) -> Any:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = export
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def cli_main(*args: Any, **kwargs: Any) -> Any:
    """Invoke the CLI without importing it during package initialization."""

    return import_module(".cli", __name__).main(*args, **kwargs)
