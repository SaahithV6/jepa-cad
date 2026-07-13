"""CAD/CAE orchestration package."""

from .backends import CadQueryBackend, MockCadBackend, build_from_spec, get_backend
from .flywheel import DataFlywheel, FlywheelEntry
from .manifest import JobManifest, ProvenanceRecord, RunRecord
from .pipeline import PipelineResult, run_pipeline
from .solver import (
    NativeProbeResult,
    SolverResult,
    probe_fea,
    probe_mbd,
    probe_native_solver,
    probe_openfoam,
    probe_solver_binary,
    run_external_command,
    run_fallback_solver,
    wrap_solver_result,
)
from .verification import VerificationReport, render_verification_report, verify_solid

__all__ = [
    "CadQueryBackend",
    "DataFlywheel",
    "FlywheelEntry",
    "JobManifest",
    "MockCadBackend",
    "NativeProbeResult",
    "PipelineResult",
    "ProvenanceRecord",
    "RunRecord",
    "SolverResult",
    "VerificationReport",
    "build_from_spec",
    "get_backend",
    "probe_fea",
    "probe_mbd",
    "probe_native_solver",
    "probe_openfoam",
    "probe_solver_binary",
    "render_verification_report",
    "run_external_command",
    "run_fallback_solver",
    "run_pipeline",
    "verify_solid",
    "wrap_solver_result",
]
