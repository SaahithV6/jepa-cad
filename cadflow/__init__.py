"""CAD/CAE orchestration package."""

from .adapters import FEAAdapter, MBDAdapter, OpenFOAMAdapter, get_adapter, run_solver
from .backends import CadQueryBackend, MockCadBackend, build_from_spec, get_backend
from .flywheel import DataFlywheel, FlywheelEntry
from .manifest import JobManifest, ProvenanceRecord, RunRecord
from .pipeline import PipelineResult, run_pipeline
from .promotion import PromotionResult, promote_verified_to_dataset
from .runtime import SolverRuntime, resolve_solver_runtime
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


def cli_main(*args, **kwargs):
    """Lazy wrapper to avoid importing cadflow.cli at package import time."""

    from .cli import main as _main

    return _main(*args, **kwargs)

__all__ = [
    "CadQueryBackend",
    "DataFlywheel",
    "FEAAdapter",
    "FlywheelEntry",
    "JobManifest",
    "MBDAdapter",
    "MockCadBackend",
    "NativeProbeResult",
    "OpenFOAMAdapter",
    "PipelineResult",
    "PromotionResult",
    "ProvenanceRecord",
    "RunRecord",
    "SolverResult",
    "SolverRuntime",
    "VerificationReport",
    "build_from_spec",
    "cli_main",
    "get_adapter",
    "get_backend",
    "probe_fea",
    "probe_mbd",
    "probe_native_solver",
    "probe_openfoam",
    "probe_solver_binary",
    "promote_verified_to_dataset",
    "render_verification_report",
    "resolve_solver_runtime",
    "run_external_command",
    "run_fallback_solver",
    "run_pipeline",
    "run_solver",
    "verify_solid",
    "wrap_solver_result",
]
