"""CAD/CAE orchestration scaffold package."""

from .backends import CadQueryBackend, MockCadBackend, get_backend
from .flywheel import DataFlywheel, FlywheelEntry
from .manifest import JobManifest, ProvenanceRecord, RunRecord
from .solver import NativeProbeResult, SolverResult, probe_native_solver, wrap_solver_result
from .verification import VerificationReport, render_verification_report, verify_solid

__all__ = [
    "CadQueryBackend",
    "DataFlywheel",
    "FlywheelEntry",
    "JobManifest",
    "MockCadBackend",
    "NativeProbeResult",
    "ProvenanceRecord",
    "RunRecord",
    "SolverResult",
    "VerificationReport",
    "get_backend",
    "probe_native_solver",
    "render_verification_report",
    "verify_solid",
    "wrap_solver_result",
]
