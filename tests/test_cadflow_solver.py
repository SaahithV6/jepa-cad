from __future__ import annotations

from cadflow.solver import NativeProbeResult, SolverResult, probe_native_solver, wrap_solver_result


def test_probe_native_solver_missing_module_falls_back() -> None:
    probe = probe_native_solver("this_module_does_not_exist_12345")

    assert isinstance(probe, NativeProbeResult)
    assert probe.available is False
    assert "missing" in probe.reason.lower()


def test_wrap_solver_result_normalizes_mapping() -> None:
    result = wrap_solver_result(
        {
            "status": "optimal",
            "objective": 1.25,
            "iterations": 7,
            "metadata": {"backend": "mock"},
        }
    )

    assert isinstance(result, SolverResult)
    assert result.ok is True
    assert result.status == "optimal"
    assert result.objective == 1.25
    assert result.metadata["backend"] == "mock"
