from __future__ import annotations

from types import SimpleNamespace

from cadflow.solver import (
    NativeProbeResult,
    SolverResult,
    probe_fea,
    probe_native_solver,
    probe_openfoam,
    probe_solver_binary,
    run_external_command,
    run_fallback_solver,
    wrap_solver_result,
)


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


def test_wrap_solver_result_attribute_object() -> None:
    obj = SimpleNamespace(status="converged", objective=0.5, iterations=3, residual=1e-6, metadata={"k": 1})
    result = wrap_solver_result(obj)
    assert result.ok is True
    assert result.iterations == 3


def test_wrap_subprocess_json_stdout() -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout='{"status": "solved", "objective": 2.5, "metadata": {"solver": "x"}}',
        stderr="",
    )
    result = wrap_solver_result(completed)
    assert result.ok is True
    assert result.objective == 2.5
    assert result.metadata["returncode"] == 0
    assert result.metadata["solver"] == "x"


def test_wrap_subprocess_nonzero_exit() -> None:
    completed = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    result = wrap_solver_result(completed)
    assert result.ok is False
    assert result.status == "failed"
    assert "boom" in result.logs[-1]


def test_probe_binaries_and_fallback_solver() -> None:
    missing = probe_solver_binary(("definitely_missing_solver_xyz",), backend="demo")
    assert missing.available is False
    foam = probe_openfoam()
    fea = probe_fea()
    assert foam.backend == "openfoam"
    assert fea.backend == "fea"

    fallback = run_fallback_solver(backend="fea", objective=0.25, probe=missing)
    assert fallback.ok is True
    assert fallback.metadata["mode"] == "fallback"


def test_run_external_command_falls_back_when_missing() -> None:
    result = run_external_command(("no_such_solver_abc", "--help"), backend="custom", allow_fallback=True)
    assert result.ok is True
    assert result.probe is not None
    assert result.probe.available is False
