"""Tests for solver runtime resolution and diagnostics."""

from __future__ import annotations

from pathlib import Path

from cadflow.runtime import resolve_solver_runtime
from cadflow.solver import probe_solver_binary


def _make_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_runtime_resolves_binaries_and_library_paths(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    lib_dir = tmp_path / "lib"
    bin_dir.mkdir()
    lib_dir.mkdir()
    _make_executable(bin_dir / "simpleFoam", "#!/usr/bin/env bash\necho simpleFoam\n")
    _make_executable(bin_dir / "ccx", "#!/usr/bin/env bash\necho ccx\n")

    runtime = resolve_solver_runtime(bin_dirs=[bin_dir], lib_dirs=[lib_dir])

    assert runtime.command_path("simpleFoam") == bin_dir / "simpleFoam"
    assert runtime.command_path("ccx") == bin_dir / "ccx"
    assert str(lib_dir) in runtime.merged_env()["LD_LIBRARY_PATH"]
    assert runtime.diagnostics()["bin_dirs"] == [str(bin_dir)]


def test_probe_solver_binary_uses_runtime_search_dirs(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_executable(bin_dir / "blockMesh", "#!/usr/bin/env bash\necho blockMesh\n")

    runtime = resolve_solver_runtime(bin_dirs=[bin_dir])
    probe = probe_solver_binary(("blockMesh",), backend="openfoam", runtime=runtime)

    assert probe.available is True
    assert probe.details["binary"] == "blockMesh"
    assert probe.details["path"] == str(bin_dir / "blockMesh")
