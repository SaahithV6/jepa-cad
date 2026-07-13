"""Solver runtime discovery and environment wiring.

The runtime is intentionally small and explicit:
- locate solver binaries from configured bin dirs / PATH
- collect library dirs for subprocess execution
- surface a compact diagnostic summary for CLI/doctor output
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


_DEFAULT_COMMANDS: tuple[str, ...] = (
    "simpleFoam",
    "foamRun",
    "blockMesh",
    "ccx",
    "CalculiX",
    "ElmerSolver",
    "elmer",
    "chrono",
    "projectchrono",
    "mbdyn",
)


def _split_env_paths(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    paths = []
    for chunk in value.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            paths.append(Path(chunk).expanduser())
    return tuple(paths)


def _unique_paths(paths: Sequence[Path | str]) -> tuple[Path, ...]:
    seen: set[str] = set()
    out: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return tuple(out)


def _discover_dirs(root: Path | None) -> tuple[Path, ...]:
    if root is None:
        return ()
    candidates: list[Path] = []
    for path in root.rglob("bin"):
        if path.is_dir():
            candidates.append(path)
    for path in root.rglob("lib"):
        if path.is_dir():
            candidates.append(path)
    for path in root.rglob("x86_64-linux-gnu"):
        if path.is_dir():
            candidates.append(path)
    return _unique_paths(candidates)


@dataclass(frozen=True, slots=True)
class SolverRuntime:
    """Resolved solver runtime configuration."""

    root: Path | None = None
    bin_dirs: tuple[Path, ...] = ()
    lib_dirs: tuple[Path, ...] = ()
    command_paths: dict[str, Path] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    def search_path(self) -> str:
        parts = [str(path) for path in self.bin_dirs if str(path)]
        if os.environ.get("PATH"):
            parts.extend(part for part in os.environ["PATH"].split(os.pathsep) if part)
        return os.pathsep.join(parts)

    def command_path(self, name: str) -> Path | None:
        if name in self.command_paths:
            return self.command_paths[name]
        search_path = self.search_path()
        resolved = shutil.which(name, path=search_path or None)
        return Path(resolved) if resolved else None

    def merged_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env)
        if self.root is not None:
            env.setdefault("CADFLOW_SOLVER_ROOT", str(self.root))
        if self.bin_dirs:
            env["CADFLOW_SOLVER_BIN_DIRS"] = os.pathsep.join(str(path) for path in self.bin_dirs)
            env["PATH"] = os.pathsep.join(
                [str(path) for path in self.bin_dirs] + [part for part in env.get("PATH", "").split(os.pathsep) if part]
            )
        if self.lib_dirs:
            env["CADFLOW_SOLVER_LIB_DIRS"] = os.pathsep.join(str(path) for path in self.lib_dirs)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(path) for path in self.lib_dirs] + [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
            )
        return env

    def diagnostics(self) -> dict[str, Any]:
        return {
            "root": str(self.root) if self.root is not None else None,
            "bin_dirs": [str(path) for path in self.bin_dirs],
            "lib_dirs": [str(path) for path in self.lib_dirs],
            "command_paths": {name: str(path) for name, path in sorted(self.command_paths.items())},
            "search_path": self.search_path(),
        }

    def is_native_ready(self, commands: Mapping[str, Sequence[str]] | None = None) -> bool:
        checks = commands or {
            "openfoam": ("simpleFoam", "foamRun", "blockMesh"),
            "fea": ("ccx", "CalculiX", "ElmerSolver", "elmer"),
            "mbd": ("chrono", "projectchrono", "mbdyn"),
        }
        return all(any(self.command_path(candidate) is not None for candidate in candidates) for candidates in checks.values())


def resolve_solver_runtime(
    *,
    root: str | Path | None = None,
    bin_dirs: Sequence[str | Path] | None = None,
    lib_dirs: Sequence[str | Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> SolverRuntime:
    """Resolve solver runtime configuration from explicit args and environment."""

    env_map = dict(env or {})
    env_root = env_map.get("CADFLOW_SOLVER_ROOT", os.environ.get("CADFLOW_SOLVER_ROOT"))
    env_bin_dirs = env_map.get("CADFLOW_SOLVER_BIN_DIRS", os.environ.get("CADFLOW_SOLVER_BIN_DIRS"))
    env_lib_dirs = env_map.get("CADFLOW_SOLVER_LIB_DIRS", os.environ.get("CADFLOW_SOLVER_LIB_DIRS"))

    resolved_root = Path(root).expanduser() if root is not None else (Path(env_root).expanduser() if env_root else None)

    resolved_bin_dirs = list(_split_env_paths(env_bin_dirs)) + [Path(path).expanduser() for path in (bin_dirs or ())]
    resolved_lib_dirs = list(_split_env_paths(env_lib_dirs)) + [Path(path).expanduser() for path in (lib_dirs or ())]

    if resolved_root is not None:
        if not resolved_bin_dirs:
            resolved_bin_dirs.extend(_discover_dirs(resolved_root))
        if not resolved_lib_dirs:
            resolved_lib_dirs.extend(_discover_dirs(resolved_root))

    resolved_bin_dirs = _unique_paths(resolved_bin_dirs)
    resolved_lib_dirs = _unique_paths(resolved_lib_dirs)

    command_paths: dict[str, Path] = {}
    runtime = SolverRuntime(root=resolved_root, bin_dirs=resolved_bin_dirs, lib_dirs=resolved_lib_dirs, env=dict(env_map))
    for command in _DEFAULT_COMMANDS:
        path = runtime.command_path(command)
        if path is not None:
            command_paths[command] = path

    return SolverRuntime(
        root=resolved_root,
        bin_dirs=resolved_bin_dirs,
        lib_dirs=resolved_lib_dirs,
        command_paths=command_paths,
        env=dict(env_map),
    )
