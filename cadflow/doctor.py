"""Runtime diagnostics for native solver readiness."""

from __future__ import annotations

import json
from typing import Any

from .runtime import SolverRuntime, resolve_solver_runtime
from .solver import probe_fea, probe_mbd, probe_openfoam


def build_doctor_report(runtime: SolverRuntime | None = None) -> dict[str, Any]:
    runtime = runtime or resolve_solver_runtime()
    probes = {
        "openfoam": probe_openfoam(runtime),
        "fea": probe_fea(runtime),
        "mbd": probe_mbd(runtime),
    }
    ready = [name for name, probe in probes.items() if probe.available]
    missing = [name for name, probe in probes.items() if not probe.available]
    return {
        "runtime": runtime.diagnostics(),
        "probes": {
            name: {
                "backend": probe.backend,
                "available": probe.available,
                "reason": probe.reason,
                "details": probe.details,
            }
            for name, probe in probes.items()
        },
        "native_ready": bool(ready),
        "ready_backends": ready,
        "missing_backends": missing,
    }


def render_doctor_report(report: dict[str, Any], *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report, indent=2, sort_keys=True)

    lines = ["CADFLOW native solver doctor", f"native_ready={report.get('native_ready', False)}"]
    runtime = report.get("runtime") or {}
    if runtime.get("root"):
        lines.append(f"root={runtime['root']}")
    if runtime.get("bin_dirs"):
        lines.append(f"bin_dirs={', '.join(runtime['bin_dirs'])}")
    if runtime.get("lib_dirs"):
        lines.append(f"lib_dirs={', '.join(runtime['lib_dirs'])}")
    for name, probe in (report.get("probes") or {}).items():
        status = "ready" if probe.get("available") else "missing"
        lines.append(f"{name}: {status} ({probe.get('reason', '')})")
    if report.get("ready_backends"):
        lines.append(f"ready_backends={', '.join(report['ready_backends'])}")
    return "\n".join(lines)
