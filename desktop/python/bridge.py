"""JSON-lines bridge between LatticeZero Electron and cadflow.

The bridge is deliberately thin: it exposes structured operations while all
geometry, solver, verification, and promotion decisions remain in Python.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any

REPO_ROOT = Path(os.environ.get("LATTICEZERO_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_ROOT = Path(os.environ.get("LATTICEZERO_DATA_DIR", Path.home() / ".local/share/latticezero"))
RUN_ROOT = APP_ROOT / "runs"
FLYWHEEL_PATH = APP_ROOT / "flywheel.jsonl"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def emit(event: str, payload: dict[str, Any]) -> None:
    # Bypass redirected library stdout so progress events stay on the protocol.
    sys.__stdout__.write(json.dumps({"event": event, "payload": payload}, default=str) + "\n")
    sys.__stdout__.flush()


def health(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "bridge": "ready",
        "repoRoot": str(REPO_ROOT),
        "appRoot": str(APP_ROOT),
        "python": sys.executable,
    }


def _safe_json(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _safe_json(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attr(module_name: str, attribute: str) -> Any:
    """Resolve repo modules at runtime so the portable sidecar stays lean."""
    return getattr(importlib.import_module(module_name), attribute)


def bootstrap(_: dict[str, Any]) -> dict[str, Any]:
    build_doctor_report = _attr("cadflow.doctor", "build_doctor_report")
    data_flywheel = _attr("cadflow.flywheel", "DataFlywheel")
    doctor = build_doctor_report()
    entries = list(data_flywheel(FLYWHEEL_PATH).load_entries())
    verified = [entry for entry in entries if entry.verified]
    momentum = min(100, len(verified) * 8 + len(entries) * 2)
    return {
        "appRoot": str(APP_ROOT),
        "repoRoot": str(REPO_ROOT),
        "doctor": doctor,
        "stats": {
            "runs": len(entries),
            "verified": len(verified),
            "promoted": len([entry for entry in entries if "promoted" in entry.manifest.tags]),
            "momentum": momentum,
            "modelVersion": _model_version(),
        },
        "recentRuns": [_entry_summary(entry) for entry in reversed(entries[-12:])],
    }


def doctor(_: dict[str, Any]) -> dict[str, Any]:
    return _attr("cadflow.doctor", "build_doctor_report")()


def _model_version() -> str:
    registry = APP_ROOT / "flywheel" / "registry" / "best.json"
    if registry.exists():
        try:
            data = json.loads(registry.read_text())
            return str(data.get("cycle_id") or data.get("checkpoint") or "JEPA α.1")
        except Exception:
            pass
    return "JEPA α.1"


def _entry_summary(entry: Any) -> dict[str, Any]:
    verification = entry.verification
    result = entry.solver_result
    return {
        "id": entry.manifest_fingerprint,
        "name": entry.manifest.name,
        "recordedAt": entry.recorded_at,
        "verified": entry.verified,
        "status": "verified" if entry.verified else "failed",
        "solver": result.metadata.get("solver") or result.metadata.get("backend") or "CAE",
        "solverMode": result.metadata.get("mode", "unknown"),
        "objective": result.objective,
        "volume": verification.metrics.get("volume"),
        "findings": list(verification.findings),
        "tags": list(entry.manifest.tags),
        "artifacts": list(entry.run.artifact_refs),
    }


def history(params: dict[str, Any]) -> dict[str, Any]:
    data_flywheel = _attr("cadflow.flywheel", "DataFlywheel")
    entries = list(data_flywheel(FLYWHEEL_PATH).load_entries())
    limit = int(params.get("limit", 100))
    return {
        "entries": [_entry_summary(entry) for entry in reversed(entries[-limit:])],
        "total": len(entries),
        "verified": sum(entry.verified for entry in entries),
    }


def run_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    get_backend = _attr("cadflow.backends", "get_backend")
    data_flywheel = _attr("cadflow.flywheel", "DataFlywheel")
    job_manifest = _attr("cadflow.manifest", "JobManifest")
    execute = _attr("cadflow.pipeline", "run_pipeline")
    spec = params.get("geometry", {})
    name = str(params.get("name", "Investor Demo"))
    solver = str(params.get("solver", "fea"))
    native_only = bool(params.get("nativeOnly", False))
    run_id = f"{name.lower().replace(' ', '-')}-{random.randint(1000, 9999)}"

    emit("run.stage", {"runId": run_id, "stage": "planning", "progress": 0.08, "message": "Freezing structured design intent"})
    manifest = job_manifest(
        name=name,
        inputs={
            "geometry": spec,
            "materials": [params.get("material", "Al 6061-T6")],
        },
        parameters={
            "solver": solver,
            "objective": float(params.get("objective", 0.42)),
            "load_n": float(params.get("load", 1200)),
            "max_stress_mpa": float(params.get("stressLimit", 240)),
            "Cd_guess": float(params.get("dragTarget", 0.24)),
            "peak_torque": float(params.get("torqueTarget", 38)),
        },
        tags=("desktop", "latticezero", "investor-demo"),
        notes="Created by LatticeZero; geometry is deterministic and verification-gated.",
    )
    emit("run.stage", {"runId": run_id, "stage": "geometry", "progress": 0.24, "message": "Building deterministic B-rep"})

    run_dir = RUN_ROOT / run_id
    flywheel = data_flywheel(FLYWHEEL_PATH)
    result = execute(
        manifest,
        backend=get_backend(prefer_real=not bool(params.get("mockCad", False))),
        workdir=run_dir,
        flywheel=flywheel,
        solver_kind=solver,
        allow_solver_fallback=not native_only,
    )
    emit("run.stage", {"runId": run_id, "stage": "verification", "progress": 0.84, "message": "Auditing topology, solver, and lineage"})

    payload = result.to_dict()
    payload["runId"] = run_id
    payload["geometry"] = spec
    payload["metrics"] = _result_metrics(payload)
    payload["ghosts"] = _ghost_variants(spec)
    payload["momentumEarned"] = 12 if result.ok else 2
    emit(
        "run.stage",
        {
            "runId": run_id,
            "stage": "complete",
            "progress": 1,
            "message": "Verified outcome added to the flywheel" if result.ok else "Run retained for forensic review",
            "ok": result.ok,
        },
    )
    return payload


def _result_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    solver = payload.get("solver_result", {})
    verification = payload.get("verification", {})
    meta = solver.get("metadata", {})
    metrics = verification.get("metrics", {})
    return {
        "objective": solver.get("objective"),
        "residual": solver.get("residual"),
        "iterations": solver.get("iterations"),
        "volume": metrics.get("volume"),
        "faces": metrics.get("face_count"),
        "watertight": metrics.get("watertight"),
        "stress": meta.get("max_von_mises_mpa"),
        "displacement": meta.get("max_displacement_mm"),
        "drag": meta.get("Cd"),
        "torque": meta.get("peak_joint_torque_nm"),
        "mode": meta.get("mode", "unknown"),
    }


def _ghost_variants(spec: dict[str, Any]) -> list[dict[str, Any]]:
    ghosts = []
    for index, scale in enumerate((0.82, 0.91, 1.06)):
        variant = json.loads(json.dumps(spec))
        for key in ("width", "height", "depth", "radius"):
            if isinstance(variant.get(key), (int, float)):
                variant[key] *= scale
        ghosts.append({"iteration": index, "scale": scale, "geometry": variant})
    return ghosts


def latent_atlas(params: dict[str, Any]) -> dict[str, Any]:
    seed = int(params.get("seed", 1701))
    rng = random.Random(seed)
    families = ["bracket", "fairing", "manifold", "linkage", "shell", "impeller"]
    points = []
    for index in range(72):
        family = families[index % len(families)]
        angle = index * 0.61
        radius = 2.5 + (index % 9) * 0.14
        points.append(
            {
                "id": f"latent-{index}",
                "family": family,
                "x": math.cos(angle) * radius + rng.uniform(-0.3, 0.3),
                "y": math.sin(angle) * radius + rng.uniform(-0.3, 0.3),
                "z": math.sin(angle * 0.37) * 1.8 + rng.uniform(-0.2, 0.2),
                "score": round(0.54 + rng.random() * 0.45, 3),
                "verified": index % 4 != 0,
            }
        )
    return {"points": points, "modelVersion": _model_version()}


def run_autopilot(params: dict[str, Any]) -> dict[str, Any]:
    execute = _attr("cadflow.autopilot", "run_autopilot")
    raw_dir = params.get("rawDir")
    emit("autopilot.stage", {"stage": "maintenance", "progress": 0.1, "message": "Running accountability gate"})
    result = execute(
        [raw_dir] if raw_dir else None,
        APP_ROOT / "flywheel",
        flywheel_path=FLYWHEEL_PATH,
        max_steps=int(params.get("maxSteps", 1)),
        skip_tests=bool(params.get("skipTests", True)),
        repair_env=False,
    )
    emit("autopilot.stage", {"stage": "complete", "progress": 1, "message": result.decision, "ok": result.ok})
    return result.to_dict()


def promote(params: dict[str, Any]) -> dict[str, Any]:
    data_flywheel = _attr("cadflow.flywheel", "DataFlywheel")
    promote_verified_to_dataset = _attr("cadflow.promotion", "promote_verified_to_dataset")
    out_dir = Path(params.get("outDir") or APP_ROOT / "curated")
    result = promote_verified_to_dataset(
        data_flywheel(FLYWHEEL_PATH),
        out_dir,
        limit=int(params.get("limit", 50)),
        num_points=int(params.get("numPoints", 1024)),
        num_fields=int(params.get("numFields", 3)),
    )
    return result.to_dict()


METHODS = {
    "health": health,
    "bootstrap": bootstrap,
    "doctor": doctor,
    "history": history,
    "flywheel": history,
    "run_pipeline": run_pipeline,
    "latent_atlas": latent_atlas,
    "run_autopilot": run_autopilot,
    "promote": promote,
}


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            handler = METHODS.get(method)
            if handler is None:
                raise ValueError(f"Unknown bridge method: {method}")
            # Keep accidental library prints off the protocol stream.
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = handler(request.get("params") or {})
            if buffer.getvalue():
                sys.stderr.write(buffer.getvalue())
                sys.stderr.flush()
            print(json.dumps({"id": request_id, "result": _safe_json(result)}, default=str), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "id": request.get("id") if "request" in locals() else None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
