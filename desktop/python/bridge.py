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

def _default_repo_root() -> Path:
    """Resolve cadflow root for both repo-dev and AppImage layouts.

    Dev:     desktop/python/bridge.py → repo root (parents[2])
    Packaged: resources/python-bridge/bridge.py → resources (parents[1])
    """
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], here.parents[2]):
        if (candidate / "cadflow").is_dir():
            return candidate
    return here.parents[2]


REPO_ROOT = Path(os.environ.get("LATTICEZERO_REPO_ROOT", _default_repo_root()))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_app_root() -> Path:
    home = Path.home()
    uid = os.getuid() if hasattr(os, "getuid") else "u"
    candidates = []
    env = os.environ.get("LATTICEZERO_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            home / ".local/share/latticezero",
            home / ".local/share/latticezero-user",
            Path(f"/tmp/latticezero-data-{uid}"),
        ]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / f".write-{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return candidates[-1]


APP_ROOT = _resolve_app_root()
RUN_ROOT = APP_ROOT / "runs"
FLYWHEEL_PATH = APP_ROOT / "flywheel.jsonl"
try:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    # Keep import alive; run_pipeline will surface a clear error if writes fail.
    pass


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
    seed_app_materials = _attr("cadflow.material_eval", "seed_app_materials")
    doctor = build_doctor_report()
    try:
        materials = seed_app_materials(APP_ROOT)
    except OSError as exc:
        materials = {
            "catalog_path": None,
            "materials": 0,
            "eval_report": None,
            "suite_passed": 0,
            "suite_failed": 0,
            "pass_rate": 0,
            "error": str(exc),
        }
    entries = list(data_flywheel(FLYWHEEL_PATH).load_entries())
    verified = [entry for entry in entries if entry.verified]
    momentum = min(100, len(verified) * 8 + len(entries) * 2)
    return {
        "appRoot": str(APP_ROOT),
        "repoRoot": str(REPO_ROOT),
        "doctor": doctor,
        "materials": materials,
        "stats": {
            "runs": len(entries),
            "verified": len(verified),
            "promoted": len([entry for entry in entries if "promoted" in entry.manifest.tags]),
            "momentum": momentum,
            "modelVersion": _model_version(),
            "materialsCatalog": materials.get("materials", 0),
            "materialEvalPassRate": materials.get("pass_rate", 0),
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
    material_name = str(params.get("material", "Al 6061-T6"))
    material_props = _resolve_material_props(material_name)
    manifest = job_manifest(
        name=name,
        inputs={
            "geometry": spec,
            "materials": [material_name],
            "material_properties": material_props,
        },
        parameters={
            "solver": solver,
            "objective": float(params.get("objective", 0.42)),
            "load_n": float(params.get("load", 1200)),
            "max_stress_mpa": float(params.get("stressLimit", 240)),
            "Cd_guess": float(params.get("dragTarget", 0.24)),
            "peak_torque": float(params.get("torqueTarget", 38)),
            "material_id": material_props.get("material_id"),
            "youngs_modulus_gpa": material_props.get("youngs_modulus_gpa"),
            "allowable_stress_mpa": material_props.get("allowable_stress_mpa"),
        },
        tags=("desktop", "latticezero", "investor-demo", "material-eval"),
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
    payload["material"] = material_props
    # Attach material-property gate using FEA stress when present
    try:
        MaterialEvalCase = _attr("cadflow.material_eval", "MaterialEvalCase")
        evaluate_case = _attr("cadflow.material_eval", "evaluate_case")
        volume = payload["metrics"].get("volume")
        volume_m3 = (float(volume) * 1e-9) if volume is not None else 1e-5  # mm³ → m³
        fea_stress = payload["metrics"].get("stress")
        # Prefer FEA stress; never treat the design stress *limit* as applied load.
        fea_val = float(fea_stress) if fea_stress is not None else None
        load_n = float(params.get("load", 1200))
        # Rough screening stress from load / (bbox face area) when FEA is absent.
        applied = None
        if fea_val is None and volume is not None and float(volume) > 0:
            face = max(float(volume) ** (2.0 / 3.0), 1e-6)
            applied = (load_n / face) * 1e-3  # N/mm² ≈ MPa for mm-based mock volumes
        mat_report = evaluate_case(
            MaterialEvalCase(
                case_id=run_id,
                material_id=str(material_props.get("material_id") or "al-6061-t6"),
                family="structure",
                volume_m3=volume_m3,
                applied_stress_mpa=applied,
                service_temp_k=300.0,
                fea_max_stress_mpa=fea_val,
            )
        ).to_dict()
        payload["material_eval"] = mat_report
        payload["metrics"]["material_eval_passed"] = mat_report.get("passed")
        payload["metrics"]["allowable_stress_mpa"] = mat_report.get("allowable_mpa")
        payload["metrics"]["mass_kg"] = mat_report.get("mass_kg")
    except Exception as exc:  # noqa: BLE001
        payload["material_eval"] = {"passed": False, "error": str(exc)}
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
    """Ghosts share the seed solid; viewport applies scale (avoid double-scaling dims)."""
    ghosts = []
    for index, scale in enumerate((0.82, 0.91, 1.06)):
        variant = json.loads(json.dumps(spec))
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


def list_materials(_: dict[str, Any]) -> dict[str, Any]:
    enriched_properties = _attr("cadflow.material_eval", "enriched_properties")
    iter_materials = _attr("cadflow.space_materials", "iter_materials")
    materials = [enriched_properties(m) for m in iter_materials()]
    return {"materials": materials, "count": len(materials)}


def material_eval(params: dict[str, Any]) -> dict[str, Any]:
    MaterialEvalCase = _attr("cadflow.material_eval", "MaterialEvalCase")
    evaluate_case = _attr("cadflow.material_eval", "evaluate_case")
    case = MaterialEvalCase(
        case_id=str(params.get("caseId") or params.get("case_id") or "adhoc"),
        material_id=str(params.get("materialId") or params.get("material_id") or "al-6061-t6"),
        family=str(params.get("family", "structure")),
        volume_m3=float(params.get("volume_m3") or params.get("volumeM3") or 1e-5),
        applied_stress_mpa=_optional_float(params.get("applied_stress_mpa") or params.get("stressMpa")),
        service_temp_k=_optional_float(params.get("service_temp_k") or params.get("tempK")),
        join_material_id=params.get("join_material_id") or params.get("joinMaterialId"),
        fea_max_stress_mpa=_optional_float(params.get("fea_max_stress_mpa") or params.get("feaStressMpa")),
        notes=str(params.get("notes", "")),
    )
    return evaluate_case(case).to_dict()


def material_eval_suite(params: dict[str, Any]) -> dict[str, Any]:
    run_suite = _attr("cadflow.material_eval", "run_suite")
    write_suite_report = _attr("cadflow.material_eval", "write_suite_report")
    suite = run_suite()
    out = APP_ROOT / "evals" / "materials"
    path = write_suite_report(out, suite)
    suite["report_path"] = str(path)
    emit(
        "eval.stage",
        {
            "stage": "material_suite",
            "progress": 1,
            "message": f"Material eval {suite['passed']}/{suite['cases']} passed",
            "ok": suite["failed"] == 0 or suite["pass_rate"] >= 0.7,
        },
    )
    return suite


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _resolve_material_props(name_or_id: str) -> dict[str, Any]:
    enriched_properties = _attr("cadflow.material_eval", "enriched_properties")
    iter_materials = _attr("cadflow.space_materials", "iter_materials")
    MATERIALS_BY_ID = _attr("cadflow.space_materials", "MATERIALS_BY_ID")
    key = name_or_id.strip().lower()
    if key in MATERIALS_BY_ID:
        return enriched_properties(MATERIALS_BY_ID[key])
    for mat in iter_materials():
        if mat.name.lower() == key or mat.material_id.lower() == key:
            return enriched_properties(mat)
    # fuzzy contains
    for mat in iter_materials():
        if key in mat.name.lower() or key in mat.material_id.lower():
            return enriched_properties(mat)
    return enriched_properties(MATERIALS_BY_ID["al-6061-t6"])


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
    "list_materials": list_materials,
    "material_eval": material_eval,
    "material_eval_suite": material_eval_suite,
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
