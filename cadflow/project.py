"""Project intake helpers for existing CAD/CAE workspaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

from .manifest import JobManifest
from data.parsers import ParseError, parse_raw_file

GEOMETRY_SUFFIXES = {
    ".step",
    ".stp",
    ".stl",
    ".obj",
    ".ply",
    ".igs",
    ".iges",
    ".glb",
    ".gltf",
    ".x_t",
    ".x_b",
}

ASSEMBLY_KEYWORDS = (
    "assembly",
    "subassembly",
    "joint",
    "mate",
    "occurrence",
    "component",
    "constraint",
    "hierarchy",
)

FASTENER_KEYWORDS = (
    "screw",
    "bolt",
    "nut",
    "washer",
    "pin",
    "stud",
    "clip",
    "circlip",
    "rivet",
    "standoff",
    "spacer",
    "bearing",
    "bracket",
    "flange",
    "clamp",
)

SPACE_KEYWORDS = (
    "space",
    "spacecraft",
    "satellite",
    "rocket",
    "nozzle",
    "thruster",
    "propulsion",
    "tank",
    "fairing",
    "adapter",
    "antenna",
    "orbiter",
    "probe",
    "iss",
    "cassini",
    "voyager",
    "hubble",
    "shuttle",
    "module",
)

SPACE_SUBSYSTEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "structures": (
        "structure",
        "structural",
        "frame",
        "beam",
        "truss",
        "panel",
        "bulkhead",
        "rib",
        "stringer",
        "frame",
    ),
    "propulsion": (
        "propulsion",
        "thruster",
        "engine",
        "nozzle",
        "injector",
        "chamber",
        "bell",
        "combustor",
        "turbopump",
    ),
    "pressurization": (
        "pressur",
        "copv",
        "regulator",
        "valve",
        "manifold",
        "pneumatic",
        "helium",
        "nitrogen",
    ),
    "tanks_and_feed": (
        "tank",
        "propellant",
        "feed",
        "feedline",
        "line",
        "umbilical",
        "header",
        "plenum",
    ),
    "thermal": (
        "thermal",
        "cooling",
        "radiator",
        "heatsink",
        "heater",
        "heat shield",
        "heatshield",
        "insulation",
        "mli",
    ),
    "tps": (
        "tile",
        "tiles",
        "ablator",
        "reentry",
        "shuttle",
        "shield",
        "heat shield",
        "thermal protection",
    ),
    "mechanisms": (
        "mechanism",
        "deploy",
        "hinge",
        "latch",
        "latching",
        "actuator",
        "linkage",
        "joint",
        "bearing",
        "motor",
    ),
    "aerodynamics": (
        "aero",
        "aerodynamic",
        "drag",
        "lift",
        "flow",
        "cfd",
        "fairing",
        "airframe",
        "wing",
        "nose cone",
    ),
    "seals_and_fluids": (
        "o-ring",
        "oring",
        "seal",
        "gasket",
        "fluid",
        "coolant",
        "compressor",
        "pump",
        "compressor",
    ),
    "integration": (
        "integration",
        "interface",
        "adapter",
        "mating",
        "coupling",
        "stack",
        "interstage",
        "fairing",
        "payload",
    ),
}

AERO_KEYWORDS = (
    "aero",
    "aerodynamic",
    "drag",
    "lift",
    "flow",
    "cfd",
    "nozzle",
    "inlet",
    "outlet",
)

STRUCTURAL_KEYWORDS = (
    "fea",
    "stress",
    "strain",
    "load",
    "bracket",
    "mount",
    "frame",
    "beam",
    "housing",
    "support",
    "bolt",
    "fastener",
    "joint",
)

MOTION_KEYWORDS = (
    "mbd",
    "motion",
    "mechanism",
    "linkage",
    "joint",
    "mate",
    "kinematic",
    "assembly",
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "project"


@dataclass(frozen=True, slots=True)
class ProjectIntakeResult:
    project_root: str
    manifest: JobManifest
    detected_files: tuple[str, ...]
    detected_assemblies: tuple[str, ...]
    detected_fasteners: tuple[str, ...]
    detected_space_assets: tuple[str, ...]
    detected_subsystems: tuple[str, ...]
    recommended_solver: str
    questions: tuple[str, ...]
    summary_path: str
    manifest_path: str

    @property
    def ok(self) -> bool:
        return bool(self.manifest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "manifest": self.manifest.to_dict(),
            "detected_files": list(self.detected_files),
            "detected_assemblies": list(self.detected_assemblies),
            "detected_fasteners": list(self.detected_fasteners),
            "detected_space_assets": list(self.detected_space_assets),
            "detected_subsystems": list(self.detected_subsystems),
            "recommended_solver": self.recommended_solver,
            "questions": list(self.questions),
            "summary_path": self.summary_path,
            "manifest_path": self.manifest_path,
            "ok": self.ok,
        }


def _lower_tokens(path: Path) -> str:
    return " ".join((path.name, *path.parts)).lower()


def _matches_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_subsystems(texts: Sequence[str]) -> tuple[str, ...]:
    haystack = " ".join(texts).lower()
    detected = [name for name, keywords in SPACE_SUBSYSTEM_KEYWORDS.items() if _matches_any(haystack, keywords)]
    return tuple(dict.fromkeys(detected))


def _infer_solver(goal: str, detected_text: str, subsystems: Sequence[str] = ()) -> str:
    goal_text = goal.lower()
    detected_text = detected_text.lower()
    subsystem_set = set(subsystems)
    if subsystem_set.intersection({"propulsion", "pressurization", "tanks_and_feed", "thermal", "tps", "structures", "seals_and_fluids", "integration"}):
        return "fea"
    if "mechanisms" in subsystem_set:
        return "mbd"
    if _matches_any(goal_text, STRUCTURAL_KEYWORDS):
        return "fea"
    if _matches_any(goal_text, AERO_KEYWORDS):
        return "openfoam"
    if _matches_any(goal_text, MOTION_KEYWORDS):
        return "mbd"
    if _matches_any(detected_text, AERO_KEYWORDS):
        return "openfoam"
    if _matches_any(detected_text, STRUCTURAL_KEYWORDS):
        return "fea"
    if _matches_any(detected_text, MOTION_KEYWORDS):
        return "mbd"
    return "fea"


def _build_questions(
    goal: str,
    solver: str,
    material: str | None,
    targets: Mapping[str, Any] | None,
    detected_files: Sequence[str],
    subsystems: Sequence[str],
) -> tuple[str, ...]:
    questions: list[str] = []
    if not goal.strip():
        questions.append("What is the design goal or optimization objective?")
    if material is None:
        questions.append("What material family should the project use?")
    if not targets:
        if solver == "openfoam":
            questions.append("What aerodynamic target should be minimized or bounded (e.g. drag, pressure drop, lift)?")
        elif solver == "fea":
            if "thermal" in subsystems or "tps" in subsystems:
                questions.append("What thermal or TPS target should be minimized or bounded (e.g. peak temperature, heat flux, mass)?")
            elif "pressurization" in subsystems or "tanks_and_feed" in subsystems:
                questions.append("What pressure or structural target should be bounded (e.g. max stress, leak margin, mass)?")
            else:
                questions.append("What structural target should be minimized or bounded (e.g. max stress, displacement, mass)?")
        elif solver == "mbd":
            questions.append("What motion target should be optimized (e.g. cycle time, peak torque, clearance)?")
    if not detected_files:
        questions.append("No geometry files were detected. Should I infer a new geometry from the spec instead?")
    return tuple(questions)


def _infer_geometry_spec(goal: str, source_paths: Sequence[str]) -> dict[str, Any]:
    if not source_paths:
        return {"kind": "box", "width": 1.0, "height": 1.0, "depth": 1.0}

    source = Path(source_paths[0])
    try:
        sample = parse_raw_file(source, num_points=1024, num_fields=3, allow_synthetic_fallback=False)
        points = sample.points
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = [max(float(delta) * 1.05, 1e-3) for delta in (maxs - mins)]
    except (FileNotFoundError, ParseError, ValueError):
        spans = [1.0, 1.0, 1.0]

    text = f"{goal} {source.name}".lower()
    if _matches_any(text, AERO_KEYWORDS):
        width, depth, height = spans[0], spans[1], spans[2]
        return {
            "kind": "extrude",
            "profile": [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)],
            "height": height,
            "source_path": str(source),
        }

    width, height, depth = spans[0], spans[1], spans[2]
    return {
        "kind": "box",
        "width": width,
        "height": height,
        "depth": depth,
        "source_path": str(source),
    }


def intake_project(
    project_root: str | Path,
    *,
    goal: str,
    family: str = "space",
    solver: str | None = None,
    material: str | None = None,
    targets: Mapping[str, Any] | None = None,
    notes: str | None = None,
    out_dir: str | Path | None = None,
    max_files: int = 2500,
) -> ProjectIntakeResult:
    """Summarize an existing project and emit a manifest that downstream loops can use."""

    root = Path(project_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"project root does not exist: {root}")

    detected_files: list[Path] = []
    detected_assemblies: list[Path] = []
    detected_fasteners: list[Path] = []
    detected_space_assets: list[Path] = []

    for path in root.rglob("*"):
        if len(detected_files) >= max_files:
            break
        if not path.is_file():
            continue
        lowered = _lower_tokens(path)
        if path.suffix.lower() in GEOMETRY_SUFFIXES:
            detected_files.append(path)
        if _matches_any(lowered, ASSEMBLY_KEYWORDS):
            detected_assemblies.append(path)
        if _matches_any(lowered, FASTENER_KEYWORDS):
            detected_fasteners.append(path)
        if _matches_any(lowered, SPACE_KEYWORDS):
            detected_space_assets.append(path)

    detected_text = " ".join(
        [
            *(str(p) for p in detected_files[:50]),
            *(str(p) for p in detected_assemblies[:50]),
            *(str(p) for p in detected_fasteners[:50]),
            *(str(p) for p in detected_space_assets[:50]),
        ]
    )
    detected_subsystems = _detect_subsystems([goal, detected_text])
    solver_kind = solver or _infer_solver(goal, detected_text, detected_subsystems)

    source_paths = [str(p) for p in detected_files[: min(25, len(detected_files))]]
    project_inputs: dict[str, Any] = {
        "project_root": str(root),
        "goal": goal,
        "family": family,
        "existing_project": str(root),
        "source_paths": source_paths,
        "subsystems": list(detected_subsystems),
        "geometry": _infer_geometry_spec(goal, source_paths),
        "summary": {
            "geometry_files": len(detected_files),
            "assemblies": len(detected_assemblies),
            "fasteners": len(detected_fasteners),
            "space_assets": len(detected_space_assets),
            "subsystems": list(detected_subsystems),
        },
    }
    if source_paths:
        project_inputs["source_path"] = source_paths[0]

    project_parameters: dict[str, Any] = {
        "solver": solver_kind,
        "family": family,
        "iteration": {"enabled": True, "resume": True},
        "targets": dict(targets or {}),
    }
    if material is not None:
        project_parameters["materials"] = [material]

    manifest = JobManifest(
        name=_slugify(root.name or goal),
        inputs=project_inputs,
        parameters=project_parameters,
        tags=(family, solver_kind, "project-intake"),
        notes=notes or goal,
    )

    questions = _build_questions(goal, solver_kind, material, targets, source_paths, detected_subsystems)
    out_root = Path(out_dir) if out_dir is not None else root / "intake"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "project_manifest.json"
    summary_path = out_root / "project_intake.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "project_root": str(root),
                "goal": goal,
                "family": family,
                "recommended_solver": solver_kind,
                "detected_files": [str(p) for p in detected_files[:200]],
                "detected_assemblies": [str(p) for p in detected_assemblies[:200]],
                "detected_fasteners": [str(p) for p in detected_fasteners[:200]],
                "detected_space_assets": [str(p) for p in detected_space_assets[:200]],
                "detected_subsystems": list(detected_subsystems),
                "questions": list(questions),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ProjectIntakeResult(
        project_root=str(root),
        manifest=manifest,
        detected_files=tuple(str(p) for p in detected_files),
        detected_assemblies=tuple(str(p) for p in detected_assemblies),
        detected_fasteners=tuple(str(p) for p in detected_fasteners),
        detected_space_assets=tuple(str(p) for p in detected_space_assets),
        detected_subsystems=detected_subsystems,
        recommended_solver=solver_kind,
        questions=questions,
        summary_path=str(summary_path),
        manifest_path=str(manifest_path),
    )
