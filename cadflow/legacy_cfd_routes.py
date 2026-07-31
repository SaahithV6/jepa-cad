"""Per-family OpenFOAM routing for legacy Parts where external aero is wrong.

Maps each Part to a CFD recipe (or an explicit skip) from family + path hints.
``wall/stress`` vs ``internal/flow`` path tags choose FEA *also*, not instead of CFD —
pressure sections, nozzles, tanks, and turbopumps still get internal duct CFD.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from cadflow.part_family import classify_part, part_blob

# OpenFOAM solvers we actually ship under cadflow-solvers (1912).
SOLVER_MATRIX: dict[str, dict[str, Any]] = {
    "external_aero": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_external",
        "regime": "incompressible_external",
        "why": "Atmospheric freestream on fins/nose/fairings during ascent.",
        "priority": 1,
        "families": ["fin", "nose_cone", "fairing"],
    },
    "nozzle_compressible": {
        "application": "rhoCentralFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "compressible_supersonic_internal",
        "why": "De Laval / bell / throat: compressible internal duct CFD.",
        "priority": 1,
        "fallback": "sonicFoam",
        "path_hints": ["nozzle", "bell", "throat", "de-laval", "delaval"],
    },
    "chamber_internal": {
        "application": "rhoSimpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "compressible_subsonic_internal",
        "why": "Chamber / pressure section / regen path (incl. wall-stress twins).",
        "priority": 1,
        "fallback": "simpleFoam",
        "path_hints": ["chamber", "combust", "enginechamber", "regen", "pressure"],
    },
    "injector_orifice": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "incompressible_orifice",
        "why": "Injector pressure drop / Cd at manifold Δp.",
        "priority": 1,
        "fallback": "interFoam",
        "path_hints": ["injector", "orifice", "showerhead"],
    },
    "valve_feed": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "incompressible_valve",
        "why": "Propellant valve / feed / manifold flow path: Δp and Cv proxy.",
        "priority": 1,
        "path_hints": ["valve", "mfv", "actuator", "manifold", "feed", "plenum"],
    },
    "igniter_internal": {
        "application": "rhoSimpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "compressible_subsonic_internal",
        "why": "Igniter gas path.",
        "priority": 2,
        "path_hints": ["igniter"],
    },
    "tank_pressure_flow": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "incompressible_vessel",
        "why": "Tank / vessel: duct Δp / fill-path proxy (FEA still owns MEOP).",
        "priority": 1,
        "families": ["tank"],
        "path_hints": ["tank", "vessel", "bottle", "ullage", "dome"],
    },
    "turbopump_internal": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "incompressible_rotating_proxy",
        "why": "Turbopump / impeller / inducer: orifice-style Δp proxy (SRF later).",
        "priority": 1,
        "fallback": "SRFSimpleFoam",
        "path_hints": ["turbopump", "impeller", "inducer", "pump"],
    },
    "radiator_loop_optional": {
        "application": "chtMultiRegionSimpleFoam",
        "mesh": "multi_region_conjugate",
        "regime": "conjugate_heat_transfer",
        "why": "Radiator / cold-plate fluid loops (rare).",
        "priority": 3,
        "path_hints": ["radiator", "heat-pipe", "heatpipe", "cold-plate", "coldplate"],
    },
    "component_duct": {
        "application": "simpleFoam",
        "mesh": "snappyHexMesh_internal_duct",
        "regime": "incompressible_component_duct",
        "why": (
            "Brackets/housings/generic hardware: duct Δp around the solid. "
            "Not freestream aero — training signal for geometry-conditioned flow."
        ),
        "priority": 2,
        "families": ["structure", "generic", "mechanism", "fastener"],
    },
    "skip_exoatmospheric": {
        "application": None,
        "mesh": None,
        "regime": "none",
        "why": (
            "CubeSat / bus / deployable / dish / solar: vacuum ops. "
            "Optimization is structural/thermal FEA, not freestream CFD."
        ),
        "priority": 0,
        "families": ["spacecraft_bus", "deployable", "antenna"],
        "primary_modality": "fea",
    },
    "skip_structure_fea": {
        "application": None,
        "mesh": None,
        "regime": "none",
        "why": "Brackets/frames/panels with no fluid path: FEA only.",
        "priority": 0,
        "families": ["structure", "fastener", "mechanism", "generic"],
        "primary_modality": "fea",
    },
}

# Back-compat alias used by older summaries / tests
SOLVER_MATRIX["tank_ullage_optional"] = SOLVER_MATRIX["tank_pressure_flow"]


@dataclass
class CfdRecipe:
    recipe_id: str
    application: str | None
    mesh: str | None
    regime: str
    why: str
    priority: int
    family: str
    subtype: str
    primary_modality: str  # cfd | fea | both
    path_hint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _path_blob(part: dict[str, Any]) -> str:
    lab = str(part.get("label") or "")
    props = part.get("properties") or {}
    sp = str(props.get("source_path") or props.get("geometry_ref") or "")
    if "raw-downloads" in lab:
        lab = lab.split("raw-downloads-", 1)[-1]
    return (lab + " " + sp + " " + part_blob(part)).lower().replace("_", "-")


def fluid_subtype(blob: str) -> str:
    """Engineering fluid noun — independent of wall-stress vs internal-flow tags."""
    if re.search(r"radiator|heat-?pipe|cold-?plate", blob):
        return "radiator_loop"
    if re.search(r"turbopump|impeller|inducer|\bpump\b", blob):
        return "turbopump"
    if re.search(r"\bvalve\b|mfv|actuator", blob):
        return "valve"
    if re.search(r"injector|orifice|showerhead", blob):
        return "injector"
    if re.search(r"igniter", blob):
        return "igniter"
    if re.search(r"nozzle|bell|throat|de-?laval", blob):
        return "nozzle"
    if re.search(r"\btank\b|vessel|bottle|ullage|\bdome\b", blob):
        return "tank"
    if re.search(r"manifold|plenum|feed-?system|\bfeed\b|plumbing|\bduct\b|\bpipe\b", blob):
        return "valve"  # share valve_feed simpleFoam duct
    if re.search(r"chamber|combust|enginechamber|regen|pressure-?section", blob):
        return "chamber"
    if "internal/flow" in blob.replace("-", "/") or "internal-flow" in blob:
        return "internal_flow_generic"
    return "unspecified"


# Back-compat name used in tests
def subtype_from_path(blob: str) -> str:
    return fluid_subtype(blob)


def recipe_for_part(part: dict[str, Any]) -> CfdRecipe:
    """Pick the OpenFOAM (or skip) recipe for one legacy Part."""
    family = classify_part(part)
    blob = _path_blob(part)
    subtype = fluid_subtype(blob)

    if family in {"fin", "nose_cone", "fairing"}:
        rid = "external_aero"
    elif subtype == "nozzle":
        rid = "nozzle_compressible"
    elif subtype == "injector":
        rid = "injector_orifice"
    elif subtype == "valve":
        rid = "valve_feed"
    elif subtype == "igniter":
        rid = "igniter_internal"
    elif subtype == "radiator_loop":
        rid = "radiator_loop_optional"
    elif subtype == "turbopump":
        rid = "turbopump_internal"
    elif subtype == "tank" or family == "tank":
        rid = "tank_pressure_flow"
    elif subtype in {"chamber", "internal_flow_generic"} or family == "combustion_chamber":
        # Includes wall/stress twins — still run internal duct CFD
        rid = "chamber_internal"
    elif family in {"spacecraft_bus", "deployable", "antenna"}:
        rid = "skip_exoatmospheric"
    elif family in {"structure", "generic", "mechanism", "fastener"}:
        # Still run duct CFD when geometry exists — user wants volume, not freestream.
        rid = "component_duct"
    else:
        rid = "skip_structure_fea"

    spec = SOLVER_MATRIX[rid]
    # Pressure hardware: CFD + FEA both matter
    if rid in {
        "chamber_internal",
        "nozzle_compressible",
        "tank_pressure_flow",
        "turbopump_internal",
        "injector_orifice",
        "valve_feed",
        "component_duct",
    }:
        primary = "both"
    else:
        primary = spec.get("primary_modality") or ("cfd" if spec.get("application") else "fea")
    return CfdRecipe(
        recipe_id=rid,
        application=spec.get("application"),
        mesh=spec.get("mesh"),
        regime=spec["regime"],
        why=spec["why"],
        priority=int(spec.get("priority") or 0),
        family=family,
        subtype=subtype,
        primary_modality=primary,
        path_hint=blob[:120],
    )


def summarize_routes(parts: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    by_recipe: Counter[str] = Counter()
    cfd_worthy = 0
    skip = 0
    for p in parts:
        r = recipe_for_part(p)
        by_recipe[r.recipe_id] += 1
        if r.application:
            cfd_worthy += 1
        else:
            skip += 1
    return {
        "n_parts": len(parts),
        "cfd_worthy": cfd_worthy,
        "skip_cfd": skip,
        "by_recipe": dict(by_recipe.most_common()),
        "solver_matrix": {
            k: {
                "application": v.get("application"),
                "regime": v.get("regime"),
                "why": v.get("why"),
                "priority": v.get("priority"),
            }
            for k, v in SOLVER_MATRIX.items()
            if k != "tank_ullage_optional"
        },
    }
