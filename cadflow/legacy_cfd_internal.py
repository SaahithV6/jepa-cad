"""Internal / orifice OpenFOAM CFD for legacy non-aero Parts.

Uses a tight duct around the STL (inlet→outlet along the long axis) with the
body as a wall — real snappyHexMesh + recipe-specific solvers:

- chamber / igniter → rhoSimpleFoam
- nozzle → rhoCentralFoam (fallback sonicFoam)
- injector / valve → simpleFoam
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadflow.legacy_cfd_routes import CfdRecipe, recipe_for_part
from cadflow.legacy_gap_fix import resolve_stl
from cadflow.physics_targets import physics_targets_for
from cadflow.legacy_cfd_bodyfit import _fix_location_in_mesh
from cadflow.rocket_cfd_bodyfit import (
    _domain_m,
    bodyfit_env,
    write_bodyfit_case,
    write_stl_meters,
)
from run_cfd_5k_proper import foam_header, run_cmd, stl_bbox, summarize_fields

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "artifacts/jepa-train-bundle/graph.json"
CFD_ROOT = ROOT / "artifacts/cfd_internal"
SUMMARY_PATH = ROOT / "data/legacy_cfd_internal_summary.json"
STL_CACHE = CFD_ROOT / "_stl_cache"

INTERNAL_RECIPES = frozenset(
    {
        "chamber_internal",
        "valve_feed",
        "nozzle_compressible",
        "injector_orifice",
        "igniter_internal",
        "tank_pressure_flow",
        "tank_ullage_optional",  # alias
        "turbopump_internal",
        "component_duct",
    }
)


def _atomic_write_graph(graph_path: Path, graph: dict[str, Any]) -> None:
    tmp = graph_path.with_suffix(graph_path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph), encoding="utf-8")
    tmp.replace(graph_path)


def case_id_for_part(part: dict[str, Any]) -> str:
    props = part.get("properties") or {}
    for key in ("cfd_case_id", "fea_case_id"):
        v = part.get(key) or props.get(key)
        if v:
            return str(v)
    gref = str(props.get("geometry_ref") or "")
    m = __import__("re").search(r"/runs/([0-9a-f]{8,})/", gref)
    if m:
        return m.group(1)
    fp = str(props.get("manifest_fingerprint") or part["id"])
    return hashlib.sha1(fp.encode()).hexdigest()[:16]


def _step_to_stl(step: Path, stl: Path, cl_max: float = 2.0) -> bool:
    import gmsh

    stl.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(step))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(0.15, cl_max / 8))
        gmsh.model.mesh.generate(2)
        tmp = stl.with_suffix(".gmsh.stl")
        gmsh.write(str(tmp))
        write_stl_meters(tmp, stl, 1.0)
        tmp.unlink(missing_ok=True)
        return stl.exists() and stl.stat().st_size > 2000
    except Exception:
        return False
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def has_geometry_source(part: dict[str, Any]) -> bool:
    """True if STEP/STL exists on disk (does not remesh)."""
    props = part.get("properties") or {}
    cid = case_id_for_part(part)
    cached = STL_CACHE / f"{cid}.stl"
    if cached.exists() and cached.stat().st_size > 2000:
        return True
    stl = resolve_stl(part)
    if stl and stl.stat().st_size > 2000:
        return True
    gref = props.get("geometry_ref")
    if not gref:
        return False
    path = Path(str(gref))
    if path.exists() and path.stat().st_size > 500:
        return True
    stl2 = path.with_suffix(".stl") if path.suffix else Path(str(gref) + ".stl")
    try:
        return stl2.exists() and stl2.stat().st_size > 2000
    except OSError:
        return False


def ensure_surface_stl(part: dict[str, Any]) -> Path | None:
    """Return a usable surface STL (remesh STEP when sweep STL is empty)."""
    props = part.get("properties") or {}
    gref = props.get("geometry_ref")
    cid = case_id_for_part(part)
    cached = STL_CACHE / f"{cid}.stl"

    if cached.exists() and cached.stat().st_size > 2000:
        return cached

    # Prefer existing non-tiny STL from resolve_stl / geometry_ref
    stl = resolve_stl(part)
    if stl and stl.stat().st_size > 2000:
        write_stl_meters(stl, cached, 1.0)
        return cached if cached.exists() else stl

    if gref:
        step = Path(str(gref))
        if step.suffix.lower() in {".step", ".stp"} and step.exists() and step.stat().st_size > 500:
            if _step_to_stl(step, cached):
                return cached
        stl2 = Path(str(gref)).with_suffix(".stl")
        if stl2.exists() and stl2.stat().st_size > 2000:
            write_stl_meters(stl2, cached, 1.0)
            return cached

    return None


def _duct_domain(stl: Path, u_inlet: float) -> dict[str, float]:
    """Tight freestream box → duct along +X with modest wake."""
    dom = _domain_m(stl)
    bb = stl_bbox(stl)
    if not bb:
        raise ValueError("bad STL bbox")
    mins, maxs = bb
    scale = dom["scale"]
    xmin, ymin, zmin = (mins[i] * scale for i in range(3))
    xmax, ymax, zmax = (maxs[i] * scale for i in range(3))
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    L = max(dx, dy, dz, 1e-3)
    thin = max(min(dx, dy, dz), 1e-4)
    # Duct: small radial pad, short inlet, longer outlet
    pad_r = max(1.5 * thin, 0.15 * L)
    pad_in = max(1.0 * thin, 0.25 * L)
    pad_out = max(2.5 * thin, 0.6 * L)
    return {
        **dom,
        "xmin": xmin - pad_in,
        "xmax": xmax + pad_out,
        "ymin": ymin - pad_r,
        "ymax": ymax + pad_r,
        "zmin": zmin - pad_r,
        "zmax": zmax + pad_r,
        "bx0": xmin,
        "bx1": xmax,
        "by0": ymin,
        "by1": ymax,
        "bz0": zmin,
        "bz1": zmax,
        "cx": 0.5 * (xmin + xmax),
        "cy": 0.5 * (ymin + ymax),
        "cz": 0.5 * (zmin + zmax),
        "L": L,
        "thin": thin,
        "U": float(u_inlet),
        "scale": scale,
    }


def _write_incompressible_fields(case_dir: Path, U: float, nu: float) -> None:
    (case_dir / "0/U").write_text(
        foam_header("volVectorField", "0", "U")
        + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U} 0 0);
boundaryField {{
    inlet  {{ type fixedValue; value uniform ({U} 0 0); }}
    outlet {{ type zeroGradient; }}
    ground {{ type noSlip; }}
    top    {{ type zeroGradient; }}
    front  {{ type zeroGradient; }}
    back   {{ type zeroGradient; }}
    body   {{ type noSlip; }}
    "body_.*" {{ type noSlip; }}
}}
"""
    )
    (case_dir / "0/p").write_text(
        foam_header("volScalarField", "0", "p")
        + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField {
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    ground { type zeroGradient; }
    top    { type zeroGradient; }
    front  { type zeroGradient; }
    back   { type zeroGradient; }
    body   { type zeroGradient; }
    "body_.*" { type zeroGradient; }
}
"""
    )
    (case_dir / "constant/transportProperties").write_text(
        foam_header("dictionary", "constant", "transportProperties")
        + f"\ntransportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};\n"
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        foam_header("dictionary", "constant", "turbulenceProperties")
        + "\nsimulationType laminar;\n"
    )


def _write_compressible_common(case_dir: Path, U: float, p_inlet: float, T: float) -> None:
    """Fields + thermo for rhoSimpleFoam / sonicFoam / rhoCentralFoam."""
    (case_dir / "constant/thermophysicalProperties").write_text(
        foam_header("dictionary", "constant", "thermophysicalProperties")
        + """
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}
mixture
{
    specie { molWeight 28.96; }
    thermodynamics { Cp 1004.5; Hf 0; }
    transport { mu 1.8e-05; Pr 0.71; }
}
"""
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        foam_header("dictionary", "constant", "turbulenceProperties")
        + "\nsimulationType laminar;\n"
    )
    # Remove incompressible transport if present
    tp = case_dir / "constant/transportProperties"
    if tp.exists():
        tp.unlink()

    (case_dir / "0/U").write_text(
        foam_header("volVectorField", "0", "U")
        + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U} 0 0);
boundaryField {{
    inlet  {{ type fixedValue; value uniform ({U} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({U} 0 0); }}
    ground {{ type noSlip; }}
    top    {{ type zeroGradient; }}
    front  {{ type zeroGradient; }}
    back   {{ type zeroGradient; }}
    body   {{ type noSlip; }}
    "body_.*" {{ type noSlip; }}
}}
"""
    )
    (case_dir / "0/p").write_text(
        foam_header("volScalarField", "0", "p")
        + f"""
dimensions [1 -1 -2 0 0 0 0];
internalField uniform {p_inlet};
boundaryField {{
    inlet  {{ type fixedValue; value uniform {p_inlet}; }}
    outlet {{ type fixedValue; value uniform {0.98 * p_inlet}; }}
    ground {{ type zeroGradient; }}
    top    {{ type zeroGradient; }}
    front  {{ type zeroGradient; }}
    back   {{ type zeroGradient; }}
    body   {{ type zeroGradient; }}
    "body_.*" {{ type zeroGradient; }}
}}
"""
    )
    (case_dir / "0/T").write_text(
        foam_header("volScalarField", "0", "T")
        + f"""
dimensions [0 0 0 1 0 0 0];
internalField uniform {T};
boundaryField {{
    inlet  {{ type fixedValue; value uniform {T}; }}
    outlet {{ type inletOutlet; inletValue uniform {T}; value uniform {T}; }}
    ground {{ type zeroGradient; }}
    top    {{ type zeroGradient; }}
    front  {{ type zeroGradient; }}
    back   {{ type zeroGradient; }}
    body   {{ type zeroGradient; }}
    "body_.*" {{ type zeroGradient; }}
}}
"""
    )


def _write_rho_simple_system(case_dir: Path) -> None:
    (case_dir / "system/controlDict").write_text(
        foam_header("dictionary", "system", "controlDict")
        + """
application     rhoSimpleFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 60;
deltaT 1; writeControl timeStep; writeInterval 60; purgeWrite 1;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""
    )
    (case_dir / "system/fvSchemes").write_text(
        foam_header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes {
    default none;
    div(phi,U) bounded Gauss upwind;
    div(phi,e) bounded Gauss upwind;
    div(phi,Ekp) bounded Gauss upwind;
    div(phi,K) bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (case_dir / "system/fvSolution").write_text(
        foam_header("dictionary", "system", "fvSolution")
        + """
solvers {
    p { solver GAMG; tolerance 1e-06; relTol 0.05; smoother GaussSeidel; }
    "(U|e|h)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
    "(rho|rhoFinal)" { solver diagonal; }
}
SIMPLE {
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-3; U 1e-3; e 1e-3; }
}
relaxationFactors {
    fields { p 0.3; rho 0.05; }
    equations { U 0.7; e 0.7; ".*" 0.7; }
}
"""
    )


def _write_rho_central_system(case_dir: Path) -> None:
    (case_dir / "system/controlDict").write_text(
        foam_header("dictionary", "system", "controlDict")
        + """
application     rhoCentralFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 0.002;
deltaT 2e-7; writeControl adjustableRunTime; writeInterval 0.002; purgeWrite 1;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
adjustTimeStep yes; maxCo 0.3; maxDeltaT 1e-5;
"""
    )
    (case_dir / "system/fvSchemes").write_text(
        foam_header("dictionary", "system", "fvSchemes")
        + """
fluxScheme Kurganov;
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes {
    default none;
    div(tauMC) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes {
    default linear;
    reconstruct(rho) vanLeer;
    reconstruct(U) vanLeerV;
    reconstruct(T) vanLeer;
}
snGradSchemes { default corrected; }
"""
    )
    (case_dir / "system/fvSolution").write_text(
        foam_header("dictionary", "system", "fvSolution")
        + """
solvers {
    "(rho|rhoU|rhoE)" { solver diagonal; }
    "(U|e|h|k|epsilon|omega)" {
        solver smoothSolver; smoother GaussSeidel; tolerance 1e-08; relTol 0;
    }
}
"""
    )


def _write_simple_system(case_dir: Path) -> None:
    (case_dir / "system/controlDict").write_text(
        foam_header("dictionary", "system", "controlDict")
        + """
application     simpleFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 80;
deltaT 1; writeControl timeStep; writeInterval 80; purgeWrite 1;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""
    )
    (case_dir / "system/fvSchemes").write_text(
        foam_header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes {
    default none;
    div(phi,U) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (case_dir / "system/fvSolution").write_text(
        foam_header("dictionary", "system", "fvSolution")
        + """
solvers {
    p { solver GAMG; tolerance 1e-06; relTol 0.1; smoother GaussSeidel; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
}
SIMPLE {
    nNonOrthogonalCorrectors 1; consistent yes;
    residualControl { p 1e-3; U 1e-4; }
}
relaxationFactors { equations { U 0.7; ".*" 0.7; } }
"""
    )


def apply_solver_physics(case_dir: Path, recipe: CfdRecipe, targets: dict[str, Any]) -> str:
    """Overwrite fields/system for the recipe. Returns application name."""
    bcs = targets.get("boundary_conditions") or {}
    app = recipe.application or "simpleFoam"

    if recipe.recipe_id in {
        "injector_orifice",
        "valve_feed",
        "tank_pressure_flow",
        "tank_ullage_optional",
        "turbopump_internal",
        "component_duct",
    }:
        U = float(bcs.get("flow_velocity_mps") or 8.0)
        U = min(40.0, max(1.0, U))
        if recipe.recipe_id.startswith("tank"):
            U = min(20.0, max(2.0, U))
        if recipe.recipe_id == "component_duct":
            U = min(25.0, max(3.0, U))
            nu = 1e-5
            _write_simple_system(case_dir)
            _write_incompressible_fields(case_dir, U, nu)
            # Faster steady RANS for volume coverage
            (case_dir / "system/controlDict").write_text(
                foam_header("dictionary", "system", "controlDict")
                + """
application     simpleFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 40;
deltaT 1; writeControl timeStep; writeInterval 40; purgeWrite 1;
writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""
            )
            return "simpleFoam"
        nu = 1e-5
        _write_simple_system(case_dir)
        _write_incompressible_fields(case_dir, U, nu)
        return "simpleFoam"

    if recipe.recipe_id == "nozzle_compressible":
        p_bar = float(bcs.get("inlet_total_pressure_bar") or 20.0)
        p = p_bar * 1e5
        T = float(bcs.get("inlet_total_temp_K") or 300.0)
        # Keep cold-flow training signal tractable
        T = min(600.0, max(280.0, T if T < 1000 else 350.0))
        U = 50.0
        _write_rho_central_system(case_dir)
        _write_compressible_common(case_dir, U, p, T)
        return "rhoCentralFoam"

    # chamber / igniter — prefer rhoSimpleFoam; fall back to simpleFoam on FPE
    p = 1.05e5  # gentle compressible Δp for RANS stability
    T = 300.0
    U = 15.0
    _write_rho_simple_system(case_dir)
    _write_compressible_common(case_dir, U, p, T)
    return "rhoSimpleFoam"


def select_internal_parts(graph: dict[str, Any], limit: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Prefer ready STLs first so the batch produces linked CFD quickly; STEPs remesh later.
    stl_ready: list[dict[str, Any]] = []
    step_only: list[dict[str, Any]] = []
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        if str(n.get("id") or "").startswith("part:rocket:"):
            continue
        cfd = n.get("simulation_results_cfd") or {}
        if cfd.get("mesh") in {
            "snappyHexMesh_external",
            "snappyHexMesh_internal_duct",
        }:
            continue
        r = recipe_for_part(n)
        if r.recipe_id not in INTERNAL_RECIPES:
            continue
        if not has_geometry_source(n):
            continue
        stl = resolve_stl(n)
        cid = case_id_for_part(n)
        cached = STL_CACHE / f"{cid}.stl"
        if (stl and stl.stat().st_size > 2000) or (cached.exists() and cached.stat().st_size > 2000):
            stl_ready.append(n)
        else:
            step_only.append(n)
    ordered = stl_ready + step_only
    for n in ordered:
        out.append(n)
        if limit > 0 and len(out) >= limit:
            break
    return out


def run_internal_case(
    part: dict[str, Any],
    cfd_root: Path = CFD_ROOT,
    *,
    force: bool = False,
    timeout_mesh: int = 360,
    timeout_solve: int = 300,
) -> dict[str, Any]:
    recipe = recipe_for_part(part)
    cid = case_id_for_part(part)
    result: dict[str, Any] = {
        "part_id": part["id"],
        "case_id": cid,
        "recipe_id": recipe.recipe_id,
        "success": False,
    }
    if recipe.recipe_id not in INTERNAL_RECIPES or not recipe.application:
        result["error"] = "not_internal_recipe"
        return result

    stl = ensure_surface_stl(part)
    if not stl:
        result["error"] = "no_stl"
        return result

    case_dir = (cfd_root / cid).resolve()
    props = part.get("properties") or {}
    fp = str(props.get("manifest_fingerprint") or part["id"])
    targets = physics_targets_for(fp, recipe.family)

    if not force:
        existing = summarize_fields(case_dir)
        boundary = case_dir / "constant/polyMesh/boundary"
        meta = case_dir / "meta.json"
        if (
            existing
            and existing.get("U_mag_max", 0) > 1e-6
            and boundary.exists()
            and b"body" in boundary.read_bytes()
            and meta.exists()
        ):
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("recipe_id") == recipe.recipe_id:
                    result.update(
                        {
                            "success": True,
                            "cached": True,
                            "family": recipe.family,
                            "metrics": {
                                **existing,
                                **(m.get("metrics") or {}),
                                "mesh": "snappyHexMesh_internal_duct",
                            },
                        }
                    )
                    return result
            except (OSError, json.JSONDecodeError):
                pass

    env = bodyfit_env()
    try:
        if case_dir.exists() and force:
            shutil.rmtree(case_dir)

        # Inlet speed for domain sizing (incompressible recipes)
        bcs = targets.get("boundary_conditions") or {}
        u0 = float(bcs.get("flow_velocity_mps") or 12.0)
        u0 = min(40.0, max(2.0, u0))
        if recipe.recipe_id in {"chamber_internal", "igniter_internal"}:
            u0 = 30.0
        if recipe.recipe_id == "nozzle_compressible":
            u0 = 50.0

        dom = _duct_domain(stl, u0)
        write_bodyfit_case(case_dir, stl, dom)
        _fix_location_in_mesh(case_dir, dom)
        app = apply_solver_physics(case_dir, recipe, targets)

        bm = run_cmd(["blockMesh", "-case", str(case_dir)], case_dir, env, 120)
        (case_dir / "log.blockMesh").write_text((bm.stdout or "") + "\n" + (bm.stderr or ""))
        if bm.returncode != 0:
            result["error"] = "blockMesh"
            return result

        sn = run_cmd(
            ["snappyHexMesh", "-overwrite", "-case", str(case_dir)],
            case_dir,
            env,
            timeout_mesh,
        )
        (case_dir / "log.snappyHexMesh").write_text((sn.stdout or "") + "\n" + (sn.stderr or ""))
        if sn.returncode != 0:
            result["error"] = "snappyHexMesh"
            return result

        boundary = case_dir / "constant/polyMesh/boundary"
        if not boundary.exists() or b"body" not in boundary.read_bytes():
            result["error"] = "no_body_patch"
            return result

        # Fallbacks: rhoCentralFoam → sonicFoam → rhoSimpleFoam;
        # rhoSimpleFoam (chamber) → simpleFoam.
        apps_try = [app]
        if app == "rhoCentralFoam":
            apps_try = ["rhoCentralFoam", "sonicFoam", "rhoSimpleFoam", "simpleFoam"]
        elif app == "rhoSimpleFoam":
            apps_try = ["rhoSimpleFoam", "simpleFoam"]

        last_err = "solve"
        for try_app in apps_try:
            if try_app == "sonicFoam":
                _write_compressible_common(case_dir, 50.0, 2e5, 300.0)
                (case_dir / "system/controlDict").write_text(
                    foam_header("dictionary", "system", "controlDict")
                    + """
application     sonicFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 0.005;
deltaT 1e-6; writeControl adjustableRunTime; writeInterval 0.005; purgeWrite 1;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
adjustTimeStep yes; maxCo 0.3; maxDeltaT 2e-5;
"""
                )
                _write_rho_central_system(case_dir)
                # controlDict overwritten above for sonicFoam
                (case_dir / "system/controlDict").write_text(
                    foam_header("dictionary", "system", "controlDict")
                    + """
application     sonicFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 0.005;
deltaT 1e-6; writeControl adjustableRunTime; writeInterval 0.005; purgeWrite 1;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
adjustTimeStep yes; maxCo 0.3; maxDeltaT 2e-5;
"""
                )
            elif try_app == "rhoSimpleFoam":
                _write_rho_simple_system(case_dir)
                _write_compressible_common(case_dir, 15.0, 1.05e5, 300.0)
            elif try_app == "simpleFoam":
                _write_simple_system(case_dir)
                _write_incompressible_fields(case_dir, 12.0, 1e-5)
            elif try_app == "rhoCentralFoam":
                pass
            sf = run_cmd([try_app, "-case", str(case_dir)], case_dir, env, timeout_solve)
            (case_dir / f"log.{try_app}").write_text((sf.stdout or "") + "\n" + (sf.stderr or ""))
            if sf.returncode == 0:
                app = try_app
                break
            last_err = try_app
        else:
            result["error"] = last_err
            return result

        metrics = summarize_fields(case_dir)
        if not metrics or metrics.get("U_mag_max", 0) < 1e-6:
            result["error"] = "empty_fields"
            return result

        metrics = {
            **metrics,
            "solver": app,
            "mesh": "snappyHexMesh_internal_duct",
            "geometry": "stl_body_wall_duct",
            "recipe_id": recipe.recipe_id,
            "family": recipe.family,
            "subtype": recipe.subtype,
            "sim_kind": targets.get("sim_kind"),
            "regime": recipe.regime,
            "U_inlet": dom["U"],
            "delta_p": float(metrics["p_max"]) - float(metrics["p_min"]),
        }
        (case_dir / "meta.json").write_text(
            json.dumps(
                {
                    "part_id": part["id"],
                    "case_id": cid,
                    "recipe_id": recipe.recipe_id,
                    "family": recipe.family,
                    "solver": app,
                    "mesh": "snappyHexMesh_internal_duct",
                    "metrics": metrics,
                    "targets": targets.get("targets"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Volume-field shard for training (legacy duct CFD often keeps the case tree).
        try:
            from cadflow.build_physics_shards import append_cfd_shard_manifest

            append_cfd_shard_manifest(case_dir, part_id=str(part["id"]))
        except Exception:  # noqa: BLE001
            pass
        result.update(
            {
                "success": True,
                "family": recipe.family,
                "recipe_id": recipe.recipe_id,
                "metrics": metrics,
                "targets": targets.get("targets"),
            }
        )
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"exception:{type(exc).__name__}:{exc}"[:180]
        return result


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    return run_internal_case(
        payload["part"],
        Path(payload["cfd_root"]),
        force=payload["force"],
        timeout_mesh=payload["timeout_mesh"],
        timeout_solve=payload["timeout_solve"],
    )


def run_batch(
    parts: list[dict[str, Any]],
    *,
    workers: int = 1,
    force: bool = False,
    timeout_mesh: int = 360,
    timeout_solve: int = 300,
    cfd_root: Path = CFD_ROOT,
) -> list[dict[str, Any]]:
    cfd_root.mkdir(parents=True, exist_ok=True)
    STL_CACHE.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for i, p in enumerate(parts, 1):
            r = run_internal_case(
                p, cfd_root, force=force, timeout_mesh=timeout_mesh, timeout_solve=timeout_solve
            )
            results.append(r)
            print(
                f"  [internal {i}/{len(parts)}] {r.get('recipe_id')} "
                f"{'OK' if r.get('success') else r.get('error')} "
                f"U={(r.get('metrics') or {}).get('U_mag_max')}",
                flush=True,
            )
        return results

    payloads = [
        {
            "part": p,
            "cfd_root": str(cfd_root),
            "force": force,
            "timeout_mesh": timeout_mesh,
            "timeout_solve": timeout_solve,
        }
        for p in parts
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, pl): i for i, pl in enumerate(payloads)}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 5 == 0 or done == len(parts):
                ok = sum(1 for x in results if x.get("success"))
                print(f"  [internal {done}/{len(parts)}] ok={ok}", flush=True)
    return results


def ingest_internal(graph_path: Path, results: list[dict[str, Any]]) -> int:
    ok = [r for r in results if r.get("success") and r.get("metrics")]
    if not ok:
        return 0
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in ok:
        node = by_id.get(r["part_id"])
        if not node:
            continue
        m = r["metrics"]
        node["simulation_results_cfd"] = {
            "solver": m.get("solver"),
            "mesh": "snappyHexMesh_internal_duct",
            "geometry": "stl_body_wall_duct",
            "recipe_id": r.get("recipe_id"),
            "family": r.get("family") or m.get("family"),
            "case_id": r.get("case_id"),
            "U_mag_max": m.get("U_mag_max"),
            "U_mag_mean": m.get("U_mag_mean"),
            "p_min": m.get("p_min"),
            "p_max": m.get("p_max"),
            "delta_p": m.get("delta_p"),
            "n_cells_sampled": m.get("n_cells_sampled"),
            "result_dir": m.get("result_dir"),
            "regime": m.get("regime"),
        }
        node["has_cfd"] = True
        node["cfd_case_id"] = r.get("case_id")
        linked += 1
    _atomic_write_graph(graph_path, graph)
    return linked
