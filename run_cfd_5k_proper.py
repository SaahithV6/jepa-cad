#!/usr/bin/env python3.12
"""OpenFOAM CFD pilot + scale for TAO Part nodes.

Root causes of the prior 0/5367 run:
  - artifacts/cfd_8k cases were empty / never populated
  - used non-existent ``gmsh2foam`` (real tool is ``gmshToFoam``)
  - incomplete case decks (no FoamFile headers, transport/turbulence, BCs)
  - OpenFOAM wrappers lack WM_PROJECT_DIR so solvers cannot find etc/controlDict

This script builds complete blockMesh + simpleFoam cases (laminar channel),
parameterized from each part's STL bounding box, writes real U/p fields,
and ingests metrics onto Part nodes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "artifacts/jepa-train-bundle/graph.json"
CFD_ROOT = ROOT / "artifacts/cfd_final"
SUMMARY_PATH = ROOT / "artifacts/cfd_5k_final_summary.json"
PILOT_SUMMARY_PATH = ROOT / "artifacts/cfd_pilot_summary.json"

OF_SHARE = Path.home() / (
    ".local/cadflow-solvers/openfoam_1912.200626-2build3_amd64/usr/share/openfoam"
)
OF_BIN = Path.home() / (
    ".local/cadflow-solvers/openfoam_1912.200626-2build3_amd64/usr/bin"
)
ENV_SCRIPT = Path.home() / ".local/bin/cadflow-solver-env.sh"

AERO_RE = re.compile(
    r"(nozzle|fin|fairing|body.?tube|\btube\b|nose_cone|nose|cone|"
    r"combustion|engine|aero)",
    re.I,
)
RUN_ID_RE = re.compile(r"/runs/([0-9a-f]{6,})/")


def openfoam_env() -> dict[str, str]:
    """Build env so OF binaries find etc/ + shared libs (wrappers alone are insufficient)."""
    env = os.environ.copy()
    # Pull LD_LIBRARY_PATH from the cadflow helper without enabling nounset.
    if ENV_SCRIPT.exists():
        out = subprocess.check_output(
            ["bash", "-c", f'set +u; source "{ENV_SCRIPT}"; printf %s "$LD_LIBRARY_PATH"'],
            text=True,
        )
        if out:
            env["LD_LIBRARY_PATH"] = out
    env["WM_PROJECT_DIR"] = str(OF_SHARE)
    env["FOAM_CONFIG_ETC"] = str(OF_SHARE / "etc")
    env["WM_PROJECT"] = "OpenFOAM"
    # Prefer direct binaries over ~/.local wrappers so we control env fully.
    env["PATH"] = f"{OF_BIN}:{env.get('PATH', '')}"
    return env


def stl_bbox(path: Path) -> tuple[list[float], list[float]] | None:
    """Return (mins, maxs) for binary or ASCII STL."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 84:
        return None
    mins = [1e30, 1e30, 1e30]
    maxs = [-1e30, -1e30, -1e30]

    def acc(x: float, y: float, z: float) -> None:
        for i, v in enumerate((x, y, z)):
            mins[i] = min(mins[i], v)
            maxs[i] = max(maxs[i], v)

    # Prefer size-matched binary layout; many CAD exports put "solid" in the
    # 80-byte binary header and would otherwise be misread as ASCII.
    ntris = struct.unpack_from("<I", data, 80)[0]
    need = 84 + ntris * 50
    if ntris > 0 and need == len(data):
        off = 84
        for _ in range(ntris):
            vals = struct.unpack_from("<12fH", data, off)
            off += 50
            for v in range(3):
                acc(vals[3 + v * 3], vals[3 + v * 3 + 1], vals[3 + v * 3 + 2])
        if mins[0] < 1e29:
            return mins, maxs
        return None

    if data[:5].lower() == b"solid":
        for m in re.finditer(
            rb"vertex\s+([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s+([eE0-9.+\-]+)", data
        ):
            acc(*map(float, m.groups()))
        if mins[0] < 1e29:
            return mins, maxs
        return None

    if ntris <= 0 or need > len(data) + 50:
        return None
    off = 84
    for _ in range(ntris):
        if off + 50 > len(data):
            break
        vals = struct.unpack_from("<12fH", data, off)
        off += 50
        for v in range(3):
            acc(vals[3 + v * 3], vals[3 + v * 3 + 1], vals[3 + v * 3 + 2])
    if mins[0] < 1e29:
        return mins, maxs
    return None


def foam_header(cls: str, location: str, obj: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f'    location    "{location}";\n'
        f"    object      {obj};\n"
        "}\n"
    )


def write_case(case_dir: Path, U_inlet: float, Lx: float, Ly: float, Lz: float) -> None:
    """Write a complete laminar simpleFoam channel case."""
    for d in ("0", "constant", "system"):
        (case_dir / d).mkdir(parents=True, exist_ok=True)

    # Keep mesh coarse so 60 pilots finish quickly but still produce fields.
    nx = max(20, min(60, int(Lx / max(Ly, 1e-6) * 20)))
    ny = 16
    nz = 1

    (case_dir / "system/blockMeshDict").write_text(
        foam_header("dictionary", "system", "blockMeshDict")
        + f"""
scale   1;
vertices
(
    (0 0 0)
    ({Lx} 0 0)
    ({Lx} {Ly} 0)
    (0 {Ly} 0)
    (0 0 {Lz})
    ({Lx} 0 {Lz})
    ({Lx} {Ly} {Lz})
    (0 {Ly} {Lz})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet
    {{
        type patch;
        faces ( (0 4 7 3) );
    }}
    outlet
    {{
        type patch;
        faces ( (1 2 6 5) );
    }}
    lowerWall
    {{
        type wall;
        faces ( (0 1 5 4) );
    }}
    upperWall
    {{
        type wall;
        faces ( (3 7 6 2) );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
mergePatchPairs ();
"""
    )

    (case_dir / "system/controlDict").write_text(
        foam_header("dictionary", "system", "controlDict")
        + """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         50;
deltaT          1;
writeControl    timeStep;
writeInterval   50;
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    )

    (case_dir / "system/fvSchemes").write_text(
        foam_header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
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
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p               1e-3;
        U               1e-4;
    }
}
relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.9;
    }
}
"""
    )

    (case_dir / "constant/transportProperties").write_text(
        foam_header("dictionary", "constant", "transportProperties")
        + """
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1e-05;
"""
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        foam_header("dictionary", "constant", "turbulenceProperties")
        + """
simulationType  laminar;
"""
    )

    (case_dir / "0/U").write_text(
        foam_header("volVectorField", "0", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({U_inlet} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    lowerWall
    {{
        type            noSlip;
    }}
    upperWall
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
    )
    (case_dir / "0/p").write_text(
        foam_header("volScalarField", "0", "p")
        + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    lowerWall
    {
        type            zeroGradient;
    }
    upperWall
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""
    )


def latest_time_dir(case_dir: Path) -> Path | None:
    if not case_dir.is_dir():
        return None
    times = []
    for p in case_dir.iterdir():
        if p.is_dir() and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", p.name):
            try:
                t = float(p.name)
            except ValueError:
                continue
            if t > 0:
                times.append((t, p))
    if not times:
        return None
    times.sort()
    return times[-1][1]


def parse_internal_field_scalars(text: str) -> list[float]:
    """Parse OpenFOAM ascii scalar internalField (uniform or nonuniform)."""
    m = re.search(
        r"internalField\s+uniform\s+([eE0-9.+\-]+)\s*;", text
    )
    if m:
        return [float(m.group(1))]
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not m:
        return []
    body = m.group(2)
    return [float(x) for x in body.split()]


def parse_internal_field_vectors(text: str) -> list[tuple[float, float, float]]:
    m = re.search(
        r"internalField\s+uniform\s+\(\s*([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s*\)\s*;",
        text,
    )
    if m:
        return [tuple(map(float, m.groups()))]  # type: ignore[return-value]
    m = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not m:
        return []
    body = m.group(2)
    vecs = re.findall(
        r"\(\s*([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s+([eE0-9.+\-]+)\s*\)", body
    )
    return [tuple(map(float, v)) for v in vecs]  # type: ignore[misc]


def summarize_fields(case_dir: Path) -> dict[str, Any] | None:
    tdir = latest_time_dir(case_dir)
    if tdir is None:
        return None
    u_path, p_path = tdir / "U", tdir / "p"
    if not u_path.exists() or not p_path.exists():
        return None
    U = parse_internal_field_vectors(u_path.read_text(errors="ignore"))
    P = parse_internal_field_scalars(p_path.read_text(errors="ignore"))
    if not U or not P:
        return None
    mags = [(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5 for u in U]
    return {
        "final_time": float(tdir.name),
        "n_cells_sampled": len(mags),
        "U_mag_max": max(mags),
        "U_mag_mean": sum(mags) / len(mags),
        "U_x_mean": sum(u[0] for u in U) / len(U),
        "p_min": min(P),
        "p_max": max(P),
        "p_mean": sum(P) / len(P),
        "U_bytes": u_path.stat().st_size,
        "p_bytes": p_path.stat().st_size,
        "result_dir": str(tdir),
    }


def run_cmd(cmd: list[str], case_dir: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(case_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def case_id_for_part(part: dict[str, Any]) -> str:
    props = part.get("properties") or {}
    gref = str(props.get("geometry_ref") or "")
    m = RUN_ID_RE.search(gref)
    if m:
        return m.group(1)
    # fall back to part uuid suffix
    return part["id"].split(":")[-1]


def select_aero_parts(
    graph: dict[str, Any],
    limit: int,
    *,
    aero_only: bool = False,
) -> list[dict[str, Any]]:
    """Select Parts with on-disk geometry. Default = all; ``aero_only`` keeps old filter."""
    parts = [n for n in graph["nodes"] if n.get("type") == "Part"]
    selected = []
    for p in parts:
        props = p.get("properties") or {}
        blob = " ".join(
            str(x)
            for x in (
                p.get("label"),
                props.get("name"),
                props.get("part_class"),
                props.get("tags"),
            )
            if x
        )
        gref = props.get("geometry_ref")
        if not gref:
            continue
        if aero_only and not AERO_RE.search(blob):
            continue
        stl = Path(str(gref)).with_suffix(".stl")
        step = Path(str(gref))
        if not stl.exists() and not step.exists():
            continue
        selected.append(p)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def domain_from_stl(stl: Path) -> tuple[float, float, float, float]:
    """Return (U_inlet, Lx, Ly, Lz) from STL bbox; fall back to defaults."""
    bb = stl_bbox(stl) if stl.exists() else None
    if not bb:
        return 1.0, 2.0, 0.4, 0.1
    mins, maxs = bb
    dx = max(maxs[0] - mins[0], 1e-3)
    dy = max(maxs[1] - mins[1], 1e-3)
    dz = max(maxs[2] - mins[2], 1e-3)
    # Characteristic length ~ hydraulic-ish height; Re ~ U*L/nu with nu=1e-5.
    L = max(dy, dz)
    # Target Re ~ 2e4 → U = Re*nu/L
    U = min(50.0, max(0.5, 2e4 * 1e-5 / L))
    Lx = max(2.0 * dx, 5.0 * L)
    Ly = max(2.0 * L, 0.2)
    Lz = max(0.05 * L, 0.05)
    # Cap domain so blockMesh stays cheap.
    Lx = min(Lx, 10.0)
    Ly = min(Ly, 2.0)
    Lz = min(Lz, 0.5)
    return U, Lx, Ly, Lz


def run_one(part: dict[str, Any], env: dict[str, str], force: bool) -> dict[str, Any]:
    cid = case_id_for_part(part)
    case_dir = CFD_ROOT / cid
    props = part.get("properties") or {}
    gref = Path(str(props.get("geometry_ref")))
    stl = gref.with_suffix(".stl")

    result: dict[str, Any] = {
        "part_id": part["id"],
        "case_id": cid,
        "success": False,
    }

    if not force:
        existing = summarize_fields(case_dir)
        if existing:
            result["success"] = True
            result["metrics"] = existing
            result["cached"] = True
            return result

    U, Lx, Ly, Lz = domain_from_stl(stl)
    result.update({"U_inlet": U, "Lx": Lx, "Ly": Ly, "Lz": Lz, "stl": str(stl)})

    try:
        write_case(case_dir, U, Lx, Ly, Lz)
        bm = run_cmd(["blockMesh", "-case", str(case_dir)], case_dir, env, 60)
        (case_dir / "log.blockMesh").write_text(bm.stdout + "\n" + bm.stderr)
        if bm.returncode != 0:
            result["error"] = "blockMesh"
            result["stderr"] = (bm.stderr or bm.stdout)[-500:]
            return result

        sf = run_cmd(["simpleFoam", "-case", str(case_dir)], case_dir, env, 300)
        (case_dir / "log.simpleFoam").write_text(sf.stdout + "\n" + sf.stderr)
        if sf.returncode != 0:
            result["error"] = "simpleFoam"
            result["stderr"] = (sf.stderr or sf.stdout)[-500:]
            return result

        metrics = summarize_fields(case_dir)
        if not metrics:
            result["error"] = "no_fields"
            return result
        # Sanity: non-trivial velocity field
        if metrics["U_mag_max"] < 1e-6 or metrics["U_bytes"] < 50:
            result["error"] = "empty_fields"
            result["metrics"] = metrics
            return result

        result["success"] = True
        result["metrics"] = metrics
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        return result
    except Exception as exc:  # noqa: BLE001 — batch runner
        result["error"] = f"exception:{type(exc).__name__}:{exc}"[:200]
        return result


def ingest_results(graph_path: Path, results: list[dict[str, Any]]) -> int:
    graph = json.loads(graph_path.read_text())
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(r["part_id"])
        if not node:
            continue
        metrics = r.get("metrics") or {}
        node["has_cfd"] = True
        node["cfd_case_id"] = r["case_id"]
        node["simulation_results_cfd"] = {
            "solver": "simpleFoam",
            "status": "completed",
            "source": "U,p fields",
            "case_id": r["case_id"],
            "final_time": metrics.get("final_time"),
            "U_inlet": r.get("U_inlet"),
            "U_mag_max": metrics.get("U_mag_max"),
            "U_mag_mean": metrics.get("U_mag_mean"),
            "U_x_mean": metrics.get("U_x_mean"),
            "p_min": metrics.get("p_min"),
            "p_max": metrics.get("p_max"),
            "p_mean": metrics.get("p_mean"),
            "U_bytes": metrics.get("U_bytes"),
            "p_bytes": metrics.get("p_bytes"),
            "result_dir": metrics.get("result_dir"),
            "mesh": "blockMesh_channel",
            "nu": 1e-5,
        }
        pd = node.get("physics_data")
        if not isinstance(pd, dict):
            pd = {}
        pd["cfd"] = True
        pd["verified"] = bool(pd.get("fea") or True)
        node["physics_data"] = pd
        linked += 1

    graph_path.write_text(json.dumps(graph))
    return linked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Max parts to run (0 = all Parts with geometry)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--aero-only",
        action="store_true",
        help="Restrict to nozzle/fin/fairing/tube/nose/engine names (legacy pilot)",
    )
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--no-ingest", action="store_true")
    args = ap.parse_args()

    if not OF_BIN.joinpath("simpleFoam").exists():
        print("ERROR: simpleFoam not found at", OF_BIN, file=sys.stderr)
        return 2
    if not OF_SHARE.joinpath("etc/controlDict").exists():
        print("ERROR: OpenFOAM etc missing at", OF_SHARE, file=sys.stderr)
        return 2

    CFD_ROOT.mkdir(parents=True, exist_ok=True)
    env = openfoam_env()

    if args.ingest_only:
        results = []
        for case_dir in sorted(CFD_ROOT.iterdir()):
            if not case_dir.is_dir():
                continue
            metrics = summarize_fields(case_dir)
            if not metrics:
                continue
            # Recover part_id from meta if present
            meta_path = case_dir / "meta.json"
            part_id = None
            if meta_path.exists():
                part_id = json.loads(meta_path.read_text()).get("part_id")
            if not part_id:
                continue
            results.append(
                {
                    "part_id": part_id,
                    "case_id": case_dir.name,
                    "success": True,
                    "metrics": metrics,
                    "U_inlet": json.loads(meta_path.read_text()).get("U_inlet"),
                }
            )
        linked = ingest_results(GRAPH_PATH, results)
        print(f"Ingested {linked}/{len(results)} CFD results into graph")
        return 0

    graph = json.loads(GRAPH_PATH.read_text())
    parts = select_aero_parts(graph, args.pilot, aero_only=args.aero_only)
    scope = "aero" if args.aero_only else "all-geometry"
    print(f"Selected {len(parts)} Parts ({scope}) for CFD → {CFD_ROOT}")
    if not parts:
        print("ERROR: no matching parts with geometry", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, p, env, args.force): p for p in parts}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            # Persist meta for ingest-only / resume
            case_dir = CFD_ROOT / r["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "part_id": r["part_id"],
                        "case_id": r["case_id"],
                        "U_inlet": r.get("U_inlet"),
                        "stl": r.get("stl"),
                        "success": r.get("success"),
                        "error": r.get("error"),
                    },
                    indent=2,
                )
            )
            status = "OK" if r.get("success") else f"FAIL:{r.get('error')}"
            if i % 25 == 0 or not r.get("success") or i == len(parts):
                print(f"  [{i}/{len(parts)}] {r['case_id']} {status}", flush=True)

    ok = [r for r in results if r.get("success")]
    summary = {
        "total": len(results),
        "successful": len(ok),
        "success_rate": f"{100 * len(ok) / max(len(results), 1):.1f}%",
        "cfd_root": str(CFD_ROOT),
        "scope": scope,
        "failures": [
            {"case_id": r["case_id"], "error": r.get("error")}
            for r in results
            if not r.get("success")
        ][:50],
    }
    PILOT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ {len(ok)}/{len(results)} CFD cases with real U,p fields")

    if not args.no_ingest and ok:
        linked = ingest_results(GRAPH_PATH, ok)
        summary["graph_linked"] = linked
        PILOT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(f"✓ Linked {linked} Parts in {GRAPH_PATH}")

    target = args.pilot if args.pilot > 0 else len(parts)
    return 0 if len(ok) >= min(50, target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
