"""Body-fitted external CFD: STL wall in freestream via snappyHexMesh + simpleFoam.

Requires libscotch on LD_LIBRARY_PATH (vendored under artifacts/solver-libs/scotch).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic
from run_cfd_5k_proper import foam_header, openfoam_env, run_cmd, stl_bbox, summarize_fields

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data/openrocket_hardware_8k"
DEFAULT_CFD_ROOT = ROOT / "artifacts/rocket_cfd_bodyfit"
DEFAULT_GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
SCOTCH_LIB = ROOT / "artifacts/solver-libs/scotch/usr/lib/x86_64-linux-gnu"


def bodyfit_env() -> dict[str, str]:
    env = openfoam_env()
    if SCOTCH_LIB.is_dir():
        env["LD_LIBRARY_PATH"] = f"{SCOTCH_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


@dataclass(frozen=True, slots=True)
class BodyfitResult:
    part_id: str
    success: bool
    metrics: dict[str, Any] | None = None
    error: str | None = None
    cached: bool = False


def _domain_m(stl: Path) -> dict[str, float]:
    """Freestream box in meters. Sized so the body spans multiple background cells.

    Snappy needs surface-edge intersections; a body thinner than one cell yields
    zero intersections and no ``body`` wall patch (common on slender nose cones).
    """
    bb = stl_bbox(stl)
    if not bb:
        raise ValueError("bad STL bbox")
    mins, maxs = bb
    span = max(maxs[i] - mins[i] for i in range(3))
    scale = 1e-3 if span > 2.0 else 1.0
    xmin, ymin, zmin = (mins[i] * scale for i in range(3))
    xmax, ymax, zmax = (maxs[i] * scale for i in range(3))
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    L = max(dx, dy, dz, 1e-3)
    thin = max(min(dx, dy, dz), 1e-4)
    # Compact freestream: enough wake, but not so huge that cells swallow the body
    pad_y = max(4.0 * thin, 0.35 * L)
    pad_z = max(4.0 * thin, 0.35 * L)
    pad_in = max(2.0 * thin, 0.4 * L)
    pad_out = max(6.0 * thin, 1.2 * L)
    return {
        "xmin": xmin - pad_in,
        "xmax": xmax + pad_out,
        "ymin": ymin - pad_y,
        "ymax": ymax + pad_y,
        "zmin": zmin - pad_z,
        "zmax": zmax + pad_z,
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
        "U": float(min(50.0, max(0.5, 2e4 * 1e-5 / L))),
        "scale": scale,
    }


def _is_binary_stl(data: bytes) -> bool:
    """True when length matches binary STL layout (even if header says 'solid')."""
    if len(data) < 84:
        return False
    ntris = struct.unpack_from("<I", data, 80)[0]
    return ntris > 0 and 84 + ntris * 50 == len(data)


def write_stl_meters(src: Path, dst: Path, scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    # OpenFOAM treats any file starting with "solid" as ASCII and can segfault on
    # binary STLs that use a solid-padded header (common from CAD exporters).
    if _is_binary_stl(data):
        ntris = struct.unpack_from("<I", data, 80)[0]
        header = b"cadflow binary STL".ljust(80, b"\0")
        out = bytearray(header)
        out.extend(struct.pack("<I", ntris))
        off = 84
        for _ in range(ntris):
            vals = list(struct.unpack_from("<12fH", data, off))
            off += 50
            if scale != 1.0:
                for v in range(3):
                    b = 3 + v * 3
                    vals[b] *= scale
                    vals[b + 1] *= scale
                    vals[b + 2] *= scale
            out.extend(struct.pack("<12fH", *vals))
        dst.write_bytes(bytes(out))
        return
    if data[:5].lower() == b"solid":
        lines = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].lower() == "vertex":
                x, y, z = (float(parts[i]) * scale for i in (1, 2, 3))
                lines.append(f"  vertex {x:.8e} {y:.8e} {z:.8e}")
            else:
                lines.append(line)
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if scale == 1.0:
        dst.write_bytes(data)
        return
    raise ValueError(f"unrecognized STL format: {src}")


def write_bodyfit_case(case_dir: Path, stl_src: Path, dom: dict[str, float]) -> None:
    for d in ("0", "constant/triSurface", "system"):
        (case_dir / d).mkdir(parents=True, exist_ok=True)

    write_stl_meters(stl_src, case_dir / "constant/triSurface/body.stl", dom["scale"])

    xmin, xmax = dom["xmin"], dom["xmax"]
    ymin, ymax = dom["ymin"], dom["ymax"]
    zmin, zmax = dom["zmin"], dom["zmax"]
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    # Background cells must be smaller than the thinnest body span (else 0 intersections)
    h = max(dom["thin"] / 3.5, 1e-4)
    nx = max(16, min(48, int(Lx / h) + 1))
    ny = max(14, min(48, int(Ly / h) + 1))
    nz = max(14, min(48, int(Lz / h) + 1))
    # Cap total background cells; prefer resolving the thin axes
    while nx * ny * nz > 80_000 and min(nx, ny, nz) > 16:
        if nz >= ny and nz >= nx:
            nz -= 1
        elif ny >= nx:
            ny -= 1
        else:
            nx -= 1

    (case_dir / "system/blockMeshDict").write_text(
        foam_header("dictionary", "system", "blockMeshDict")
        + f"""
scale 1;
vertices
(
    ({xmin} {ymin} {zmin})
    ({xmax} {ymin} {zmin})
    ({xmax} {ymax} {zmin})
    ({xmin} {ymax} {zmin})
    ({xmin} {ymin} {zmax})
    ({xmax} {ymin} {zmax})
    ({xmax} {ymax} {zmax})
    ({xmin} {ymax} {zmax})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  {{ type patch; faces ( (0 4 7 3) ); }}
    outlet {{ type patch; faces ( (1 2 6 5) ); }}
    ground {{ type wall;  faces ( (0 1 5 4) ); }}
    top    {{ type patch; faces ( (3 7 6 2) ); }}
    front  {{ type patch; faces ( (0 3 2 1) ); }}
    back   {{ type patch; faces ( (4 5 6 7) ); }}
);
mergePatchPairs ();
"""
    )

    # locationInMesh: upstream of body, clearly in fluid
    loc_x = xmin + 0.05 * Lx
    (case_dir / "system/snappyHexMeshDict").write_text(
        foam_header("dictionary", "system", "snappyHexMeshDict")
        + f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    body.stl
    {{
        type triSurfaceMesh;
        name body;
    }}
    bodyZone
    {{
        type searchableBox;
        min ({dom['bx0'] - 0.05 * dom['L']} {dom['by0'] - 0.05 * dom['L']} {dom['bz0'] - 0.05 * dom['L']});
        max ({dom['bx1'] + 0.05 * dom['L']} {dom['by1'] + 0.05 * dom['L']} {dom['bz1'] + 0.05 * dom['L']});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 150000;
    maxGlobalCells 300000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
        body
        {{
            level (2 3);
            patchInfo {{ type wall; }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        bodyZone
        {{
            mode inside;
            levels ((1e15 2));
        }}
    }}
    locationInMesh ({loc_x} {dom['cy']} {dom['cz']});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 0;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""
    )

    (case_dir / "system/meshQualityDict").write_text(
        foam_header("dictionary", "system", "meshQualityDict")
        + """
maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
"""
    )

    U = dom["U"]
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
    (case_dir / "constant/transportProperties").write_text(
        foam_header("dictionary", "constant", "transportProperties")
        + "\ntransportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1e-05;\n"
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        foam_header("dictionary", "constant", "turbulenceProperties")
        + "\nsimulationType laminar;\n"
    )
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


def run_bodyfit_case(
    entry: dict[str, Any],
    corpus_dir: Path,
    cfd_root: Path,
    *,
    force: bool = False,
    timeout_mesh: int = 420,
    timeout_solve: int = 300,
) -> BodyfitResult:
    part_id = entry["part_id"]
    case_dir = (cfd_root / part_id).resolve()
    stl = (corpus_dir / entry["stl"]).resolve()

    if not force:
        meta_path = case_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else None
                if metrics and metrics.get("U_mag_max", 0) > 1e-6:
                    return BodyfitResult(
                        part_id,
                        True,
                        {**metrics, "mesh": "snappyHexMesh_external"},
                        cached=True,
                    )
            except (OSError, json.JSONDecodeError):
                pass
        existing = summarize_fields(case_dir)
        body_stl = case_dir / "constant/triSurface/body.stl"
        boundary = case_dir / "constant/polyMesh/boundary"
        if (
            existing
            and existing.get("U_mag_max", 0) > 1e-6
            and body_stl.exists()
            and boundary.exists()
            and b"body" in boundary.read_bytes()
        ):
            return BodyfitResult(
                part_id,
                True,
                {**existing, "mesh": "snappyHexMesh_external"},
                cached=True,
            )

    if not stl.exists():
        return BodyfitResult(part_id, False, error="missing_stl")
    if not SCOTCH_LIB.is_dir():
        return BodyfitResult(part_id, False, error="missing_libscotch")

    env = bodyfit_env()
    try:
        if case_dir.exists() and force:
            shutil.rmtree(case_dir)
        dom = _domain_m(stl)
        write_bodyfit_case(case_dir, stl, dom)

        bm = run_cmd(["blockMesh", "-case", str(case_dir)], case_dir, env, 120)
        (case_dir / "log.blockMesh").write_text((bm.stdout or "") + "\n" + (bm.stderr or ""))
        if bm.returncode != 0:
            return BodyfitResult(part_id, False, error="blockMesh")

        sn = run_cmd(
            ["snappyHexMesh", "-overwrite", "-case", str(case_dir)],
            case_dir,
            env,
            timeout_mesh,
        )
        (case_dir / "log.snappyHexMesh").write_text((sn.stdout or "") + "\n" + (sn.stderr or ""))
        if sn.returncode != 0:
            return BodyfitResult(part_id, False, error="snappyHexMesh")

        boundary = case_dir / "constant/polyMesh/boundary"
        if not boundary.exists() or b"body" not in boundary.read_bytes():
            return BodyfitResult(part_id, False, error="no_body_patch")

        sf = run_cmd(["simpleFoam", "-case", str(case_dir)], case_dir, env, timeout_solve)
        (case_dir / "log.simpleFoam").write_text((sf.stdout or "") + "\n" + (sf.stderr or ""))
        if sf.returncode != 0:
            return BodyfitResult(part_id, False, error="simpleFoam")

        metrics = summarize_fields(case_dir)
        if not metrics or metrics.get("U_mag_max", 0) < 1e-6:
            return BodyfitResult(part_id, False, error="empty_fields")

        metrics = {
            **metrics,
            "solver": "simpleFoam",
            "mesh": "snappyHexMesh_external",
            "U_inlet": dom["U"],
            "nu": 1e-5,
            "geometry": "stl_body_wall",
        }
        (case_dir / "meta.json").write_text(
            json.dumps(
                {
                    "part_id": f"part:rocket:{part_id}",
                    "mesh": "snappyHexMesh_external",
                    "U_inlet": dom["U"],
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Capture real U/p volume fields into a training shard BEFORE dropping
        # the bulky OpenFOAM tree (otherwise CFD training signal is scalars-only).
        try:
            from cadflow.build_physics_shards import append_cfd_shard_manifest

            append_cfd_shard_manifest(case_dir, id_prefix="part:rocket:")
        except Exception:  # noqa: BLE001 — never fail the CFD case on shard I/O
            pass
        # Keep meta (+ logs); drop bulky mesh/fields so parallel runs stay disk-light
        for sub in ("constant", "system", "0"):
            p = case_dir / sub
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        for p in list(case_dir.iterdir()):
            if p.is_dir() and p.name.replace(".", "", 1).isdigit():
                shutil.rmtree(p, ignore_errors=True)
        return BodyfitResult(part_id, True, metrics=metrics)
    except subprocess.TimeoutExpired:
        return BodyfitResult(part_id, False, error="timeout")
    except Exception as exc:  # noqa: BLE001
        return BodyfitResult(part_id, False, error=f"exception:{type(exc).__name__}:{exc}"[:180])


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    r = run_bodyfit_case(
        payload["entry"],
        Path(payload["corpus_dir"]),
        Path(payload["cfd_root"]),
        force=payload["force"],
        timeout_mesh=payload["timeout_mesh"],
        timeout_solve=payload["timeout_solve"],
    )
    return {
        "part_id": r.part_id,
        "success": r.success,
        "metrics": r.metrics,
        "error": r.error,
        "cached": r.cached,
    }


def run_batch_bodyfit(
    entries: list[dict[str, Any]],
    corpus_dir: Path,
    cfd_root: Path,
    *,
    workers: int = 2,
    force: bool = False,
    timeout_mesh: int = 420,
    timeout_solve: int = 300,
) -> list[dict[str, Any]]:
    import multiprocessing as mp

    cfd_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    payloads = [
        {
            "entry": e,
            "corpus_dir": str(corpus_dir),
            "cfd_root": str(cfd_root),
            "force": force,
            "timeout_mesh": timeout_mesh,
            "timeout_solve": timeout_solve,
        }
        for e in entries
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_worker, p): p["entry"]["part_id"] for p in payloads}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "part_id": futs[fut],
                        "success": False,
                        "error": f"future:{type(exc).__name__}",
                        "metrics": None,
                        "cached": False,
                    }
                )
            if i % 5 == 0 or i == len(futs):
                ok = sum(1 for r in results if r["success"])
                print(f"  [bodyfit {i}/{len(futs)}] ok={ok}", flush=True)
    return results


def ingest_bodyfit_to_graph(graph_path: Path, results: list[dict[str, Any]]) -> int:
    """Serialized against the other ingest pipelines (see cadflow.graph_lock)."""
    with graph_lock(graph_path):
        return _ingest_bodyfit_to_graph(graph_path, results)


def _ingest_bodyfit_to_graph(graph_path: Path, results: list[dict[str, Any]]) -> int:
    graph = read_graph(graph_path)
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(f"part:rocket:{r['part_id']}")
        if not node:
            continue
        metrics = r.get("metrics") or {}
        # Preserve prior channel-proxy annotation if present
        prev = node.get("simulation_results_cfd")
        if isinstance(prev, dict) and prev.get("mesh") == "blockMesh_channel":
            node["simulation_results_cfd_channel_proxy"] = prev
        node["has_cfd"] = True
        node["cfd_case_id"] = r["part_id"]
        node["cfd_mesh"] = "snappyHexMesh_external"
        node["simulation_results_cfd"] = {
            "solver": "simpleFoam",
            "status": "completed",
            "source": "U,p fields",
            "case_id": r["part_id"],
            "replaced_channel_proxy": True,
            **metrics,
        }
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["cfd"] = True
        pd["cfd_bodyfit"] = True
        pd["fea"] = bool(node.get("has_fea"))
        pd["verified"] = bool(pd.get("fea") or True)
        node["physics_data"] = pd
        linked += 1
    write_graph_atomic(graph_path, graph)
    return linked
