"""Physics suite for OpenRocket / TPS hardware corpus (CalculiX + OpenFOAM).

Meshes STL parts to solid MSH2 (mm → m), runs CalculiX FEA with catalog
material elasticity, runs laminar simpleFoam channel CFD, and ingests
results onto new Part nodes in the TAO training graph.

Owns ``artifacts/rocket_fea_8k`` / ``artifacts/rocket_cfd_8k`` only —
does not touch ``artifacts/fea_final`` or ``artifacts/cfd_final``.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic
from cadflow.msh_to_calculix import (
    DEFAULT_CCX,
    case_has_valid_frd,
    generate_fea_case_inp,
    parse_frd_summary,
    parse_msh2_solid,
    run_calculix_case,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data/openrocket_hardware_8k"
DEFAULT_GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
DEFAULT_FEA_ROOT = ROOT / "artifacts/rocket_fea_8k"
DEFAULT_CFD_ROOT = ROOT / "artifacts/rocket_cfd_8k"
DEFAULT_SUMMARY = ROOT / "data/rocket_physics_8k_summary.json"

# Reuse proven OpenFOAM channel helpers from the existing CFD runner.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_cfd_5k_proper import (  # noqa: E402
    openfoam_env,
    run_cmd,
    stl_bbox,
    summarize_fields,
    write_case,
)


@dataclass(frozen=True, slots=True)
class MeshResult:
    part_id: str
    success: bool
    msh_path: Path | None
    node_count: int = 0
    tet_count: int = 0
    error: str | None = None
    # True when the mesh came from the convex-hull proxy rather than the actual
    # geometry. This matters for thin-walled parts: the convex hull of a tube is
    # a solid cylinder, so a hull-substituted result reports the stresses of a
    # billet and says nothing about the shell. Callers that care about the
    # structure must check this rather than just `success`.
    used_hull: bool = False


@dataclass(frozen=True, slots=True)
class PhysicsResult:
    part_id: str
    success: bool
    kind: str  # "fea" | "cfd"
    metrics: dict[str, Any] | None = None
    error: str | None = None
    cached: bool = False


def load_manifest(corpus_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(corpus_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def material_elastic_props(entry: dict[str, Any]) -> tuple[float, float]:
    """Return (E_Pa, poisson) from a corpus manifest entry."""
    mat = entry.get("material") or {}
    e_gpa = mat.get("youngs_modulus_gpa")
    if e_gpa is None:
        e_gpa = 70.0
    nu = mat.get("poisson_ratio")
    if nu is None:
        # category fallbacks
        cat = str(entry.get("material_category") or mat.get("category") or "")
        nu = 0.33 if cat in {"aluminum", "titanium", "copper", "steel", "superalloy"} else 0.30
    return float(e_gpa) * 1e9, float(nu)


def scale_load_n(entry: dict[str, Any], youngs_pa: float) -> float:
    """Pick a compressive load that yields meaningful stress on small parts."""
    extents = entry.get("extents_mm") or [50.0, 50.0, 50.0]
    # Characteristic cross-section ~ min lateral extent squared (mm² → m²)
    lat_mm = max(min(float(extents[0]), float(extents[1])), 5.0)
    area_m2 = (lat_mm * 1e-3) ** 2
    # Target ~50 MPa nominal stress
    target_pa = 50e6
    load = target_pa * area_m2
    # Clamp: tiny antennas stay ≥50 N; big tanks ≤ 5e5 N
    return float(min(5e5, max(50.0, load)))


# Hard ceiling on tet count before CalculiX. Above this, PaStiX/SPOOLES dies with
# "Failed during initial partitioning" under typical box RAM (seen at ~100k–800k).
MAX_FEA_TETS = 40_000
MIN_FEA_TETS = 800


def cl_for_target_tets(
    extents_mm: list[float] | tuple[float, ...] | None,
    *,
    target_tets: int = 15_000,
    cl_min_floor: float = 0.35,
    cl_max_ceil: float = 80.0,
) -> tuple[float, float]:
    """Pick Gmsh cl_max/cl_min (mm) aiming for ~target_tets (typ. 10k–25k).

    Uses bbox volume as a cheap proxy for solid volume. ``h*1.7`` with a high
    ceiling is required for large tanks (390³ mm) — the old 5.5 mm ceiling forced
    ~800k tets and CalculiX partitioning deaths.
    """
    ext = list(extents_mm or [50.0, 50.0, 50.0])
    while len(ext) < 3:
        ext.append(50.0)
    dx, dy, dz = (max(float(ext[i]), 0.5) for i in range(3))
    vol = dx * dy * dz
    n = max(int(target_tets), 1_000)
    h = (vol / n) ** (1.0 / 3.0)
    cl_max = float(max(cl_min_floor, min(cl_max_ceil, h * 1.7)))
    cl_min = float(max(0.15, min(cl_max / 5.0, cl_max * 0.35)))
    return cl_max, cl_min


def mesh_stl_volume(
    stl_path: Path | str,
    msh_path: Path | str,
    *,
    cl_max_mm: float = 4.0,
    cl_min_mm: float = 1.0,
    scale_to_meters: bool = True,
    angle_deg: float = 40.0,
    mesh_timeout_s: int = 120,
    allow_hull_fallback: bool = True,
) -> MeshResult:
    """Classify STL surfaces, create a volume, tet-mesh, write MSH2 (SI meters).

    Runs gmsh in a child process so a hung classify/generate cannot stall the pool.
    On stubborn PLC intersection errors (common on tank STLs), optionally remesh a
    convex-hull proxy so CalculiX still gets a solid for training fields.
    """
    import multiprocessing as mp

    stl = Path(stl_path)
    out = Path(msh_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part_id = out.parent.name if out.parent.name else stl.stem

    if not stl.exists():
        return MeshResult(part_id, False, None, error="missing_stl")

    # fork is fine on Linux and works inside spawn pool workers; spawn breaks
    # when the parent was started via ``python -c`` / stdin.
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_mesh_stl_volume_child,
        args=(str(stl), str(out), cl_max_mm, cl_min_mm, scale_to_meters, angle_deg, q),
    )
    proc.start()
    proc.join(timeout=max(30, int(mesh_timeout_s)))
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return MeshResult(part_id, False, None, error="mesh_timeout")
    try:
        payload = q.get_nowait()
    except Exception:  # noqa: BLE001
        return MeshResult(part_id, False, None, error="mesh_no_result")
    if payload.get("success"):
        return MeshResult(
            part_id=part_id,
            success=True,
            msh_path=out,
            node_count=int(payload.get("node_count") or 0),
            tet_count=int(payload.get("tet_count") or 0),
        )

    err = str(payload.get("error") or "mesh_failed")
    el = err.lower()
    # Thin lofted fins / concatenated fairings often fail with "overlapping facets",
    # Invalid boundary mesh, or classify/partition errors — still hull-recoverable.
    # Fairing STLs are tube+nose concatenates (not watertight) so ANY mesh failure
    # on a fairing_* case should try the convex-hull solid proxy.
    recoverable = (
        "PLC" in err
        or "intersect" in el
        or "overlapping" in el
        or "invalid boundary" in el
        or "self-intersect" in el
        or "singular matrix" in el
        or "partition" in el
        or "wrong topology" in el
        or "parametrization" in el
        or "no volume" in el
        or "classify" in el
        or part_id.startswith("fairing_")
    )
    if not (allow_hull_fallback and recoverable):
        return MeshResult(part_id, False, None, error=err)

    # Convex-hull proxy: still a solid of similar envelope; unblocks CalculiX.
    try:
        import trimesh

        mesh = trimesh.load_mesh(stl, force="mesh")
        hull = mesh.convex_hull
        hull_stl = out.with_suffix(".hull.stl")
        hull.export(hull_stl)
    except Exception as exc:  # noqa: BLE001
        return MeshResult(part_id, False, None, error=f"hull:{type(exc).__name__}")

    q2: mp.Queue = ctx.Queue()
    proc2 = ctx.Process(
        target=_mesh_stl_volume_child,
        args=(str(hull_stl), str(out), cl_max_mm * 1.2, cl_min_mm * 1.2, scale_to_meters, 40.0, q2),
    )
    proc2.start()
    proc2.join(timeout=max(30, int(mesh_timeout_s)))
    if proc2.is_alive():
        proc2.terminate()
        proc2.join(5)
        return MeshResult(part_id, False, None, error="hull_mesh_timeout")
    try:
        payload2 = q2.get_nowait()
    except Exception:  # noqa: BLE001
        return MeshResult(part_id, False, None, error="hull_mesh_no_result")
    if not payload2.get("success"):
        return MeshResult(part_id, False, None, error=f"hull:{payload2.get('error') or err}")
    return MeshResult(
        part_id=part_id,
        success=True,
        msh_path=out,
        node_count=int(payload2.get("node_count") or 0),
        tet_count=int(payload2.get("tet_count") or 0),
        used_hull=True,
    )


def _mesh_stl_volume_child(
    stl: str,
    out: str,
    cl_max_mm: float,
    cl_min_mm: float,
    scale_to_meters: bool,
    angle_deg: float,
    q: Any,
) -> None:
    """Child-process gmsh body (must be top-level for spawn)."""
    import gmsh

    stl_p = Path(stl)
    out_p = Path(out)
    part_id = out_p.parent.name if out_p.parent.name else stl_p.stem
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add(part_id)
        gmsh.merge(str(stl_p))
        rad = math.radians
        gmsh.model.mesh.classifySurfaces(rad(angle_deg), True, True, rad(180.0))
        gmsh.model.mesh.createGeometry()
        ents2 = gmsh.model.getEntities(2)
        if not ents2:
            q.put({"success": False, "error": "no_surfaces"})
            return
        try:
            sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in ents2])
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
        except Exception as exc:  # noqa: BLE001
            q.put({"success": False, "error": f"volume:{type(exc).__name__}"})
            return

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max_mm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min_mm)
        # Delaunay first; on PLC-ish geometry HXT occasionally surfaces-only, so
        # we keep Delaunay as primary and let the hull fallback handle hard STLs.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out_p))
    except Exception as exc:  # noqa: BLE001
        q.put({"success": False, "error": f"gmsh:{type(exc).__name__}:{exc}"[:180]})
        return
    finally:
        try:
            gmsh.finalize()
        except Exception:  # noqa: BLE001
            pass

    try:
        mesh = parse_msh2_solid(out_p)
    except Exception as exc:  # noqa: BLE001
        q.put({"success": False, "error": f"parse:{type(exc).__name__}"})
        return

    if not mesh.elements:
        # Leave no hollow surface-only MSH behind — it confuses retries/caches.
        try:
            out_p.unlink(missing_ok=True)
        except OSError:
            pass
        q.put({"success": False, "error": "no_tets", "node_count": len(mesh.nodes)})
        return

    if scale_to_meters:
        lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(mesh.nodes))]
        for nid in sorted(mesh.nodes):
            x, y, z = mesh.nodes[nid]
            lines.append(f"{nid} {x * 1e-3:.10e} {y * 1e-3:.10e} {z * 1e-3:.10e}")
        lines += ["$EndNodes", "$Elements", str(len(mesh.elements))]
        for eid, nids in mesh.elements:
            lines.append(f"{eid} 4 2 0 0 " + " ".join(map(str, nids)))
        lines.append("$EndElements")
        out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    q.put(
        {
            "success": True,
            "node_count": len(mesh.nodes),
            "tet_count": len(mesh.elements),
        }
    )


def domain_from_stl_mm(stl: Path) -> tuple[float, float, float, float]:
    """Channel domain in meters from an STL whose coordinates are millimeters."""
    bb = stl_bbox(stl)
    if not bb:
        return 1.0, 2.0, 0.4, 0.1
    mins, maxs = bb
    # Heuristic: extents > 2 → treat as mm
    span = max(maxs[i] - mins[i] for i in range(3))
    scale = 1e-3 if span > 2.0 else 1.0
    dx = max((maxs[0] - mins[0]) * scale, 1e-3)
    dy = max((maxs[1] - mins[1]) * scale, 1e-3)
    dz = max((maxs[2] - mins[2]) * scale, 1e-3)
    L = max(dy, dz)
    U = min(50.0, max(0.5, 2e4 * 1e-5 / L))
    Lx = min(max(2.0 * dx, 5.0 * L), 10.0)
    Ly = min(max(2.0 * L, 0.2), 2.0)
    Lz = min(max(0.05 * L, 0.05), 0.5)
    return U, Lx, Ly, Lz


def prepare_fea_case(
    entry: dict[str, Any],
    corpus_dir: Path,
    fea_root: Path,
    *,
    force: bool = False,
    cl_max_mm: float = 4.0,
    cl_min_mm: float | None = None,
    target_tets: int = 0,
    mesh_timeout_s: int = 120,
    max_tets: int = MAX_FEA_TETS,
) -> MeshResult:
    """Mesh STL into ``fea_root/<part_id>/mesh.msh`` if needed.

    Rejects (and remeshes) caches whose tet count exceeds ``max_tets`` — oversized
    meshes are the dominant CalculiX partitioning failure mode on this corpus.
    """
    part_id = entry["part_id"]
    case_dir = fea_root / part_id
    case_dir.mkdir(parents=True, exist_ok=True)
    msh = case_dir / "mesh.msh"
    if msh.exists() and not force:
        try:
            mesh = parse_msh2_solid(msh)
            n_tets = len(mesh.elements)
            if mesh.elements and n_tets <= max_tets:
                return MeshResult(part_id, True, msh, len(mesh.nodes), n_tets)
            # Oversized or empty → fall through and remesh coarser.
            try:
                msh.unlink()
            except OSError:
                pass
        except (OSError, ValueError):
            pass
    stl = corpus_dir / entry["stl"]
    # Fairings are open concatenates — prefer a coarser tet budget so CalculiX
    # PaStiX partitioning succeeds on the hull proxy.
    fam = str(entry.get("family") or "")
    if fam == "fairing" and target_tets > 8000:
        target_tets = 6000
    if target_tets > 0:
        cl_max_mm, cl_auto_min = cl_for_target_tets(
            entry.get("extents_mm"), target_tets=target_tets
        )
        if cl_min_mm is None:
            cl_min_mm = cl_auto_min
    if cl_min_mm is None:
        cl_min_mm = max(0.15, float(cl_max_mm) / 5.0)
    # Progressive coarsening: keep going when mesh is *too fine* (tanks) or fails
    # to classify (thin fins / open shells).
    attempts: list[tuple[float, float, float]] = [
        (float(cl_max_mm), float(cl_min_mm), 40.0),
        (float(cl_max_mm) * 1.6, float(cl_min_mm) * 1.5, 40.0),
        (float(cl_max_mm) * 2.4, float(cl_min_mm) * 2.0, 25.0),
        (float(cl_max_mm) * 3.5, float(cl_min_mm) * 2.5, 55.0),
        (float(cl_max_mm) * 5.0, float(cl_min_mm) * 3.0, 40.0),
        (float(cl_max_mm) * 8.0, float(cl_min_mm) * 4.0, 55.0),
    ]
    if fam == "fairing":
        # Extra-coarse last-ditch attempts for open fairing shells.
        attempts.extend(
            [
                (float(cl_max_mm) * 12.0, float(cl_min_mm) * 6.0, 55.0),
                (float(cl_max_mm) * 20.0, float(cl_min_mm) * 8.0, 60.0),
            ]
        )
    last = MeshResult(part_id, False, None, error="mesh_failed")
    best_ok: MeshResult | None = None
    for cmax, cmin, angle in attempts:
        if msh.exists():
            try:
                msh.unlink()
            except OSError:
                pass
        last = mesh_stl_volume(
            stl,
            msh,
            cl_max_mm=cmax,
            cl_min_mm=cmin,
            angle_deg=angle,
            mesh_timeout_s=mesh_timeout_s,
        )
        if not last.success or (last.tet_count or 0) <= 0:
            continue
        tets = int(last.tet_count or 0)
        if tets <= max_tets:
            return last
        # Too fine — keep coarsening, but remember the smallest oversized mesh.
        if best_ok is None or tets < int(best_ok.tet_count or 10**12):
            best_ok = last
    if best_ok is not None and (best_ok.tet_count or 0) <= max_tets * 2:
        # Last resort: accept up to 2× the soft cap rather than producing nothing.
        return best_ok
    return last if last.success else (best_ok or last)


def run_fea_for_entry(
    entry: dict[str, Any],
    corpus_dir: Path,
    fea_root: Path,
    *,
    force: bool = False,
    timeout: int = 300,
    ccx: Path = DEFAULT_CCX,
    cl_max_mm: float = 4.0,
    target_tets: int = 0,
    min_frd_bytes: int = 50_000,
    mesh_timeout_s: int = 120,
) -> PhysicsResult:
    part_id = entry["part_id"]
    case_dir = fea_root / part_id
    if not force and case_has_valid_frd(case_dir, min_bytes=min_frd_bytes):
        summary = parse_frd_summary(case_dir / "case.frd", min_bytes=min_frd_bytes)
        metrics = None if summary is None else {
            "max_stress_mpa": summary.max_von_mises_mpa,
            "mean_stress_mpa": summary.mean_von_mises_mpa,
            "max_displacement_mm": summary.max_displacement_mm,
            "frd_bytes": summary.frd_bytes,
            "result_nodes": summary.node_count,
        }
        return PhysicsResult(part_id, True, "fea", metrics=metrics, cached=True)

    stl = corpus_dir / entry["stl"]
    try:
        # Degenerate fin/blanket STLs are ~0.5KB and only waste gmsh cycles.
        if not stl.is_file() or stl.stat().st_size < 2000:
            return PhysicsResult(part_id, False, "fea", error="stl_too_small")
    except OSError:
        return PhysicsResult(part_id, False, "fea", error="stl_missing")

    mesh = prepare_fea_case(
        entry,
        corpus_dir,
        fea_root,
        force=force,
        cl_max_mm=cl_max_mm,
        target_tets=target_tets,
        mesh_timeout_s=mesh_timeout_s,
    )
    # One refine pass if adaptive sizing undershot the floor — but never on
    # already-large meshes (that path used to push tanks back into 100k+ tets).
    if (
        mesh.success
        and target_tets > 0
        and MIN_FEA_TETS <= int(mesh.tet_count or 0) < 8_000
        and mesh.msh_path is not None
    ):
        cl_max, cl_min = cl_for_target_tets(
            entry.get("extents_mm"), target_tets=target_tets
        )
        mesh.msh_path.unlink(missing_ok=True)
        mesh = mesh_stl_volume(
            corpus_dir / entry["stl"],
            mesh.msh_path,
            cl_max_mm=max(0.3, cl_max * 0.65),
            cl_min_mm=max(0.12, cl_min * 0.65),
            mesh_timeout_s=mesh_timeout_s,
        )
    if not mesh.success:
        return PhysicsResult(part_id, False, "fea", error=mesh.error)
    if int(mesh.tet_count or 0) > MAX_FEA_TETS * 2:
        return PhysicsResult(
            part_id,
            False,
            "fea",
            error=f"mesh_too_fine:{mesh.tet_count}",
        )

    e_pa, nu = material_elastic_props(entry)
    load = scale_load_n(entry, e_pa)
    try:
        generate_fea_case_inp(
            case_dir,
            total_load=load,
            youngs_modulus=e_pa,
            poisson=nu,
        )
        result = run_calculix_case(case_dir, ccx_binary=ccx, timeout=timeout)
    except (OSError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PhysicsResult(part_id, False, "fea", error=f"run:{type(exc).__name__}")

    if not (result.converged and case_has_valid_frd(case_dir, min_bytes=min_frd_bytes)):
        return PhysicsResult(part_id, False, "fea", error="no_valid_frd")

    summary = parse_frd_summary(case_dir / "case.frd", min_bytes=min_frd_bytes)
    metrics = {
        "solver": "calculix",
        "load_n": load,
        "youngs_modulus_pa": e_pa,
        "poisson": nu,
        "material_id": entry.get("material_id"),
        "max_stress_mpa": None if summary is None else round(summary.max_von_mises_mpa, 4),
        "mean_stress_mpa": None if summary is None else round(summary.mean_von_mises_mpa, 4),
        "max_displacement_mm": None if summary is None else round(summary.max_displacement_mm, 6),
        "frd_bytes": result.frd_bytes,
        "result_nodes": None if summary is None else summary.node_count,
        "mesh_nodes": mesh.node_count,
        "mesh_tets": mesh.tet_count,
        "target_tets": target_tets or None,
    }
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "part_id": part_id,
                "load_n": load,
                "material_id": entry.get("material_id"),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    strip_fea_mesh_intermediates(case_dir)
    return PhysicsResult(part_id, True, "fea", metrics=metrics)


def run_cfd_for_entry(
    entry: dict[str, Any],
    corpus_dir: Path,
    cfd_root: Path,
    *,
    force: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> PhysicsResult:
    part_id = entry["part_id"]
    case_dir = (cfd_root / part_id).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    stl = (corpus_dir / entry["stl"]).resolve()

    if not force:
        existing = summarize_fields(case_dir)
        if existing and existing.get("U_mag_max", 0) > 1e-6:
            return PhysicsResult(part_id, True, "cfd", metrics=existing, cached=True)

    if not stl.exists():
        return PhysicsResult(part_id, False, "cfd", error="missing_stl")

    of_env = env or openfoam_env()
    U, Lx, Ly, Lz = domain_from_stl_mm(stl)
    try:
        write_case(case_dir, U, Lx, Ly, Lz)
        bm = run_cmd(["blockMesh", "-case", str(case_dir)], case_dir, of_env, 60)
        (case_dir / "log.blockMesh").write_text((bm.stdout or "") + "\n" + (bm.stderr or ""))
        if bm.returncode != 0:
            return PhysicsResult(part_id, False, "cfd", error="blockMesh")
        sf = run_cmd(["simpleFoam", "-case", str(case_dir)], case_dir, of_env, timeout)
        (case_dir / "log.simpleFoam").write_text((sf.stdout or "") + "\n" + (sf.stderr or ""))
        if sf.returncode != 0:
            return PhysicsResult(part_id, False, "cfd", error="simpleFoam")
        metrics = summarize_fields(case_dir)
        if not metrics or metrics.get("U_mag_max", 0) < 1e-6:
            return PhysicsResult(part_id, False, "cfd", error="empty_fields")
        metrics = {
            **metrics,
            "solver": "simpleFoam",
            "U_inlet": U,
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "mesh": "blockMesh_channel",
            "nu": 1e-5,
        }
        (case_dir / "meta.json").write_text(
            json.dumps({"part_id": f"part:rocket:{part_id}", "U_inlet": U}, indent=2),
            encoding="utf-8",
        )
        return PhysicsResult(part_id, True, "cfd", metrics=metrics)
    except subprocess.TimeoutExpired:
        return PhysicsResult(part_id, False, "cfd", error="timeout")
    except Exception as exc:  # noqa: BLE001
        return PhysicsResult(part_id, False, "cfd", error=f"exception:{type(exc).__name__}"[:120])


def _fea_worker(payload: dict[str, Any]) -> dict[str, Any]:
    entry = payload["entry"]
    result = run_fea_for_entry(
        entry,
        Path(payload["corpus_dir"]),
        Path(payload["fea_root"]),
        force=payload["force"],
        timeout=payload["timeout"],
        cl_max_mm=payload["cl_max_mm"],
        target_tets=int(payload.get("target_tets") or 0),
        mesh_timeout_s=int(payload.get("mesh_timeout_s") or 120),
    )
    return {
        "part_id": result.part_id,
        "success": result.success,
        "kind": result.kind,
        "metrics": result.metrics,
        "error": result.error,
        "cached": result.cached,
    }


def _cfd_worker(payload: dict[str, Any]) -> dict[str, Any]:
    # Rebuild OpenFOAM env inside the worker process.
    entry = payload["entry"]
    result = run_cfd_for_entry(
        entry,
        Path(payload["corpus_dir"]),
        Path(payload["cfd_root"]),
        force=payload["force"],
        timeout=payload["timeout"],
    )
    return {
        "part_id": result.part_id,
        "success": result.success,
        "kind": result.kind,
        "metrics": result.metrics,
        "error": result.error,
        "cached": result.cached,
    }


def ensure_parts_in_graph(
    graph_path: Path | str,
    manifest: list[dict[str, Any]],
    *,
    corpus_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Create Part nodes (+ MADE_OF edges) for corpus entries missing from the graph."""
    graph_file = Path(graph_path)
    corpus = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])

    existing_ids = {n["id"] for n in nodes}
    material_ids = {
        n["id"] for n in nodes if n.get("type") == "Material"
    }
    existing_edge_ids = {e.get("id") for e in edges if e.get("id")}
    added_parts = 0
    added_edges = 0

    for entry in manifest:
        part_id = f"part:rocket:{entry['part_id']}"
        if part_id not in existing_ids:
            nodes.append(
                {
                    "id": part_id,
                    "label": entry["part_id"],
                    "type": "Part",
                    "properties": {
                        "name": entry["part_id"],
                        "part_class": entry.get("family"),
                        "geometry_ref": str((corpus / entry["stl"]).resolve()),
                        "stl": entry.get("stl"),
                        "params": entry.get("params") or {},
                        "faces": entry.get("faces"),
                        "watertight": entry.get("watertight"),
                        "manifest_fingerprint": entry["part_id"],
                        "tags": list(entry.get("tags") or [])
                        + ["openrocket_hardware_8k", "raw_geometry"],
                        "material_id": entry.get("material_id"),
                        "material_name": entry.get("material_name"),
                        "material_category": entry.get("material_category"),
                        "family": entry.get("family"),
                        "extents_mm": entry.get("extents_mm"),
                        "source_corpus": "openrocket_hardware_8k",
                        "raw_geometry": {
                            "format": "stl",
                            "path": str((corpus / entry["stl"]).resolve()),
                            "path_rel": entry.get("stl"),
                            "family": entry.get("family"),
                            "params": entry.get("params") or {},
                            "extents_mm": entry.get("extents_mm"),
                            "faces": entry.get("faces"),
                            "watertight": entry.get("watertight"),
                        },
                    },
                    "raw_geometry": {
                        "format": "stl",
                        "path": str((corpus / entry["stl"]).resolve()),
                        "path_rel": entry.get("stl"),
                        "family": entry.get("family"),
                        "params": entry.get("params") or {},
                        "extents_mm": entry.get("extents_mm"),
                        "faces": entry.get("faces"),
                        "watertight": entry.get("watertight"),
                    },
                    "has_fea": False,
                    "has_cfd": False,
                    "physics_verified": False,
                }
            )
            existing_ids.add(part_id)
            added_parts += 1

        mat_key = entry.get("material_id")
        if mat_key:
            mat_node = f"material:{mat_key}"
            if mat_node in material_ids:
                edge_id = f"edge:rocket:{entry['part_id']}:made_of:{mat_key}"
                if edge_id not in existing_edge_ids:
                    edges.append(
                        {
                            "id": edge_id,
                            "type": "MADE_OF",
                            "source": part_id,
                            "target": mat_node,
                            "properties": {"role": "corpus_material"},
                        }
                    )
                    existing_edge_ids.add(edge_id)
                    added_edges += 1

    # Only rewrite graph when something changed (full dump is expensive).
    if added_parts or added_edges:
        write_graph_atomic(graph_file, graph)
    return {
        "parts_added": added_parts,
        "edges_added": added_edges,
        "parts_total": sum(1 for n in nodes if n.get("type") == "Part"),
    }

def ingest_fea_to_graph(
    graph_path: Path | str,
    fea_root: Path | str,
    results: list[dict[str, Any]] | None = None,
    *,
    min_bytes: int = 50_000,
) -> int:
    """Serialized against the other ingest pipelines (see cadflow.graph_lock)."""
    with graph_lock(graph_path):
        return _ingest_fea_to_graph(graph_path, fea_root, results, min_bytes=min_bytes)


def _ingest_fea_to_graph(
    graph_path: Path | str,
    fea_root: Path | str,
    results: list[dict[str, Any]] | None = None,
    *,
    min_bytes: int = 50_000,
) -> int:
    graph_file = Path(graph_path)
    graph = read_graph(graph_file)
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0

    if results is None:
        results = []
        for case_dir in sorted(Path(fea_root).iterdir()):
            if not case_dir.is_dir():
                continue
            frd = case_dir / "case.frd"
            try:
                frd_bytes = frd.stat().st_size if frd.is_file() else 0
            except OSError:
                continue
            if frd_bytes < min_bytes:
                continue

            meta: dict[str, Any] = {}
            meta_path = case_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}

            # Prefer metrics written by the solver worker (avoids re-parsing FRD).
            cached = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else None
            if cached and cached.get("max_stress_mpa") is not None:
                results.append(
                    {
                        "part_id": case_dir.name,
                        "success": True,
                        "metrics": {
                            **cached,
                            "load_n": meta.get("load_n", cached.get("load_n")),
                            "material_id": meta.get("material_id", cached.get("material_id")),
                            "solver": "calculix",
                            "frd_bytes": cached.get("frd_bytes", frd_bytes),
                        },
                    }
                )
                continue

            # Fallback: light validity + FRD parse (slower).
            if not case_has_valid_frd(case_dir, min_bytes=min_bytes):
                continue
            summary = parse_frd_summary(frd, min_bytes=min_bytes)
            if summary is None:
                continue
            results.append(
                {
                    "part_id": case_dir.name,
                    "success": True,
                    "metrics": {
                        "max_stress_mpa": round(summary.max_von_mises_mpa, 4),
                        "mean_stress_mpa": round(summary.mean_von_mises_mpa, 4),
                        "max_displacement_mm": round(summary.max_displacement_mm, 6),
                        "frd_bytes": summary.frd_bytes,
                        "result_nodes": summary.node_count,
                        "load_n": meta.get("load_n"),
                        "material_id": meta.get("material_id"),
                        "solver": "calculix",
                    },
                }
            )

    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(f"part:rocket:{r['part_id']}")
        if node is None:
            continue
        metrics = r.get("metrics") or {}
        node["has_fea"] = True
        node["fea_case_id"] = r["part_id"]
        node["fea_status"] = "completed"
        node["fea_complete"] = True
        node["fea_verified"] = True
        node["physics_verified"] = True
        node["physics_ready"] = True
        node["simulation_results_fea"] = {
            "solver": "calculix",
            "status": "completed",
            "source": "case.frd",
            "case_id": r["part_id"],
            **{k: v for k, v in metrics.items() if k != "solver"},
        }
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["fea"] = True
        pd["cfd"] = bool(node.get("has_cfd"))
        pd["verified"] = True
        node["physics_data"] = pd
        linked += 1

    # Compact JSON — indent=2 on 12k+ Parts is multi-minute. Atomic so concurrent
    # readers (dataset, shard registration) never observe a torn 250MB document.
    write_graph_atomic(graph_file, graph)
    # FEA ingest historically wiped mass_kg; re-merge from sidecar if present.
    try:
        from cadflow.enrich_tao_mass_properties import apply_sidecar_to_graph

        apply_sidecar_to_graph(graph_file)
    except Exception:  # noqa: BLE001
        pass
    return linked


def ingest_cfd_to_graph(
    graph_path: Path | str,
    results: list[dict[str, Any]],
) -> int:
    """Serialized against the other ingest pipelines (see cadflow.graph_lock)."""
    with graph_lock(graph_path):
        return _ingest_cfd_to_graph(graph_path, results)


def _ingest_cfd_to_graph(
    graph_path: Path | str,
    results: list[dict[str, Any]],
) -> int:
    graph_file = Path(graph_path)
    graph = read_graph(graph_file)
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(f"part:rocket:{r['part_id']}")
        if node is None:
            continue
        metrics = r.get("metrics") or {}
        node["has_cfd"] = True
        node["cfd_case_id"] = r["part_id"]
        node["simulation_results_cfd"] = {
            "solver": "simpleFoam",
            "status": "completed",
            "source": "U,p fields",
            "case_id": r["part_id"],
            **metrics,
        }
        pd = node.get("physics_data") if isinstance(node.get("physics_data"), dict) else {}
        pd["cfd"] = True
        pd["fea"] = bool(node.get("has_fea"))
        pd["verified"] = bool(pd.get("fea") or True)
        node["physics_data"] = pd
        if node.get("has_fea"):
            node["physics_verified"] = True
        linked += 1
    write_graph_atomic(graph_file, graph)
    try:
        from cadflow.enrich_tao_mass_properties import apply_sidecar_to_graph

        apply_sidecar_to_graph(graph_file)
    except Exception:  # noqa: BLE001
        pass
    return linked


def select_entries(
    manifest: list[dict[str, Any]],
    *,
    limit: int = 0,
    families: list[str] | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    entries = manifest
    if families:
        fam = {f.lower() for f in families}
        entries = [e for e in entries if str(e.get("family", "")).lower() in fam]
    entries = entries[offset:]
    if limit > 0:
        entries = entries[:limit]
    return entries


def filter_fea_skip_safe_dupes(
    entries: list[dict[str, Any]],
    fea_root: Path | str,
    maps_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Skip parts whose (stl_sha1, material) canonical sibling already has a valid FRD."""
    fea_root = Path(fea_root)
    maps_file = Path(maps_path) if maps_path else ROOT / "artifacts/rocket_dedupe_maps.json"
    if not maps_file.is_file():
        return entries
    try:
        maps = json.loads(maps_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return entries
    fea_canon = maps.get("fea_canon") or {}
    kept: list[dict[str, Any]] = []
    skipped = 0
    for e in entries:
        pid = e["part_id"]
        canon = fea_canon.get(pid, pid)
        if pid != canon and case_has_valid_frd(fea_root / canon, min_bytes=50_000):
            skipped += 1
            continue
        kept.append(e)
    if skipped:
        print(f"FEA safe-dedupe skip: {skipped} (canonical FRD exists)", flush=True)
    return kept


def strip_fea_mesh_intermediates(case_dir: Path | str) -> None:
    """Drop bulky mesh inputs once a valid FRD exists."""
    case_dir = Path(case_dir)
    if not case_has_valid_frd(case_dir, min_bytes=50_000):
        return
    for name in (
        "mesh.msh",
        "mesh_solid.inp",
        "case.12d",
        "case.dat",
        "case.cvg",
        "case.sta",
        "spooles.out",
    ):
        p = case_dir / name
        if p.is_file():
            p.unlink(missing_ok=True)

def run_batch_fea(
    entries: list[dict[str, Any]],
    corpus_dir: Path,
    fea_root: Path,
    *,
    workers: int = 4,
    force: bool = False,
    timeout: int = 300,
    cl_max_mm: float = 4.0,
    target_tets: int = 0,
    chunk_size: int = 40,
    mesh_timeout_s: int = 120,
) -> list[dict[str, Any]]:
    import multiprocessing as mp

    fea_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(entries)
    if total == 0:
        return results

    # spawn avoids fork+OpenMP/gmsh deadlocks that stall result collection
    # even after case.frd is already on disk.
    ctx = mp.get_context("spawn")

    for start in range(0, total, chunk_size):
        chunk = entries[start : start + chunk_size]
        payloads = [
            {
                "entry": e,
                "corpus_dir": str(corpus_dir),
                "fea_root": str(fea_root),
                "force": force,
                "timeout": timeout,
                "cl_max_mm": cl_max_mm,
                "target_tets": target_tets,
                "mesh_timeout_s": mesh_timeout_s,
            }
            for e in chunk
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_fea_worker, p): p["entry"]["part_id"] for p in payloads
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        {
                            "part_id": pid,
                            "success": False,
                            "kind": "fea",
                            "metrics": None,
                            "error": f"future:{type(exc).__name__}",
                            "cached": False,
                        }
                    )
                done = len(results)
                if done % 10 == 0 or done == total:
                    ok = sum(1 for x in results if x["success"])
                    print(f"  [fea {done}/{total}] ok={ok}", flush=True)
    return results


def run_batch_cfd(
    entries: list[dict[str, Any]],
    corpus_dir: Path,
    cfd_root: Path,
    *,
    workers: int = 4,
    force: bool = False,
    timeout: int = 300,
) -> list[dict[str, Any]]:
    cfd_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "entry": e,
            "corpus_dir": str(corpus_dir),
            "cfd_root": str(cfd_root),
            "force": force,
            "timeout": timeout,
        }
        for e in entries
    ]
    results: list[dict[str, Any]] = []
    # CFD is mostly subprocess-bound; threads are fine, but keep process pool
    # for consistency / isolation of OpenFOAM env.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_cfd_worker, p): p["entry"]["part_id"] for p in payloads}
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            if i % 25 == 0 or i == len(futures):
                ok = sum(1 for x in results if x["success"])
                print(f"  [cfd {i}/{len(futures)}] ok={ok}", flush=True)
    return results
