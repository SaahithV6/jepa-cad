"""Class-conditioned FEA/CFD alternate decks for existing ~2k Parts.

Baseline ``fea_final`` / ``cfd_final`` used one global BC. This module builds
situation-aware alternates from Part labels → ``physics_targets`` families and
stores them under ``artifacts/fea_alt`` / ``artifacts/cfd_alt`` without touching
the rocket 8k lane.
"""

from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadflow.msh_to_calculix import (
    DEFAULT_CCX,
    SolidMesh,
    parse_frd_summary,
    parse_msh2_solid,
    pick_face_boundary_nodes,
    run_calculix_case,
    write_solid_mesh_inp,
    _nset_lines,
)
from cadflow.part_family import classify_part, preferred_modality
from cadflow.physics_targets import physics_targets_for, resolve_family

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "artifacts/jepa-train-bundle/graph.json"
FEA_BASE = ROOT / "artifacts/fea_final"
FEA_ALT = ROOT / "artifacts/fea_alt"
CFD_ALT = ROOT / "artifacts/cfd_alt"
SUMMARY_PATH = ROOT / "artifacts/physics_alternates_summary.json"

# C3D4 local face node index triples (1-based face convention unused; we use nodes).
_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _alt_frd_valid(case_dir: Path, frd_name: str = "case_alt.frd", min_bytes: int = 20_000) -> bool:
    frd = case_dir / frd_name
    if not frd.exists() or frd.stat().st_size < min_bytes:
        return False
    text = frd.read_text(errors="ignore")
    return "DISP" in text and "STRESS" in text


def _centroid(nodes: dict[int, tuple[float, float, float]]) -> tuple[float, float, float]:
    xs = ys = zs = 0.0
    n = len(nodes)
    for x, y, z in nodes.values():
        xs += x
        ys += y
        zs += z
    return xs / n, ys / n, zs / n


def _free_tris(mesh: SolidMesh) -> list[tuple[int, int, int]]:
    """Boundary triangles (node-id ordered) appearing on exactly one tet."""
    counts: dict[tuple[int, int, int], int] = {}
    for _, nids in mesh.elements:
        for a, b, c in _TET_FACES:
            key = tuple(sorted((nids[a], nids[b], nids[c])))
            counts[key] = counts.get(key, 0) + 1
    return [f for f, c in counts.items() if c == 1]  # type: ignore[misc]


def _tri_geom(
    nodes: dict[int, tuple[float, float, float]], tri: tuple[int, int, int]
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    p0, p1, p2 = (nodes[i] for i in tri)
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    area = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
    if area < 1e-18:
        return 0.0, (0.0, 0.0, 0.0), p0
    inv = 1.0 / (2.0 * area)
    nrm = (nx * inv, ny * inv, nz * inv)
    cen = ((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0)
    return area, nrm, cen


def _material_e(part: dict[str, Any]) -> tuple[float, float, float, str]:
    """Return (E_pa, nu, density_kgm3, safe_material_name)."""
    props = part.get("properties") or {}
    cat = str(props.get("material_category") or "").lower()
    name = str(props.get("material_name") or "Steel")
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:32] or "Steel"
    if "aluminum" in cat or "alumin" in name.lower() or name.lower().startswith("al_"):
        return 68.9e9, 0.33, 2700.0, safe
    if "titanium" in cat or "ti-" in name.lower() or name.lower().startswith("ti"):
        return 110e9, 0.34, 4500.0, safe
    if "copper" in cat or "cu" in name.lower() or "cfrp" in name.lower() or "compos" in cat:
        # CFRP: orthotropic ignored; use laminate-ish isotropic proxy + low density
        if "cfrp" in name.lower() or "compos" in cat:
            return 70e9, 0.30, 1550.0, safe
        return 110e9, 0.34, 8960.0, safe
    return 210e9, 0.30, 7850.0, safe


def _inner_pressure_loads(
    mesh: SolidMesh, pressure_pa: float
) -> dict[int, tuple[float, float, float]]:
    """Nodal forces from internal pressure on the *inner* free-surface band.

    Solid meshes often have both inner and outer free faces. We keep faces whose
    centroid is closer to the part centroid than the median free-face distance
    (inner shell), and push along the inward normal (fluid → wall).
    """
    center = _centroid(mesh.nodes)
    faces: list[tuple[tuple[int, int, int], float, tuple[float, float, float], tuple[float, float, float], float]] = []
    for tri in _free_tris(mesh):
        area, nrm, cen = _tri_geom(mesh.nodes, tri)
        if area <= 0:
            continue
        dist = math.sqrt(
            (cen[0] - center[0]) ** 2 + (cen[1] - center[1]) ** 2 + (cen[2] - center[2]) ** 2
        )
        faces.append((tri, area, nrm, cen, dist))
    if not faces:
        return {}
    dists = sorted(f[4] for f in faces)
    med = dists[len(dists) // 2]
    forces: dict[int, list[float]] = {}
    for tri, area, nrm, cen, dist in faces:
        if dist > med:
            continue  # outer envelope
        to_c = (center[0] - cen[0], center[1] - cen[1], center[2] - cen[2])
        # Inward normal (toward cavity / center)
        if nrm[0] * to_c[0] + nrm[1] * to_c[1] + nrm[2] * to_c[2] < 0:
            nrm = (-nrm[0], -nrm[1], -nrm[2])
        # Pressure on wall: fluid pushes wall *away* from cavity → opposite of inward
        # For a pressure vessel, stress is hoop tension: wall sees outward force from internal P.
        fx = pressure_pa * area * (-nrm[0]) / 3.0
        fy = pressure_pa * area * (-nrm[1]) / 3.0
        fz = pressure_pa * area * (-nrm[2]) / 3.0
        for nid in tri:
            slot = forces.setdefault(nid, [0.0, 0.0, 0.0])
            slot[0] += fx
            slot[1] += fy
            slot[2] += fz
    return {nid: (v[0], v[1], v[2]) for nid, v in forces.items()}


def write_alternate_fea_deck(
    mesh: SolidMesh,
    out_dir: Path,
    family: str,
    targets: dict[str, Any],
    *,
    youngs: float,
    poisson: float,
    density: float,
    material_name: str,
) -> dict[str, Any]:
    """Write mesh_solid.inp + case_alt.inp for the family load case."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_solid_mesh_inp(mesh, out_dir / "mesh_solid.inp")
    fam = resolve_family(family)
    bcs = targets["boundary_conditions"]
    sim_kind = targets["sim_kind"]
    mat = material_name
    meta: dict[str, Any] = {
        "family": fam,
        "sim_kind": sim_kind,
        "load_case": bcs.get("load_case"),
        "material": mat,
        "youngs_pa": youngs,
        "density": density,
    }

    lines = [
        "*HEADING",
        f"Alternate FEA family={fam} kind={sim_kind}",
        "*INCLUDE, INPUT=mesh_solid.inp",
        f"*MATERIAL, NAME={mat}",
        "*ELASTIC",
        f"{youngs:.6e}, {poisson}",
        "*DENSITY",
        f"{density:.6e}",
        f"*SOLID SECTION, ELSET=ALL, MATERIAL={mat}",
        "*STEP",
        "*STATIC",
    ]

    if fam in {"tank", "combustion_chamber"} or sim_kind in {"pressure_vessel", "thermo_structural"}:
        # Internal pressure (Pa). chamber_pressure_bar or meop_bar from targets.
        bar = float(
            targets["targets"].get("chamber_pressure_bar")
            or targets["targets"].get("meop_bar")
            or targets["targets"].get("proof_pressure_bar")
            or 30.0
        )
        # Proof factor ~1.5 when catalog says so.
        if "proof" in str(bcs.get("load_case", "")).lower():
            bar *= 1.5
        pressure_pa = bar * 1e5
        meta["pressure_bar"] = bar
        meta["pressure_pa"] = pressure_pa
        fixed, _ = pick_face_boundary_nodes(mesh.nodes, axis="x")
        if len(fixed) < 3:
            fixed, _ = pick_face_boundary_nodes(mesh.nodes, axis="z")
        forces = _inner_pressure_loads(mesh, pressure_pa)
        # Keep only loaded nodes that aren't purely fixed (allow overlap).
        lines.extend(_nset_lines("FIXED", fixed))
        lines.extend(["*BOUNDARY", "FIXED, 1, 3, 0.0", "*CLOAD"])
        n_load = 0
        for nid, (fx, fy, fz) in forces.items():
            if abs(fx) + abs(fy) + abs(fz) < 1e-9:
                continue
            if fx:
                lines.append(f"{nid}, 1, {fx:.6e}")
            if fy:
                lines.append(f"{nid}, 2, {fy:.6e}")
            if fz:
                lines.append(f"{nid}, 3, {fz:.6e}")
            n_load += 1
        meta["loaded_nodes"] = n_load
        meta["fixed_nodes"] = len(fixed)

    elif fam == "fin" or "aero" in str(bcs.get("load_case", "")).lower():
        # Root fix (min-x) + tip lateral force approximating aero load.
        fixed, tip = pick_face_boundary_nodes(mesh.nodes, axis="x")
        # Lateral along Y; magnitude from target stress window → force scale via bbox.
        xs = [p[0] for p in mesh.nodes.values()]
        ys = [p[1] for p in mesh.nodes.values()]
        zs = [p[2] for p in mesh.nodes.values()]
        area = max(max(ys) - min(ys), 1e-4) * max(max(zs) - min(zs), 1e-4)
        q = 0.5 * 1.225 * (float(bcs.get("freestream_velocity_mps") or 100.0) ** 2)
        total = min(5e4, max(10.0, q * area * 0.5))  # crude lift proxy
        meta["aero_force_n"] = total
        per = total / max(len(tip), 1)
        lines.extend(_nset_lines("FIXED", fixed))
        lines.extend(_nset_lines("TIP", tip))
        lines.extend(["*BOUNDARY", "FIXED, 1, 3, 0.0", "*CLOAD"])
        for nid in tip:
            lines.append(f"{nid}, 2, {per:.6e}")
        meta["fixed_nodes"] = len(fixed)
        meta["loaded_nodes"] = len(tip)

    else:
        # structure / spacecraft_bus / generic / deployable / aero shells:
        # compressive face load ≈ mass * Ng (more reliable than GRAV on coarse tets).
        g_mult = 6.0
        lc = str(bcs.get("load_case") or "")
        if "6g" in lc or "launch" in lc or "axial" in lc or "lateral" in lc:
            g_mult = 6.0
        elif fam == "deployable":
            g_mult = float(targets["targets"].get("shock_load_g") or 10.0)
        elif sim_kind == "external_aero":
            # Axial aero push on nose/fairing
            g_mult = 0.0
        vols = 0.0
        # tet volume sum for mass
        for _, nids in mesh.elements:
            p0, p1, p2, p3 = (mesh.nodes[i] for i in nids)
            ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
            vols += abs(
                ax * (by * cz - bz * cy)
                - ay * (bx * cz - bz * cx)
                + az * (bx * cy - by * cx)
            ) / 6.0
        mass = density * vols
        if sim_kind == "external_aero":
            xs = [p[0] for p in mesh.nodes.values()]
            ys = [p[1] for p in mesh.nodes.values()]
            zs = [p[2] for p in mesh.nodes.values()]
            area = max(max(ys) - min(ys), 1e-4) * max(max(zs) - min(zs), 1e-4)
            q = 0.5 * float(bcs.get("air_density_kgm3") or 1.225) * (
                float(bcs.get("freestream_velocity_mps") or 100.0) ** 2
            )
            total = min(5e4, max(5.0, q * area))
            meta["aero_force_n"] = total
        else:
            total = max(5.0, mass * 9.80665 * g_mult)
            meta["g_load"] = g_mult
            meta["mass_kg"] = mass
        fixed, loaded = pick_face_boundary_nodes(mesh.nodes, axis="x")
        if len(fixed) < 3 or len(loaded) < 3:
            fixed, loaded = pick_face_boundary_nodes(mesh.nodes, axis="z")
        per = total / max(len(loaded), 1)
        lines.extend(_nset_lines("FIXED", fixed))
        lines.extend(_nset_lines("LOADED", loaded))
        lines.extend(["*BOUNDARY", "FIXED, 1, 3, 0.0", "*CLOAD"])
        for nid in loaded:
            lines.append(f"{nid}, 1, {per:.6e}")
        meta["fixed_nodes"] = len(fixed)
        meta["loaded_nodes"] = len(loaded)
        meta["total_load_n"] = total

    lines.extend(["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"])
    (out_dir / "case_alt.inp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _case_id_from_part(part: dict[str, Any]) -> str | None:
    props = part.get("properties") or {}
    # Prefer existing fea case id
    for key in ("fea_case_id", "cfd_case_id"):
        v = part.get(key) or props.get(key)
        if v:
            return str(v)
    gref = str(props.get("geometry_ref") or "")
    m = __import__("re").search(r"/runs/([0-9a-f]{8,})/", gref)
    if m:
        return m.group(1)
    # fingerprint suffix
    fp = str(props.get("manifest_fingerprint") or "")
    if len(fp) >= 12:
        return fp[:16]
    return None


def run_fea_alternate(part: dict[str, Any], *, force: bool = False, timeout: int = 300) -> dict[str, Any]:
    cid = _case_id_from_part(part)
    result: dict[str, Any] = {"part_id": part["id"], "success": False, "modality": "fea"}
    if not cid:
        result["error"] = "no_case_id"
        return result
    result["case_id"] = cid
    src = FEA_BASE / cid
    msh = src / "mesh.msh"
    if not msh.exists():
        result["error"] = "missing_mesh"
        return result

    out = FEA_ALT / cid
    frd = out / "case_alt.frd"
    if not force and _alt_frd_valid(out, min_bytes=50_000):
        summary = parse_frd_summary(frd, min_bytes=50_000)
        result["success"] = True
        result["cached"] = True
        if summary:
            result["metrics"] = {
                "max_stress_mpa": summary.max_von_mises_mpa,
                "max_displacement_mm": summary.max_displacement_mm,
            }
        return result

    family = classify_part(part)
    props = part.get("properties") or {}
    fp = str(props.get("manifest_fingerprint") or part["id"])
    targets = physics_targets_for(fp, family)
    youngs, poisson, density, mat = _material_e(part)
    mesh = parse_msh2_solid(msh)
    # Copy mesh.msh for provenance
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "mesh.msh").exists():
        shutil.copy2(msh, out / "mesh.msh")
    meta = write_alternate_fea_deck(
        mesh,
        out,
        family,
        targets,
        youngs=youngs,
        poisson=poisson,
        density=density,
        material_name=mat,
    )
    # Also stamp part_id into FEA meta for disk→graph ingest
    meta["part_id"] = part["id"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    result["family"] = family
    result["sim_kind"] = targets["sim_kind"]
    result["meta"] = meta

    run = run_calculix_case(out, job_name="case_alt", ccx_binary=DEFAULT_CCX, timeout=timeout)
    if not run.converged or not _alt_frd_valid(out, min_bytes=20_000):
        result["error"] = "calculix"
        result["frd_bytes"] = run.frd_bytes
        return result
    summary = parse_frd_summary(frd, min_bytes=20_000)
    if not summary:
        result["error"] = "parse_frd"
        return result
    result["success"] = True
    result["metrics"] = {
        "max_stress_mpa": summary.max_von_mises_mpa,
        "mean_stress_mpa": summary.mean_von_mises_mpa,
        "max_displacement_mm": summary.max_displacement_mm,
        "frd_bytes": summary.frd_bytes,
    }
    result["targets"] = targets["targets"]
    return result


def run_cfd_alternate(part: dict[str, Any], *, force: bool = False, timeout: int = 180) -> dict[str, Any]:
    """Family-tagged channel CFD proxy with freestream/internal BC from physics_targets."""
    # Import OpenFOAM helpers lazily to avoid circular imports at module load.
    from run_cfd_5k_proper import (
        domain_from_stl,
        openfoam_env,
        run_cmd,
        summarize_fields,
        write_case,
    )

    cid = _case_id_from_part(part)
    result: dict[str, Any] = {"part_id": part["id"], "success": False, "modality": "cfd"}
    if not cid:
        result["error"] = "no_case_id"
        return result
    result["case_id"] = cid
    out = CFD_ALT / cid
    if not force:
        existing = summarize_fields(out)
        if existing:
            result["success"] = True
            result["cached"] = True
            result["metrics"] = existing
            return result

    family = classify_part(part)
    props = part.get("properties") or {}
    fp = str(props.get("manifest_fingerprint") or part["id"])
    targets = physics_targets_for(fp, family)
    bcs = targets["boundary_conditions"]
    sim_kind = targets["sim_kind"]
    gref = Path(str(props.get("geometry_ref") or ""))
    stl = gref.with_suffix(".stl")
    U_default, Lx, Ly, Lz = domain_from_stl(stl) if stl.exists() else (1.0, 2.0, 0.4, 0.1)

    if sim_kind == "external_aero":
        U = float(bcs.get("freestream_velocity_mps") or U_default)
        nu = float(bcs.get("kinematic_viscosity_m2s") or 1.48e-5)
        tag = "external_aero_proxy"
    elif sim_kind == "internal_flow":
        # Higher Re pipe-like proxy from pressure drop / velocity targets.
        U = float(targets["targets"].get("flow_velocity_mps") or max(U_default, 5.0))
        if "inlet_total_pressure_bar" in bcs or "manifold_pressure_bar" in bcs:
            U = max(U, 10.0)
        nu = 1e-5
        tag = "internal_flow_proxy"
    else:
        U = U_default
        nu = 1e-5
        tag = "channel_proxy"

    U = min(80.0, max(0.5, U))
    env = openfoam_env()
    write_case(out, U, Lx, Ly, Lz)
    # Patch viscosity for family
    (out / "constant/transportProperties").write_text(
        (out / "constant/transportProperties").read_text().replace("1e-05", f"{nu:.6e}")
    )
    bm = run_cmd(["blockMesh", "-case", str(out)], out, env, 60)
    if bm.returncode != 0:
        result["error"] = "blockMesh"
        return result
    sf = run_cmd(["simpleFoam", "-case", str(out)], out, env, timeout)
    if sf.returncode != 0:
        result["error"] = "simpleFoam"
        return result
    metrics = summarize_fields(out)
    if not metrics:
        result["error"] = "no_fields"
        return result
    meta = {
        "family": family,
        "sim_kind": sim_kind,
        "proxy_tag": tag,
        "U_inlet": U,
        "nu": nu,
        "note": "bbox channel proxy — geometry body not immersed; BC magnitudes are family-conditioned",
    }
    (out / "meta.json").write_text(
        json.dumps({"part_id": part["id"], **meta}, indent=2), encoding="utf-8"
    )
    result.update({"success": True, "family": family, "sim_kind": sim_kind, "meta": meta, "metrics": metrics})
    result["targets"] = targets["targets"]
    return result


def ingest_alternates(graph_path: Path, results: list[dict[str, Any]]) -> int:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    linked = 0
    for r in results:
        if not r.get("success"):
            continue
        node = by_id.get(r["part_id"])
        if not node:
            continue
        family = r.get("family") or classify_part(node)
        node["physics_family"] = family
        node["physics_sim_kind"] = r.get("sim_kind")
        metrics = r.get("metrics") or {}
        meta = r.get("meta") or {}
        if r.get("modality") == "fea":
            node["simulation_results_fea_alt"] = {
                "solver": "CalculiX",
                "status": "completed",
                "source": "case_alt.frd",
                "case_id": r.get("case_id"),
                "family": family,
                "sim_kind": r.get("sim_kind"),
                "load_case": meta.get("load_case"),
                "pressure_bar": meta.get("pressure_bar"),
                "g_load": meta.get("g_load"),
                "aero_force_n": meta.get("aero_force_n"),
                "total_load_n": meta.get("total_load_n"),
                "max_stress_mpa": metrics.get("max_stress_mpa"),
                "mean_stress_mpa": metrics.get("mean_stress_mpa"),
                "max_displacement_mm": metrics.get("max_displacement_mm"),
                "targets": r.get("targets"),
            }
            node["has_fea_alt"] = True
        elif r.get("modality") == "cfd":
            node["simulation_results_cfd_alt"] = {
                "solver": "simpleFoam",
                "status": "completed",
                "source": "U,p fields",
                "case_id": r.get("case_id"),
                "family": family,
                "sim_kind": r.get("sim_kind"),
                "proxy_tag": meta.get("proxy_tag"),
                "U_inlet": meta.get("U_inlet"),
                "U_mag_max": metrics.get("U_mag_max"),
                "p_mean": metrics.get("p_mean"),
                "note": meta.get("note"),
                "targets": r.get("targets"),
            }
            node["has_cfd_alt"] = True
        pd = node.get("physics_data")
        if not isinstance(pd, dict):
            pd = {}
        pd["alternates"] = True
        node["physics_data"] = pd
        linked += 1
    # Atomic write — avoids truncating TAO graph.json on ENOSPC mid-write.
    tmp = graph_path.with_suffix(graph_path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph), encoding="utf-8")
    tmp.replace(graph_path)
    return linked


def _part_id_by_case(graph: dict[str, Any]) -> dict[str, str]:
    """Map FEA/CFD case ids → Part node ids."""
    import re

    run_re = re.compile(r"/runs/([0-9a-f]{8,})/")
    out: dict[str, str] = {}
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        props = n.get("properties") or {}
        for key in ("fea_case_id", "cfd_case_id"):
            v = n.get(key) or props.get(key)
            if v:
                out[str(v)] = n["id"]
        gref = str(props.get("geometry_ref") or "")
        m = run_re.search(gref)
        if m:
            out[m.group(1)] = n["id"]
        fp = str(props.get("manifest_fingerprint") or "")
        if len(fp) >= 12:
            out[fp[:16]] = n["id"]
    return out


def ingest_alternates_from_disk(graph_path: Path = GRAPH_PATH) -> dict[str, int]:
    """Rebuild alternate annotations on TAO Parts from fea_alt/ + cfd_alt/."""
    from run_cfd_5k_proper import summarize_fields

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_case = _part_id_by_case(graph)
    by_id = {n["id"]: n for n in graph["nodes"] if n.get("type") == "Part"}
    results: list[dict[str, Any]] = []

    if FEA_ALT.exists():
        for case_dir in sorted(FEA_ALT.iterdir()):
            if not case_dir.is_dir():
                continue
            frd = case_dir / "case_alt.frd"
            if not _alt_frd_valid(case_dir):
                continue
            summary = parse_frd_summary(frd, min_bytes=20_000)
            if not summary:
                continue
            meta: dict[str, Any] = {}
            if (case_dir / "meta.json").exists():
                meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            part_id = meta.get("part_id") or by_case.get(case_dir.name)
            if not part_id or part_id not in by_id:
                continue
            node = by_id[part_id]
            family = meta.get("family") or classify_part(node)
            props = node.get("properties") or {}
            fp = str(props.get("manifest_fingerprint") or part_id)
            targets = physics_targets_for(fp, family)
            results.append(
                {
                    "part_id": part_id,
                    "case_id": case_dir.name,
                    "success": True,
                    "modality": "fea",
                    "family": family,
                    "sim_kind": meta.get("sim_kind") or targets["sim_kind"],
                    "meta": meta,
                    "metrics": {
                        "max_stress_mpa": summary.max_von_mises_mpa,
                        "mean_stress_mpa": summary.mean_von_mises_mpa,
                        "max_displacement_mm": summary.max_displacement_mm,
                        "frd_bytes": summary.frd_bytes,
                    },
                    "targets": targets["targets"],
                }
            )

    if CFD_ALT.exists():
        for case_dir in sorted(CFD_ALT.iterdir()):
            if not case_dir.is_dir():
                continue
            metrics = summarize_fields(case_dir)
            if not metrics:
                continue
            meta = {}
            if (case_dir / "meta.json").exists():
                meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            part_id = meta.get("part_id") or by_case.get(case_dir.name)
            if not part_id or part_id not in by_id:
                continue
            node = by_id[part_id]
            family = meta.get("family") or classify_part(node)
            props = node.get("properties") or {}
            fp = str(props.get("manifest_fingerprint") or part_id)
            targets = physics_targets_for(fp, family)
            results.append(
                {
                    "part_id": part_id,
                    "case_id": case_dir.name,
                    "success": True,
                    "modality": "cfd",
                    "family": family,
                    "sim_kind": meta.get("sim_kind") or targets["sim_kind"],
                    "meta": meta,
                    "metrics": metrics,
                    "targets": targets["targets"],
                }
            )

    linked = ingest_alternates(graph_path, results)
    fea_n = sum(1 for r in results if r["modality"] == "fea")
    cfd_n = sum(1 for r in results if r["modality"] == "cfd")
    summary = {"fea_disk": fea_n, "cfd_disk": cfd_n, "graph_linked": linked}
    SUMMARY_PATH.write_text(json.dumps({**summary, "mode": "ingest_from_disk"}, indent=2), encoding="utf-8")
    return summary


def select_parts_for_alternates(graph: dict[str, Any], limit: int = 0) -> list[dict[str, Any]]:
    parts = [n for n in graph["nodes"] if n.get("type") == "Part"]
    # Prefer parts that already have baseline FEA mesh linkage
    chosen = []
    for p in parts:
        cid = _case_id_from_part(p)
        if not cid:
            continue
        if not (FEA_BASE / cid / "mesh.msh").exists():
            continue
        chosen.append(p)
        if limit > 0 and len(chosen) >= limit:
            break
    return chosen


def run_batch(
    *,
    pilot: int = 0,
    workers: int = 4,
    force: bool = False,
    fea_only: bool = False,
    cfd_only: bool = False,
) -> dict[str, Any]:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    parts = select_parts_for_alternates(graph, limit=pilot)
    FEA_ALT.mkdir(parents=True, exist_ok=True)
    CFD_ALT.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, dict[str, Any]]] = []
    for p in parts:
        fam = classify_part(p)
        mode = preferred_modality(fam)
        if not cfd_only and mode in {"fea", "both"}:
            jobs.append(("fea", p))
        if not fea_only and mode in {"cfd", "both"}:
            jobs.append(("cfd", p))

    results: list[dict[str, Any]] = []

    def _work(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        kind, part = item
        if kind == "fea":
            return run_fea_alternate(part, force=force)
        return run_cfd_alternate(part, force=force)

    print(f"Alternate physics jobs={len(jobs)} parts={len(parts)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            status = "OK" if r.get("success") else f"FAIL:{r.get('error')}"
            if i % 10 == 0 or not r.get("success"):
                print(
                    f"  [{i}/{len(jobs)}] {r.get('modality')} {r.get('case_id')} "
                    f"{r.get('family')} {status}",
                    flush=True,
                )

    ok = [r for r in results if r.get("success")]
    linked = ingest_alternates(GRAPH_PATH, ok)
    summary = {
        "jobs": len(jobs),
        "parts": len(parts),
        "successful": len(ok),
        "fea_ok": sum(1 for r in ok if r.get("modality") == "fea"),
        "cfd_ok": sum(1 for r in ok if r.get("modality") == "cfd"),
        "graph_linked": linked,
        "fea_alt": str(FEA_ALT),
        "cfd_alt": str(CFD_ALT),
        "failures": [
            {"part_id": r["part_id"], "modality": r.get("modality"), "error": r.get("error")}
            for r in results
            if not r.get("success")
        ][:40],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary