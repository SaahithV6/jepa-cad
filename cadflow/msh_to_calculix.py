"""Convert Gmsh MSH2 meshes to CalculiX-compatible solid-only INP decks.

Gmsh exports mixed 1D/2D/3D elements when written via ``gmsh.write()``. CalculiX
then creates millions of spurious constraints from surface/shell elements. This
module parses MSH2 directly and keeps only tetrahedral (type 4) volume elements.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path
import subprocess
from typing import Any, Iterable

# Gmsh MSH2 element type -> node count (only types we parse).
_GMSH_NODE_COUNTS: dict[int, int] = {
    1: 2,   # line
    2: 3,   # triangle
    4: 4,   # tetrahedron
    9: 6,   # 6-node triangle (second order)
    11: 10,  # 10-node tetrahedron (second order)
    15: 1,  # point
}

#: Gmsh element types that carry a solid tetrahedron, linear then quadratic.
_TET_TYPES = (4, 11)

DEFAULT_CCX = Path.home() / ".local" / "bin" / "ccx"


@dataclass(frozen=True, slots=True)
class SolidMesh:
    nodes: dict[int, tuple[float, float, float]]
    elements: list[tuple[int, list[int]]]


@dataclass(frozen=True, slots=True)
class FEASetupResult:
    case_dir: Path
    mesh_inp: Path
    case_inp: Path
    node_count: int
    element_count: int
    fixed_nodes: int
    loaded_nodes: int


@dataclass(frozen=True, slots=True)
class FEARunResult:
    case_dir: Path
    converged: bool
    frd_path: Path | None
    dat_path: Path | None
    frd_bytes: int
    dat_bytes: int


@dataclass(frozen=True, slots=True)
class FRDSummary:
    case_id: str
    max_displacement_mm: float
    max_von_mises_mpa: float
    mean_von_mises_mpa: float
    node_count: int
    frd_bytes: int


_RUN_ID_RE = re.compile(r"/runs/([0-9a-f]{8,})/")


def parse_msh2_solid(msh_file: Path | str) -> SolidMesh:
    """Parse a Gmsh MSH2 file and return nodes plus tetrahedral elements only."""
    lines = Path(msh_file).read_text().splitlines()
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, list[int]]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            i += 1
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                parts = lines[i].split()
                i += 1
                node_id = int(parts[0])
                nodes[node_id] = (float(parts[1]), float(parts[2]), float(parts[3]))
            continue

        if line == "$Elements":
            i += 1
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                parts = lines[i].split()
                i += 1
                elem_id = int(parts[0])
                elem_type = int(parts[1])
                num_tags = int(parts[2])
                node_count = _GMSH_NODE_COUNTS.get(elem_type)
                if node_count is None:
                    continue
                node_ids = [int(parts[3 + num_tags + j]) for j in range(node_count)]
                if elem_type in _TET_TYPES:
                    elements.append((elem_id, node_ids))
            continue

        i += 1

    return SolidMesh(nodes=nodes, elements=elements)


def _nset_lines(name: str, node_ids: Iterable[int], per_line: int = 16) -> list[str]:
    ids = list(node_ids)
    if not ids:
        return []
    lines = [f"*NSET, NSET={name}"]
    for start in range(0, len(ids), per_line):
        chunk = ids[start : start + per_line]
        lines.append(", ".join(str(nid) for nid in chunk))
    return lines


def longest_axis(nodes: dict[int, tuple[float, float, float]]) -> str:
    """Axis with the largest extent -- the part's load-carrying direction.

    Airframe sections are axial members: a tank carries thrust along its length.
    Loading one transversely instead bends it as a cantilever, which for a thin
    shell is a completely different (and far more severe) problem.
    """
    def robust_span(vals: list[float]) -> float:
        """5th-95th percentile span, so a protrusion does not define the axis.

        Plain min/max is dominated by outliers: on a short tank section a fin
        sticking out radially made the x extent (83.6 mm) exceed the axial z
        extent (58.8 mm), so the part was loaded across its diameter instead of
        along its length. That reported 2,538 MPa on a wall whose nominal stress
        is 116 MPa. A fin is a handful of nodes and the body is thousands, so
        trimming the tails picks the structural axis.
        """
        v = sorted(vals)
        if len(v) < 20:
            return v[-1] - v[0]
        return v[int(0.95 * len(v))] - v[int(0.05 * len(v))]

    spans = {
        "x": robust_span([p[0] for p in nodes.values()]),
        "y": robust_span([p[1] for p in nodes.values()]),
        "z": robust_span([p[2] for p in nodes.values()]),
    }
    return max(spans, key=spans.get)


def pick_face_boundary_nodes(
    nodes: dict[int, tuple[float, float, float]],
    axis: str | None = None,
    tol_fraction: float = 0.01,
) -> tuple[list[int], list[int]]:
    """Return node ids on the min/max faces along ``axis``.

    Defaults to the longest axis. It used to default to x regardless of how the
    part was oriented, so a tube lying along z was gripped at one side and
    pushed on the other -- bending rather than compressing it. On a 1.1 mm shell
    that reported 75,935 MPa and 178 mm of deflection against a nominal wall
    stress of 197 MPa.
    """
    if axis is None:
        axis = longest_axis(nodes)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    coords = {nid: pos[axis_idx] for nid, pos in nodes.items()}
    min_val = min(coords.values())
    max_val = max(coords.values())
    tol = (max_val - min_val) * tol_fraction + 1e-9
    fixed = sorted(nid for nid, val in coords.items() if abs(val - min_val) <= tol)
    loaded = sorted(nid for nid, val in coords.items() if abs(val - max_val) <= tol)
    return fixed, loaded


#: gmsh and CalculiX order the last two midside nodes of a quadratic tetrahedron
#: differently: gmsh puts mid(3,4) in slot 9 and mid(2,4) in slot 10, CalculiX
#: the other way round. Verified against element geometry rather than recalled,
#: because a mis-ordered C3D10 mesh solves without complaint and returns a
#: plausible field -- there is no symptom to notice.
_GMSH_TO_CCX_TET10 = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)


def _verify_tet10_order(mesh: SolidMesh, permuted: list[list[int]],
                        sample: int = 200) -> None:
    """Check a sample of permuted elements against the CalculiX face convention.

    Uses a nearest-midpoint test with a wide tolerance rather than an equality
    test: gmsh projects midside nodes onto curved geometry, so they do not sit
    at straight-line midpoints -- on a 50 mm bore with 8 mm edges the offset is
    about 0.16 mm. An exact test fires on correct meshes. A wrong slot, by
    contrast, is a whole edge away, so the two are easy to separate.
    """
    pairs = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
    for nds in permuted[:sample]:
        corners = nds[:4]
        for slot, (i, j) in enumerate(pairs, start=5):
            p, q = mesh.nodes[corners[i - 1]], mesh.nodes[corners[j - 1]]
            want = tuple(0.5 * (p[k] + q[k]) for k in range(3))
            got = mesh.nodes[nds[slot - 1]]
            d = math.dist(got, want)
            edge = math.dist(p, q)
            if edge > 0 and d > 0.25 * edge:
                raise ValueError(
                    f"C3D10 node ordering is wrong at slot {slot}: midside node "
                    f"sits {d:.3e} m from the ({i},{j}) midpoint on a "
                    f"{edge:.3e} m edge")


def write_solid_mesh_inp(mesh: SolidMesh, output_file: Path | str) -> None:
    """Write a CalculiX include deck with *NODE and tetrahedral *ELEMENT blocks.

    Element type follows the connectivity: four nodes give C3D4, ten give
    C3D10. Both are supported because C3D4 is a poor stress element -- its own
    solver manual says so, and a Lame verification case measured it converging
    first-order and reading 9.8% low on surface stress at 34,493 elements,
    where C3D10 does better with 661. See
    artifacts/verification/fea_mesh_convergence.json.
    """
    output = Path(output_file)
    if not mesh.elements:
        raise ValueError("mesh carries no solid elements")

    width = len(mesh.elements[0][1])
    if any(len(n) != width for _, n in mesh.elements):
        raise ValueError("mixed element orders in one mesh")
    if width == 4:
        etype, conn = "C3D4", [list(n) for _, n in mesh.elements]
    elif width == 10:
        etype = "C3D10"
        conn = [[n[i] for i in _GMSH_TO_CCX_TET10] for _, n in mesh.elements]
        _verify_tet10_order(mesh, conn)
    else:
        raise ValueError(f"unsupported tetrahedral connectivity of {width} nodes")

    with output.open("w", encoding="utf-8") as handle:
        handle.write("*NODE\n")
        for node_id in sorted(mesh.nodes):
            x, y, z = mesh.nodes[node_id]
            handle.write(f"{node_id}, {x:.10e}, {y:.10e}, {z:.10e}\n")
        handle.write(f"*ELEMENT, TYPE={etype}, ELSET=ALL\n")
        for (elem_id, _), nds in zip(mesh.elements, conn):
            handle.write(f"{elem_id}, {', '.join(map(str, nds))}\n")


def generate_fea_case_inp(
    case_dir: Path | str,
    mesh_filename: str = "mesh_solid.inp",
    case_filename: str = "case.inp",
    load_axis: str | None = "z",
    total_load: float = 5_000_000.0,  # N; sized for meter-scale meshes + steel E
    youngs_modulus: float = 210_000_000_000.0,  # Pa; meshes are in meters
    poisson: float = 0.3,
) -> FEASetupResult:
    """Build ``mesh_solid.inp`` and ``case.inp`` for a case directory."""
    case_path = Path(case_dir)
    msh_file = case_path / "mesh.msh"
    if not msh_file.exists():
        raise FileNotFoundError(f"Missing mesh: {msh_file}")

    mesh = parse_msh2_solid(msh_file)
    if not mesh.nodes or not mesh.elements:
        raise ValueError(f"No solid elements found in {msh_file}")

    mesh_inp = case_path / mesh_filename
    write_solid_mesh_inp(mesh, mesh_inp)

    # Axis is specified, not inferred. Every part constraints_to_geometry
    # builds is axial along z, and inference gets squat parts wrong: the thrust
    # structure is 71.4 mm across and 58.9 mm long, so "longest axis" is
    # legitimately x, and loading it across the diameter reported 2,538 MPa on a
    # wall whose nominal stress is 116 MPa. Callers with genuinely non-axial
    # geometry can pass load_axis=None to fall back to inference.
    axis = load_axis or longest_axis(mesh.nodes)
    fixed, loaded = pick_face_boundary_nodes(mesh.nodes, axis=axis)
    if not fixed or not loaded:
        raise ValueError(f"Could not derive boundary nodes for {case_path}")
    # CalculiX degrees of freedom are 1=x, 2=y, 3=z; load along the part's axis.
    load_dof = {"x": 1, "y": 2, "z": 3}[axis]

    load_per_node = total_load / len(loaded)
    lines = [
        "*HEADING",
        "FEA solid-only analysis",
        f"*INCLUDE, INPUT={mesh_filename}",
        "*MATERIAL, NAME=Steel",
        "*ELASTIC",
        f"{youngs_modulus:.6e}, {poisson}",
        "*SOLID SECTION, ELSET=ALL, MATERIAL=Steel",
        "*STEP",
        "*STATIC",
    ]
    lines.extend(_nset_lines("FIXED", fixed))
    lines.extend(_nset_lines("LOADED", loaded))
    lines.extend(["*BOUNDARY", "FIXED, 1, 3, 0.0", "*CLOAD"])
    for node_id in loaded:
        lines.append(f"{node_id}, {load_dof}, {load_per_node:.6f}")
    lines.extend(["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"])

    case_inp = case_path / case_filename
    case_inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return FEASetupResult(
        case_dir=case_path,
        mesh_inp=mesh_inp,
        case_inp=case_inp,
        node_count=len(mesh.nodes),
        element_count=len(mesh.elements),
        fixed_nodes=len(fixed),
        loaded_nodes=len(loaded),
    )


def generate_modal_case_inp(
    case_dir: Path | str,
    mesh_filename: str = "mesh_solid.inp",
    case_filename: str = "modal.inp",
    fix_axis: str | None = "z",
    modes: int = 6,
    youngs_modulus: float = 210_000_000_000.0,
    poisson: float = 0.3,
    density: float = 7850.0,
) -> FEASetupResult:
    """Build a CalculiX ``*FREQUENCY`` deck for the mesh in ``case_dir``.

    The static check cannot see a resonance. A fin sized only for steady load
    can still be destroyed by flutter, which is set by where its natural
    frequencies sit relative to the flight condition -- and ``first_mode_hz`` is
    already a conditioning slot with nothing populating it.

    A modal analysis needs mass, so this writes ``*DENSITY``, which the static
    deck has no reason to carry and does not.
    """
    case_path = Path(case_dir)
    msh_file = case_path / "mesh.msh"
    if not msh_file.exists():
        raise FileNotFoundError(f"Missing mesh: {msh_file}")

    mesh = parse_msh2_solid(msh_file)
    if not mesh.nodes or not mesh.elements:
        raise ValueError(f"No solid elements found in {msh_file}")

    mesh_inp = case_path / mesh_filename
    write_solid_mesh_inp(mesh, mesh_inp)

    axis = fix_axis or longest_axis(mesh.nodes)
    fixed, _ = pick_face_boundary_nodes(mesh.nodes, axis=axis)
    if not fixed:
        raise ValueError(f"Could not derive fixed nodes for {case_path}")

    lines = [
        "*HEADING",
        "FEA modal analysis",
        f"*INCLUDE, INPUT={mesh_filename}",
        "*MATERIAL, NAME=Steel",
        "*ELASTIC",
        f"{youngs_modulus:.6e}, {poisson}",
        "*DENSITY",
        f"{density:.6e}",
        "*SOLID SECTION, ELSET=ALL, MATERIAL=Steel",
    ]
    lines.extend(_nset_lines("FIXED", fixed))
    lines.extend(["*BOUNDARY", "FIXED, 1, 3, 0.0"])
    lines.extend([
        "*STEP",
        "*FREQUENCY",
        f"{int(max(1, modes))}",
        "*NODE FILE",
        "U",
        "*END STEP",
    ])

    case_inp = case_path / case_filename
    case_inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FEASetupResult(
        case_dir=case_path,
        mesh_inp=mesh_inp,
        case_inp=case_inp,
        node_count=len(mesh.nodes),
        element_count=len(mesh.elements),
        fixed_nodes=len(fixed),
        loaded_nodes=0,
    )


def parse_eigenfrequencies(dat_path: Path | str) -> list[float]:
    """Natural frequencies in Hz from a CalculiX ``.dat`` eigenvalue block.

    CalculiX prints mode number, eigenvalue, then the frequency in rad/s and in
    cycles/s; the last column is the one wanted. Rigid-body modes come out at
    essentially zero and are dropped, so the first entry is the first *elastic*
    mode -- which is what "first_mode_hz" is supposed to mean.
    """
    path = Path(dat_path)
    if not path.exists():
        return []
    freqs: list[float] = []
    in_block = False
    for line in path.read_text(errors="ignore").splitlines():
        upper = line.upper()
        if "EIGENVALUE" in upper.replace(" ", ""):
            in_block = True
            continue
        if not in_block:
            continue
        parts = line.split()
        if not parts:
            continue
        if not parts[0].lstrip("-").isdigit():
            if freqs:
                break
            continue
        # Columns are: mode, eigenvalue, frequency (rad/time), frequency
        # (cycles/time), and -- for the real solver, though not in every
        # CalculiX build -- an imaginary part. Taking parts[-1] read that
        # imaginary column, which is identically zero for an undamped
        # eigenproblem, so every frequency was filtered out as rigid-body.
        if len(parts) < 4:
            continue
        try:
            freqs.append(float(parts[3]))
        except ValueError:
            continue
    if not freqs:
        return []
    # Drop rigid-body modes. An absolute epsilon is not enough on its own: a
    # rigid-body mode comes out at whatever the solver's round-off gives, which
    # was 5e-6 Hz in one case and sailed past a 1e-6 cut. Real modes of a rocket
    # part are Hz to kHz and a cantilever's second mode is only ~6x its first,
    # so anything four orders below the highest computed mode is numerical.
    ceiling = max(freqs)
    floor = max(1e-3, 1e-4 * ceiling)
    return [f for f in freqs if f > floor]


def run_calculix_case(
    case_dir: Path | str,
    job_name: str = "case",
    ccx_binary: Path | str = DEFAULT_CCX,
    timeout: int = 600,
) -> FEARunResult:
    """Run CalculiX in ``case_dir`` and report whether result files were produced."""
    case_path = Path(case_dir)
    ccx = Path(ccx_binary)
    if not ccx.exists():
        raise FileNotFoundError(f"CalculiX binary not found: {ccx}")

    env = dict(os.environ)
    # PaStiX/OpenMP under ProcessPool + OpenFOAM load → "Failed during initial
    # partitioning" / double-free. Force CalculiX to one thread (override, don't
    # inherit a polluted parent OMP_NUM_THREADS).
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMBER_OF_CPUS"] = "1"
    proc = subprocess.run(
        [str(ccx), job_name],
        cwd=str(case_path),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    # Persist solver chatter so partitioning deaths aren't silent empty .dat files.
    try:
        (case_path / "ccx.log").write_text(
            (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""),
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        pass

    frd_path = case_path / f"{job_name}.frd"
    dat_path = case_path / f"{job_name}.dat"
    frd_bytes = frd_path.stat().st_size if frd_path.exists() else 0
    dat_bytes = dat_path.stat().st_size if dat_path.exists() else 0

    return FEARunResult(
        case_dir=case_path,
        converged=frd_bytes > 0,
        frd_path=frd_path if frd_bytes > 0 else None,
        dat_path=dat_path if dat_bytes > 0 else None,
        frd_bytes=frd_bytes,
        dat_bytes=dat_bytes,
    )


def prepare_fea_workdir_from_stl(
    stl_path: Path | str,
    workdir: Path | str,
    *,
    cl_max_mm: float = 6.0,
    cl_min_mm: float = 1.5,
    total_load_n: float = 1_000.0,
    youngs_modulus: float = 70e9,
    poisson: float = 0.33,
    # Thin shells legitimately take longer to mesh than solids: a wall needs
    # several elements through its thickness, so element counts run to hundreds
    # of thousands and 180 s was timing out on the nose cone and stage 2 tank,
    # which then reported nothing at all.
    mesh_timeout_s: int = 900,
    scale_to_meters: bool = True,
    allow_hull_fallback: bool = True,
    load_axis: str | None = "z",
) -> FEASetupResult:
    """Mesh an STL (mm) → MSH2 (meters) → CalculiX ``case.inp`` in ``workdir``."""
    from cadflow.rocket_physics_suite import mesh_stl_volume

    case_path = Path(workdir)
    case_path.mkdir(parents=True, exist_ok=True)
    stl = Path(stl_path)
    if not stl.exists():
        raise FileNotFoundError(f"geometry STL missing: {stl}")

    # Keep a local copy so case dirs are self-contained.
    local_stl = case_path / "geometry.stl"
    if stl.resolve() != local_stl.resolve():
        local_stl.write_bytes(stl.read_bytes())

    msh = case_path / "mesh.msh"
    # Try the requested element size, then progressively coarser ones, and only
    # accept a convex hull if none of them produce a real mesh.
    #
    # A hull is not a degraded answer, it is a different part: for a thin shell
    # it is a solid billet of the same envelope, so every stress it returns is
    # about something that was never designed. A mesh two steps coarser is still
    # the actual geometry. The fin set showed why this matters -- the same
    # design meshed cleanly at one tessellation and hulled at another, differing
    # only in the last digits of its dimensions, so meshability was turning on
    # luck rather than on anything about the part.
    mesh_result = None
    for factor in (1.0, 1.5, 2.25, 3.0):
        attempt = mesh_stl_volume(
            local_stl,
            msh,
            cl_max_mm=cl_max_mm * factor,
            cl_min_mm=cl_min_mm * factor,
            scale_to_meters=scale_to_meters,
            mesh_timeout_s=mesh_timeout_s,
            allow_hull_fallback=False,
        )
        if attempt.success and msh.exists() and not getattr(
                attempt, "used_hull", False):
            mesh_result = attempt
            if factor > 1.0:
                (case_path / "MESH_COARSENED").write_text(
                    f"Meshed at {factor:g}x the requested element size "
                    f"({cl_max_mm * factor:.3f}/{cl_min_mm * factor:.3f} mm). "
                    f"The requested size did not produce a real mesh, and a "
                    f"coarser mesh of the actual part is worth more than a "
                    f"convex hull of it.\n")
            break
    if mesh_result is None and allow_hull_fallback:
        mesh_result = mesh_stl_volume(
            local_stl, msh,
            cl_max_mm=cl_max_mm, cl_min_mm=cl_min_mm,
            scale_to_meters=scale_to_meters, mesh_timeout_s=mesh_timeout_s,
            allow_hull_fallback=True,
        )
    if mesh_result is None or not mesh_result.success or not msh.exists():
        err = mesh_result.error if mesh_result else "no element size produced a mesh"
        raise RuntimeError(f"gmsh mesh failed: {err}")

    # Record when the mesh is a convex-hull proxy rather than the real
    # geometry. For a thin-walled part the hull is a solid billet, so the
    # stresses that come back describe something else entirely and must not be
    # reported as verifying the shell.
    if getattr(mesh_result, "used_hull", False):
        (case_path / "MESH_IS_CONVEX_HULL").write_text(
            "The volume mesh came from a convex-hull proxy, not the actual "
            "geometry. For a hollow part this is a solid of the same envelope, "
            "so any stress result describes a billet rather than the shell.\n")

    return generate_fea_case_inp(
        case_path,
        load_axis=load_axis,
        total_load=float(total_load_n),
        youngs_modulus=float(youngs_modulus),
        poisson=float(poisson),
    )


def convert_and_run_case(
    case_dir: Path | str,
    ccx_binary: Path | str = DEFAULT_CCX,
    timeout: int = 600,
) -> tuple[FEASetupResult, FEARunResult]:
    """Convert MSH2 to solid INP, generate BCs, and run CalculiX for one case."""
    setup = generate_fea_case_inp(case_dir)
    result = run_calculix_case(case_dir, ccx_binary=ccx_binary, timeout=timeout)
    return setup, result


def _von_mises(sxx: float, syy: float, szz: float, sxy: float, syz: float, szx: float) -> float:
    return math.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + szx**2)
        )
    )


def parse_frd_summary(frd_file: Path | str, *, min_bytes: int = 100_000) -> FRDSummary | None:
    """Extract max displacement and von Mises stress from a CalculiX FRD file."""
    path = Path(frd_file)
    if not path.exists() or path.stat().st_size < min_bytes:
        return None

    max_disp = 0.0
    max_vm = 0.0
    vm_sum = 0.0
    vm_count = 0
    node_count = 0
    mode: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.lstrip()
            if stripped.startswith("-4"):
                upper = stripped.upper()
                if "DISP" in upper:
                    mode = "disp"
                elif "STRESS" in upper:
                    mode = "stress"
                else:
                    mode = None
                continue
            if stripped.startswith("-3"):
                mode = None
                continue
            if not stripped.startswith("-1") or mode is None:
                continue

            # FRD data lines: " -1" + 10-char node id + values
            payload = line[3:]
            try:
                _node = int(payload[:10])
                values = [float(tok) for tok in payload[10:].split()]
            except ValueError:
                continue

            if mode == "disp" and len(values) >= 3:
                mag = math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)
                if mag > max_disp:
                    max_disp = mag
                node_count += 1
            elif mode == "stress" and len(values) >= 6:
                vm = _von_mises(*values[:6])
                if vm > max_vm:
                    max_vm = vm
                vm_sum += vm
                vm_count += 1

    if node_count == 0 and vm_count == 0:
        return None

    # Mesh coords are meters; E is Pascals → stress in Pa, disp in m.
    return FRDSummary(
        case_id=path.parent.name,
        max_displacement_mm=max_disp * 1000.0,
        max_von_mises_mpa=max_vm / 1e6,
        mean_von_mises_mpa=((vm_sum / vm_count) / 1e6) if vm_count else 0.0,
        node_count=node_count,
        frd_bytes=path.stat().st_size,
    )


def case_has_valid_frd(case_dir: Path | str, *, min_bytes: int = 100_000) -> bool:
    """True only when FRD exists, is large enough, and contains result blocks.

    Size alone is insufficient: aborted CalculiX runs often leave multi‑MB FRDs
    with mesh/header data but no DISP/STRESS fields.
    """
    frd = Path(case_dir) / "case.frd"
    if not frd.exists() or frd.stat().st_size < min_bytes:
        return False
    has_disp = False
    has_stress = False
    with frd.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            upper = chunk.upper()
            if b"DISP" in upper:
                has_disp = True
            if b"STRESS" in upper:
                has_stress = True
            if has_disp and has_stress:
                return True
    return False


def collect_fea_summaries(
    fea_dir: Path | str,
    *,
    min_bytes: int = 100_000,
) -> dict[str, FRDSummary]:
    """Parse all valid FRD files under ``fea_dir``."""
    root = Path(fea_dir)
    summaries: dict[str, FRDSummary] = {}
    for case_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if not case_has_valid_frd(case_dir, min_bytes=min_bytes):
            continue
        summary = parse_frd_summary(case_dir / "case.frd", min_bytes=min_bytes)
        if summary is not None:
            summaries[case_dir.name] = summary
    return summaries


def ingest_fea_results_to_graph(
    graph_path: Path | str,
    fea_dir: Path | str,
    *,
    index_path: Path | str | None = None,
    min_bytes: int = 100_000,
    total_load_n: float = 5_000_000.0,
) -> dict[str, Any]:
    """Attach real CalculiX FRD metrics to Part nodes.

    Mapping order:
      1. sweep run id in ``geometry_ref`` (``.../runs/<id>/...``)
      2. ``manifest_fingerprint`` / Part id suffix matching FEA case dir name
    """
    graph_file = Path(graph_path)
    with graph_file.open(encoding="utf-8") as handle:
        graph = json.load(handle)

    summaries = collect_fea_summaries(fea_dir, min_bytes=min_bytes)
    linked = 0
    for node in graph.get("nodes", []):
        if node.get("type") != "Part":
            continue
        props = node.get("properties") or {}
        geometry_ref = str(props.get("geometry_ref") or "")
        match = _RUN_ID_RE.search(geometry_ref)
        case_id = match.group(1) if match else None
        if case_id is None or case_id not in summaries:
            fp = str(props.get("manifest_fingerprint") or "")
            if not fp:
                fp = str(node.get("id") or "").rsplit(":", 1)[-1]
            if fp in summaries:
                case_id = fp
        summary = summaries.get(case_id) if case_id else None

        if summary is None:
            # Keep topology; mark as not yet physics-verified from real FRD.
            node["has_fea"] = False
            node["physics_verified"] = False
            node["fea_status"] = "pending"
            node["physics_data"] = {"fea": False, "cfd": bool(node.get("has_cfd")), "verified": False}
            continue

        node["has_fea"] = True
        node["physics_verified"] = True
        node["fea_status"] = "completed"
        node["fea_complete"] = True
        node["fea_verified"] = True
        node["physics_ready"] = True
        node["fea_case_id"] = summary.case_id
        node["physics_data"] = {"fea": True, "cfd": bool(node.get("has_cfd")), "verified": True}
        node["simulation_results_fea"] = {
            "solver": "calculix",
            "status": "completed",
            "source": "case.frd",
            "case_id": summary.case_id,
            "load_n": total_load_n,
            "max_stress_mpa": round(summary.max_von_mises_mpa, 4),
            "mean_stress_mpa": round(summary.mean_von_mises_mpa, 4),
            "max_displacement_mm": round(summary.max_displacement_mm, 6),
            "frd_bytes": summary.frd_bytes,
            "result_nodes": summary.node_count,
        }
        linked += 1

    with graph_file.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2)

    index = {
        "fea_cases_with_frd": len(summaries),
        "parts_linked": linked,
        "parts_total": sum(1 for n in graph.get("nodes", []) if n.get("type") == "Part"),
        "cases": {
            case_id: {
                "max_stress_mpa": s.max_von_mises_mpa,
                "mean_stress_mpa": s.mean_von_mises_mpa,
                "max_displacement_mm": s.max_displacement_mm,
                "frd_bytes": s.frd_bytes,
            }
            for case_id, s in summaries.items()
        },
    }
    if index_path is not None:
        Path(index_path).write_text(json.dumps(index, indent=2), encoding="utf-8")

    return {
        "fea_cases_with_frd": len(summaries),
        "parts_linked": linked,
        "parts_total": index["parts_total"],
    }
