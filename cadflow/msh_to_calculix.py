"""Convert Gmsh MSH2 meshes to CalculiX-compatible solid-only INP decks.

Gmsh exports mixed 1D/2D/3D elements when written via ``gmsh.write()``. CalculiX
then creates millions of spurious constraints from surface/shell elements. This
module parses MSH2 directly and keeps only tetrahedral (type 4) volume elements.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
import subprocess
from typing import Any, Iterable

# Gmsh MSH2 element type -> node count (only types we parse).
_GMSH_NODE_COUNTS: dict[int, int] = {
    1: 2,   # line
    2: 3,   # triangle
    4: 4,   # tetrahedron
    15: 1,  # point
}

DEFAULT_CCX = Path("/home/best/.local/bin/ccx")


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
                if elem_type == 4:
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


def pick_face_boundary_nodes(
    nodes: dict[int, tuple[float, float, float]],
    axis: str = "x",
    tol_fraction: float = 0.01,
) -> tuple[list[int], list[int]]:
    """Return node ids on the min/max faces along ``axis`` (default: x)."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    coords = {nid: pos[axis_idx] for nid, pos in nodes.items()}
    min_val = min(coords.values())
    max_val = max(coords.values())
    tol = (max_val - min_val) * tol_fraction + 1e-9
    fixed = sorted(nid for nid, val in coords.items() if abs(val - min_val) <= tol)
    loaded = sorted(nid for nid, val in coords.items() if abs(val - max_val) <= tol)
    return fixed, loaded


def write_solid_mesh_inp(mesh: SolidMesh, output_file: Path | str) -> None:
    """Write a CalculiX include deck with *NODE and C3D4 *ELEMENT blocks only."""
    output = Path(output_file)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("*NODE\n")
        for node_id in sorted(mesh.nodes):
            x, y, z = mesh.nodes[node_id]
            handle.write(f"{node_id}, {x:.10e}, {y:.10e}, {z:.10e}\n")
        handle.write("*ELEMENT, TYPE=C3D4, ELSET=ALL\n")
        for elem_id, node_ids in mesh.elements:
            handle.write(f"{elem_id}, {', '.join(map(str, node_ids))}\n")


def generate_fea_case_inp(
    case_dir: Path | str,
    mesh_filename: str = "mesh_solid.inp",
    case_filename: str = "case.inp",
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

    fixed, loaded = pick_face_boundary_nodes(mesh.nodes)
    if not fixed or not loaded:
        raise ValueError(f"Could not derive boundary nodes for {case_path}")

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
        lines.append(f"{node_id}, 1, {load_per_node:.6f}")
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

    subprocess.run(
        [str(ccx), job_name],
        cwd=str(case_path),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

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
    frd = Path(case_dir) / "case.frd"
    return frd.exists() and frd.stat().st_size >= min_bytes


def collect_fea_summaries(
    fea_dir: Path | str,
    *,
    min_bytes: int = 100_000,
) -> dict[str, FRDSummary]:
    """Parse all valid FRD files under ``fea_dir``."""
    root = Path(fea_dir)
    summaries: dict[str, FRDSummary] = {}
    for case_dir in sorted(d for d in root.iterdir() if d.is_dir()):
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
    """Attach real CalculiX FRD metrics to Part nodes via sweep run ids in geometry_ref."""
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
