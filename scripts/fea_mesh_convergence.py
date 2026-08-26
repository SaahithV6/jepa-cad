"""Mesh convergence and discretisation error for the structural solver.

This project sizes pressure vessels and thrust structures with CalculiX and then
reports the resulting stress as if it were the stress. It is not: it is the
stress *this mesh* produced, and the gap between the two is discretisation
error. Nothing in the repository measured that gap. The one prior data point --
"across a 16x refinement on one part the p95 moved 13.8%" -- is a sensitivity
range, not a convergence study: it has no exact answer to converge toward, no
observed order, and no error band.

The case here is a thick-walled cylinder under internal pressure, quarter
symmetric. It is chosen because it is the tank problem this project actually
solves, and because Lame gives its stress field in closed form, so every mesh
can be scored against truth rather than against a finer mesh. It is also free
of re-entrant corners, which means peak stress is a convergent quantity here --
unlike the production corpus, where p95 is used precisely because peak is not.

Two element types are compared. The production path writes C3D4, the linear
tetrahedron, whose own solver manual advises against it for stress. Whether that
advice matters at the mesh densities this project can afford is an empirical
question, and this script answers it.

Reported per element type: observed order of convergence from Richardson
extrapolation, the Grid Convergence Index in Roache's form as used by ASME
V&V 20, and the signed error against Lame at every refinement level.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Problem definition. SI throughout except where noted.
# ---------------------------------------------------------------------------

A_INNER = 0.050          # m, inner radius
B_OUTER = 0.075          # m, outer radius -- 25 mm wall, thick enough that the
                         # through-thickness gradient is real and must be resolved
LENGTH = 0.020           # m, axial extent of the modelled slice
PRESSURE = 10.0e6        # Pa, internal

E_MODULUS = 73.1e9       # Pa, Al 2219-T87
POISSON = 0.33

#: CalculiX C3D4 face node sets, 1-based within the element connectivity.
#: Used to map a boundary triangle onto the *DLOAD face label that carries it.
C3D4_FACES = {1: (1, 2, 3), 2: (1, 4, 2), 3: (2, 4, 3), 4: (3, 4, 1)}


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def lame_stress(r: float) -> tuple[float, float, float]:
    """Radial, hoop and axial stress at radius ``r``, in Pa.

    Plane strain: the modelled slice has both end faces held at uz = 0, so the
    axial stress is the Poisson reaction nu*(sr + st) rather than a free-end
    zero or a closed-end pa^2/(b^2-a^2). Matching this to the boundary
    conditions actually applied is the whole point -- a closed-end formula
    against a plane-strain model would show a fixed offset that never converges
    away, and would read exactly like solver error.
    """
    a2, b2 = A_INNER ** 2, B_OUTER ** 2
    k = PRESSURE * a2 / (b2 - a2)
    sr = k * (1.0 - b2 / (r * r))
    st = k * (1.0 + b2 / (r * r))
    sz = POISSON * (sr + st)
    return sr, st, sz


def von_mises(sr: float, st: float, sz: float) -> float:
    return math.sqrt(0.5 * ((sr - st) ** 2 + (st - sz) ** 2 + (sz - sr) ** 2))


def exact_peak_vm() -> float:
    """Peak von Mises, which Lame places at the bore."""
    return von_mises(*lame_stress(A_INNER))


# ---------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    nodes: dict[int, tuple[float, float, float]]
    elements: list[tuple[int, list[int]]]
    inner_faces: list[tuple[int, int]]       # (element id, ccx face label)
    sym_x: list[int]                         # nodes on x=0
    sym_y: list[int]                         # nodes on y=0
    sym_z: list[int]                         # nodes on both z faces
    order: int


def build_mesh(cl_mm: float, order: int) -> Mesh:
    """Quarter annulus, tet-meshed at characteristic length ``cl_mm``.

    The solid is built from OCC primitives rather than remeshed from an STL.
    That is deliberate: STL faceting introduces a geometric error that shrinks
    with the *tessellation*, not with the mesh, so a convergence study run on
    faceted input measures the wrong thing and converges to the faceted body's
    answer instead of the cylinder's.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("cyl")

        quarter = math.pi / 2.0
        outer = gmsh.model.occ.addCylinder(
            0, 0, 0, 0, 0, LENGTH, B_OUTER, angle=quarter)
        inner = gmsh.model.occ.addCylinder(
            0, 0, 0, 0, 0, LENGTH, A_INNER, angle=quarter)
        gmsh.model.occ.cut([(3, outer)], [(3, inner)])
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_mm * 1e-3)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_mm * 1e-3 * 0.5)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        nodes = {int(t): (ncoords[3 * i], ncoords[3 * i + 1], ncoords[3 * i + 2])
                 for i, t in enumerate(ntags)}

        tet_type = 4 if order == 1 else 11
        npe = 4 if order == 1 else 10
        etags, enodes = gmsh.model.mesh.getElementsByType(tet_type)
        elements = [(int(t), [int(x) for x in enodes[i * npe:(i + 1) * npe]])
                    for i, t in enumerate(etags)]

        # Classify boundary surfaces geometrically. Tag order out of OCC is not
        # contractual, so nothing here depends on it.
        #
        # FLAT_TOL is 1e-6 m and not something tighter because OCC inflates
        # every bounding box by about 1e-7: a face lying exactly in z = 0 comes
        # back 2e-7 thick. A 1e-9 test therefore matches no planar face at all,
        # leaves all three constraint sets empty, and hands CalculiX an
        # unconstrained body -- which it solves without complaint, returning
        # stresses in the right units and roughly the right places that are
        # wrong by 350%. The tolerance sits far above OCC's padding and far
        # below the smallest real dimension here, 20 mm.
        flat_tol = 1e-6
        inner_surf, sx, sy, sz = None, [], [], []
        for dim, tag in gmsh.model.getEntities(2):
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
            if abs(z1 - z0) < flat_tol:                   # a z end cap
                sz.append(tag)
            elif abs(x1 - x0) < flat_tol:                 # the x = 0 symmetry cut
                sx.append(tag)
            elif abs(y1 - y0) < flat_tol:                 # the y = 0 symmetry cut
                sy.append(tag)
            elif abs(x1 - A_INNER) < 1e-5:                # bore: bbox reaches a, not b
                inner_surf = tag
        if inner_surf is None:
            raise RuntimeError("could not identify the bore surface")
        if len(sz) != 2 or len(sx) != 1 or len(sy) != 1:
            raise RuntimeError(
                f"surface classification found {len(sx)} x-symmetry, {len(sy)} "
                f"y-symmetry and {len(sz)} end faces; expected 1, 1 and 2")

        def surf_nodes(tags: list[int]) -> list[int]:
            out: set[int] = set()
            for t in tags:
                nt, _, _ = gmsh.model.mesh.getNodes(2, t, includeBoundary=True)
                out.update(int(x) for x in nt)
            return sorted(out)

        sym_x, sym_y, sym_z = surf_nodes(sx), surf_nodes(sy), surf_nodes(sz)

        tri_type = 2 if order == 1 else 9
        ttags, tnodes = gmsh.model.mesh.getElementsByType(tri_type, inner_surf)
        tpe = 3 if order == 1 else 6
        bore_tris = [frozenset(int(x) for x in tnodes[i * tpe:i * tpe + 3])
                     for i in range(len(ttags))]
    finally:
        gmsh.finalize()

    inner_faces = _map_faces_to_elements(bore_tris, elements)
    return Mesh(nodes, elements, inner_faces, sym_x, sym_y, sym_z, order)


def _map_faces_to_elements(
    tris: list[frozenset[int]], elements: list[tuple[int, list[int]]]
) -> list[tuple[int, int]]:
    """Attach each bore triangle to the tet face that carries it.

    CalculiX takes pressure as ``element, Pn``, so a surface triangle is not
    enough -- the face index within its owning element is required, and getting
    it wrong loads the wrong face silently.
    """
    node_to_elem: dict[int, list[int]] = {}
    conn: dict[int, list[int]] = {}
    for eid, nds in elements:
        conn[eid] = nds
        for n in nds[:4]:                      # corners identify the face
            node_to_elem.setdefault(n, []).append(eid)

    out: list[tuple[int, int]] = []
    for tri in tris:
        any_node = next(iter(tri))
        for eid in node_to_elem.get(any_node, ()):
            corners = conn[eid][:4]
            if not tri.issubset(corners):
                continue
            for label, idx in C3D4_FACES.items():
                if frozenset(corners[i - 1] for i in idx) == tri:
                    out.append((eid, label))
                    break
            break
    if len(out) != len(tris):
        raise RuntimeError(f"mapped {len(out)} of {len(tris)} bore faces")
    return out


def reorder_for_calculix(mesh: Mesh) -> None:
    """Permute gmsh tet10 midside nodes into CalculiX C3D10 order, and verify.

    gmsh and CalculiX disagree on the last two midside nodes. Rather than trust
    a remembered permutation, this applies the swap and then checks every
    element against the geometry: each midside node must sit at the midpoint of
    the corner pair its slot claims. A wrong permutation produces a mesh that
    still solves and still looks plausible, so the assertion is the only thing
    standing between a silent 20% error and a correct one.
    """
    if mesh.order != 2:
        return

    # CalculiX C3D10 midside slots, as corner pairs.
    pairs = [(1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4)]

    # Assign by nearest straight midpoint rather than by a remembered gmsh
    # permutation. Two reasons. A hard-coded permutation is exactly the kind of
    # fact that is easy to recall wrongly and impossible to notice afterwards --
    # a mis-ordered C3D10 mesh solves happily and returns a plausible field.
    # And setOrder(2) projects midside nodes onto the true curved surface, so
    # they do *not* sit at straight-line midpoints: on the bore the offset is
    # about h^2/8r, some 0.16 mm at h = 8 mm. Any exact-midpoint assertion
    # therefore fires on correct meshes. Nearest-match tolerates that offset
    # while still separating the alternatives, because a wrong slot is a whole
    # edge away -- roughly h/2, more than an order of magnitude further.
    for eid, nds in mesh.elements:
        corners, mids = nds[:4], nds[4:]
        chosen: list[int] = []
        for (i, j) in pairs:
            p, q = mesh.nodes[corners[i - 1]], mesh.nodes[corners[j - 1]]
            want = tuple(0.5 * (p[k] + q[k]) for k in range(3))
            edge_len = math.dist(p, q)
            ranked = sorted(mids, key=lambda n: math.dist(mesh.nodes[n], want))
            best, runner = ranked[0], ranked[1]
            d_best = math.dist(mesh.nodes[best], want)
            d_next = math.dist(mesh.nodes[runner], want)
            if d_best > 0.25 * edge_len or d_next < 2.0 * max(d_best, 1e-12):
                raise RuntimeError(
                    f"element {eid}: cannot identify midside node for edge "
                    f"({i},{j}) -- nearest {d_best:.3e} m, next {d_next:.3e} m, "
                    f"edge {edge_len:.3e} m")
            chosen.append(best)
        if len(set(chosen)) != 6:
            raise RuntimeError(
                f"element {eid}: midside assignment is not a bijection")
        nds[4:] = chosen


# ---------------------------------------------------------------------------
# Deck and solve
# ---------------------------------------------------------------------------

def write_deck(mesh: Mesh, case: Path) -> None:
    # An empty constraint set is not a degenerate case to tolerate, it is a
    # rigid body mode. CalculiX will still return a field for one, and that
    # field looks like an answer. Refusing here is the only cheap place to
    # catch it: every check downstream reads "0 nodes misplaced" and passes.
    if not (mesh.sym_x and mesh.sym_y and mesh.sym_z):
        raise RuntimeError(
            f"unconstrained model: symx={len(mesh.sym_x)} symy={len(mesh.sym_y)} "
            f"symz={len(mesh.sym_z)} -- refusing to solve a free-floating body")
    if not mesh.inner_faces:
        raise RuntimeError("no pressure faces: the model carries no load")

    etype = "C3D4" if mesh.order == 1 else "C3D10"
    L = [f"*NODE, NSET=NALL"]
    for nid in sorted(mesh.nodes):
        x, y, z = mesh.nodes[nid]
        L.append(f"{nid}, {x:.10e}, {y:.10e}, {z:.10e}")
    L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for eid, nds in mesh.elements:
        L.append(f"{eid}, " + ", ".join(str(n) for n in nds))

    for name, ids in (("SYMX", mesh.sym_x), ("SYMY", mesh.sym_y),
                      ("SYMZ", mesh.sym_z)):
        L.append(f"*NSET, NSET={name}")
        for i in range(0, len(ids), 8):
            L.append(", ".join(str(n) for n in ids[i:i + 8]) + ",")

    L += [
        "*MATERIAL, NAME=AL",
        "*ELASTIC",
        f"{E_MODULUS:.6e}, {POISSON}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=AL",
        "*STEP",
        "*STATIC",
        # Quarter symmetry plus plane strain. Together these remove all six
        # rigid body modes, so no artificial anchor node is needed -- an anchor
        # would introduce a local stress spike right where the field is read.
        "*BOUNDARY",
        "SYMX, 1, 1",
        "SYMY, 2, 2",
        "SYMZ, 3, 3",
        "*DLOAD",
    ]
    for eid, face in mesh.inner_faces:
        L.append(f"{eid}, P{face}, {PRESSURE:.6e}")
    L += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP", ""]
    (case / "job.inp").write_text("\n".join(L))


def solve(case: Path, timeout: int) -> bool:
    ccx = Path.home() / ".local" / "bin" / "ccx"
    try:
        subprocess.run([str(ccx), "-i", "job"], cwd=case, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False
    return (case / "job.frd").exists() and (case / "job.frd").stat().st_size > 5000


def read_stress(case: Path) -> dict[int, tuple[float, ...]]:
    """Extract the nodal stress tensor from the .frd STRESS block.

    Fixed-width 12-character fields after a 13-character key+id prefix. The
    format is column oriented, not whitespace delimited: a value large enough to
    fill its field abuts its neighbour and any split() based reader silently
    mis-associates every component after it.
    """
    out: dict[int, tuple[float, ...]] = {}
    inside = False
    for line in (case / "job.frd").read_text(errors="ignore").splitlines():
        if line.startswith("  1PSTEP") or line.startswith(" -4  DISP"):
            inside = False
        if line.startswith(" -4  STRESS"):
            inside = True
            continue
        if not inside:
            continue
        if line.startswith(" -3"):
            inside = False
            continue
        if line.startswith(" -1"):
            nid = int(line[3:13])
            vals = [float(line[13 + 12 * i:25 + 12 * i]) for i in range(6)]
            out[nid] = tuple(vals)
    return out


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def evaluate(mesh: Mesh, stress: dict[int, tuple[float, ...]]) -> dict:
    """Score a solved mesh against Lame, on the peak and in an L2 norm."""
    exact_peak = exact_peak_vm()
    peak, sq, n = 0.0, 0.0, 0
    bore_sr: list[float] = []
    for nid, (sxx, syy, szz, sxy, syz, szx) in stress.items():
        x, y, z = mesh.nodes[nid]
        vm = math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
        peak = max(peak, vm)
        r = math.hypot(x, y)
        if abs(r - A_INNER) < 1e-6 and r > 0:
            # Rotate the tensor onto the radial direction. Traction equilibrium
            # forces this to equal -p at the loaded surface no matter how coarse
            # the mesh is, so it tests the load and the restraints rather than
            # the discretisation -- which is why it is reported separately from
            # the convergence metrics and not folded into them.
            c, s = x / r, y / r
            bore_sr.append(c * c * sxx + s * s * syy + 2.0 * c * s * sxy)
        if A_INNER + 1e-6 < r < B_OUTER - 1e-6:   # interior only; the L2 norm is
            sr, st, sz_ = lame_stress(r)          # a field measure, and the two
            sq += (vm - von_mises(sr, st, sz_)) ** 2   # bounding radii are where
            n += 1                                     # nodal averaging is weakest
    volume = (math.pi / 4.0) * (B_OUTER ** 2 - A_INNER ** 2) * LENGTH
    bore_mean = (sum(bore_sr) / len(bore_sr)) if bore_sr else float("nan")
    return {
        "nodes": len(mesh.nodes),
        "elements": len(mesh.elements),
        "h_mm": 1e3 * (volume / len(mesh.elements)) ** (1.0 / 3.0),
        "peak_vm_mpa": peak / 1e6,
        "exact_peak_vm_mpa": exact_peak / 1e6,
        "peak_error_pct": 100.0 * (peak - exact_peak) / exact_peak,
        "l2_error_mpa": math.sqrt(sq / max(1, n)) / 1e6,
        "bore_radial_mpa": bore_mean / 1e6,
        "bore_radial_target_mpa": -PRESSURE / 1e6,
        "bore_radial_error_pct": 100.0 * (bore_mean + PRESSURE) / PRESSURE,
    }


def gci(fine: dict, mid: dict, coarse: dict) -> dict | None:
    """Observed order and Grid Convergence Index, Roache's formulation.

    The 1.25 factor is the standard safety factor for a three-grid study with a
    computed order. GCI is reported as a percentage band on the finest result:
    it is a numerical uncertainty estimate, not a bound on the answer.
    """
    f1, f2, f3 = fine["peak_vm_mpa"], mid["peak_vm_mpa"], coarse["peak_vm_mpa"]
    h1, h2, h3 = fine["h_mm"], mid["h_mm"], coarse["h_mm"]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if abs(e21) < 1e-12 or r21 <= 1.0 or r32 <= 1.0:
        return None

    # Classify before extrapolating. Richardson assumes the solution is in the
    # asymptotic range and approaching its limit monotonically; fed an
    # oscillating sequence it still returns a number, and that number is
    # meaningless. This case produced exactly that -- C3D4 peak stress went
    # 31.94, 31.74, 31.78 across three refinements and the formula reported an
    # order of 5.5, which for a linear tetrahedron is impossible. Reporting a
    # tight GCI derived from it would have understated the real uncertainty by
    # a wide margin, so the convergence ratio is checked first.
    ratio = e21 / e32 if abs(e32) > 1e-12 else float("inf")
    if ratio < 0:
        span = max(f1, f2, f3) - min(f1, f2, f3)
        return {
            "convergence": "oscillatory",
            "convergence_ratio_R": round(ratio, 4),
            "note": "not in the asymptotic range; Richardson extrapolation does "
                    "not apply. Uncertainty is bounded by the oscillation "
                    "amplitude instead.",
            "oscillation_band_pct": round(100.0 * span / f1, 3),
        }
    if ratio >= 1.0:
        return {
            "convergence": "divergent",
            "convergence_ratio_R": round(ratio, 4),
            "note": "refinement is not reducing the change between grids; no "
                    "converged value can be claimed.",
        }
    s = math.copysign(1.0, e32 / e21)
    p = 2.0
    for _ in range(200):                      # fixed point on Roache's implicit p
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
            p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        except (ValueError, ZeroDivisionError):
            return None
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    if not (0.1 < p < 8.0):
        return {"order_p": round(p, 3), "note": "order outside a credible range"}
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1.0)
    band = 1.25 * abs(e21 / f1) / (r21 ** p - 1.0)
    return {
        "convergence": "monotone",
        "convergence_ratio_R": round(ratio, 4),
        "order_p": round(p, 3),
        "richardson_extrapolated_mpa": round(ext, 4),
        "gci_fine_pct": round(100.0 * band, 3),
        "refinement_ratio_21": round(r21, 3),
    }


def order_from_norm(rows: list[dict], key: str) -> dict | None:
    """Least-squares convergence order of ``key`` against element size.

    The three-grid GCI reads only the last three points of a sequence and is
    fragile when the finest ones sit close together, which is exactly where
    peak stress starts oscillating. A regression of log(error) on log(h) uses
    every refinement level and is what the L2 field norm is for: the peak is a
    single node's value and can wander, whereas the norm integrates the whole
    field and falls smoothly. Where the two disagree, the norm is the honest
    description of how the solver converges.
    """
    pts = [(math.log(r["h_mm"]), math.log(abs(r[key]))) for r in rows
           if r.get(key) and abs(r[key]) > 1e-12]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx < 1e-15:
        return None
    slope = sxy / sxx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in pts)
    return {
        "metric": key,
        "order_p": round(slope, 3),
        "r_squared": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "levels": n,
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/verification/fea_mesh_convergence.json")
    ap.add_argument("--cl", type=float, nargs="+",
                    default=[8.0, 6.0, 4.5, 3.4, 2.5],
                    help="characteristic lengths in mm, coarse to fine")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print(f"exact peak von Mises (Lame, plane strain): "
          f"{exact_peak_vm()/1e6:.4f} MPa\n")

    results: dict[str, list[dict]] = {}
    work = Path(tempfile.mkdtemp(prefix="fea_conv_"))
    try:
        for order in args.orders:
            etype = "C3D4" if order == 1 else "C3D10"
            results[etype] = []
            print(f"--- {etype} ---")
            for cl in args.cl:
                case = work / f"{etype}_cl{cl:g}"
                case.mkdir(parents=True, exist_ok=True)
                try:
                    mesh = build_mesh(cl, order)
                    reorder_for_calculix(mesh)
                except Exception as exc:  # noqa: BLE001
                    print(f"  cl={cl:<5g} mesh failed: {type(exc).__name__}: {exc}")
                    continue
                write_deck(mesh, case)
                if not solve(case, args.timeout):
                    print(f"  cl={cl:<5g} solve failed "
                          f"({len(mesh.elements)} elements)")
                    continue
                row = evaluate(mesh, read_stress(case))
                row["cl_mm"] = cl
                results[etype].append(row)
                print(f"  cl={cl:<5g} {row['elements']:>7d} el  "
                      f"h={row['h_mm']:.3f} mm  peak={row['peak_vm_mpa']:8.4f} MPa  "
                      f"err={row['peak_error_pct']:+7.2f}%  "
                      f"L2={row['l2_error_mpa']:.4f} MPa  "
                      f"bore_sr={row['bore_radial_mpa']:+7.3f} MPa", flush=True)
            rows = results[etype]
            if len(rows) >= 3:
                g = gci(rows[-1], rows[-2], rows[-3])
                if g:
                    if g.get("convergence") == "monotone":
                        print(f"  peak stress: monotone, order p = {g['order_p']}, "
                              f"GCI(fine) = {g['gci_fine_pct']}%")
                    else:
                        print(f"  peak stress: {g['convergence']} (R = "
                              f"{g['convergence_ratio_R']}) -- {g['note']}")
                results[etype + "_gci"] = g  # type: ignore[assignment]
                orders = [o for o in (order_from_norm(rows, "l2_error_mpa"),
                                      order_from_norm(rows, "bore_radial_error_pct"))
                          if o]
                for o in orders:
                    print(f"  {o['metric']}: order p = {o['order_p']} "
                          f"(R2 = {o['r_squared']}, {o['levels']} levels)")
                results[etype + "_orders"] = orders  # type: ignore[assignment]
            print()
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"cases kept in {work}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "case": "thick-walled cylinder under internal pressure, quarter symmetric",
        "geometry": {"a_m": A_INNER, "b_m": B_OUTER, "length_m": LENGTH},
        "load_pa": PRESSURE,
        "material": {"E_pa": E_MODULUS, "nu": POISSON},
        "boundary_conditions": "quarter symmetry (ux=0 on x=0, uy=0 on y=0), "
                               "plane strain (uz=0 on both end faces)",
        "exact_peak_vm_mpa": exact_peak_vm() / 1e6,
        "results": results,
    }, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
