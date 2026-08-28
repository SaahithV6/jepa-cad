"""Flight-scale stress on a thin shell, using shell elements instead of solids.

Every component this project analyses is a coupon. Body radius is clamped to
about 50 mm so the parts stay meshable, while the flight vehicle is 335 mm, and
the packet says so plainly -- but it means the flight parts have never been
analysed at all.

The clamp is not laziness, it is arithmetic. Resolving a 0.8 mm wall with solid
tetrahedra needs elements around a quarter of a millimetre, and a 4.37 m tank
at 335 mm radius then takes something like two billion of them against a budget
of forty thousand. No amount of patience closes a gap of fifty thousand times.

Solid elements are simply the wrong tool for a thin wall. A shell element
carries its thickness as a property rather than meshing through it, so the same
tank becomes a few thousand elements on its mid-surface and solves in seconds.
That is what launch vehicle structures are actually analysed with.

The mesh here is generated directly rather than through gmsh: a barrel is a
structured grid in two parameters, and building it explicitly avoids the
surface classification that makes STL-derived meshing fragile. It also means
the element quality is known rather than hoped for.

Verification is exact and cheap. A cylinder under axial load carries
N/(2 pi r t) everywhere, and under internal pressure the hoop stress is p r / t.
Both are closed form, so the solver can be checked against truth rather than
against a finer version of itself.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: CalculiX's four-node shell. S4 rather than S4R: reduced integration is
#: cheaper and admits hourglass modes, and on a coarse barrel mesh those show
#: up as a stress field that looks plausible and is wrong.
SHELL_TYPE = "S4"

DEFAULT_CCX = Path.home() / ".local" / "bin" / "ccx"


@dataclass
class ShellResult:
    nodes: int
    elements: int
    max_von_mises_mpa: float
    mean_von_mises_mpa: float
    analytic_mpa: float | None
    error_pct: float | None
    converged: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "elements": self.elements,
            "max_von_mises_mpa": round(self.max_von_mises_mpa, 3),
            "mean_von_mises_mpa": round(self.mean_von_mises_mpa, 3),
            "analytic_mpa": (round(self.analytic_mpa, 3)
                             if self.analytic_mpa is not None else None),
            "error_pct": (round(self.error_pct, 3)
                          if self.error_pct is not None else None),
            "converged": self.converged,
            "notes": list(self.notes),
        }


def cylinder_shell_mesh(radius_m: float, length_m: float, *,
                        n_theta: int = 32, n_axial: int = 24):
    """Structured quad mesh of a cylindrical mid-surface.

    Returns (nodes, elements) with 1-based ids. The seam closes on itself
    rather than duplicating a column of nodes -- a barrel that is not actually
    connected round its circumference carries no hoop load, and would report a
    pressure vessel as a flat plate rolled up.
    """
    if radius_m <= 0 or length_m <= 0:
        raise ValueError("radius and length must be positive")
    if n_theta < 8 or n_axial < 2:
        raise ValueError("mesh is too coarse to represent a barrel")

    nodes: dict[int, tuple[float, float, float]] = {}
    nid = 1
    grid = []
    for i in range(n_axial + 1):
        z = length_m * i / n_axial
        row = []
        for j in range(n_theta):
            th = 2.0 * math.pi * j / n_theta
            nodes[nid] = (radius_m * math.cos(th), radius_m * math.sin(th), z)
            row.append(nid)
            nid += 1
        grid.append(row)

    elements = []
    eid = 1
    for i in range(n_axial):
        for j in range(n_theta):
            k = (j + 1) % n_theta          # wraps, closing the seam
            elements.append((eid, [grid[i][j], grid[i][k],
                                   grid[i + 1][k], grid[i + 1][j]]))
            eid += 1
    return nodes, elements


def write_deck(case: Path, nodes, elements, *, thickness_m: float,
               youngs_pa: float, poisson: float, axial_n: float = 0.0,
               pressure_pa: float = 0.0, length_m: float = 1.0) -> None:
    """CalculiX deck for the barrel under axial load and/or internal pressure."""
    case.mkdir(parents=True, exist_ok=True)
    L = ["*NODE, NSET=NALL"]
    for n in sorted(nodes):
        x, y, z = nodes[n]
        L.append(f"{n}, {x:.9e}, {y:.9e}, {z:.9e}")
    L.append(f"*ELEMENT, TYPE={SHELL_TYPE}, ELSET=EALL")
    for eid, ns in elements:
        L.append(f"{eid}, " + ", ".join(str(v) for v in ns))

    base = sorted(n for n, p in nodes.items() if abs(p[2]) < 1e-12)
    top = sorted(n for n, p in nodes.items() if abs(p[2] - length_m) < 1e-9)
    if not base or not top:
        raise ValueError("could not identify the end rings of the barrel")

    L.append("*NSET, NSET=BASE")
    for i in range(0, len(base), 8):
        L.append(", ".join(str(v) for v in base[i:i + 8]) + ",")
    L.append("*NSET, NSET=TOP")
    for i in range(0, len(top), 8):
        L.append(", ".join(str(v) for v in top[i:i + 8]) + ",")

    L += [
        "*MATERIAL, NAME=SKIN",
        "*ELASTIC",
        f"{youngs_pa:.6e}, {poisson}",
        f"*SHELL SECTION, ELSET=EALL, MATERIAL=SKIN",
        f"{thickness_m:.9e}",
        "*STEP",
        "*STATIC",
        # Fixed at the base, axially loaded at the top. The base is fully
        # clamped, which is what a thrust structure does to a tank skirt.
        "*BOUNDARY",
        "BASE, 1, 6",
    ]
    if axial_n:
        per = -abs(float(axial_n)) / len(top)     # compression, along -z
        L.append("*CLOAD")
        for n in top:
            L.append(f"{n}, 3, {per:.6e}")
    if pressure_pa:
        L.append("*DLOAD")
        for eid, _ in elements:
            L.append(f"{eid}, P, {-abs(float(pressure_pa)):.6e}")
    L += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP", ""]
    (case / "job.inp").write_text("\n".join(L))


def _read_stress(frd: Path) -> list[float]:
    """Von Mises per node from the FRD stress block, fixed-width."""
    out, inside = [], False
    for line in frd.read_text(errors="ignore").splitlines():
        if line.startswith(" -4  STRESS"):
            inside = True
            continue
        if inside and line.startswith(" -3"):
            break
        if inside and line.startswith(" -1"):
            v = [float(line[13 + 12 * i:25 + 12 * i]) for i in range(6)]
            sxx, syy, szz, sxy, syz, szx = v
            out.append(math.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)) / 1e6)
    return out


def analyse_barrel(case_dir: Path, *, radius_m: float, length_m: float,
                   thickness_m: float, youngs_pa: float, poisson: float = 0.33,
                   axial_n: float = 0.0, pressure_pa: float = 0.0,
                   n_theta: int = 32, n_axial: int = 24,
                   timeout_s: int = 600) -> ShellResult:
    """Solve the flight-scale barrel and score it against closed form."""
    nodes, elements = cylinder_shell_mesh(radius_m, length_m,
                                          n_theta=n_theta, n_axial=n_axial)
    write_deck(Path(case_dir), nodes, elements, thickness_m=thickness_m,
               youngs_pa=youngs_pa, poisson=poisson, axial_n=axial_n,
               pressure_pa=pressure_pa, length_m=length_m)
    try:
        subprocess.run([str(DEFAULT_CCX), "-i", "job"], cwd=case_dir,
                       timeout=timeout_s, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return ShellResult(len(nodes), len(elements), 0.0, 0.0, None, None,
                           False, ("solver timed out",))
    frd = Path(case_dir) / "job.frd"
    if not frd.exists() or frd.stat().st_size < 2000:
        return ShellResult(len(nodes), len(elements), 0.0, 0.0, None, None,
                           False, ("no usable FRD",))

    vm = _read_stress(frd)
    if not vm:
        return ShellResult(len(nodes), len(elements), 0.0, 0.0, None, None,
                           False, ("FRD carried no stress block",))

    # Closed form for whichever load case was applied alone. Combined loading
    # has no one-line answer, so no analytic column is offered rather than an
    # approximate one presented as exact.
    analytic = None
    notes: list[str] = []
    if axial_n and not pressure_pa:
        analytic = abs(axial_n) / (2.0 * math.pi * radius_m * thickness_m) / 1e6
    elif pressure_pa and not axial_n:
        # Hoop only, because this barrel is open-ended.
        #
        # The closed-end formula adds an axial pr/2t and gives sqrt(3)/2 times
        # the hoop stress -- and using it here produced an 8.4% error that
        # looked like solver inaccuracy. It was not: this model has no end
        # caps, so pressure develops hoop stress and the base clamp reacts the
        # rest. Comparing a closed-end reference against an open-ended model
        # shows a fixed offset that no refinement removes, which is exactly the
        # trap the Lame verification case was written to avoid and which this
        # module walked into anyway.
        analytic = abs(pressure_pa) * radius_m / thickness_m / 1e6
    else:
        notes.append(
            "Combined axial and pressure loading has no single closed form, so "
            "no analytic comparison is offered for this case.")

    # The clamped base is a stress concentration the closed form does not have,
    # so the mean over the barrel is the fair comparison and the peak belongs
    # to the boundary layer at the fixed end.
    # Trim both tails, not the top one. Under axial load the clamped base
    # raises stress; under pressure it lowers it, by restraining the radial
    # growth that produces the hoop stress in the first place. A one-sided trim
    # therefore keeps the wrong outliers for one of the two load cases.
    ordered = sorted(vm)
    lo = int(0.05 * len(ordered))
    hi = max(lo + 1, int(0.95 * len(ordered)))
    interior = ordered[lo:hi] or vm
    mean = sum(interior) / len(interior)
    err = (100.0 * (mean - analytic) / analytic) if analytic else None
    if analytic:
        notes.append(
            "Compared on the middle 90% of nodes: the clamped base introduces "
            "a bending boundary layer the membrane solution does not model, "
            "and it perturbs the stress in opposite directions under axial "
            "load and under pressure.")
    return ShellResult(len(nodes), len(elements), max(vm), mean, analytic, err,
                       True, tuple(notes))
