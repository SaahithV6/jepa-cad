"""Compressible CFD verification of the analytic nozzle.

The nozzle sizing in this project is the ideal-rocket isentropic relations and
nothing has ever checked them against a flow solve. The existing CFD routes
cannot: they run simpleFoam, which is incompressible, and a nozzle is nothing
but compressibility. rhoCentralFoam is a density-based solver built for exactly
this, and it is installed.

The case is a planar converging-diverging nozzle, inviscid, with a symmetry
plane on the centreline. Planar rather than axisymmetric because the quantity
being checked -- exit Mach as a function of area ratio -- comes from the
quasi-1D area-Mach relation, which holds for area however the area is made, and
a planar duct needs no wedge patches to get right. Inviscid with slip walls
because that is precisely the ideal the analytic relation assumes: any
difference is then the numerics or the multidimensionality, not a boundary
layer that the analytic model never claimed to include.

The contour is cosine-blended into the throat from both sides, so dh/dx is zero
there and the geometry introduces no kink for the solver to make a shock out of.

For a choked nozzle running full supersonic, exit Mach depends only on the area
ratio -- not on chamber pressure -- so the check is independent of how well the
inlet stagnation condition is imposed. The outlet is fully supersonic and every
variable is extrapolated, which is the correct characteristic treatment.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GAMMA = 1.4
R_SPECIFIC = 287.0


def area_ratio_for_mach(mach: float, gamma: float = GAMMA) -> float:
    """A/A* from the isentropic area-Mach relation."""
    m = float(mach)
    g = float(gamma)
    return (1.0 / m) * (
        (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * m * m)
    ) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def supersonic_mach_for_area_ratio(ratio: float, gamma: float = GAMMA) -> float:
    """Invert the area-Mach relation on the supersonic branch by bisection."""
    target = float(ratio)
    if target < 1.0:
        raise ValueError(f"area ratio must be >= 1, got {target}")
    lo, hi = 1.0 + 1e-9, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio_for_mach(mid, gamma) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def diverging_contour(
    throat_half_height: float,
    inlet_ratio: float,
    area_ratio: float,
    length: float = 20.0,
    points: int = 120,
) -> list[tuple[float, float]]:
    """Diverging section only, from A/A* = inlet_ratio out to area_ratio.

    Cosine-blended so dh/dx is zero at the inlet, which keeps the imposed inlet
    state from meeting a corner.
    """
    h_t = float(throat_half_height)
    h_in = h_t * float(inlet_ratio)
    h_e = h_t * float(area_ratio)
    ell = h_t * float(length)
    n = max(8, int(points))
    return [
        (ell * i / n,
         h_in + (h_e - h_in) * 0.5 * (1.0 - math.cos(math.pi * i / n)))
        for i in range(n + 1)
    ]


def isentropic_state(mach: float, p0: float, t0: float,
                     gamma: float = GAMMA) -> tuple[float, float, float]:
    """(p, T, U) at a station, from stagnation conditions."""
    m = float(mach)
    ratio = 1.0 + 0.5 * (gamma - 1.0) * m * m
    t = float(t0) / ratio
    p = float(p0) * ratio ** (-gamma / (gamma - 1.0))
    u = m * math.sqrt(gamma * R_SPECIFIC * t)
    return p, t, u


def nozzle_contour(
    throat_half_height: float,
    area_ratio: float,
    inlet_ratio: float = 6.0,
    converging_length: float = 3.0,
    diverging_length: float = 6.0,
    points: int = 120,
) -> list[tuple[float, float]]:
    """Half-height h(x) of a planar CD nozzle, cosine-blended at the throat.

    Lengths are in units of the throat half-height. The inlet is made generous
    so the inlet Mach is small and a static inlet condition is close to
    stagnation -- which does not affect exit Mach for a choked nozzle, but does
    keep the inlet from being a hard boundary-condition problem.
    """
    h_t = float(throat_half_height)
    h_in = h_t * float(inlet_ratio)
    h_e = h_t * float(area_ratio)
    lc = h_t * float(converging_length)
    ld = h_t * float(diverging_length)

    pts: list[tuple[float, float]] = []
    n_c = max(4, points // 3)
    n_d = max(4, points - n_c)
    for i in range(n_c + 1):
        x = lc * i / n_c
        h = h_t + (h_in - h_t) * 0.5 * (1.0 + math.cos(math.pi * i / n_c))
        pts.append((x, h))
    for i in range(1, n_d + 1):
        x = lc + ld * i / n_d
        h = h_t + (h_e - h_t) * 0.5 * (1.0 - math.cos(math.pi * i / n_d))
        pts.append((x, h))
    return pts


_HEADER = (
    "FoamFile\n{{\n    version 2.0;\n    format ascii;\n"
    "    class {cls};\n    object {obj};\n}}\n"
)


def _dict_file(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER.format(cls=cls, obj=obj) + body + "\n", encoding="utf-8")


def _box_block(x0: float, x1: float, height: float, thickness: float,
               nx: int, ny: int) -> str:
    """A plain orthogonal box; warp_points_to_contour bends it to the wall."""
    return f"""
scale 1;

vertices
(
    ({x0:.8g} 0 0)
    ({x1:.8g} 0 0)
    ({x1:.8g} {height:.8g} 0)
    ({x0:.8g} {height:.8g} 0)
    ({x0:.8g} 0 {thickness:.8g})
    ({x1:.8g} 0 {thickness:.8g})
    ({x1:.8g} {height:.8g} {thickness:.8g})
    ({x0:.8g} {height:.8g} {thickness:.8g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5)); }}
    topWall  {{ type wall; faces ((3 7 6 2)); }}
    symmetry {{ type symmetryPlane; faces ((0 1 5 4)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
"""


def _write_common_system(case: Path) -> None:
    _dict_file(
        case / "system" / "fvSchemes", "dictionary", "fvSchemes",
        """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(tauMC) Gauss linear; }
interpolationSchemes
{
    default         linear;
    reconstruct(rho) vanLeer;
    reconstruct(U)   vanLeerV;
    reconstruct(T)   vanLeer;
}
snGradSchemes   { default corrected; }
laplacianSchemes { default Gauss linear corrected; }
""",
    )
    _dict_file(
        case / "system" / "fvSolution", "dictionary", "fvSolution",
        """
solvers
{
    "(rho|rhoU|rhoE)" { solver diagonal; }
    U { solver smoothSolver; smoother GaussSeidel; tolerance 1e-09; relTol 0; }
    e { solver smoothSolver; smoother GaussSeidel; tolerance 1e-10; relTol 0; }
}
""",
    )


def _write_thermo(case: Path) -> None:
    """Inviscid perfect gas -- exactly the ideal the analytic relation assumes."""
    cv = R_SPECIFIC / (GAMMA - 1.0)
    mol_weight = 8314.0 / R_SPECIFIC
    _dict_file(
        case / "constant" / "thermophysicalProperties", "dictionary",
        "thermophysicalProperties",
        f"""
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          eConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie      {{ molWeight {mol_weight:.6f}; }}
    thermodynamics {{ Cv {cv:.6f}; Hf 0; }}
    transport   {{ mu 0; Pr 1; }}
}}
""",
    )
    _dict_file(
        case / "constant" / "turbulenceProperties", "dictionary",
        "turbulenceProperties", "simulationType laminar;",
    )


@dataclass(frozen=True)
class NozzleCase:
    case_dir: Path
    area_ratio: float
    throat_half_height: float
    contour: tuple[tuple[float, float], ...]
    nx: int = 0
    ny: int = 0


def write_nozzle_case(
    case_dir: Path | str,
    area_ratio: float = 4.0,
    throat_half_height: float = 0.01,
    chamber_pressure_pa: float = 2.0e6,
    chamber_temp_k: float = 3000.0,
    nx: int = 260,
    ny: int = 40,
    end_time: float = 3.0e-3,
    write_interval: float = 3.0e-3,
    initial_pressure_fraction: float = 0.02,
    ramp_time: float = 2.0e-4,
) -> NozzleCase:
    """Write a complete rhoCentralFoam case for a planar CD nozzle.

    The interior starts at a fraction of chamber pressure, isentropically cold
    to match, and the inlet is ramped from that state up to chamber conditions
    over ``ramp_time`` -- a valve opening rather than a diaphragm bursting.

    Both details were forced by failures. Initialising the interior at chamber
    pressure, with a fixed-value inlet at that same pressure and an
    extrapolating outlet, leaves no pressure gradient anywhere: the solver ran
    to completion having moved nothing, exit Mach 1.8e-14 after 152 s of
    compute. Dropping the interior to 2% of chamber without the ramp puts a 50x
    pressure jump across the inlet face, and with velocity extrapolated there
    the momentum flux is ill-posed; that died in sqrt on a negative temperature.

    The outlet is ``waveTransmissive`` rather than plain ``zeroGradient``.
    Extrapolating everything is the right characteristic treatment once the exit
    is supersonic, but during startup it is still subsonic, and then nothing
    anchors the pressure at all: it drifts down, takes the temperature negative
    with it, and the solver dies in sqrt. Lowering the Courant number or
    switching to a more diffusive limiter only moved when that happened, which
    is how it was clear the problem was not the timestep.

    The ramp presupposes nothing about the answer -- the interior is a uniform
    low-pressure state with no Mach structure in it -- so converging to the
    isentropic exit Mach remains a real result.
    """
    case = Path(case_dir)
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    (case / "system").mkdir(parents=True)

    contour = nozzle_contour(throat_half_height, area_ratio)
    x0, x1 = contour[0][0], contour[-1][0]
    thickness = throat_half_height

    # A plain rectangle, meshed orthogonally, then warped onto the contour by
    # scaling each point's y with h(x). The obvious approach -- one block with
    # the top wall as a polyLine edge -- does not work: blockMesh distributes
    # points along the curved top edge by arc length and along the straight
    # bottom edge by x, so the two disagree about where each column sits and the
    # cells shear. checkMesh on that mesh reported 140 negative-volume cells,
    # max skewness 262 and 97 degree non-orthogonality, and rhoCentralFoam died
    # in sqrt after nine steps with a max Courant number 1,950x its mean --
    # one tangled cell setting the timestep for the whole domain.
    height = max(h for _, h in contour)
    block = f"""
scale 1;

vertices
(
    ({x0:.8g} 0 0)
    ({x1:.8g} 0 0)
    ({x1:.8g} {height:.8g} 0)
    ({x0:.8g} {height:.8g} 0)
    ({x0:.8g} 0 {thickness:.8g})
    ({x1:.8g} 0 {thickness:.8g})
    ({x1:.8g} {height:.8g} {thickness:.8g})
    ({x0:.8g} {height:.8g} {thickness:.8g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    topWall
    {{
        type wall;
        faces ((3 7 6 2));
    }}
    symmetry
    {{
        type symmetryPlane;
        faces ((0 1 5 4));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);
"""
    _dict_file(case / "system" / "blockMeshDict", "dictionary", "blockMeshDict", block)

    _dict_file(
        case / "system" / "controlDict", "dictionary", "controlDict",
        f"""
application     rhoCentralFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:g};
deltaT          1e-9;
writeControl    adjustableRunTime;
writeInterval   {write_interval:g};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           0.3;
maxDeltaT       1e-6;
""",
    )

    _dict_file(
        case / "system" / "fvSchemes", "dictionary", "fvSchemes",
        """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(tauMC) Gauss linear; }
interpolationSchemes
{
    default         linear;
    reconstruct(rho) vanLeer;
    reconstruct(U)   vanLeerV;
    reconstruct(T)   vanLeer;
}
snGradSchemes   { default corrected; }
laplacianSchemes { default Gauss linear corrected; }
""",
    )

    _dict_file(
        case / "system" / "fvSolution", "dictionary", "fvSolution",
        """
solvers
{
    "(rho|rhoU|rhoE)"
    {
        solver          diagonal;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-09;
        relTol          0;
    }
    e
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-10;
        relTol          0;
    }
}
""",
    )

    # Inviscid: mu = 0 makes the solve the exact ideal the analytic relation
    # assumes, so any discrepancy is numerical rather than a boundary layer the
    # analytic model never claimed to resolve.
    cv = R_SPECIFIC / (GAMMA - 1.0)
    mol_weight = 8314.0 / R_SPECIFIC
    _dict_file(
        case / "constant" / "thermophysicalProperties", "dictionary",
        "thermophysicalProperties",
        f"""
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          eConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie      {{ molWeight {mol_weight:.6f}; }}
    thermodynamics {{ Cv {cv:.6f}; Hf 0; }}
    transport   {{ mu 0; Pr 1; }}
}}
""",
    )

    _dict_file(
        case / "constant" / "turbulenceProperties", "dictionary",
        "turbulenceProperties", "simulationType laminar;",
    )

    # Supersonic outlet: every characteristic leaves the domain, so everything
    # is extrapolated. That is why this check does not depend on knowing the
    # exit pressure.
    p_init = float(chamber_pressure_pa) * float(initial_pressure_fraction)
    t_init = float(chamber_temp_k) * (
        float(initial_pressure_fraction) ** ((GAMMA - 1.0) / GAMMA))
    _dict_file(
        case / "0" / "p", "volScalarField", "p",
        f"""
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {p_init:g};
boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue    table ((0 {p_init:g}) ({ramp_time:g} {chamber_pressure_pa:g}));
    }}
    outlet
    {{
        type            waveTransmissive;
        field           p;
        psi             thermo:psi;
        gamma           {GAMMA};
        fieldInf        {p_init:g};
        lInf            {(x1 - x0):.8g};
        value           uniform {p_init:g};
    }}
    topWall     {{ type zeroGradient; }}
    symmetry    {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _dict_file(
        case / "0" / "T", "volScalarField", "T",
        f"""
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {t_init:g};
boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue    table ((0 {t_init:g}) ({ramp_time:g} {chamber_temp_k:g}));
    }}
    outlet      {{ type zeroGradient; }}
    topWall     {{ type zeroGradient; }}
    symmetry    {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _dict_file(
        case / "0" / "U", "volVectorField", "U",
        """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type zeroGradient; }
    topWall     { type slip; }
    symmetry    { type symmetryPlane; }
    frontAndBack { type empty; }
}
""",
    )

    return NozzleCase(
        case_dir=case,
        area_ratio=float(area_ratio),
        throat_half_height=float(throat_half_height),
        contour=tuple(contour),
        nx=int(nx),
        ny=int(ny),
    )


def warp_points_to_contour(case_dir: Path | str,
                           contour: list[tuple[float, float]]) -> int:
    """Scale each mesh point's y by h(x)/H so the box becomes the nozzle.

    Because every column keeps its x and every point keeps its relative height,
    the result is as orthogonal as the box it came from: no negative volumes, no
    shear, and the wall lands exactly on the contour.
    """
    points = Path(case_dir) / "constant" / "polyMesh" / "points"
    text = points.read_text()
    start = text.index("(", text.index("\n(", text.index("FoamFile")))
    head, body = text[:start + 1], text[start + 1:]
    end = body.rindex(")")
    tail = body[end:]
    xs = [c[0] for c in contour]
    hs = [c[1] for c in contour]
    height = max(hs)

    out, moved = [], 0
    for line in body[:end].splitlines():
        stripped = line.strip()
        if not (stripped.startswith("(") and stripped.endswith(")")):
            out.append(line)
            continue
        x, y, z = (float(v) for v in stripped[1:-1].split())
        if y > 0.0:
            # piecewise-linear interpolation of the contour at this station
            h = hs[0]
            for i in range(len(xs) - 1):
                if xs[i] <= x <= xs[i + 1]:
                    t = 0.0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
                    h = hs[i] + t * (hs[i + 1] - hs[i])
                    break
            else:
                h = hs[-1] if x > xs[-1] else hs[0]
            y = y * h / height
            moved += 1
        out.append(f"({x:.10g} {y:.10g} {z:.10g})")
    points.write_text(head + "\n".join(out) + tail)
    return moved


def run_nozzle_case(case_dir: Path | str, timeout_s: int = 3600,
                    contour: list[tuple[float, float]] | None = None) -> dict:
    """blockMesh, warp the box onto the contour, then rhoCentralFoam."""
    case = Path(case_dir)
    out = {}
    for name in ("blockMesh", "rhoCentralFoam"):
        if name == "rhoCentralFoam" and contour is not None:
            warp_points_to_contour(case, contour)
        proc = subprocess.run(
            [name, "-case", str(case)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        (case / f"{name}.log").write_text(
            (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
        )
        out[name] = proc.returncode
        if proc.returncode != 0:
            break
    return out


def exit_mach_from_case(case_dir: Path | str, nx: int = 0) -> float | None:
    """Area-averaged exit Mach from the last written time directory.

    Reads the field files directly rather than shelling out to a post-processing
    utility, so this works the same whether or not the OpenFOAM tools are on the
    path.

    ``nx`` selects the exit column properly. blockMesh numbers cells with i
    varying fastest, so simply averaging the last few percent of the list takes
    the top row of cells along the whole wall rather than the exit plane -- the
    wrong plane entirely, and one that happens to look plausible.
    """
    case = Path(case_dir)
    times = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            t = float(child.name)
        except ValueError:
            continue
        if t > 0.0 and (child / "U").exists():
            times.append((t, child))
    if not times:
        return None
    _, latest = max(times, key=lambda pair: pair[0])

    u = _internal_field_mean_vector_mag(latest / "U", nx)
    temp = _internal_field_mean_scalar(latest / "T", nx)
    if u is None or temp is None or temp <= 0.0:
        return None
    return abs(u) / math.sqrt(GAMMA * R_SPECIFIC * temp)


def _tail_values(path: Path) -> list[str]:
    """The last internalField block's values, as raw tokens."""
    text = path.read_text(errors="ignore")
    key = "internalField"
    idx = text.find(key)
    if idx < 0:
        return []
    start = text.find("(", idx)
    if start < 0:
        return []
    depth, i = 0, start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start + 1 : i].split()


def _exit_slice(vals: list, nx: int) -> list:
    """Cells in the last axial station, given i varies fastest."""
    if nx and nx > 1 and len(vals) >= nx:
        return [v for idx, v in enumerate(vals) if idx % nx == nx - 1]
    return vals[-max(1, len(vals) // 50):]


def _internal_field_mean_scalar(path: Path, nx: int = 0) -> float | None:
    toks = _tail_values(path)
    vals = []
    for tok in toks:
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    if not vals:
        return None
    sel = _exit_slice(vals, nx)
    return sum(sel) / len(sel)


def _internal_field_mean_vector_mag(path: Path, nx: int = 0) -> float | None:
    """Mean |U|, not mean Ux.

    In a diverging nozzle the flow has a real radial component, so the axial
    component understates the Mach number -- the quantity the area-Mach relation
    predicts is the speed, not its projection on the axis.
    """
    toks = _tail_values(path)
    xs = []
    buf = []
    for tok in toks:
        tok = tok.strip("()")
        if not tok:
            continue
        try:
            buf.append(float(tok))
        except ValueError:
            continue
        if len(buf) == 3:
            xs.append(math.sqrt(buf[0] ** 2 + buf[1] ** 2 + buf[2] ** 2))
            buf = []
    if not xs:
        return None
    sel = _exit_slice(xs, nx)
    return sum(sel) / len(sel)


def write_supersonic_expansion_case(
    case_dir: Path | str,
    inlet_area_ratio: float = 1.2,
    exit_area_ratio: float = 4.0,
    throat_half_height: float = 0.01,
    chamber_pressure_pa: float = 2.0e6,
    chamber_temp_k: float = 3000.0,
    nx: int = 240,
    ny: int = 40,
    end_time: float = 1.0e-3,
    length: float = 20.0,
) -> NozzleCase:
    """A well-posed supersonic expansion, for verifying the area-Mach relation.

    Why this and not a full converging-diverging nozzle with a subsonic inlet:
    that problem is genuinely hard to pose for a density-based solver. Fixing p
    and T at a subsonic inlet while extrapolating U leaves the momentum flux
    there under-determined, and during the choking transient it pumps energy in
    -- a diagnostic dump caught the interior at 26,347 K and 4.9 MPa against a
    3,000 K, 2 MPa chamber, i.e. hotter and higher-pressure than the reservoir
    feeding it, which is thermodynamically impossible and unmistakably the
    boundary rather than the mesh. Lowering the Courant number, switching
    limiters and adding a wave-transmissive outlet each only moved the moment it
    blew up.

    A supersonic inlet has every characteristic entering the domain, so fixing
    p, T and U there is the correct and complete specification; a supersonic
    outlet has every characteristic leaving, so extrapolating everything is
    correct. Both boundaries are then exactly determined and there is no
    subsonic pocket anywhere.

    What it tests is unchanged in substance: the solver is handed a state at
    A/A* = 1.2 and must expand it to A/A* = 4.0, and the exit Mach it produces
    is compared against the isentropic area-Mach relation. Only the inlet state
    is given; the expansion is the solver's own.
    """
    case = Path(case_dir)
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    (case / "system").mkdir(parents=True)

    contour = diverging_contour(throat_half_height, inlet_area_ratio,
                                exit_area_ratio, length=length)
    x0, x1 = contour[0][0], contour[-1][0]
    thickness = throat_half_height
    height = max(h for _, h in contour)

    mach_in = supersonic_mach_for_area_ratio(inlet_area_ratio)
    p_in, t_in, u_in = isentropic_state(mach_in, chamber_pressure_pa,
                                        chamber_temp_k)

    _dict_file(case / "system" / "blockMeshDict", "dictionary", "blockMeshDict",
               _box_block(x0, x1, height, thickness, nx, ny))
    _dict_file(
        case / "system" / "controlDict", "dictionary", "controlDict",
        f"""
application     rhoCentralFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:g};
deltaT          1e-9;
writeControl    adjustableRunTime;
writeInterval   {end_time:g};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           0.2;
maxDeltaT       1e-6;
""",
    )
    _write_common_system(case)
    _write_thermo(case)

    _dict_file(
        case / "0" / "p", "volScalarField", "p",
        f"""
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {p_in:g};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {p_in:g}; }}
    outlet      {{ type zeroGradient; }}
    topWall     {{ type zeroGradient; }}
    symmetry    {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _dict_file(
        case / "0" / "T", "volScalarField", "T",
        f"""
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {t_in:g};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {t_in:g}; }}
    outlet      {{ type zeroGradient; }}
    topWall     {{ type zeroGradient; }}
    symmetry    {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    _dict_file(
        case / "0" / "U", "volVectorField", "U",
        f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u_in:g} 0 0);
boundaryField
{{
    inlet       {{ type fixedValue; value uniform ({u_in:g} 0 0); }}
    outlet      {{ type zeroGradient; }}
    topWall     {{ type slip; }}
    symmetry    {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
""",
    )
    return NozzleCase(
        case_dir=case,
        area_ratio=float(exit_area_ratio),
        throat_half_height=float(throat_half_height),
        contour=tuple(contour),
        nx=int(nx),
        ny=int(ny),
    )
