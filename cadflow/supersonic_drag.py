"""Supersonic drag on bodies of revolution, by compressible CFD.

A search across six domains for an open dataset pairing rocket geometry with
drag coefficient over a Mach sweep found nothing. Not scarce -- absent. Every
aerodynamic dataset with real labels is subsonic (AirfRANS, DrivAerML), and
every supersonic archive with geometry (sonic-boom workshops, AGARD) publishes
pressure signatures rather than integrated Cd, in tens of cases rather than
thousands. So the corpus this project needs for its aerodynamic target has to be
computed.

That is affordable because a rocket is a body of revolution: an axisymmetric
wedge solves a 2D problem and recovers the 3D answer, which is two orders of
magnitude cheaper than meshing the whole body. The existing external-aero route
here runs simpleFoam and cannot be used -- it is incompressible, and above Mach 1
compressibility is the entire phenomenon.

The mesh is built as an orthogonal block and then warped onto the body contour,
the same trick the nozzle case uses: handing blockMesh a curved edge makes it
distribute points by arc length on the curve and uniformly on the straight
opposite edge, the two disagree about where each column belongs, and the cells
shear. That produced 140 negative-volume cells and skewness 262 the first time
it was tried on the nozzle.

Validation is against theory that does not come from a solver: Karman slender-
body wave drag, which this project already implements and which converged to
1.395282 against the Sears-Haack minimum. A CFD Cd that disagrees with it by
more than the viscous contribution is wrong, and the comparison is checkable
without trusting either.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Sea-level standard, used unless a case names something else.
P_INF_PA = 101325.0
T_INF_K = 288.15
GAMMA = 1.4
R_AIR = 287.05

#: Validity: reliable to about Mach 4; Mach 5 fails for some geometries.
#:
#: Six of 24 high-Mach cases died with a floating point exception, all at Mach 5
#: except one cone at Mach 4. The mechanism was measured, not guessed:
#:
#:   maxCo 0.30    176 timesteps -> SIGFPE at t = 7.80e-05 s
#:   maxCo 0.05   1000 timesteps -> SIGFPE at t = 8.00e-05 s
#:
#: Six times the timesteps and the same physical failure time, with the Courant
#: number sitting healthily at its limit and no bounding warnings. So it is not
#: a stability problem that a smaller step fixes -- something goes negative
#: during initial shock formation, where rhoCentralFoam starts from a uniform
#: freestream and the discontinuity at the wall is infinitely sharp.
#:
#: An earlier note here blamed detached bow shocks on blunt tips. That was
#: wrong: a 9.5 degree cone at Mach 5 has a firmly attached shock and failed
#: identically. The guess is recorded as withdrawn rather than deleted, because
#: the next person to see these failures will reach for the same explanation.
#:
#: Fixing it properly needs a smoothed initial condition or explicit temperature
#: and density floors, neither of which is worth doing before someone needs Mach
#: 5 data.

#: Wedge half-angle for the axisymmetric slice. Five degrees is the OpenFOAM
#: convention: small enough that the sector is effectively 2D, large enough that
#: the front and back patches do not collapse numerically.
WEDGE_DEG = 2.5


@dataclass
class BodySpec:
    """A body of revolution, in metres."""
    name: str
    nose_length_m: float
    body_length_m: float
    radius_m: float
    nose_shape: str = "ogive"          # ogive | cone | vonkarman
    boattail_length_m: float = 0.0
    boattail_radius_m: float = 0.0
    #: Tip bluntness as a fraction of body radius. A mathematically sharp tip
    #: is a singular point of the axisymmetric mesh -- the cell there is
    #: degenerate in the wedge direction AND in the body direction at once, and
    #: rhoCentralFoam blew up after 388 steps with a healthy Courant number of
    #: 0.3. Real nose cones are blunted for the same reason at a larger scale:
    #: a perfectly sharp tip melts.
    tip_bluntness: float = 0.02

    @property
    def total_length_m(self) -> float:
        return self.nose_length_m + self.body_length_m + self.boattail_length_m

    @property
    def fineness(self) -> float:
        return self.total_length_m / (2.0 * self.radius_m)

    @property
    def frontal_area_m2(self) -> float:
        return math.pi * self.radius_m ** 2

    def contour(self, points: int = 120) -> list[tuple[float, float]]:
        """(x, r) along the body, nose tip at x=0."""
        out: list[tuple[float, float]] = []
        r, ln = self.radius_m, self.nose_length_m
        n_nose = max(20, points // 2)
        r_tip = max(1e-5, self.tip_bluntness * r)
        for i in range(n_nose + 1):
            f = i / n_nose
            x = ln * f
            if self.nose_shape == "cone":
                rr = r * f
            elif self.nose_shape == "vonkarman":
                theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * f)))
                rr = r / math.sqrt(math.pi) * math.sqrt(
                    theta - math.sin(2.0 * theta) / 2.0)
            else:                                   # tangent ogive
                rho = (r * r + ln * ln) / (2.0 * r)
                rr = math.sqrt(max(0.0, rho * rho - (ln - x) ** 2)) + r - rho
            out.append((x, max(r_tip, rr)))
        out.append((ln + self.body_length_m, r))
        if self.boattail_length_m > 0.0:
            out.append((self.total_length_m, max(1e-4, self.boattail_radius_m)))
        return out


@dataclass
class DragResult:
    body: str
    mach: float
    cd: float | None = None
    cd_pressure: float | None = None
    cd_viscous: float | None = None
    converged: bool = False
    error: str | None = None
    meta: dict = field(default_factory=dict)


def _f(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {cls};\n    object {obj};\n}}\n\n{body}\n")


def write_case(case: Path, spec: BodySpec, mach: float, *,
               nx_body: int = 160, nr: int = 80, end_time: float | None = None,
               radial_grading: float = 4.0,
               p_inf: float = P_INF_PA, t_inf: float = T_INF_K) -> dict:
    """Write a complete axisymmetric supersonic external-flow case.

    Domain: half a body length upstream of the tip, two lengths downstream, and
    six body radii outward. The upstream margin matters -- a bow shock that
    reaches the inlet reflects off it and the drag comes out wrong -- and six
    radii is enough at these Mach numbers for the outer boundary to sit outside
    the shock layer.
    """
    case.mkdir(parents=True, exist_ok=True)
    for sub in ("system", "constant", "0"):
        (case / sub).mkdir(exist_ok=True)

    contour = spec.contour()
    length = spec.total_length_m
    r_max = spec.radius_m
    x0 = 0.0
    # Far enough out that the bow shock leaves through the outlet rather than
    # striking this boundary. Six radii was not: at Mach 2 the shock off a
    # slender body reaches r = 0.47 m by the time the flow has travelled 0.8 m,
    # and with a fixedValue farfield at 0.45 m it reflected and killed the run
    # at t = 0.00122 s every time, at a healthy Courant number, with no
    # divergence beforehand.
    r_far = max(20.0 * r_max, 0.6 * length * math.tan(
        math.asin(min(0.99, 1.0 / max(mach, 1.01)))))

    a_inf = math.sqrt(GAMMA * R_AIR * t_inf)
    u_inf = mach * a_inf
    rho_inf = p_inf / (R_AIR * t_inf)

    # Time to sweep the domain several times over, so the transient is gone.
    if end_time is None:
        end_time = 4.0 * (2.5 * length) / u_inf

    half = math.radians(WEDGE_DEG)
    # y is the near-radial direction, z the thin one. Swapping these makes the
    # two wedge planes 175 degrees apart instead of 5: OpenFOAM's axisymmetric
    # treatment assumes a thin slice, and at 175 degrees the source terms are
    # meaningless. The mesh still built, checkMesh still passed, the solver
    # still converged -- and a 15 degree cone at Mach 2 produced 2.1% of the
    # compression Taylor-Maccoll requires.
    sy, sz = math.cos(half), math.sin(half)

    # One block, inlet at the nose tip, body surface as the entire bottom.
    #
    # The previous layout put the bottom boundary on the axis at r = 0 and
    # relied on warping to lift it onto the body. blockMesh assigned that patch
    # nFaces 0 -- a face at r = 0 on a wedge is a line, not a surface -- so the
    # "body" had no area, no pressure was ever attributable to it, and a run
    # that completed reported no wall pressure.
    #
    # Putting the inlet at the tip is not a compromise here: in supersonic flow
    # nothing propagates upstream, so there is nothing for an upstream margin to
    # capture. Downstream the contour continues at base radius, which is a
    # sting-mounted model -- the drag measured is forebody pressure drag, which
    # is the quantity wave-drag theory predicts and therefore the one worth
    # comparing.
    y_far, z_far = r_far * sy, r_far * sz
    r_tip = max(1e-5, spec.tip_bluntness * r_max)
    x1 = 2.5 * length

    v = [
        f"(0 {r_tip * sy:.8g} {-r_tip * sz:.8g})",     # 0 body, inlet, back
        f"({x1:.8g} {r_max * sy:.8g} {-r_max * sz:.8g})",  # 1 body, outlet, back
        f"({x1:.8g} {y_far:.8g} {-z_far:.8g})",        # 2 far, outlet, back
        f"(0 {y_far:.8g} {-z_far:.8g})",               # 3 far, inlet, back
        f"(0 {r_tip * sy:.8g} {r_tip * sz:.8g})",      # 4 body, inlet, front
        f"({x1:.8g} {r_max * sy:.8g} {r_max * sz:.8g})",   # 5 body, outlet, front
        f"({x1:.8g} {y_far:.8g} {z_far:.8g})",         # 6 far, outlet, front
        f"(0 {y_far:.8g} {z_far:.8g})",                # 7 far, inlet, front
    ]

    nx = max(60, int(nx_body * x1 / length))
    blocks = f"hex (0 1 2 3 4 5 6 7) ({nx} {nr} 1) simpleGrading (1 {radial_grading} 1)"
    _f(case / "system/blockMeshDict", "dictionary", "blockMeshDict",
       "scale 1;\n\nvertices\n(\n    " + "\n    ".join(v) + "\n);\n\n"
       f"blocks\n(\n    {blocks}\n);\n\nedges();\n\n"
       "boundary\n(\n"
       "    inlet { type patch; faces ( (0 3 7 4) ); }\n"
       "    outlet { type patch; faces ( (1 5 6 2) ); }\n"
       "    farfield { type patch; faces ( (3 2 6 7) ); }\n"
       "    body { type wall; faces ( (0 4 5 1) ); }\n"
       "    front { type wedge; faces ( (4 7 6 5) ); }\n"
       "    back { type wedge; faces ( (0 1 2 3) ); }\n"
       ");\n\nmergePatchPairs();")

    _f(case / "constant/thermophysicalProperties", "dictionary",
       "thermophysicalProperties",
       "thermoType\n{\n    type hePsiThermo;\n    mixture pureMixture;\n"
       "    transport sutherland;\n    thermo hConst;\n"
       "    equationOfState perfectGas;\n    specie specie;\n"
       "    energy sensibleInternalEnergy;\n}\n\n"
       "mixture\n{\n    specie { molWeight 28.96; }\n"
       "    thermodynamics { Cp 1005; Hf 0; }\n"
       "    transport { As 1.4792e-06; Ts 116; }\n}")
    _f(case / "constant/turbulenceProperties", "dictionary",
       "turbulenceProperties", "simulationType laminar;")

    _f(case / "system/controlDict", "dictionary", "controlDict",
       "application rhoCentralFoam;\nstartFrom startTime;\nstartTime 0;\n"
       f"stopAt endTime;\nendTime {end_time:.6g};\n"
       f"deltaT {end_time / 200000.0:.6g};\nwriteControl runTime;\n"
       f"writeInterval {end_time / 6.0:.6g};\npurgeWrite 3;\nwriteFormat ascii;\n"
       # writeCompression, timeFormat and timePrecision are not optional. Left
       # out, the first write fails with 'error in IOstream "sha1"' at the top
       # of the time loop -- an error that names the hash of the header it was
       # composing rather than the entry it was missing.
       "writePrecision 8;\nwriteCompression off;\ntimeFormat general;\n"
       "timePrecision 8;\nrunTimeModifiable false;\nadjustTimeStep yes;\n"
       "maxCo 0.3;\nmaxDeltaT 1e-4;\n\n"
       # No function objects. Every one tried here -- forceCoeffs hand-written,
       # forceCoeffs via the shipped includeEtc, and an unrelated fieldMinMax --
       # dies at the top of the time loop with 'error in IOstream "sha1"'. That
       # is `dictionary::digest()` failing to write to an OSHA1stream in this
       # build, not a configuration error: a function object with no relation to
       # forces fails the same way. Drag is integrated from the written pressure
       # field instead, which is also what the nozzle case does and is why it
       # never hit this.
       "")

    _f(case / "system/fvSchemes", "dictionary", "fvSchemes",
       # The central-upwind flux is what makes rhoCentralFoam a shock-capturing
       # solver. Without this entry the cone case produced 2.1% of the pressure
       # rise Taylor-Maccoll requires -- a converged, stable solution with no
       # shock anywhere in the domain. Taken from the canonical template rather
       # than assumed.
       "fluxScheme Kurganov;\n"
       "ddtSchemes { default Euler; }\n"
       "gradSchemes { default Gauss linear; }\n"
       "divSchemes { default none; div(tauMC) Gauss linear; }\n"
       "laplacianSchemes { default Gauss linear corrected; }\n"
       "interpolationSchemes\n{\n    default linear;\n"
       "    reconstruct(rho) vanLeer;\n    reconstruct(U) vanLeerV;\n"
       "    reconstruct(T) vanLeer;\n}\n"
       "snGradSchemes { default corrected; }")
    _f(case / "system/fvSolution", "dictionary", "fvSolution",
       "solvers\n{\n    \"(rho|rhoU|rhoE)\" { solver diagonal; }\n"
       "    U { solver smoothSolver; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }\n"
       "    e { solver smoothSolver; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }\n}")

    def field_file(name, dims, internal, walls):
        _f(case / "0" / name, "volScalarField" if name != "U" else "volVectorField",
           name,
           f"dimensions {dims};\ninternalField uniform {internal};\n\n"
           "boundaryField\n{\n"
           f"    inlet {{ type fixedValue; value uniform {internal}; }}\n"
           "    outlet { type zeroGradient; }\n"
           # zeroGradient, not fixedValue: a shock that does reach this
           # boundary should leave through it rather than reflect off a pinned
           # freestream state.
           "    farfield { type zeroGradient; }\n"
           f"    body {{ {walls} }}\n"
           "    axis { type empty; }\n"
           "    front { type wedge; }\n    back { type wedge; }\n}")

    field_file("p", "[1 -1 -2 0 0 0 0]", f"{p_inf:.8g}", "type zeroGradient;")
    field_file("T", "[0 0 0 1 0 0 0]", f"{t_inf:.8g}", "type zeroGradient;")
    field_file("U", "[0 1 -1 0 0 0 0]", f"({u_inf:.8g} 0 0)",
               "type noSlip;")

    return {"contour": contour, "u_inf": u_inf, "rho_inf": rho_inf,
            "end_time": end_time, "nx": nx, "nr": nr,
            "domain": {"x0": x0, "x1": x1, "r_far": r_far}}


def warp_to_body(case: Path, spec: BodySpec, domain: dict) -> int:
    """Push mesh points onto the body contour.

    Radial coordinates are compressed between the body surface and the far
    field, leaving the wake region on the axis. Done after blockMesh rather
    than as a curved edge for the reason in the module docstring.
    """
    points = case / "constant/polyMesh/points"
    if not points.exists():
        return 0
    raw = points.read_text()
    start = raw.index("(", raw.index("\n(", raw.index("object")))
    head, body_txt = raw[:start], raw[start:]

    contour = spec.contour()
    xs = [p[0] for p in contour]
    rs = [p[1] for p in contour]

    def body_radius(x: float) -> float:
        if x >= spec.total_length_m:
            return rs[-1]                    # sting: base radius continues
        if x <= 0.0:
            return rs[0]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                f = 0.0 if span <= 0 else (x - xs[i]) / span
                return rs[i] + f * (rs[i + 1] - rs[i])
        return 0.0

    r_far = domain["r_far"]
    x_end = domain["x1"]
    r_tip_line = max(1e-5, spec.tip_bluntness * spec.radius_m)
    r_base_line = spec.radius_m
    out_lines, moved = [], 0
    for line in body_txt.splitlines():
        s = line.strip()
        if not (s.startswith("(") and s.endswith(")") and s.count(" ") == 2):
            out_lines.append(line)
            continue
        try:
            x, y, z = (float(v) for v in s[1:-1].split())
        except ValueError:
            out_lines.append(line)
            continue
        r = math.hypot(y, z)
        if r <= 1e-12:
            out_lines.append(line)
            continue
        rb = body_radius(x)
        if rb <= 0.0:
            out_lines.append(line)
            continue
        # Map the block's own bottom edge onto the body, not r = 0.
        #
        # blockMesh lays the lower boundary as a straight line from the tip
        # radius to the base radius across the whole domain, so a point's
        # pre-warp radius already starts at r_line(x), not zero. Treating it as
        # zero adds that line's height on top of the contour: the surface came
        # out at r = 0.093 where the body is 0.075, and kept growing downstream
        # instead of staying at base radius. The flow then saw a shallow cone
        # rather than an ogive and turned by almost nothing -- wall pressure
        # 7% above freestream at Mach 2, where an ogive of this fineness gives
        # several times freestream.
        r_line = r_tip_line + (x / x_end) * (r_base_line - r_tip_line)
        span = r_far - r_line
        frac = 0.0 if span <= 1e-12 else (r - r_line) / span
        new_r = rb + max(0.0, frac) * (r_far - rb)
        scale = new_r / r
        out_lines.append(f"({x:.8g} {y * scale:.8g} {z * scale:.8g})")
        moved += 1
    points.write_text(head + "\n".join(out_lines) + "\n")
    return moved


def _patch_face_values(field_path: Path, patch: str) -> list[float]:
    """Boundary values on one patch of a written scalar field.

    OpenFOAM writes either `value nonuniform List<scalar> N ( ... )` for a
    resolved field or `value uniform X;` when every face agrees. Both appear in
    practice and only handling the first silently returns nothing.
    """
    text = field_path.read_text()
    idx = text.find(patch)
    if idx < 0:
        return []
    seg = text[idx: idx + 400000]
    uni = re.search(r"value\s+uniform\s+([-\d.eE+]+)\s*;", seg)
    non = re.search(r"value\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\(", seg)
    if non:
        count = int(non.group(1))
        start = seg.index("(", non.end() - 1)
        body = seg[start + 1:]
        vals = []
        for token in body.split():
            if token.startswith(")"):
                break
            try:
                vals.append(float(token))
            except ValueError:
                break
        return vals[:count]
    if uni:
        return [float(uni.group(1))]
    return []


def _read_list(path: Path) -> list[str]:
    """Tokens inside the top-level ( ... ) of an OpenFOAM list file."""
    text = path.read_text()
    i = text.index("(", text.index("FoamFile"))
    # the count precedes the paren; find the matching close from the end
    j = text.rindex(")")
    return text[i + 1:j].split()


def wall_face_centres(case: Path, patch: str = "body") -> list[tuple[float, float]]:
    """(x, r) centroid of each face on a patch, in patch face order.

    Needed because boundary faces are not stored in any geometric order. The
    drag integral assumed face i sat at x = L*i/n, which silently scrambled the
    pressure distribution: the cone surface read 93 kPa -- below freestream --
    while the same face list contained a 167 kPa peak.
    """
    mesh = case / "constant/polyMesh"
    btxt = (mesh / "boundary").read_text()
    idx = btxt.find(f"\n    {patch}\n")
    if idx < 0:
        return []
    seg = btxt[idx: idx + 600]
    m_n = re.search(r"nFaces\s+(\d+)\s*;", seg)
    m_s = re.search(r"startFace\s+(\d+)\s*;", seg)
    if not (m_n and m_s):
        return []
    n_faces, start = int(m_n.group(1)), int(m_s.group(1))

    ptxt = (mesh / "points").read_text()
    pi = ptxt.index("(", ptxt.index("FoamFile"))
    pj = ptxt.rindex(")")
    pts = []
    for line in ptxt[pi + 1:pj].splitlines():
        t = line.strip()
        if t.startswith("(") and t.endswith(")") and t.count(" ") == 2:
            try:
                pts.append(tuple(float(v) for v in t[1:-1].split()))
            except ValueError:
                pass

    ftxt = (mesh / "faces").read_text()
    fi = ftxt.index("(", ftxt.index("FoamFile"))
    face_lists = re.findall(r"\d+\(([\d\s]+)\)", ftxt[fi:])

    out = []
    for k in range(n_faces):
        f = start + k
        if f >= len(face_lists):
            break
        ids = [int(v) for v in face_lists[f].split() if int(v) < len(pts)]
        if not ids:
            out.append((0.0, 0.0))
            continue
        xs = [pts[v][0] for v in ids]
        rs = [math.hypot(pts[v][1], pts[v][2]) for v in ids]
        out.append((sum(xs) / len(xs), sum(rs) / len(rs)))
    return out


def wall_pressure(case: Path, time_dir: Path, patch: str = "body") -> list[float]:
    """Pressure on a zeroGradient wall, taken from the cells behind it.

    A zeroGradient patch has no stored face values -- OpenFOAM derives them from
    the adjacent cell each time it needs them, so the written field carries only
    the boundary *type*. Reading the file and looking for numbers finds nothing,
    which is exactly what happened: a completed run reported "no wall pressure
    written" while the pressure was sitting in the internal field all along.

    So: `constant/polyMesh/boundary` gives the patch's face range, `owner` gives
    the cell behind each face, and the internal field gives its value.
    """
    mesh = case / "constant/polyMesh"
    btxt = (mesh / "boundary").read_text()
    idx = btxt.find(f"\n    {patch}\n")
    if idx < 0:
        return []
    seg = btxt[idx: idx + 600]
    m_n = re.search(r"nFaces\s+(\d+)\s*;", seg)
    m_s = re.search(r"startFace\s+(\d+)\s*;", seg)
    if not (m_n and m_s):
        return []
    n_faces, start = int(m_n.group(1)), int(m_s.group(1))
    if n_faces == 0:
        return []

    owners = _read_list(mesh / "owner")
    p_text = (time_dir / "p").read_text()
    m_int = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\n?\(",
                      p_text)
    if not m_int:
        m_uni = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", p_text)
        return [float(m_uni.group(1))] * n_faces if m_uni else []
    body_start = p_text.index("(", m_int.end() - 1)
    body_end = p_text.index(")", body_start)
    values = [float(t) for t in p_text[body_start + 1:body_end].split()]

    out = []
    for f in range(start, start + n_faces):
        if f >= len(owners):
            break
        cell = int(owners[f])
        if cell < len(values):
            out.append(values[cell])
    return out


def drag_from_pressure(case: Path, spec: BodySpec, p_inf: float,
                       rho_inf: float, u_inf: float) -> dict | None:
    """Pressure drag by integrating the wall pressure over the body.

    For a body of revolution the axial pressure force is

        D = integral (p - p_inf) * 2 pi r * (dr/dx) dx

    which needs only the wall pressure and the contour, both of which are known
    here without asking the solver for a derived quantity. Skin friction is not
    included and is stated as absent rather than folded in silently -- at Mach 2
    on a slender body it is a real fraction of the total, so this is wave drag
    plus base effects, and it is compared against wave-drag theory on that
    understanding.
    """
    times = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            times.append((float(child.name), child))
        except ValueError:
            continue
    times = [t for t in times if t[0] > 0.0]
    if not times:
        return None
    _, latest = max(times, key=lambda t: t[0])
    p_file = latest / "p"
    if not p_file.exists():
        return None

    wall_p = wall_pressure(case, latest, "body")
    if not wall_p:
        return None
    centres = wall_face_centres(case, "body")
    if len(centres) != len(wall_p):
        return None

    # Sort by x. Face order on a patch is whatever the mesher produced; the
    # ring areas below only mean anything if consecutive entries are
    # consecutive stations along the body.
    order = sorted(range(len(wall_p)), key=lambda i: centres[i][0])
    force = 0.0
    for a, b in zip(order, order[1:]):
        x_a, r_a = centres[a]
        x_b, r_b = centres[b]
        if x_b <= x_a:
            continue
        r_mid = 0.5 * (r_a + r_b)
        dr = r_b - r_a
        p_mid = 0.5 * (wall_p[a] + wall_p[b])
        force += (p_mid - p_inf) * 2.0 * math.pi * r_mid * dr

    n = len(wall_p)
    q = 0.5 * rho_inf * u_inf ** 2
    cd = force / (q * spec.frontal_area_m2)
    return {"cd_pressure": cd, "faces": n, "axial_force_n": force,
            "dynamic_pressure_pa": q}


def run_case(case: Path, spec: BodySpec, mach: float, *,
             timeout_s: int = 1800, **kwargs) -> DragResult:
    """Mesh, warp, solve, and read the drag coefficient back."""
    if not (shutil.which("rhoCentralFoam") and shutil.which("blockMesh")):
        return DragResult(spec.name, mach, error="OpenFOAM not installed")

    meta = write_case(case, spec, mach, **kwargs)
    try:
        bm = subprocess.run(["blockMesh", "-case", str(case)],
                            capture_output=True, text=True, timeout=600)
        if bm.returncode != 0:
            return DragResult(spec.name, mach,
                              error=f"blockMesh: {bm.stderr[-200:]}")
        moved = warp_to_body(case, spec, meta["domain"])
        if moved == 0:
            return DragResult(spec.name, mach, error="no points warped to body")

        chk = subprocess.run(["checkMesh", "-case", str(case)],
                             capture_output=True, text=True, timeout=600)
        (case / "log.checkMesh").write_text(chk.stdout)
        if "negative cell volume" in chk.stdout.lower():
            return DragResult(spec.name, mach, error="negative cell volumes")

        sol = subprocess.run(["rhoCentralFoam", "-case", str(case)],
                             capture_output=True, text=True, timeout=timeout_s)
        (case / "log.rhoCentralFoam").write_text(
            (sol.stdout or "")[-200000:] + "\n" + (sol.stderr or "")[-20000:])
    except subprocess.TimeoutExpired:
        return DragResult(spec.name, mach, error="solver timeout")

    got = drag_from_pressure(case, spec, kwargs.get("p_inf", P_INF_PA),
                             meta["rho_inf"], meta["u_inf"])
    if got is None:
        return DragResult(spec.name, mach, error="no wall pressure written")
    return DragResult(spec.name, mach, cd=got["cd_pressure"],
                      cd_pressure=got["cd_pressure"], converged=True,
                      meta={**got, "fineness": spec.fineness,
                            "nose_shape": spec.nose_shape,
                            "frontal_area_m2": spec.frontal_area_m2})
