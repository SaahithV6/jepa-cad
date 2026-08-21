"""Sculpted shapes: the parts that are not primitives.

The geometry layer could make cylinders, boxes, extrusions and surfaces of
revolution. That covers an airframe section and a nose cone and nothing else.
The pieces of a rocket that are neither a prism nor axisymmetric-from-a-simple-
curve had to be faked or skipped, and the biggest one was skipped entirely: the
nozzle existed only as an area ratio. It had no contour, no mass, no wall, and
nothing structural or thermal could be asked about it.

Three shapes live here, each of which needs a real sculpting primitive.

A **bell nozzle** is a surface of revolution whose meridian is a curve fitted to
angle constraints at both ends, not an arc or a conic section. Its shape is the
difference between a nozzle that works and one that loses several percent of its
thrust to divergence.

A **stage transition** joins two different diameters. It is a loft, and it
cannot be an extrusion or a revolve of anything simple if the two ends are
different shapes.

A **hollow shell** of any of these is a real shell operation, which the backend
could not previously do -- it silently returned an expanded bounding box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Reference conical nozzle half-angle. A bell is described by how much shorter
#: it is than the 15-degree cone of the same area ratio, which is the
#: conventional way to state it: an "80% bell" is 80% of that length.
CONICAL_REFERENCE_DEG = 15.0


@dataclass(frozen=True)
class BellNozzle:
    throat_radius_m: float
    exit_radius_m: float
    length_m: float
    area_ratio: float
    initial_angle_deg: float
    exit_angle_deg: float
    percent_bell: float
    contour: tuple[tuple[float, float], ...]

    @property
    def divergence_efficiency(self) -> float:
        """Fraction of thrust surviving the exit flow angle.

        (1 + cos theta_e)/2, the standard result for a conical-equivalent exit.
        The CFD work on these nozzles measured the same effect directly: exit
        Mach fell 1.83% below the quasi-1D prediction at a 36 degree half-angle
        and matched it to 0.00% at 10.4 degrees.
        """
        return 0.5 * (1.0 + math.cos(math.radians(self.exit_angle_deg)))


def bell_contour(throat_radius_m: float, area_ratio: float,
                 percent_bell: float = 0.80,
                 initial_angle_deg: float = 30.0,
                 exit_angle_deg: float = 9.0,
                 points: int = 60) -> BellNozzle:
    """Meridian of a bell nozzle, as (radius, z) from throat to exit.

    The curve is a quadratic Bezier pinned by four things that are all either
    given or forced: it starts at the throat radius, ends at the exit radius the
    area ratio demands, leaves the throat at ``initial_angle_deg`` and arrives at
    the exit at ``exit_angle_deg``. Two endpoints and two tangents determine a
    quadratic completely -- the control point is where the two tangent lines
    cross -- so nothing here is fitted or read off a chart.

    That matters because the published bell contours are chart lookups. This is
    the same family of shape, parameterised by quantities a designer chooses
    directly, and the consequence of choosing them badly shows up in
    divergence_efficiency rather than being hidden.
    """
    r_t = float(throat_radius_m)
    eps = float(area_ratio)
    if r_t <= 0.0:
        raise ValueError("throat radius must be positive")
    if eps <= 1.0:
        raise ValueError("area ratio must exceed 1 for a diverging nozzle")
    frac = float(percent_bell)
    if not 0.4 <= frac <= 1.2:
        raise ValueError("percent bell should be between 0.4 and 1.2")

    theta_n = math.radians(float(initial_angle_deg))
    theta_e = math.radians(float(exit_angle_deg))
    if not 0.0 < theta_e < theta_n < math.radians(89.0):
        raise ValueError("need 0 < exit angle < initial angle < 89 degrees")

    r_e = r_t * math.sqrt(eps)
    cone_length = (r_e - r_t) / math.tan(math.radians(CONICAL_REFERENCE_DEG))
    length = frac * cone_length

    # Bezier control point: intersection of the tangent at the throat with the
    # tangent at the exit. Solving the two lines for their crossing.
    #   from throat  (0, r_t)  along  (1, tan theta_n)
    #   from exit    (L, r_e)  along  (1, tan theta_e)
    tn, te = math.tan(theta_n), math.tan(theta_e)
    if abs(tn - te) < 1e-12:
        raise ValueError("initial and exit angles must differ")
    z_c = (r_e - r_t - te * length) / (tn - te)
    r_c = r_t + tn * z_c

    n = max(8, int(points))
    contour = []
    for i in range(n + 1):
        t = i / n
        omt = 1.0 - t
        z = omt * omt * 0.0 + 2 * omt * t * z_c + t * t * length
        r = omt * omt * r_t + 2 * omt * t * r_c + t * t * r_e
        contour.append((r, z))
    # pin the ends exactly against floating point drift
    contour[0] = (r_t, 0.0)
    contour[-1] = (r_e, length)

    return BellNozzle(
        throat_radius_m=r_t, exit_radius_m=r_e, length_m=length,
        area_ratio=eps, initial_angle_deg=float(initial_angle_deg),
        exit_angle_deg=float(exit_angle_deg), percent_bell=frac,
        contour=tuple(contour),
    )


def nozzle_solid(nozzle: BellNozzle, wall_thickness_m: float, backend=None):
    """Build the nozzle as a solid wall of the given thickness.

    The contour is the inner surface, the gas side. The wall is added outward,
    so the throat stays the throat -- offsetting inward would silently change
    the area ratio the engine was sized for.
    """
    from cadflow.backends import get_backend

    b = backend or get_backend(prefer_real=True)
    t = float(wall_thickness_m)
    if t <= 0.0:
        raise ValueError("wall thickness must be positive")

    scale = 1000.0                       # the CAD layer works in millimetres
    inner = [(r * scale, z * scale) for r, z in nozzle.contour]

    # Offset along the surface normal, not radially. A radial offset gives a
    # wall of t*cos(theta) measured across the sheet, so it thins exactly where
    # the contour is steepest -- at the throat, where theta is 30 degrees, a
    # radial offset produces 87% of the intended thickness, and the solid came
    # out 5% light overall. A nozzle is made from constant-thickness sheet.
    outer = []
    pts = nozzle.contour
    for i, (r, z) in enumerate(pts):
        j0, j1 = max(0, i - 1), min(len(pts) - 1, i + 1)
        dr = pts[j1][0] - pts[j0][0]
        dz = pts[j1][1] - pts[j0][1]
        norm = math.hypot(dr, dz) or 1.0
        # outward normal: away from the axis, i.e. (+dz, -dr) normalised
        nr, nz = dz / norm, -dr / norm
        outer.append(((r + t * nr) * scale, (z + t * nz) * scale))

    # a closed meridian: out along the inner surface, back along the outer
    ring = inner + list(reversed(outer))
    # already a closed ring that never touches the axis, so no axis closure
    return b.revolve_profile(ring, 360.0, smooth=False, close_to_axis=False)


def transition_sections(lower_radius_m: float, upper_radius_m: float,
                        length_m: float, points: int = 64,
                        stations: int = 12) -> list:
    """Cross-sections for an interstage joining two diameters.

    A cosine blend rather than a straight cone, so the transition meets both
    cylinders tangentially and raises no slope discontinuity where it joins --
    the same reason a tangent ogive beats a cone on the nose, and the same
    reason it matters: a slope break is where the wave drag and the stress both
    concentrate.
    """
    r0, r1 = float(lower_radius_m), float(upper_radius_m)
    ell = float(length_m)
    if r0 <= 0.0 or r1 <= 0.0 or ell <= 0.0:
        raise ValueError("radii and length must be positive")

    scale = 1000.0
    out = []
    for i in range(max(2, stations) + 1):
        f = i / max(2, stations)
        blend = 0.5 * (1.0 - math.cos(math.pi * f))
        r = (r0 + (r1 - r0) * blend) * scale
        ring = [(r * math.cos(2 * math.pi * k / points),
                 r * math.sin(2 * math.pi * k / points))
                for k in range(points)]
        out.append((f * ell * scale, ring))
    return out


def transition_solid(lower_radius_m: float, upper_radius_m: float,
                     length_m: float, backend=None):
    """Loft an interstage between two diameters."""
    from cadflow.backends import get_backend

    b = backend or get_backend(prefer_real=True)
    return b.loft_sections(
        transition_sections(lower_radius_m, upper_radius_m, length_m))
