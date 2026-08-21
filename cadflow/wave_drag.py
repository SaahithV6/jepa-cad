"""Supersonic wave drag of a body of revolution, from its actual meridian.

Nose shape used to be decorative: Cd was drawn from the CFD corpus without
reference to geometry and the planner used a hard CD = 0.42, so a cone and a
von Karman ogive flew identically. A world model cannot learn that shape matters
if the physics it trains on does not depend on it.

Method
------
Karman's slender-body result. For a body of revolution at zero incidence in
linearised supersonic flow, with cross-sectional area S(x),

    D/q = -(1/2 pi) INT INT S''(x1) S''(x2) ln|x1 - x2| dx1 dx2

Two things make this hard to evaluate and both bit on the first attempt.

The kernel is log-singular on the diagonal. That is handled exactly: a cell of
width h contributes INT INT_cell ln|x-y| = h^2 (ln h - 3/2), which is the term a
naive sampling scheme silently drops. Checked in isolation against the closed
form for a constant integrand, this converges to +0.03% at n=1600.

S'' is singular at the ends of the shapes that matter -- the Haack family has
S ~ x^(3/2) at the tip, so S'' ~ x^(-1/2). A uniform grid cannot represent that,
and a first attempt using one over-predicted Sears-Haack by a factor that drifted
4.59 -> 5.20 across n=200..6400 without settling. The fix is to integrate in
theta under x = (L/2)(1 - cos theta) at *midpoints*, so no node ever lands on an
end. The Jacobian (L/2) sin(theta) vanishes at the ends exactly fast enough to
cancel the singularity: S'' * w stays finite. With that, Sears-Haack converges
to six figures and Richardson-extrapolates to the same value at every n.

On the absolute constant
------------------------
Evaluated this way, Sears-Haack gives D/q = (9 pi^3 / 2) R^4 / L^2, equivalently
128 V^2 / (pi L^4), with the scaling exactly constant in L and R. That is 16/3
times the 24 V^2 / (pi L^4) that is often quoted. The quadrature is validated
independently, so the discrepancy is a constant in the physics, not numerics --
and rather than pick a side, this module is used *relatively*: shape factors are
normalised to the tangent ogive, where any common prefactor cancels. The
absolute level of Cd continues to come from the measured CFD corpus.

The relative use is validated by a fact that holds whatever the prefactor is:
the von Karman ogive is by construction the minimum-wave-drag nose for a given
length and base radius, and it comes out minimal here at every fineness ratio.

Validity
--------
Slender-body theory requires a *pointed* nose: S'(0) = 0 at the tip. Ogive,
conical and von Karman all satisfy this. An elliptical nose does not -- it meets
the axis with infinite slope, S'(0) = 2 pi R^2 / L, and a blunt nose at
supersonic speed has a detached bow shock that this theory does not model at all.
Its number is reported as an upper bound and flagged, not trusted.

The theory also over-predicts sharp cones, where the exact conical solution
differs, and degrades below fineness ratios of about 3. Factors are therefore
computed on the real vehicle -- nose, cylinder, and one fixed closure shared by
every shape -- so that what varies between shapes is only the nose and the joint
it makes with the body, which is itself real: a tangent ogive meets the cylinder
with no slope break, a cone does not.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .profiles import NOSE_SHAPES, nose_profile

#: Nose families whose tip satisfies the pointed-body requirement S'(0) = 0.
POINTED_SHAPES = ("ogive", "conical", "vonkarman")

#: Reference vehicle proportions for shape comparison, in base radii.
_CYLINDER_RADII = 10.0
_TAIL_RADII = 6.0


def karman_drag_area(
    zs: Sequence[float], rs: Sequence[float], n: int = 3000
) -> float:
    """D/q for a closed body of revolution given its meridian, in length^2.

    zs must be increasing and the body must close (r = 0) at both ends; the
    slender-body integral is only defined for a closed body.
    """
    zf = np.asarray(zs, dtype=float)
    rf = np.asarray(rs, dtype=float)
    total_length = float(zf[-1] - zf[0])
    if total_length <= 0.0:
        raise ValueError("meridian has non-positive length")

    dth = math.pi / n
    th = dth * (np.arange(n) + 0.5)              # midpoints: no node on an end
    x = zf[0] + (total_length / 2.0) * (1.0 - np.cos(th))
    w = (total_length / 2.0) * np.sin(th) * dth  # exact Jacobian

    s = math.pi * np.interp(x, zf, rf) ** 2
    # Second derivative by the 2nd-order formula for a non-uniform grid.
    d2 = np.zeros(n)
    h1 = x[1:-1] - x[:-2]
    h2 = x[2:] - x[1:-1]
    d2[1:-1] = 2.0 * (
        s[:-2] * h2 - s[1:-1] * (h1 + h2) + s[2:] * h1
    ) / (h1 * h2 * (h1 + h2))
    d2[0], d2[-1] = d2[1], d2[-2]

    g = d2 * w                                    # finite even where S'' is not
    dist = np.abs(x[:, None] - x[None, :])
    kern = np.where(
        dist > 0.0,
        np.log(np.where(dist > 0.0, dist, 1.0)),
        np.log(w) - 1.5,                          # exact log-singular self term
    )
    return float(-(g @ kern @ g) / (2.0 * math.pi))


def _vehicle(shape: str, radius: float, nose_length: float, n: int = 3000):
    """Nose + cylinder + one fixed closure, as a closed meridian."""
    r = float(radius)
    cyl = _CYLINDER_RADII * r
    tail = _TAIL_RADII * r
    prof = nose_profile(r, nose_length, shape, n)
    zs = np.array([z for _, z in prof])
    rs = np.array([q for q, _ in prof])
    z_nose = nose_length - zs[::-1]               # tip at 0, base at nose_length
    r_nose = rs[::-1]
    # The same von Karman closure for every shape, so the difference between
    # shapes is the nose and its joint with the body, not the tail.
    tprof = nose_profile(r, tail, "vonkarman", n)
    z_tail = nose_length + cyl + np.array([z for _, z in tprof])
    r_tail = np.array([q for q, _ in tprof])
    return (
        np.concatenate([z_nose, [nose_length + cyl], z_tail]),
        np.concatenate([r_nose, [r], r_tail]),
    )


def wave_drag_coefficient(
    shape: str, fineness: float, radius: float = 1.0, n: int = 3000
) -> float:
    """Wave-drag coefficient on base area for a nose of the given fineness.

    fineness is nose length over base *diameter*, the usual rocketry convention.
    """
    r = float(radius)
    nose_length = 2.0 * r * float(fineness)
    zs, rs = _vehicle(shape, r, nose_length, n)
    return karman_drag_area(zs, rs, n) / (math.pi * r * r)


def shape_factor(shape: str, fineness: float, n: int = 1500) -> float:
    """Wave drag of `shape` relative to a tangent ogive of the same fineness.

    Normalising to the ogive is what makes this usable despite the unresolved
    absolute constant: any common prefactor cancels, and the ogive is the shape
    the existing Cd corpus is implicitly calibrated on, so the default vehicle
    keeps the drag it always had.
    """
    shape = str(shape).lower()
    if shape not in NOSE_SHAPES:
        raise ValueError(f"unknown nose shape {shape!r}")
    if shape == "ogive":
        return 1.0
    ref = wave_drag_coefficient("ogive", fineness, n=n)
    if ref <= 0.0:
        return 1.0
    return wave_drag_coefficient(shape, fineness, n=n) / ref


def is_trustworthy(shape: str) -> bool:
    """Whether slender-body theory applies to this nose family at all."""
    return str(shape).lower() in POINTED_SHAPES


def sears_haack(length: float, max_radius: float, n: int = 400):
    """Sears-Haack meridian r(x) = R [4 x/L (1 - x/L)]^(3/4), as (zs, rs)."""
    ell, r = float(length), float(max_radius)
    t = np.linspace(0.0, 1.0, n + 1)
    return ell * t, r * (4.0 * t * (1.0 - t)) ** 0.75


def sears_haack_drag_area(length: float, max_radius: float) -> float:
    """(9 pi^3 / 2) R^4 / L^2 -- the value this quadrature converges to."""
    return 4.5 * math.pi**3 * float(max_radius) ** 4 / float(length) ** 2


#: Share of zero-lift drag attributable to wave drag near max-Q, for a slender
#: supersonic body. This is a stated assumption, not a derived quantity: the
#: absolute wave-drag constant is unresolved (see the module docstring), so the
#: split between wave and everything else cannot be recovered from the corpus.
#: Standard drag build-ups for slender rockets around M = 2 put it near half.
#: It is a module constant so it can be moved in one place.
WAVE_DRAG_SHARE = 0.5

#: Slender-body theory over-predicts sharp cones, where the exact conical
#: solution differs, and it is applied here at fineness ratios below the ~3 it
#: is really good for. Cap the multiplier so a cone is penalised but not absurdly.
MAX_SHAPE_FACTOR = 2.0


def cd_multiplier(shape: str, fineness: float) -> float:
    """Factor on zero-lift Cd for a nose shape, relative to a tangent ogive.

    Only the wave-drag share is scaled, so an ogive returns exactly 1.0 and the
    existing corpus calibration is preserved for the default vehicle.
    """
    factor = min(shape_factor(shape, fineness), MAX_SHAPE_FACTOR)
    return (1.0 - WAVE_DRAG_SHARE) + WAVE_DRAG_SHARE * factor
