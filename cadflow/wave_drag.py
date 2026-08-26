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
Two conditions, and the second is the one that bites.

The nose must be *pointed*: S'(0) = 0 at the tip. An elliptical nose is not -- it
meets the axis with infinite slope -- and a blunt nose at supersonic speed has a
detached bow shock this theory does not model at all.

The nose must also meet the cylinder *tangentially*. A slope break there puts a
jump in S', and the wave drag of a body with a slope discontinuity is
logarithmically divergent in linearised slender-body theory: it has no limit, so
no amount of quadrature produces a number. This is visible directly. At fineness
3, refining the quadrature from n=750 to n=12000 gives

    ogive        0.2442  0.2447  0.2449  0.2450  0.2451     converged
    vonkarman    0.2249  0.2257  0.2260  0.2262  0.2263     converged
    conical      0.4266  0.4596  0.5109  0.5281  0.5846     still climbing

and the reason is exactly the joint slope dr/dz, which is -1e-5 for the ogive
and -1e-3 for von Karman -- both tangent to within rounding -- but -0.1667 for
the cone, a real break. An earlier version of this module priced cones anyway.
The number it produced moved 7% between adjacent fineness values and grew
without bound under refinement; it was noise with a plausible magnitude.

So only tangent noses are priced. What remains is a real and useful comparison:
von Karman against the tangent ogive is smooth in fineness, n-converged, and
behaves correctly -- 13% better at fineness 1.5, 3.5% at 5.5, the advantage
shrinking as the nose slims, which is what must happen as every smooth shape
converges in the slender limit.

Factors are computed on the real vehicle -- nose, cylinder, and one fixed closure
shared by every shape -- so what varies between shapes is only the nose and its
joint with the body.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np

from .profiles import NOSE_SHAPES, nose_profile

#: Nose families this model can actually price. The requirement is *tangency
#: where the nose meets the cylinder*, which is stricter than having a pointed
#: tip, and it is the condition under which the integral has a limit at all.
#: See the "Validity" section below -- a cone fails it and its wave drag is
#: logarithmically divergent, not merely inaccurate.
TANGENT_SHAPES = ("ogive", "vonkarman")

#: Kept as an alias: the pointed-tip condition S'(0) = 0 is necessary but, as it
#: turned out, nowhere near sufficient.
POINTED_SHAPES = TANGENT_SHAPES

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


#: Fineness is quantised to this step before the factor is looked up, so the
#: memo actually hits. The factor varies slowly and smoothly with fineness --
#: the conical/ogive ratio moves about 0.03 per unit -- so 0.02 is far below
#: any resolution the drag model can honestly claim.
_FINENESS_STEP = 0.02


@lru_cache(maxsize=8192)
def _shape_factor_quantised(shape: str, steps: int, n: int) -> float:
    fineness = steps * _FINENESS_STEP
    ref = wave_drag_coefficient("ogive", fineness, n=n)
    if ref <= 0.0:
        return 1.0
    return wave_drag_coefficient(shape, fineness, n=n) / ref


def shape_factor(shape: str, fineness: float, n: int = 1500) -> float:
    """Wave drag of `shape` relative to a tangent ogive of the same fineness.

    Normalising to the ogive is what makes this usable despite the unresolved
    absolute constant: any common prefactor cancels, and the ogive is the shape
    the existing Cd corpus is implicitly calibrated on, so the default vehicle
    keeps the drag it always had.

    Memoised on a quantised fineness. Each call is two dense n x n quadratures,
    56 ms at the default n, and corpus sampling calls this twice per record --
    which turned generating 1500 records into minutes of pure drag integration
    for a handful of distinct answers.
    """
    shape = str(shape).lower()
    if shape not in NOSE_SHAPES:
        raise ValueError(f"unknown nose shape {shape!r}")
    if shape not in TANGENT_SHAPES:
        raise ValueError(
            f"{shape!r} does not meet the body tangentially, so its slender-body "
            f"wave drag is divergent; only {TANGENT_SHAPES} can be priced")
    if shape == "ogive":
        return 1.0
    steps = max(1, int(round(float(fineness) / _FINENESS_STEP)))
    return _shape_factor_quantised(shape, steps, int(n))


def is_trustworthy(shape: str) -> bool:
    """Whether this model can price the shape at all.

    False for a cone -- slope break at the body joint, divergent -- and for an
    elliptical nose -- blunt tip, detached shock. Both are excluded rather than
    given a number that looks like an answer.
    """
    return str(shape).lower() in TANGENT_SHAPES


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

#: Guard rail. With only tangent shapes priced the factor stays in 0.86..1.0,
#: so this should never bind; it is here so a future shape cannot quietly move
#: Cd by an arbitrary amount.
MAX_SHAPE_FACTOR = 2.0


#: Forebody shape factors measured by CFD, relative to a tangent ogive.
#:
#: `shape_factor` compares CLOSED bodies -- the slender-body integral needs a
#: body that closes at both ends -- so it prices a von Karman tail that every
#: shape shares, and that common tail pulls every ratio toward 1.0. A launch
#: vehicle has no such tail; it ends in a blunt base with an engine. For
#: choosing a nose on an open-based vehicle the forebody ratio is the relevant
#: one, and it is 9-14 points more favourable to von Karman.
#:
#: Measured from 48 axisymmetric rhoCentralFoam cases over fineness 1.5-4 and
#: Mach 1.5-3, validated on cones to -1.8% against exact Taylor-Maccoll. The
#: ratio is near-constant over that whole range (0.777-0.823 for von Karman),
#: which is what a shape factor should be. See
#: data/drag_corpus/shape_factor_calibration.json.
CFD_FOREBODY_FACTOR = {
    "ogive": 1.000,        # reference
    "vonkarman": 0.801,    # 16 cases, spread 0.777-0.823
    "cone": 0.913,         # 16 cases, spread 0.812-1.088 -- Mach dependent
}

#: Cones are Mach-dependent in a way the others are not: their factor crosses
#: above 1.0 at Mach 1.5 and falls to 0.82 by Mach 3, because a cone's own drag
#: is Mach-sensitive while slender-body wave drag is not. The single number
#: above is the mean; callers wanting the trend should read the corpus.
CONE_FACTOR_IS_MACH_DEPENDENT = True


def cd_multiplier(shape: str, fineness: float, *, forebody: bool = False) -> float:
    """Factor on zero-lift Cd for a nose shape, relative to a tangent ogive.

    Only the wave-drag share is scaled, so an ogive returns exactly 1.0 and the
    existing corpus calibration is preserved for the default vehicle.

    `forebody=True` uses the CFD-measured open-base ratio instead of the
    closed-body slender-body factor. That is the right choice for a launch
    vehicle and the wrong one for a body that actually closes; it is opt-in so
    no existing caller silently changes its answer.
    """
    if forebody:
        key = str(shape).lower()
        if key not in CFD_FOREBODY_FACTOR:
            raise ValueError(
                f"no CFD forebody factor for {shape!r}; measured shapes are "
                f"{sorted(CFD_FOREBODY_FACTOR)}")
        factor = CFD_FOREBODY_FACTOR[key]
    else:
        factor = min(shape_factor(shape, fineness), MAX_SHAPE_FACTOR)
    return (1.0 - WAVE_DRAG_SHARE) + WAVE_DRAG_SHARE * factor
