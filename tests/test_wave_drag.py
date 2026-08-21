"""Validation for the slender-body wave-drag integral.

These are the checks that decide whether the module is telling the truth. The
first attempt at this integral failed all of them and was deleted; keeping them
executable is what stops that happening quietly again.
"""

import math

import numpy as np
import pytest

from cadflow.profiles import NOSE_SHAPES, nose_profile
from cadflow.wave_drag import (
    TANGENT_SHAPES,
    is_trustworthy,
    karman_drag_area,
    sears_haack,
    sears_haack_drag_area,
    shape_factor,
    wave_drag_coefficient,
)


def test_log_kernel_quadrature_is_exact():
    """The log-singular diagonal, isolated from any physics.

    INT INT over [0,L]^2 of ln|x-y| with a constant integrand is L^2(ln L - 3/2).
    Getting this right is what a naive sampling scheme misses.
    """
    for length in (1.0, 10.0):
        n = 1600
        xs = np.linspace(0.0, length, n + 1)
        h = length / n
        w = np.ones(n + 1)
        w[0] = w[-1] = 0.5
        dist = np.abs(xs[:, None] - xs[None, :])
        kern = np.where(
            dist > 0.0, np.log(np.where(dist > 0.0, dist, 1.0)), math.log(h) - 1.5
        ) * h * h
        got = w @ kern @ w
        want = length * length * (math.log(length) - 1.5)
        assert abs(got - want) / abs(want) < 1e-3


@pytest.mark.parametrize("length,radius", [(10.0, 1.0), (8.0, 1.0), (12.0, 1.5)])
def test_sears_haack_converges_to_closed_form(length, radius):
    """Sears-Haack has S'' ~ x^(-1/2) at both ends -- the case that broke v1."""
    got = karman_drag_area(*sears_haack(length, radius, 4000), n=3000)
    want = sears_haack_drag_area(length, radius)
    assert abs(got - want) / want < 5e-3


def test_sears_haack_scaling_is_exact_in_length_and_radius():
    """D/q * L^2 / R^4 must be one constant for every body in the family."""
    vals = [
        karman_drag_area(*sears_haack(ell, r, 4000), n=2000) * ell**2 / r**4
        for ell, r in [(10.0, 1.0), (8.0, 1.0), (12.0, 1.5), (20.0, 2.0)]
    ]
    assert max(vals) - min(vals) < 1e-6 * max(vals)


@pytest.mark.parametrize("fineness", [1.0, 2.0, 3.0, 5.0])
def test_von_karman_is_the_minimum_drag_nose(fineness):
    """The one check that holds whatever the absolute prefactor is.

    The von Karman ogive is by construction the minimum-wave-drag shape for a
    given length and base radius, so it must come out lowest at every fineness.
    """
    drags = {s: wave_drag_coefficient(s, fineness, n=1500) for s in TANGENT_SHAPES}
    assert min(drags, key=drags.get) == "vonkarman", drags


def test_tangent_noses_converge_under_refinement():
    """The condition that decides whether a shape can be priced at all.

    A tangent nose has no jump in S' at the body joint and the integral has a
    limit. Refining the quadrature must stop moving the answer.
    """
    for shape in TANGENT_SHAPES:
        vals = [wave_drag_coefficient(shape, 3.0, n=n) for n in (750, 1500, 3000)]
        spread = (max(vals) - min(vals)) / min(vals)
        assert spread < 0.01, (shape, vals)


def test_a_cone_is_divergent_and_is_refused():
    """Why conical is excluded, asserted rather than asserted-about.

    A cone meets the cylinder with a real slope break, dr/dz = -R/L, which puts
    a jump in S'. Linearised slender-body wave drag is logarithmically divergent
    at such a discontinuity: the value climbs without settling as the quadrature
    refines, so there is no number to report. An earlier version of this module
    reported one anyway.
    """
    vals = [wave_drag_coefficient("conical", 3.0, n=n) for n in (750, 3000, 12000)]
    assert vals[0] < vals[1] < vals[2], vals
    assert vals[-1] / vals[0] > 1.2, vals          # still climbing, not converging
    with pytest.raises(ValueError, match="tangent"):
        shape_factor("conical", 3.0)


def test_joint_slope_is_what_separates_the_two_cases():
    """Tangency is a property of the profile, so check it there."""
    from cadflow.profiles import nose_profile
    for shape in ("ogive", "vonkarman", "conical"):
        prof = nose_profile(1.0, 6.0, shape, 20000)
        (r0, z0), (r1, z1) = prof[0], prof[1]
        slope = abs((r1 - r0) / (z1 - z0))
        if shape in TANGENT_SHAPES:
            assert slope < 5e-3, (shape, slope)
        else:
            assert slope > 0.1, (shape, slope)


def test_shape_factor_is_smooth_in_fineness():
    """A jagged conditioning signal is noise the model would try to fit.

    The cone version moved 7% between adjacent fineness values. The tangent
    families must vary smoothly and monotonically.
    """
    ratios = [shape_factor("vonkarman", 3.30 + 0.02 * i) for i in range(10)]
    deltas = [b - a for a, b in zip(ratios, ratios[1:])]
    assert all(d > 0 for d in deltas), ratios
    assert max(deltas) < 5.0 * min(deltas), deltas


def test_ogive_is_near_optimal_and_improves_with_fineness():
    """A tangent ogive is close to von Karman, and closer as it gets slender."""
    ratios = [
        wave_drag_coefficient("ogive", f, n=1500)
        / wave_drag_coefficient("vonkarman", f, n=1500)
        for f in (1.0, 2.0, 3.0, 5.0)
    ]
    assert all(r >= 1.0 for r in ratios)
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 1.10


def test_von_karman_advantage_shrinks_as_the_nose_gets_slender():
    """Shape matters most when blunt; smooth shapes converge in the slender limit."""
    ratios = [shape_factor("vonkarman", f) for f in (1.5, 2.5, 3.5, 4.5, 5.5)]
    assert all(r < 1.0 for r in ratios), ratios
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[0] < 0.90 and ratios[-1] > 0.95, ratios


def test_wave_drag_falls_with_fineness():
    for shape in TANGENT_SHAPES:
        vals = [wave_drag_coefficient(shape, f, n=1500) for f in (1.0, 2.0, 3.0, 5.0)]
        assert all(a > b for a, b in zip(vals, vals[1:])), (shape, vals)


def test_ogive_is_the_normalisation_reference():
    for f in (1.0, 3.0, 5.0):
        assert shape_factor("ogive", f) == 1.0


def test_untrustworthy_shapes_are_flagged():
    """Elliptical is blunt at the tip; conical breaks slope at the joint."""
    assert not is_trustworthy("elliptical")
    assert not is_trustworthy("conical")
    for shape in TANGENT_SHAPES:
        assert is_trustworthy(shape)
    assert set(TANGENT_SHAPES) < set(NOSE_SHAPES)


def _tip_area_slope(shape: str, n: int) -> float:
    """dS/dx sampled at the first node off the tip."""
    prof = nose_profile(1.0, 4.0, shape, n)
    zs = np.array([4.0 - z for _, z in prof])[::-1]
    ss = math.pi * np.array([r for r, _ in prof])[::-1] ** 2
    return abs((ss[3] - ss[0]) / (zs[3] - zs[0]))


#: Pointed at the *tip*, S'(0) = 0. This is a different property from tangency
#: at the body joint: a cone is pointed but not tangent, which is why it is
#: excluded for divergence rather than for bluntness.
POINTED_TIP_SHAPES = ("ogive", "conical", "vonkarman")


def test_pointed_noses_really_are_pointed():
    """The tip condition, checked on the profiles rather than assumed.

    S'(0) = 0 is a limit, so a fixed threshold is the wrong test: von Karman has
    S ~ x^(3/2), whose finite-difference slope near the tip decays only as
    sqrt(x) and is still 0.037 three nodes in at n=4000. What separates pointed
    from blunt is that the sampled slope vanishes under refinement for one and
    is bounded below for the other.
    """
    for shape in NOSE_SHAPES:
        coarse = _tip_area_slope(shape, 1000)
        fine = _tip_area_slope(shape, 16000)
        if shape in POINTED_TIP_SHAPES:
            assert fine < coarse / 2.0, (shape, coarse, fine)
            assert fine < 1e-1, (shape, fine)
        else:
            # blunt: S'(0) = 2 pi R^2 / L = pi/2 here, and refinement does not
            # move it, because the limit is simply not zero
            assert fine > 1.0, (shape, fine)
            assert abs(fine - coarse) < 0.1 * fine, (shape, coarse, fine)
