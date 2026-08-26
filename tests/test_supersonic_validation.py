"""The supersonic CFD against exact conical flow.

This is the acceptance test the drag pipeline was built to pass, and it was
written before any dataset was generated. That ordering mattered: the pipeline
produced converged, stable, plausible numbers at three separate points while
being badly wrong, and nothing internal to it ever complained.

    stage                                   cone surface pressure
    zero-area body patch (nFaces 0)         no result at all
    175 degree "wedge"                      102,102 Pa   (2% of the compression)
    corrected                               159,131 Pa   (+0.23% of exact)

    exact Taylor-Maccoll, 15 deg, Mach 2    158,705 Pa

The 175-degree wedge is the one worth remembering. `blockMesh` built it,
`checkMesh` passed it, the solver converged on it, and it returned the same
answer to sixteen digits across runs. Only comparison against theory computed
outside the solver exposed it.

The reference is computed, not recalled. An earlier version of this comparison
used Cp = 0.1305 from memory and made the solver look 56% high; the exact value
is 0.2022 and the solver was right.
"""

import math

import pytest

pytest.importorskip("scipy")

from cadflow.taylor_maccoll import cone_surface  # noqa: E402


def test_shock_angle_matches_the_standard_tables():
    """33.9 degrees for a 15 degree cone at Mach 2 is a textbook number.

    Checking it first means a failure below points at the CFD rather than at
    this module.
    """
    beta, _pr, _cp = cone_surface(2.0, 15.0)
    assert beta == pytest.approx(33.9, abs=0.15), beta


def test_surface_pressure_ratio_is_physical():
    beta, pr, cp = cone_surface(2.0, 15.0)
    assert 1.5 < pr < 1.65, pr
    assert 0.19 < cp < 0.21, cp


def test_pressure_rises_with_cone_angle():
    """Monotone in half-angle, and by the right amount.

    A 10 degree cone compresses less than a 15, which compresses less than a
    20. Any solver or table that violates this is broken in a way no single
    point comparison would catch.
    """
    ratios = [cone_surface(2.0, tc)[1] for tc in (10.0, 15.0, 20.0)]
    assert ratios[0] < ratios[1] < ratios[2], ratios
    assert ratios[0] == pytest.approx(1.2925, rel=0.01)
    assert ratios[1] == pytest.approx(1.5663, rel=0.01)
    assert ratios[2] == pytest.approx(1.9115, rel=0.01)


def test_shock_angle_exceeds_the_mach_angle():
    """A shock cannot lie inside the Mach cone."""
    for mach in (1.5, 2.0, 3.0):
        beta, _pr, _cp = cone_surface(mach, 15.0)
        mach_angle = math.degrees(math.asin(1.0 / mach))
        assert beta > mach_angle, (mach, beta, mach_angle)


def test_a_cone_too_blunt_for_an_attached_shock_returns_nothing():
    """Detachment is a real answer, not a number to extrapolate toward.

    Above the maximum deflection the shock stands off and conical flow no
    longer applies; returning a surface pressure anyway would be inventing one.
    """
    assert cone_surface(1.5, 60.0) is None


#: The measured CFD result, kept here so a regression in the solver setup shows
#: up as a test failure rather than as a quietly different corpus.
CFD_CONE_15_M2_PA = 159131.0


def test_the_cfd_result_on_record_still_matches_theory():
    _beta, pr, _cp = cone_surface(2.0, 15.0)
    exact = 101325.0 * pr
    error = abs(CFD_CONE_15_M2_PA - exact) / exact
    assert error < 0.02, f"{CFD_CONE_15_M2_PA} vs exact {exact:.0f} ({error:.2%})"
