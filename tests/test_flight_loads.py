"""Bending along the assembled vehicle, checked against a closed form.

Every structural check before this one was a component check: a part meshed
alone, gripped at one end and pushed on the other. That answers a question about
a coupon. A launch vehicle fails as a stack -- flying at incidence through
max-Q, with the air pushing where the mass is not -- and the station carrying
the largest bending moment is typically in the middle of the body, where no
component analysis is looking.

The load set has to balance, and that is what makes the answer checkable. A
free vehicle reacts side load by accelerating, not by pushing against a
support, so integrating the net load from nose to tail must bring shear and
moment back to zero at the aft end. An unbalanced distribution still draws a
smooth, entirely plausible moment curve, so closure is asserted rather than
eyeballed.

There is also an exact answer to compare against. A uniform free rod carrying a
point load at its tip has peak bending moment 4FL/27 at one third of its length,
which the solver has to reproduce without being told.
"""

import math

import pytest

from cadflow.flight_loads import (
    PointLoad, mass_per_length, skin_stress_mpa, solve)


def uniform_rod(mass_kg: float = 1000.0, length_m: float = 10.0) -> dict:
    """A single uniform section, which is the case with a closed-form answer."""
    return {
        "mass_kg": mass_kg,
        "length_m": length_m,
        "radius_m": 0.5,
        "cg_z_m": length_m / 2.0,
        "section_extents": [("rod", 0.0, length_m, mass_kg)],
    }


def test_uniform_rod_matches_the_closed_form():
    """Peak moment 4FL/27 at L/3, from a tip load on a free rod.

    This is the check that the solver is solving the right problem rather than
    merely a smooth one. Both the magnitude and the location follow from the
    inertial relief distribution; get that wrong in any way and the peak moves.
    """
    F, L = 5000.0, 10.0
    res = solve(uniform_rod(1000.0, L), [PointLoad("tip", 0.0, F)],
                n_stations=2001)
    assert res.peak_moment_nm == pytest.approx(4.0 * F * L / 27.0, rel=2e-3)
    assert res.peak_moment_station_m == pytest.approx(L / 3.0, abs=0.02)


def test_the_load_set_closes():
    """Shear and moment must return to zero at a free aft end.

    They will not if the inertial relief is distributed wrongly, and the moment
    curve will look no different. An early version integrated the mass with
    rectangle sums while integrating the load with trapezoids, and left 466 N of
    shear on 100 kN applied while drawing a perfectly reasonable curve.
    """
    F, L = 5000.0, 10.0
    res = solve(uniform_rod(1000.0, L), [PointLoad("tip", 0.0, F)],
                n_stations=1001)
    assert res.balanced
    assert abs(res.closure_shear_n) < 1e-3 * F
    assert abs(res.closure_moment_nm) < 1e-3 * F * L
    assert res.moment_nm[0] == pytest.approx(0.0, abs=1e-9)


def test_closure_converges_under_refinement():
    """First order, which is what a point load allows.

    Trapezoid integration is second-order on a smooth integrand, and this one is
    not smooth: a concentrated load makes the shear jump discontinuously, and no
    grid resolves a jump better than to within a cell. The residual therefore
    falls as 1/N, and expecting 1/N^2 here was an error in the expectation
    rather than in the solver -- 98.8 N m at 251 stations became 24.9 at 1001,
    a ratio of 3.96 against the 4.0 the refinement bought.

    What matters is that it shrinks at all. A residual that stays put under
    refinement is a modelling error, and this code had one: point loads snapped
    to the nearest node left the shear closing to 1e-11 while the moment sat at
    a fixed offset, because a load displaced half a cell is a moment no
    refinement removes.
    """
    F, L = 5000.0, 10.0
    veh = uniform_rod(1000.0, L)
    coarse = abs(solve(veh, [PointLoad("tip", 0.0, F)], n_stations=251
                       ).closure_moment_nm)
    fine = abs(solve(veh, [PointLoad("tip", 0.0, F)], n_stations=1001
                     ).closure_moment_nm)
    assert fine < coarse / 3.5, (coarse, fine)
    # And in absolute terms it must be negligible against the peak the same run
    # reports, or "converging" would be no comfort.
    peak = abs(solve(veh, [PointLoad("tip", 0.0, F)], n_stations=1001
                     ).peak_moment_nm)
    assert fine < 0.01 * peak, (fine, peak)


def test_a_balanced_pair_of_loads_still_bends_the_body():
    """Zero net force is not zero bending.

    Equal and opposite loads at the ends produce no acceleration at all, and a
    check that only looked at net force or at the centre of gravity would call
    this case unloaded. The body is in pure bending.
    """
    F, L = 4000.0, 8.0
    res = solve(uniform_rod(600.0, L),
                [PointLoad("nose", 0.0, F), PointLoad("tail", L, -F)],
                n_stations=1001)
    assert res.lateral_accel_m_s2 == pytest.approx(0.0, abs=1e-9)
    assert abs(res.peak_moment_nm) > 0.05 * F * L
    assert res.balanced


def test_mass_distribution_is_not_the_centre_of_gravity():
    """Two vehicles with one centre of gravity bend differently.

    The mass properties model reports a centre of gravity and an inertia. That
    is enough to fly the vehicle and not enough to bend it: where the mass sits
    along the body decides how much of the aerodynamic load each station has to
    carry past.
    """
    spread = {"mass_kg": 1000.0, "length_m": 10.0, "radius_m": 0.5,
              "cg_z_m": 5.0,
              "section_extents": [("all", 0.0, 10.0, 1000.0)]}
    barbell = {"mass_kg": 1000.0, "length_m": 10.0, "radius_m": 0.5,
               "cg_z_m": 5.0,
               "section_extents": [("fwd", 0.0, 1.0, 500.0),
                                   ("aft", 9.0, 10.0, 500.0)]}
    load = [PointLoad("mid", 5.0, 10_000.0)]
    a = solve(spread, load, n_stations=1001)
    b = solve(barbell, load, n_stations=1001)
    assert a.balanced and b.balanced
    assert abs(b.peak_moment_nm) > 1.3 * abs(a.peak_moment_nm), (
        a.peak_moment_nm, b.peak_moment_nm)


def test_it_refuses_a_vehicle_with_no_mass_distribution():
    """A centre of gravity cannot stand in for the distribution.

    Silently assuming a uniform body would produce a number for any input,
    which is the failure this module exists to avoid.
    """
    with pytest.raises(ValueError, match="section_extents"):
        solve({"mass_kg": 100.0, "length_m": 5.0, "cg_z_m": 2.5}, [])


def test_mass_per_length_integrates_back_to_the_mass():
    """The distribution has to describe the same vehicle the stack does."""
    extents = [("a", 0.0, 3.0, 300.0), ("b", 3.0, 5.0, 200.0)]
    n = 5001
    xs = [5.0 * i / (n - 1) for i in range(n)]
    mu = mass_per_length(extents, xs)
    dx = 5.0 / (n - 1)
    total = dx * (sum(mu) - 0.5 * (mu[0] + mu[-1]))
    assert total == pytest.approx(500.0, rel=1e-3)


def test_axial_and_bending_stress_add_in_the_skin():
    """Sizing against either alone passes sections that fail together.

    Thrust compresses the whole barrel; bending compresses one side of it. The
    windward fibre carries the sum, and a check that looked only at the axial
    term would clear a section the bending term governs.
    """
    st = skin_stress_mpa(moment_nm=70_000.0, axial_n=80_000.0,
                         radius_m=0.5, wall_m=0.0024)
    assert st["combined_mpa"] == pytest.approx(
        st["axial_mpa"] + st["bending_mpa"], rel=1e-9)
    assert st["bending_share"] > 0.4
    # Thin-shell section modulus is pi r^2 t; check the bending term directly
    # rather than trusting the implementation to agree with itself.
    assert st["bending_mpa"] == pytest.approx(
        70_000.0 / (math.pi * 0.5 ** 2 * 0.0024) / 1e6, rel=1e-9)


def test_zero_wall_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        skin_stress_mpa(1000.0, 1000.0, 0.5, 0.0)
