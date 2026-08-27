"""Buckling of the skin, which governs a thin shell long before yield does.

Every structural check in this project compared a stress against a yield
allowable. For a monocoque cylinder in compression that is the wrong failure
mode: the skin does not tear, it goes unstable and folds, at a stress that can
be an order of magnitude lower.

The packet that prompted this reported 67.8 MPa of combined skin stress against
a 700 MPa yield allowable and called the margin 10.3. The same wall -- 0.80 mm
at 335 mm radius -- has a buckling margin of 1.79. The design survives, but it
survives by a factor the packet was not reporting and for a reason it was not
modelling.

The knockdowns are the empirical lower bounds from NASA SP-8007. They exist
because classical shell theory overpredicts real cylinders by factors of two to
five: an imperfection a fraction of a wall thickness deep is enough, and no
amount of analysis of a perfect cylinder finds that.
"""

import math

import pytest

from cadflow.shell_buckling import (
    check, classical_axial_stress_pa, knockdown_bending,
    knockdown_compression, wall_for_buckling_m)


def test_classical_stress_matches_the_formula():
    """sigma = E t / (r sqrt(3(1-nu^2))), checked independently.

    Pinned against the arithmetic rather than a remembered number, since the
    knockdown is applied to this and an error here scales everything.
    """
    E, r, t, nu = 200e9, 0.335, 0.0008, 0.33
    got = classical_axial_stress_pa(E, r, t, nu)
    want = E * t / (r * math.sqrt(3.0 * (1.0 - nu * nu)))
    assert got == pytest.approx(want, rel=1e-12)
    assert got / 1e6 == pytest.approx(292.1, abs=1.0)


def test_the_knockdown_is_severe_and_gets_worse_as_the_shell_thins():
    """This is the whole reason the module exists.

    A perfect-cylinder calculation would clear a wall that folds in practice.
    At R/t = 419 the allowable is about a third of classical, and thinner is
    worse.
    """
    thin = knockdown_compression(0.5, 0.0008)     # R/t = 625
    thick = knockdown_compression(0.335, 0.0024)  # R/t = 140
    assert 0.2 < thin < 0.35
    assert thick > thin
    assert knockdown_compression(0.335, 0.0008) == pytest.approx(0.350, abs=0.01)


def test_bending_is_less_imperfection_sensitive_than_compression():
    """0.731 against 0.901, and the ordering has to survive into the allowable.

    Under bending only part of the circumference carries peak compression, so
    an imperfection is less likely to sit where it does damage. Treating the two
    identically would size the vehicle for a load case it does not have.
    """
    for r, t in ((0.335, 0.0008), (0.5, 0.0024), (1.0, 0.003)):
        assert knockdown_bending(r, t) > knockdown_compression(r, t)
    res = check(axial_mpa=10.0, bending_mpa=10.0, radius_m=0.335,
                wall_m=0.0008, youngs_pa=200e9)
    assert res.allowable_bending_mpa > res.allowable_compression_mpa


def test_buckling_governs_over_yield_for_this_vehicle():
    """The finding, pinned so it cannot quietly stop being true.

    67.8 MPa of skin stress against a 700 MPa yield allowable reads as a margin
    of ten. The buckling margin is under two.
    """
    res = check(axial_mpa=25.8, bending_mpa=42.1, radius_m=0.335,
                wall_m=0.0008, youngs_pa=200e9)
    assert res.passes
    assert 1.5 < res.margin < 2.2, res.margin
    yield_margin = 700.4 / (25.8 + 42.1)
    assert yield_margin > 4.0 * res.margin, (yield_margin, res.margin)
    assert res.governs == "bending"


def test_a_shell_that_buckles_is_reported_as_failing():
    """The check has to be capable of saying no.

    An interaction above one means the skin folds, and nothing about the stress
    being far below yield changes that.
    """
    res = check(axial_mpa=60.0, bending_mpa=60.0, radius_m=0.75,
                wall_m=0.0006, youngs_pa=70e9)
    assert not res.passes
    assert res.interaction > 1.0
    assert res.margin < 1.0


def test_sizing_inverts_the_check_consistently():
    """The wall the sizer returns must be the wall the checker accepts.

    The knockdown depends on the thickness being solved for, so there is no
    closed-form inverse and the bisection could drift from the forward check
    without either of them looking wrong on its own.
    """
    r, E = 0.4, 200e9
    axial_n, moment_nm = 120_000.0, 25_000.0
    t = wall_for_buckling_m(axial_n=axial_n, moment_nm=moment_nm,
                            radius_m=r, youngs_pa=E, target_margin=1.4)
    assert t is not None
    area = 2.0 * math.pi * r * t
    modulus = math.pi * r * r * t
    res = check(axial_mpa=(axial_n / area) / 1e6,
                bending_mpa=(moment_nm / modulus) / 1e6,
                radius_m=r, wall_m=t, youngs_pa=E)
    assert res.margin == pytest.approx(1.4, rel=0.02), res.margin


def test_an_impossible_load_returns_nothing_rather_than_a_wall():
    """No thickness within the cap survives, so none is offered.

    Returning the cap regardless would hand the design loop a wall that does
    not work and look like a successful sizing.
    """
    assert wall_for_buckling_m(axial_n=5e7, moment_nm=5e7, radius_m=3.0,
                               youngs_pa=70e9, max_wall_m=0.002) is None


def test_pressure_stabilisation_is_declined_out_loud():
    """Not crediting it is conservative; not saying so would not be.

    A pressurised tank buckles well above this. An unpressurised interstage does
    not, and claiming the tank's benefit for it is the dangerous direction.
    """
    res = check(axial_mpa=10.0, bending_mpa=10.0, radius_m=0.5, wall_m=0.002,
                youngs_pa=70e9)
    assert any("pressure" in n.lower() for n in res.notes)


def test_a_thick_wall_is_flagged_as_outside_the_thin_shell_regime():
    """Below R/t of 20 the correlation stops describing the failure mode."""
    res = check(axial_mpa=10.0, bending_mpa=5.0, radius_m=0.1, wall_m=0.02,
                youngs_pa=70e9)
    assert any("R/t" in n and "yield" in n for n in res.notes)
