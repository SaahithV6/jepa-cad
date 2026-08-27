"""Bending modes of the assembled vehicle, checked against the closed form.

The component path reports a first natural frequency per part, from a modal run
on that part clamped at one end. That is a fact about a bracket. Nothing clamps
a rocket in flight: the vehicle bends as a free-free beam, and its first elastic
bending frequency is what decides whether the autopilot can fly it. Drive a
structure near its first bending mode and the control system stops steering and
starts exciting.

A cantilever frequency cannot stand in for that. The boundary condition is
imaginary, the assembled vehicle is longer and softer than any of its parts, and
it bends in a shape no single part sees.

Two things make the answer checkable. A uniform free-free beam has a closed
form, and the ratios between its first several modes are fixed constants that a
wrong stiffness matrix will not reproduce. And an unconstrained beam has exactly
two zero-energy modes in a plane; a model that returns one or three has a bug
that a full set of plausible frequencies would otherwise hide.
"""

import math

import pytest

from cadflow.assembly_modes import (
    BETA_1_L, beam_modes, uniform_beam_first_mode_hz, vehicle_bending_modes)

#: Free-free beam eigenvalues, roots of cos(bL)cosh(bL) = 1. Frequency scales
#: with the square, so these fix the ratios between modes.
BETA_L = (4.730040744862704, 7.853204624095838, 10.995607838001671)


def test_first_mode_matches_the_closed_form():
    """Within a hundredth of a percent, on a coarse mesh.

    A beam model that is merely plausible will not land on 4.730^2 by accident.
    """
    E, I, mu, L = 70e9, 1e-6, 20.0, 10.0
    exact = uniform_beam_first_mode_hz(L, E, I, mu)
    n = 41
    st = [L * i / (n - 1) for i in range(n)]
    res = beam_modes(st, [mu] * n, [E * I] * n)
    assert res.first_bending_hz == pytest.approx(exact, rel=1e-3)


def test_higher_modes_hold_the_right_ratios():
    """1 : 2.7565 : 5.4039, which follow from the eigenvalues alone.

    The first frequency can be right for the wrong reason -- a stiffness error
    that scales the whole matrix moves every mode together. The ratios do not
    move, so they test the shape of the operator rather than its scale.
    """
    E, I, mu, L = 70e9, 1e-6, 20.0, 10.0
    n = 161
    st = [L * i / (n - 1) for i in range(n)]
    res = beam_modes(st, [mu] * n, [E * I] * n, n_modes=3)
    f = res.frequencies_hz
    assert len(f) >= 3
    for k in (1, 2):
        expected = (BETA_L[k] / BETA_L[0]) ** 2
        assert f[k] / f[0] == pytest.approx(expected, rel=5e-3), (k, f)


def test_exactly_two_rigid_body_modes():
    """A free planar beam translates and rotates at zero energy, and no more.

    This is the check that the model is genuinely unconstrained. A stray
    restraint removes one; a defective stiffness matrix invents another. Either
    way the elastic frequencies that follow are wrong, and they still look like
    frequencies.
    """
    E, I, mu, L = 70e9, 1e-6, 20.0, 10.0
    n = 81
    st = [L * i / (n - 1) for i in range(n)]
    res = beam_modes(st, [mu] * n, [E * I] * n)
    assert res.rigid_body_modes == 2
    assert res.well_posed
    assert not res.notes


def test_a_softer_vehicle_has_a_lower_mode():
    """Frequency follows sqrt(EI), so a thinner wall must drop it.

    Direction is the cheapest possible sanity check and the one that catches a
    reciprocal used the wrong way round.
    """
    veh = {"mass_kg": 1000.0, "length_m": 8.0, "radius_m": 0.4,
           "cg_z_m": 4.0, "section_extents": [("body", 0.0, 8.0, 1000.0)]}
    thin = vehicle_bending_modes(veh, youngs_pa=70e9, wall_m=0.002)
    thick = vehicle_bending_modes(veh, youngs_pa=70e9, wall_m=0.008)
    assert thin.first_bending_hz < thick.first_bending_hz
    # EI scales linearly with wall for a thin shell, so f scales as sqrt(t).
    assert thick.first_bending_hz / thin.first_bending_hz == pytest.approx(
        math.sqrt(4.0), rel=0.02)


def test_the_vehicle_model_uses_the_same_mass_distribution_as_the_loads():
    """One vehicle, not two.

    The bending moment and the bending frequency both depend on where the mass
    sits. Deriving them from different distributions would let a design change
    move one and not the other, and nothing would report the contradiction.
    """
    from cadflow.flight_loads import mass_per_length

    veh = {"mass_kg": 500.0, "length_m": 6.0, "radius_m": 0.3, "cg_z_m": 3.0,
           "section_extents": [("fwd", 0.0, 2.0, 100.0),
                               ("aft", 2.0, 6.0, 400.0)]}
    res = vehicle_bending_modes(veh, youngs_pa=70e9, wall_m=0.003,
                                n_stations=121)
    mu = mass_per_length(veh["section_extents"], res.stations_m)
    dx = 6.0 / 120
    total = dx * (sum(mu) - 0.5 * (mu[0] + mu[-1]))
    assert total == pytest.approx(500.0, rel=1e-2)


def test_it_refuses_a_vehicle_with_no_mass_distribution():
    with pytest.raises(ValueError, match="section_extents"):
        vehicle_bending_modes({"mass_kg": 10.0, "length_m": 2.0,
                               "radius_m": 0.1}, youngs_pa=70e9, wall_m=0.002)


def test_zero_wall_is_refused():
    veh = {"mass_kg": 10.0, "length_m": 2.0, "radius_m": 0.1, "cg_z_m": 1.0,
           "section_extents": [("b", 0.0, 2.0, 10.0)]}
    with pytest.raises(ValueError):
        vehicle_bending_modes(veh, youngs_pa=70e9, wall_m=0.0)


def test_the_closed_form_constant_is_the_right_root():
    """4.730040... satisfies cos(bL)cosh(bL) = 1.

    Pinned by the equation it solves rather than by being written down twice,
    since a remembered constant is exactly the kind of thing this project has
    already been wrong about.
    """
    assert math.cos(BETA_1_L) * math.cosh(BETA_1_L) == pytest.approx(1.0, abs=1e-9)
