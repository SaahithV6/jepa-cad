"""Propellant slosh, which the bending-mode section said was missing.

Liquid in a partly full tank has lateral modes of its own. They carry a real
fraction of the vehicle's mass and are almost undamped, so one that lands on a
structural frequency or on control bandwidth couples -- and that coupling
destroyed several early launch vehicles. It is a routine design check that this
program did not have.

The two properties worth pinning are the ones a plausible-looking wrong model
would get wrong. Frequency scales with the square root of *axial acceleration*,
not gravity, so a vehicle under 4.5 g sloshes at twice the rate a ground
calculation suggests. And the participating mass has known limits at both ends:
a shallow pan sloshes almost entirely, a deep tank barely at all.
"""

import math

import pytest

from cadflow.slosh import (
    G0, first_eigenvalue, frequency_hz, mass_fraction, separation_from,
    tank_mode)


def test_the_eigenvalue_is_solved_not_quoted():
    """1.8412 is the first zero of J1', and that is checkable.

    This project has already spent a day on a remembered constant that was
    wrong. Root-finding it costs nothing.
    """
    from scipy.special import jvp

    lam = first_eigenvalue()
    assert jvp(1, lam) == pytest.approx(0.0, abs=1e-10)
    assert lam == pytest.approx(1.8412, abs=1e-4)


def test_frequency_follows_axial_acceleration_not_gravity():
    """Under 4.5 g the modes sit at sqrt(4.5) times the ground value.

    Using 9.81 in flight would place every slosh mode a factor of two low, and
    could report clearance from a structural mode that in flight coincides
    with it.
    """
    ground = frequency_hz(0.335, 1.0, G0)
    flight = frequency_hz(0.335, 1.0, 4.5 * G0)
    assert flight / ground == pytest.approx(math.sqrt(4.5), rel=1e-6)


def test_the_shallow_limit_matches_the_closed_form():
    """A shallow layer slings 2/(lam^2 - 1) of its mass, about 84%.

    The limit follows from tanh(x) -> x, so it tests the algebra rather than
    the implementation agreeing with itself.
    """
    lam = first_eigenvalue()
    expected = 2.0 / (lam * lam - 1.0)
    assert mass_fraction(1.0, 0.001) == pytest.approx(expected, rel=1e-3)
    assert expected == pytest.approx(0.8368, abs=1e-3)


def test_a_deep_tank_barely_sloshes():
    """Only a surface layer moves, and the fraction falls off as R/h.

    A model that reported most of a deep tank as sloshing mass would put an
    enormous participating mass at a low frequency and manufacture a coupling
    problem that does not exist.
    """
    assert mass_fraction(1.0, 10.0) < 0.06
    assert mass_fraction(1.0, 1.0) > mass_fraction(1.0, 3.0)
    assert mass_fraction(1.0, 3.0) > mass_fraction(1.0, 10.0)


def test_a_fuller_tank_sloshes_faster_and_relatively_less():
    """Frequency rises with depth and saturates; participation falls.

    Both follow from the same tanh, and getting one inverted would be easy to
    miss without checking them together.
    """
    shallow = tank_mode("t", radius_m=0.5, propellant_kg=1000.0,
                        bulk_density=1000.0, fill_ratio=0.2,
                        axial_accel_m_s2=4.0 * G0)
    full = tank_mode("t", radius_m=0.5, propellant_kg=1000.0,
                     bulk_density=1000.0, fill_ratio=1.0,
                     axial_accel_m_s2=4.0 * G0)
    assert full.frequency_hz > shallow.frequency_hz
    assert full.slosh_mass_fraction < shallow.slosh_mass_fraction


def test_free_fall_is_refused_rather_than_answered():
    """With no settling acceleration the model does not apply.

    Returning zero, or a number computed from 9.81, would both be wrong in a
    coast phase where the propellant is not against the tank floor at all.
    """
    with pytest.raises(ValueError, match="free fall"):
        frequency_hz(0.5, 1.0, 0.0)
    with pytest.raises(ValueError, match="empty tank"):
        frequency_hz(0.5, 0.0, 4.0 * G0)


def test_coincidence_with_a_structural_mode_is_called_out():
    """Proximity is the whole question, because slosh is barely damped.

    A ratio near one has to read as a coupling risk rather than as two numbers
    that happen to be similar.
    """
    mode = tank_mode("t", radius_m=0.5, propellant_kg=1000.0,
                     bulk_density=1000.0, fill_ratio=0.5,
                     axial_accel_m_s2=4.0 * G0)
    near = separation_from(mode, mode.frequency_hz * 1.05)
    far = separation_from(mode, mode.frequency_hz * 20.0)
    assert near["coupled"] and "couple" in near["verdict"]
    assert not far["coupled"]
    assert "separated" in far["verdict"]


def test_the_real_vehicle_is_clear_of_its_bending_mode():
    """2.5 Hz of slosh against 50 Hz of first bending.

    Recorded because the separation is what makes the design acceptable, and if
    a future change shortens the vehicle or thins the wall enough to drag the
    bending mode down, this should fail rather than pass quietly.
    """
    mode = tank_mode("stage 1", radius_m=0.335, propellant_kg=1200.0,
                     bulk_density=1030.0, fill_ratio=0.5,
                     axial_accel_m_s2=4.5 * G0)
    assert 1.5 < mode.frequency_hz < 4.0
    assert not separation_from(mode, 50.4)["coupled"]
