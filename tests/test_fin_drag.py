"""Fin drag, which the trajectory had been flying without.

The vehicle's drag coefficient is a function of nose shape and fineness. That
describes a body of revolution, and the vehicle this program builds is not one:
it carries four fins whose combined planform is three and a half times the
body's frontal area. The fins exist because the stability model needs them, and
the loop that adds them never charged the trajectory for them, so every apogee
reported so far is a finless rocket's apogee.

Measured on the packet vehicle the effect is about 6% of body drag and 42 km of
apogee on a 4,000 km mission -- around one percent, and always in the
optimistic direction. Small enough that reporting the correction is
proportionate and large enough that pretending it is zero is not.
"""

import math

import pytest

from cadflow.fin_drag import (
    REPRESENTATIVE_CF, fin_drag, planform_area_m2, wave_drag_coefficient)

# The packet vehicle's fin set, after the control-authority trade.
FINS = dict(span_m=0.615, root_chord_m=0.670, tip_chord_m=0.335,
            thickness_m=0.005, n_fins=4, body_radius_m=0.335)


def test_the_fins_are_not_a_small_correction_to_the_reference_area():
    """Three and a half times the body frontal area, in planform.

    This is why the omission mattered. A fin set that was a few percent of the
    body could be neglected with a clear conscience; this one cannot.
    """
    planform = planform_area_m2(span_m=0.615, root_chord_m=0.670,
                                tip_chord_m=0.335, n_fins=4)
    frontal = math.pi * 0.335 ** 2
    assert planform / frontal > 3.0


def test_wave_drag_is_zero_below_mach_one():
    """There is no wave drag to have, and the formula must not invent any."""
    assert wave_drag_coefficient(0.5, 0.06) == 0.0
    assert wave_drag_coefficient(0.99, 0.06) == 0.0
    assert wave_drag_coefficient(1.5, 0.06) > 0.0


def test_wave_drag_follows_thickness_squared_and_falls_with_mach():
    """4 (t/c)^2 / sqrt(M^2 - 1), checked against the arithmetic.

    Both dependencies matter: thickness squared is why a thin fin is cheap
    supersonically, and the Mach term is why the penalty is worst just above
    Mach 1 rather than at the top of the ascent.
    """
    thin = wave_drag_coefficient(2.0, 0.03)
    thick = wave_drag_coefficient(2.0, 0.06)
    assert thick / thin == pytest.approx(4.0, rel=1e-9)
    assert wave_drag_coefficient(4.0, 0.06) < wave_drag_coefficient(2.0, 0.06)
    assert wave_drag_coefficient(2.0, 0.06) == pytest.approx(
        4.0 * 0.06 ** 2 / math.sqrt(3.0), rel=1e-9)


def test_the_transonic_singularity_is_clamped_not_returned():
    """Linearised theory blows up at Mach 1; that is a model failure, not physics.

    Returning an enormous number there would let a trajectory integrator take a
    step through Mach 1 and record a drag spike that does not exist.
    """
    at_one = wave_drag_coefficient(1.0001, 0.06)
    assert math.isfinite(at_one)
    assert at_one < 100.0 * wave_drag_coefficient(2.0, 0.06)


def test_friction_dominates_for_a_thin_fin():
    """The packet's fins are 5 mm on a half-metre chord, so 1% thick.

    A model that reported wave drag as the larger term for a fin this thin
    would be wrong about which lever matters -- thinning these fins buys almost
    nothing, while shrinking their area buys everything.
    """
    fd = fin_drag(mach=2.0, **FINS)
    assert fd.cd_friction > 10.0 * fd.cd_wave
    assert fd.cd_total == pytest.approx(fd.cd_friction + fd.cd_wave, rel=1e-12)


def test_the_total_is_referenced_to_body_frontal_area():
    """So it adds directly to the vehicle Cd the trajectory integrates.

    Fin area is the natural aerodynamic denominator and the wrong one here;
    mixing the two would overstate the correction by the 3.5x area ratio.
    """
    fd = fin_drag(mach=2.0, **FINS)
    assert fd.reference_m2 == pytest.approx(math.pi * 0.335 ** 2, rel=1e-9)
    expected_friction = REPRESENTATIVE_CF * fd.wetted_m2 / fd.reference_m2
    assert fd.cd_friction == pytest.approx(expected_friction, rel=1e-9)


def test_the_measured_penalty_is_a_few_percent_of_body_drag():
    """About 6%, which is the number the packet has to carry.

    Pinned so that a change to the fin sizing or the reference area cannot
    quietly move it without this failing.
    """
    from cadflow.planner import drag_coefficient

    fd = fin_drag(mach=2.0, **FINS)
    body = drag_coefficient("vonkarman", 3.0)
    assert 0.03 < fd.cd_total / body < 0.10


def test_the_estimate_declares_itself_a_floor():
    """Interference drag is real, positive, and not modelled.

    Presenting a partial drag build-up as complete is the same error as
    presenting a yield margin as a buckling margin.
    """
    fd = fin_drag(mach=2.0, **FINS)
    assert any("floor" in n for n in fd.notes)
    assert any("Interference" in n for n in fd.notes)


def test_degenerate_geometry_is_refused():
    with pytest.raises(ValueError):
        planform_area_m2(span_m=0.0, root_chord_m=0.5, tip_chord_m=0.2, n_fins=4)
    with pytest.raises(ValueError):
        fin_drag(mach=2.0, **{**FINS, "body_radius_m": 0.0})
