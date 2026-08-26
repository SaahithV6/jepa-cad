"""Nose shape as a design variable, priced by measured drag.

The planner held `NOSE_SHAPE = "ogive"` as a module constant. `plan()` accepted
the argument and the drag model supported three shapes, but nothing in the
design loop could choose between them -- so every vehicle this system produced
flew an ogive because that was the default, not because it was better.

It is a variable now, and the price list is measured rather than assumed: 48
axisymmetric CFD cases over fineness 1.5-4 and Mach 1.5-3, validated on cones to
-1.8% against exact Taylor-Maccoll. For 25 kg to 4,000 km the loop picks von
Karman and the vehicle drops 59.9 kg, 3.2% of gross.

The subtlety these tests pin is *which* drag ratio applies. `shape_factor`
compares closed bodies -- the slender-body integral needs a body that closes at
both ends -- so it prices a tail that every shape shares, and that common tail
pulls every ratio toward 1.0. A launch vehicle has no tail closure; it ends in a
blunt base with an engine. Using the closed-body factor for that decision
understates von Karman by 9-14 points and refuses to price a cone at all.
"""

import pytest

from cadflow.wave_drag import CFD_FOREBODY_FACTOR, cd_multiplier, shape_factor


def test_an_ogive_is_still_exactly_one():
    """The reference shape must not move.

    Every previously-computed vehicle and the existing Cd corpus calibration
    depend on an ogive returning exactly 1.0. If this drifts, designs change
    silently and no test elsewhere would notice.
    """
    for fineness in (1.5, 2.0, 3.0, 4.0):
        assert cd_multiplier("ogive", fineness) == pytest.approx(1.0)
        assert cd_multiplier("ogive", fineness, forebody=True) == pytest.approx(1.0)


def test_forebody_credits_von_karman_more_than_closed_body():
    """The 9-14 point difference, in the direction the physics requires.

    Both numbers are correct for their own body. The forebody one is correct
    for a rocket.
    """
    for fineness in (2.0, 3.0, 4.0):
        closed = cd_multiplier("vonkarman", fineness)
        fore = cd_multiplier("vonkarman", fineness, forebody=True)
        assert fore < closed, (fineness, fore, closed)


def test_cones_are_priceable_by_cfd_and_refused_by_slender_body():
    """A cone's slender-body integral diverges; CFD measures it fine.

    This is the capability the corpus adds, not just a better number: the
    analytic route cannot price a conical nose at all, so the design loop could
    never have considered one.
    """
    with pytest.raises(ValueError):
        shape_factor("cone", 3.0)
    assert 0.9 < cd_multiplier("cone", 3.0, forebody=True) < 1.0


def test_measured_factors_are_ordered_as_the_corpus_found_them():
    """von Karman < cone < ogive, from 16 cases each.

    The cone sits between rather than above: its factor is Mach dependent and
    crosses above 1.0 only at Mach 1.5, so its mean lands below the ogive.
    """
    f = CFD_FOREBODY_FACTOR
    assert f["vonkarman"] < f["cone"] < f["ogive"]
    assert f["ogive"] == 1.0
    assert f["vonkarman"] == pytest.approx(0.801, abs=0.01)


def test_an_unmeasured_shape_is_refused_rather_than_guessed():
    """No factor, no answer.

    Interpolating a shape the corpus never covered would put a fabricated
    number into vehicle sizing, which is the failure this project spent a day
    removing from its stress labels.
    """
    with pytest.raises(ValueError, match="no CFD forebody factor"):
        cd_multiplier("power_series", 3.0, forebody=True)


def test_the_multiplier_only_scales_the_wave_drag_share():
    """A 20% shape advantage is not a 20% vehicle drag advantage.

    Only the wave-drag share of Cd responds to nose shape; the rest is friction
    and base drag that a different nose does not change. With the share at 0.5,
    the measured 0.801 ratio becomes 0.90 on total Cd -- and reporting the raw
    ratio as a vehicle-level saving would overstate it twofold.
    """
    from cadflow.wave_drag import WAVE_DRAG_SHARE

    raw = CFD_FOREBODY_FACTOR["vonkarman"]
    got = cd_multiplier("vonkarman", 3.0, forebody=True)
    expected = (1.0 - WAVE_DRAG_SHARE) + WAVE_DRAG_SHARE * raw
    assert got == pytest.approx(expected)
    assert got > raw, "the multiplier must be closer to 1 than the raw ratio"
