"""Baffles, and the difference between "needs more" and "cannot work".

The control-authority check left one failure the fins could not fix. Trading
static margin bought gimbal authority and moved the vehicle's pitch mode away
from the slosh mode, but the bandwidth window stayed shut.

That cap is a fact about *undamped* slosh, not about slosh. The five-to-one
separation the window check applies is the rule for a mode with essentially no
damping, which is what a bare tank has. Baffles raise damping by an order of
magnitude and a damped mode can be approached far more closely.

The distinction these tests exist to protect is between two failures that both
end in "no baffle helps". One means the baffles are too small. The other means
the slosh mode sits inside the control band rather than above it, where damping
is irrelevant because damping does not move a frequency. Reporting them the
same way would send a design loop looking for bigger baffles when it needs a
notch filter.
"""

import math

import pytest

from cadflow.slosh_baffles import (
    BARE_TANK_DAMPING, SEPARATION_UNDAMPED, SEPARATION_WELL_DAMPED,
    required_separation, ring_baffle_damping, size_baffles)


def test_a_bare_tank_needs_the_undamped_separation():
    """Half a percent of a percent is effectively zero damping."""
    assert required_separation(BARE_TANK_DAMPING) == pytest.approx(
        SEPARATION_UNDAMPED, rel=0.02)


def test_damping_relaxes_the_separation_requirement():
    """This is the whole mechanism by which a baffle opens a window.

    A fixed factor of five makes a bandwidth window look impossible when it is
    merely un-baffled. Making the requirement a function of damping is what
    turns "different vehicle" into "hardware in the tank".
    """
    loose = required_separation(0.05)
    tight = required_separation(0.0005)
    assert loose < tight
    assert loose == pytest.approx(SEPARATION_WELL_DAMPED)
    assert required_separation(0.5) == pytest.approx(SEPARATION_WELL_DAMPED), (
        "beyond well-damped the requirement must not keep falling")


def test_baffle_damping_dies_with_depth():
    """A baffle only works where the fluid is moving.

    Lateral slosh decays fast below the surface, so a ring a tank radius down
    contributes almost nothing. A model without that exponential would let a
    design loop bury baffles anywhere and still claim the damping.
    """
    shallow = ring_baffle_damping(width_ratio=0.1, depth_ratio=0.1)
    deep = ring_baffle_damping(width_ratio=0.1, depth_ratio=1.0)
    assert deep < 0.1 * shallow
    assert ring_baffle_damping(width_ratio=0.2, depth_ratio=0.2) > \
        ring_baffle_damping(width_ratio=0.05, depth_ratio=0.2)


def test_damping_is_capped_rather_than_summed_without_limit():
    """Stacked baffles interact; summing them is an over-credit.

    The deeper rings sit in fluid the shallower ones have already quietened, so
    an uncapped sum would report damping no real tank achieves.
    """
    assert ring_baffle_damping(width_ratio=0.2, depth_ratio=0.05,
                               n_baffles=8) <= 0.25


def test_a_mode_inside_the_control_band_is_named_not_just_refused():
    """The failure that matters, and the one a bare None would disguise.

    Needing 4.62 Hz of bandwidth against a 2.48 Hz slosh mode is not a baffle
    sizing problem. The mode is inside the band, damping does not move
    frequencies, and no baffle of any size changes it.
    """
    with pytest.raises(ValueError, match="inside the control band"):
        size_baffles(tank_radius_m=0.335, fill_depth_m=1.5, slosh_hz=2.48,
                     required_bandwidth_hz=4.62)


def test_a_reachable_window_gets_baffles_and_a_mass():
    """The check must be able to succeed, and baffles must not look free.

    A design loop that could add damping at no cost would baffle everything.
    """
    d = size_baffles(tank_radius_m=1.0, fill_depth_m=3.0, slosh_hz=6.0,
                     required_bandwidth_hz=2.5)
    assert d is not None
    assert d.damping_ratio > BARE_TANK_DAMPING
    assert d.mass_kg > 0.0
    assert 6.0 / d.achieved_separation >= 2.5
    assert any("empirical" in n or "test" in n for n in d.notes)


def test_the_smallest_workable_baffle_is_chosen():
    """Wider than necessary is mass for nothing.

    The search walks width upward and returns on the first fit, so a generous
    window must not produce a heavier baffle than a marginal one.
    """
    easy = size_baffles(tank_radius_m=1.0, fill_depth_m=3.0, slosh_hz=10.0,
                        required_bandwidth_hz=2.0)
    hard = size_baffles(tank_radius_m=1.0, fill_depth_m=3.0, slosh_hz=6.0,
                        required_bandwidth_hz=3.0)
    assert easy is not None and hard is not None
    assert easy.width_m <= hard.width_m


def test_an_impossible_geometry_returns_nothing():
    """Distinct from the in-band case: here bigger baffles simply run out.

    Returning a design anyway would hand the loop baffles that do not work.
    """
    assert size_baffles(tank_radius_m=1.0, fill_depth_m=3.0, slosh_hz=2.0,
                        required_bandwidth_hz=1.9, max_width_ratio=0.001) is None
