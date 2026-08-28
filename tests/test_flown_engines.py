"""Engine thrust-to-weight, placed against engines that have flown.

The engine turned out to be about half of stage structure -- twice the whole
shell -- so the 60 in structural_sizing carries more of the answer than any
other single number in it. This is the same treatment flown_envelope gives the
structural coefficient: place the assumption against hardware rather than
asserting it.

Two tests here exist because I got the corresponding claims wrong first. I said
flown engines run "80 to 180" and argued the model was badly pessimistic; the
low end is 66. And I hypothesised that thrust-to-weight scales with thrust,
which would have neatly explained why the solved structural coefficient does not
improve with vehicle size. It does not scale, and the neat explanation was
wrong.
"""

import pytest

from cadflow.flown_engines import (
    FLOWN_ENGINES, check, flown_ratios, scales_with_thrust, thrust_to_weight)


def test_the_ratios_are_computed_from_thrust_and_mass_not_quoted():
    """Every number in the table is derived from two published figures.

    A quoted ratio can be anything; thrust over weight can be checked.
    """
    assert thrust_to_weight(845_000.0, 470.0) == pytest.approx(183.0, abs=1.0)
    assert thrust_to_weight(2_280_000.0, 3527.0) == pytest.approx(66.0, abs=1.0)
    assert len(flown_ratios()) == len(FLOWN_ENGINES)


def test_the_flown_range_is_66_to_183_not_80_to_180():
    """The correction. I claimed the low end was 80; the RS-25 is 66.

    It matters because the argument built on it -- that a T/W of 60 is badly
    pessimistic -- was too strong. 60 sits just under a real floor of 66, not
    far under a supposed floor of 80.
    """
    r = sorted(flown_ratios())
    assert r[0] == pytest.approx(66.0, abs=1.0)
    assert r[-1] == pytest.approx(183.0, abs=1.0)
    assert r[len(r) // 2] == pytest.approx(82.0, abs=1.0)


def test_the_projects_assumption_is_reported_as_below_the_flown_floor():
    """60 is conservative, and the report has to say by how much.

    Not a fault -- a conservative engine is a defensible choice -- but at half
    of stage structure it is the most expensive conservatism in the budget, so
    it should be visible rather than buried in a module constant.
    """
    from cadflow.structural_sizing import ENGINE_THRUST_TO_WEIGHT

    v = check(ENGINE_THRUST_TO_WEIGHT)
    assert not v.inside
    assert v.assumed < v.flown_min
    assert "below every engine" in v.note


def test_a_mid_range_assumption_reads_as_inside():
    """The check must be able to come out the other way."""
    v = check(100.0)
    assert v.inside and 0.0 < v.percentile < 1.0


def test_claiming_better_than_anything_flown_is_flagged():
    v = check(250.0)
    assert not v.inside
    assert "exceeds every engine flown" in v.note


def test_thrust_to_weight_does_not_scale_with_thrust():
    """The appealing explanation, refuted by its own data.

    A thrust-scaled engine model would explain why this project's structural
    coefficient fails to improve with vehicle size. The table says no: Merlin at
    845 kN reaches 183 while the RD-180 at 3830 kN sits at 71, and Rutherford at
    24 kN beats both relative to its size.
    """
    got = scales_with_thrust()
    assert not got["supports_scaling"]
    assert abs(got["pearson_r"]) < 0.7


def test_a_zero_mass_engine_is_refused():
    with pytest.raises(ValueError):
        thrust_to_weight(1000.0, 0.0)
