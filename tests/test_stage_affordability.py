"""Can the design loop reject an architecture physics does not allow?

Architecture selection in the planner is `if gross < best_plan.gross_kg` --
lowest gross wins, full stop. Gross mass cannot see whether a stage can afford
the tank it needs, and it never will: a structural coefficient is a fraction, so
structure scales with propellant, while minimum gauge scales with nothing. Below
some size the two cross and a stage's tank ends alone cost more than its entire
structural allowance. The optimiser picks that architecture every time, because
every number it reads says it is lighter.

This is the knob that lets the loop say otherwise, and the step that uses it.
The step matters as much as the knob: for 25 kg to 4,000 km there is no shorter
stack that closes the mission, so the useful output is not a repair but a stated
conflict -- the mission is over-specified for the technology. A loop that
returned silently there would report a converged design whose smallest stage
cannot be built, with no record that an alternative was looked for.
"""

import pytest

from cadflow.autodesign import Knobs, _afford_tankage, _tankage_shares


class _Stage:
    def __init__(self, prop, struct):
        self.prop_mass_kg = prop
        self.struct_mass_kg = struct


class _Plan:
    def __init__(self, stack):
        self.stack = stack


class _Ev:
    def __init__(self, stack, gross, stages):
        self.plan = _Plan(stack)
        self.gross_kg = gross
        self.stages = stages
        self.feasible = True
        self.violations = []


def _four_stage():
    return _Ev([_Stage(1006.6, 349.5), _Stage(261.7, 90.9),
                _Stage(68.0, 23.6), _Stage(23.9, 8.3)], 1857.5, 4)


def _two_stage():
    return _Ev([_Stage(1100.0, 380.0), _Stage(280.0, 97.0)], 1900.0, 2)


def test_the_knob_exists_and_defaults_to_unconstrained():
    """None means the planner chooses freely, as it always did."""
    assert Knobs().max_stages is None


def test_the_planner_honours_a_stage_cap():
    """And it has to be in the cache key, or the second call lies.

    plan() is memoised on every argument that changes its result. A cap left out
    of the key would make the first call's architecture the answer to every
    subsequent call with a different cap -- which is exactly how the loop would
    come to believe a shorter stack was impossible without ever trying one.
    """
    from cadflow import planner as pl

    pl.clear_plan_cache()
    free = pl.plan(300.0, 10.0)
    capped = pl.plan(300.0, 10.0, max_stages=1)
    if free is None:
        pytest.skip("mission does not close at all")
    assert capped is None or capped.stages <= 1
    # and asking again unconstrained must not return the capped answer
    again = pl.plan(300.0, 10.0)
    assert again is not None and again.stages == free.stages


def test_an_impossible_cap_returns_no_plan_rather_than_a_wrong_one():
    """Zero stages is not an architecture."""
    from cadflow import planner as pl

    pl.clear_plan_cache()
    assert pl.plan(4000.0, 25.0, max_stages=0) is None


def test_the_shares_are_computed_at_each_stage_own_radius():
    """Dome mass goes as R^2 at fixed gauge.

    Using one radius for a tapered stack would overstate the upper stages and
    could invent an infeasibility that is not there -- which is the opposite
    error from the one this check exists to catch, and just as bad.
    """
    ev, kn = _four_stage(), Knobs(skin_material="inconel-718")
    shares = _tankage_shares(ev, kn)
    assert shares is not None and len(shares) == 4
    # tankage share rises monotonically toward the top of the stack
    fracs = [s["fraction_of_allowance"] for s in shares]
    assert fracs == sorted(fracs)
    assert not shares[-1]["feasible"]


def test_the_alloy_the_loop_chose_is_the_alloy_the_domes_are_weighed_in():
    """Inconel domes are three times aluminium's at the same gauge.

    A verdict that ignored the material knob would be answering about a vehicle
    the loop is not designing.
    """
    ev = _four_stage()
    heavy = _tankage_shares(ev, Knobs(skin_material="inconel-718"))
    light = _tankage_shares(ev, Knobs(skin_material="al-6061-t6"))
    assert heavy[-1]["fraction_of_allowance"] > light[-1]["fraction_of_allowance"]


def test_an_affordable_stack_produces_no_intervention():
    """The step must be quiet on a design that is fine."""
    ev, kn = _two_stage(), Knobs(skin_material="al-6061-t6")
    shares = _tankage_shares(ev, kn)
    assert all(s["feasible"] for s in shares)
    assert _afford_tankage(10.0, 100.0, kn, ev, None) is None


def test_an_unaffordable_stack_with_no_shorter_option_reports_a_conflict():
    """The output that matters for this mission.

    Nothing shorter than four stages reaches 4,000 km, and four stages cannot
    pay for its tanks. Returning None there would leave a converged design whose
    smallest stage cannot be built and no record that an alternative was sought.
    """
    ev, kn = _four_stage(), Knobs(skin_material="inconel-718")
    got = _afford_tankage(25.0, 4000.0, kn, ev, None)
    assert got is not None and got["fixed"] is False
    assert "cannot afford its own tankage" in got["conflict"]
    # and it says what it tried, not merely that it failed
    assert got["alternatives"]
    assert all("stage(s):" in a for a in got["alternatives"])


def test_the_conflict_names_the_stage_and_the_number():
    """A conflict a reader cannot act on is a complaint."""
    ev, kn = _four_stage(), Knobs(skin_material="inconel-718")
    got = _afford_tankage(25.0, 4000.0, kn, ev, None)
    assert "stage 4" in got["conflict"]
    assert "%" in got["conflict"]


def test_missing_pressurisation_leaves_the_loop_alone():
    """This is an addition, not a gate on designing anything."""
    ev = _Ev([], 100.0, 0)
    assert _tankage_shares(ev, Knobs()) is None
    assert _afford_tankage(1.0, 1.0, Knobs(), ev, None) is None
