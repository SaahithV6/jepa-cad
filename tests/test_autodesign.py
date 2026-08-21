"""The design loop: find violations across disciplines, then act on them.

The packet could find violations and report them. It could not act, so a design
that ran its payload at 16 g or could not cool its chamber came out as a
well-documented failure rather than as a design.

Each constraint here has a knob, and each knob's direction was measured before
being used rather than reasoned about -- one of them is counter-intuitive.
Raising chamber pressure *improves* regenerative cooling, because coolant flow
rises linearly with pressure while heat load rises as pressure^0.8. It also
raises throat heat flux. So the two thermal constraints oppose each other and
between them define a window rather than a bound, which is why the loop has to
be able to recognise when that window is empty.
"""

import pytest

from cadflow.autodesign import (
    DEFAULT_LIMITS,
    Knobs,
    Violation,
    autodesign,
    evaluate,
    remedy,
)
from generate_propulsion_trajectory_corpus import load_coupling

PAYLOAD, APOGEE = 25.0, 4000.0


@pytest.fixture(scope="module", autouse=True)
def _coupling():
    load_coupling()


def _limits(**over):
    lim = dict(DEFAULT_LIMITS)
    # aeroheating has no knob in this loop, so it would block every run
    lim["skin_temp_k"] = 1e9
    lim.update(over)
    return lim


# --- remedy directions ------------------------------------------------------

def test_excess_acceleration_lowers_thrust_to_weight():
    v = Violation("loads", "peak axial acceleration", 16.0, 10.0,
                  "lower thrust-to-weight")
    out = remedy(Knobs(), [v])
    assert out.twr_by_stage[0] < Knobs().twr_by_stage[0]
    assert out.chamber_bar == Knobs().chamber_bar


def test_excess_throat_flux_lowers_chamber_pressure():
    v = Violation("thermal", "throat heat flux", 120.0, 90.0,
                  "lower chamber pressure")
    out = remedy(Knobs(), [v])
    assert out.chamber_bar < Knobs().chamber_bar


def test_thin_coolant_margin_raises_chamber_pressure():
    """The counter-intuitive one, and the reason the direction was measured."""
    v = Violation("thermal", "coolant margin", 10.0, 25.0,
                  "raise chamber pressure")
    out = remedy(Knobs(), [v])
    assert out.chamber_bar > Knobs().chamber_bar


def test_remedies_respect_their_floors():
    """A runaway correction must not produce a vehicle that cannot fly."""
    v = Violation("loads", "peak axial acceleration", 1e6, 1.0,
                  "lower thrust-to-weight")
    out = remedy(Knobs(), [v])
    assert all(t >= 1.2 for t in out.twr_by_stage)
    v2 = Violation("thermal", "throat heat flux", 1e6, 1.0,
                   "lower chamber pressure")
    assert remedy(Knobs(), [v2]).chamber_bar >= 5.0


# --- evaluation -------------------------------------------------------------

def test_evaluation_covers_every_discipline():
    ev = evaluate(PAYLOAD, APOGEE, Knobs(), _limits())
    assert ev.plan is not None
    assert ev.peak_g > 1.0
    assert ev.throat_flux_mw_m2 > 0.0
    assert ev.gross_kg > PAYLOAD
    assert ev.stages >= 1


def test_a_loose_specification_is_already_feasible():
    ev = evaluate(PAYLOAD, APOGEE, Knobs(),
                  _limits(payload_g=1e9, throat_flux_mw_m2=1e9,
                          coolant_margin_k=-1e9))
    assert ev.feasible, [str(v) for v in ev.violations]


def test_a_tight_acceleration_limit_is_a_violation():
    ev = evaluate(PAYLOAD, APOGEE, Knobs(), _limits(payload_g=5.0))
    assert not ev.feasible
    assert any(v.quantity == "peak axial acceleration" for v in ev.violations)


# --- convergence ------------------------------------------------------------

def test_the_loop_fixes_an_acceleration_violation():
    """The whole point: 16 g on a 10 g payload becomes a design, not a report."""
    res = autodesign(PAYLOAD, APOGEE, limits=_limits(), max_iters=8)
    assert res["converged"], res["history"][-1]["violations"]
    assert res["evaluation"].peak_g <= DEFAULT_LIMITS["payload_g"]
    assert res["history"][0]["peak_g"] > DEFAULT_LIMITS["payload_g"]


def test_fixing_acceleration_costs_gross_mass():
    """Lower thrust-to-weight means longer under gravity. The loop should pay
    that price, and it should be visible rather than hidden."""
    res = autodesign(PAYLOAD, APOGEE, limits=_limits(), max_iters=8)
    assert res["converged"]
    assert res["history"][-1]["gross_kg"] > res["history"][0]["gross_kg"]
    assert res["knobs"].twr_by_stage[0] < Knobs().twr_by_stage[0]


def test_a_tight_flux_limit_drives_pressure_down():
    res = autodesign(PAYLOAD, APOGEE,
                     limits=_limits(throat_flux_mw_m2=20.0), max_iters=8)
    assert res["converged"], res["history"][-1]["violations"]
    assert res["knobs"].chamber_bar < Knobs().chamber_bar
    assert res["evaluation"].throat_flux_mw_m2 <= 20.0


def test_an_already_feasible_design_converges_immediately():
    res = autodesign(PAYLOAD, APOGEE,
                     limits=_limits(payload_g=1e9, throat_flux_mw_m2=1e9,
                                    coolant_margin_k=-1e9), max_iters=5)
    assert res["converged"]
    assert res["iterations"] == 1


# --- over-constrained problems ---------------------------------------------

def test_opposing_constraints_are_reported_as_a_conflict():
    """Not 'did not converge' -- which is true and useless -- but which pair of
    requirements cannot coexist and on which knob."""
    res = autodesign(PAYLOAD, APOGEE,
                     limits=_limits(coolant_margin_k=120.0), max_iters=10)
    assert not res["converged"]
    conflict = res.get("conflict")
    assert conflict is not None, res["history"][-1]
    assert conflict["knob"] == "chamber pressure"
    assert len(conflict["constraints"]) == 2
    assert "opposite directions" in conflict["message"]


def test_a_conflict_stops_the_loop_early():
    """Iterating an over-constrained problem just oscillates; an earlier
    version bounced between 167 and 172 bar until it ran out of iterations."""
    res = autodesign(PAYLOAD, APOGEE,
                     limits=_limits(coolant_margin_k=120.0), max_iters=20)
    assert res["conflict"] is not None
    assert res["iterations"] < 20


def test_an_extreme_mission_stays_physically_coherent():
    """10,000 t to lunar distance. I expected this to be refused; it is not,
    and the answer it gives is the right shape -- six stages at a gross-to-
    payload ratio of 114, which is what the rocket equation demands for that
    delta-v. The property worth asserting is coherence under extrapolation,
    not refusal."""
    res = autodesign(1e7, 500_000.0, limits=_limits(), max_iters=3)
    ev = res["evaluation"]
    assert ev is not None and ev.plan is not None
    assert ev.stages > 2, ev.stages
    assert ev.gross_kg > 10.0 * 1e7, ev.gross_kg
    # and it must still be flying the mission it was asked for
    assert ev.plan.achieved_km > 0.8 * 500_000.0


def test_a_mission_that_cannot_close_is_reported_not_crashed():
    """The planner caps its stage count, so some missions genuinely do not
    close. That has to come back as a violation rather than an exception."""
    from cadflow.autodesign import Knobs, evaluate

    ev = evaluate(1.0, 5_000_000.0, Knobs(), _limits())
    assert ev is not None
    if ev.plan is None:
        assert any(v.discipline == "architecture" for v in ev.violations)
