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
    # Aeroheating now HAS a knob -- the loop picks a skin material that
    # survives the temperature -- but these tests predate it and several assert
    # on iteration counts that a material swap would change. Left pinned so each
    # test exercises the constraint it is about.
    lim["skin_temp_k"] = 1e9
    lim.update(over)
    return lim


def _unconstrained(**over):
    """Every limit loosened, including the ones added after these tests.

    `_limits` loosens what existed when it was written. Mass closure came later,
    and it fires on the default 25 kg / 4,000 km vehicle -- correctly, since
    that design is 27.9 kg short of containing its own skin and engine. A test
    that means "nothing is binding" has to say so about every constraint, not
    just the ones that existed the day it was written.
    """
    lim = _limits(payload_g=1e9, throat_flux_mw_m2=1e9, coolant_margin_k=-1e9)
    lim["struct_closure_tol"] = 1e9
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
    ev = evaluate(PAYLOAD, APOGEE, Knobs(), _unconstrained())
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
    res = autodesign(PAYLOAD, APOGEE, limits=_unconstrained(), max_iters=5)
    assert res["converged"]
    assert res["iterations"] == 1
    assert res["conflict"] is None


# --- over-constrained problems ---------------------------------------------

def test_opposing_constraints_are_reported_as_a_conflict():
    """Not 'did not converge' -- which is true and useless -- but which pair of
    requirements cannot coexist and on which knob."""
    # 120 K used to be unsatisfiable and no longer is: with structural
    # coefficient and skin material as knobs the vehicle can restructure until
    # both thermal constraints hold, and it converges at 120.3 K. That is a real
    # gain in reach, so the test moved to a margin the enlarged design space
    # still cannot deliver rather than being deleted -- conflict detection is
    # the feature here, and it needs a genuine conflict to detect.
    res = autodesign(PAYLOAD, APOGEE,
                     limits=_limits(coolant_margin_k=250.0), max_iters=10)
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
                     limits=_limits(coolant_margin_k=250.0), max_iters=20)
    assert res["conflict"] is not None
    assert res["iterations"] < 20


def test_an_extreme_mission_stays_physically_coherent():
    """Far outside the mission this system was tuned on, the answer must still
    have the right shape: several stages at the gross-to-payload ratio the
    rocket equation demands for that delta-v.

    This was 10,000 t to lunar distance, which cost 326 s -- a third of the
    entire test suite -- to check a property that 5 t to 20,000 km checks in
    11 s. Extrapolating two orders of magnitude past the tuning point is the
    point; extrapolating five is just slow.
    """
    res = autodesign(5000.0, 20_000.0, limits=_limits(), max_iters=2)
    ev = res["evaluation"]
    assert ev is not None and ev.plan is not None
    assert ev.stages > 2, ev.stages
    assert ev.gross_kg > 10.0 * 5000.0, ev.gross_kg
    assert ev.plan.achieved_km > 0.8 * 20_000.0


def test_a_mission_that_cannot_close_is_reported_as_a_violation(monkeypatch):
    """A mission with no architecture must come back as a violation rather than
    an exception or a crash.

    Tested by making the planner return None rather than by hunting for a
    mission it genuinely cannot close -- that hunt cost 152 s and exercised the
    search, not the error path this test is about.
    """
    import cadflow.autodesign as autodesign_mod
    from cadflow.autodesign import Knobs, evaluate

    monkeypatch.setattr("cadflow.planner.plan", lambda *a, **k: None)
    ev = evaluate(25.0, 4000.0, Knobs(), _limits())
    assert ev.plan is None
    assert not ev.feasible
    assert any(v.discipline == "architecture" for v in ev.violations)
    _ = autodesign_mod
