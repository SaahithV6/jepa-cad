"""A pass by less than its own error bar is not a pass.

Packet v40 reports the tank wall as passing at von Mises 130.0 MPa against a 131
MPa allowable -- a margin of 1.008. The same document states that swapping
element order moves the p95 stress this loop sizes against by -13.9% to +14.5%,
and the pressure driving that hoop stress rests on a suction head the
pressurisation module labels ASSUMED, not derived.

Both statements are true and they cannot both be load-bearing. This is the check
that says so.
"""

import pytest

from cadflow.margin_audit import (
    MEASURED_STRESS_UNCERTAINTY, RESOLVED_MARGIN, audit, judge, summary,
    unresolved)


def test_the_threshold_is_the_projects_own_measurement():
    """Not a safety factor and not an opinion.

    The number comes from element_order_ab.json: twelve real components solved
    at identical meshes under both element orders. Inventing a rounder
    threshold would make this a house style rather than a statement about what
    the analysis can resolve.
    """
    assert MEASURED_STRESS_UNCERTAINTY == pytest.approx(0.145)
    assert RESOLVED_MARGIN == pytest.approx(1.145)


def test_the_tank_wall_verdict_is_reported_as_unresolved():
    """The case that prompted the module: 130.0 against 131 MPa."""
    v = judge("tank wall under pressure", 131.0 / 130.0)
    assert v.margin == pytest.approx(1.0077, abs=1e-3)
    assert not v.resolved
    assert "not established either way" in v.note


def test_a_comfortable_margin_is_left_alone():
    """A check that fires on healthy designs is one that gets switched off.

    The interstage clears by 131%. Nothing about that is ambiguous.
    """
    v = judge("interstage buckling", 2.31)
    assert v.resolved
    assert "outside the" in v.note


def test_a_margin_just_past_the_threshold_resolves():
    """The boundary has to fall where the measurement puts it."""
    assert judge("x", 1.146).resolved
    assert not judge("x", 1.144).resolved


def test_an_outright_failure_is_not_called_unresolved():
    """Failing is a verdict. Only passes can be inside the noise.

    Reporting a genuine failure as 'unresolved' would soften it, which is the
    opposite of what this module is for.
    """
    v = judge("overloaded wall", 0.8)
    assert not v.resolved
    assert "fails outright" in v.note
    assert unresolved([v]) == []


def test_the_audit_orders_worst_first():
    """The thinnest margin decides what the overall verdict is worth."""
    got = audit({"comfortable": 4.85, "thin": 1.008, "middling": 1.40})
    assert [v.check for v in got] == ["thin", "middling", "comfortable"]


def test_the_summary_names_the_thinnest_and_stays_silent_when_clear():
    """Silence on a clean packet, and something actionable otherwise."""
    clear = audit({"a": 2.0, "b": 1.5})
    assert summary(clear) == ""

    thin = audit({"tank wall under pressure": 1.008, "tvc authority": 1.10,
                  "interstage": 2.31})
    text = summary(thin)
    assert "tank wall under pressure" in text
    assert "2 passing check(s)" in text
    assert "open questions rather than as passes" in text


def test_the_packet_margins_are_judged_the_way_the_packet_reports_them():
    """The four structural verdicts from v40, as published.

    Two of them are inside the measured uncertainty. That is the finding, and
    it survives only if the numbers come from the packet rather than from
    memory.
    """
    got = audit({
        "tank wall under pressure": 131.0 / 130.0,
        "skin buckling": 1.40,
        "interstage buckling": 2.31,
        "flight barrel stress": 131.0 / 27.0,
    })
    thin = [v.check for v in unresolved(got)]
    assert "tank wall under pressure" in thin
    assert "flight barrel stress" not in thin
