"""Proving the self-check catches the bugs it was written for.

Five defects this session shared one shape -- one part of the program knew
something and another reported otherwise -- and all five were invisible to five
hundred tests, because every individual model was right and only the connection
between them was wrong.

The audit added for that class reported "13 of 13 agree" on its first run. That
is worth almost nothing on its own: a check that has only been run against
correct data is not known to work, it is known to be quiet, and a broken check
is quiet too. So every test here feeds it a packet that is wrong in a specific,
historically real way and requires it to say so.
"""

import pytest

from cadflow.packet_audit import audit, failures


class _Stage:
    def __init__(self, prop, struct):
        self.prop_mass_kg = prop
        self.struct_mass_kg = struct


def _good_stack():
    """Four stages at a uniform 0.2613, as the repair loop produces."""
    return [_Stage(973.89, 344.53), _Stage(253.21, 89.58),
            _Stage(65.83, 23.29), _Stage(23.13, 8.18)]


def _component(name, mass, volume):
    return {"name": name,
            "mass_properties": {"mass_kg": mass, "volume_m3": volume}}


def _kwargs(**over):
    stack = over.pop("stack", None) or _good_stack()
    gross = over.pop("gross_kg", None)
    if gross is None:
        gross = sum(s.prop_mass_kg + s.struct_mass_kg for s in stack) + 25.0
    base = dict(gross_kg=gross, stack=stack, payload_kg=25.0,
                flight_vehicle_mass_kg=gross,
                components=[_component("stage 1 tank", 8.19, 0.001)],
                skin_density_kg_m3=8190.0, skin_material="inconel-718")
    base.update(over)
    return base


def test_a_consistent_packet_passes():
    """The baseline, which is necessary and not sufficient."""
    assert not failures(audit(**_kwargs()))


def test_it_catches_a_component_weighed_in_the_wrong_alloy():
    """The three-fold mass error, caught without inspection.

    Component masses were computed with aluminium's 2,700 kg/m3 on an Inconel
    vehicle. Mass over volume is 2,700 where the design says 8,190, and no
    amount of the individual models being right conceals that.
    """
    bad = audit(**_kwargs(
        components=[_component("stage 1 tank", 2.70, 0.001)]))
    bad_ones = failures(bad)
    assert bad_ones, "the density mismatch has to be caught"
    assert "weighed in inconel-718" in bad_ones[0].check
    assert bad_ones[0].got == pytest.approx(2700.0)


def test_it_catches_a_gross_mass_that_does_not_match_its_own_stack():
    """The planner reporting one number while listing stages summing to another.

    This is the shape that first made struct_coeff_used suspicious: a headline
    figure that its own detail does not support.
    """
    assert failures(audit(**_kwargs(gross_kg=5974.3)))


def test_it_catches_a_flight_vehicle_that_disagrees_with_the_plan():
    """Mass properties describing a different vehicle from the trajectory."""
    bad = failures(audit(**_kwargs(flight_vehicle_mass_kg=1200.0)))
    assert any("flight vehicle mass" in c.check for c in bad)


def test_it_catches_a_repair_that_reached_some_stages_and_not_others():
    """A partially applied structural coefficient.

    The repair loop sets one coefficient for the whole stack. A stage left at
    the old value would still produce a plausible vehicle, and every per-stage
    number would still be internally fine.
    """
    stack = _good_stack()
    stack[2] = _Stage(65.83, 10.72)      # 0.14 where the others are 0.2613
    bad = failures(audit(**_kwargs(stack=stack)))
    assert any("structural coefficient" in c.check for c in bad)


def test_a_component_without_volume_is_skipped_not_guessed():
    """A check that invents an input to stay green is worse than one that
    does not run.

    Hull-substituted parts have no usable volume. Defaulting it would either
    manufacture a failure or, worse, manufacture a pass.
    """
    out = audit(**_kwargs(components=[
        {"name": "fairing", "mass_properties": {"mass_kg": 3.0}},
        {"name": "nose cone"},
    ]))
    assert not failures(out)
    assert not any("fairing" in c.check for c in out)


def test_the_tolerance_is_tight_enough_to_matter():
    """A 1% density error is a real error, not rounding.

    Loose tolerances are how a check reports agreement it has not established.
    """
    assert failures(audit(**_kwargs(
        components=[_component("tank", 8.19 * 1.01, 0.001)])))
