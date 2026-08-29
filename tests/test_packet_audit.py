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


def test_it_catches_a_part_reported_where_it_is_not():
    """The nozzle overlapping the tank by 324 mm, on paper only.

    The assembly table showed the nozzle at 0.000 m while its solid sits
    between -0.324 and 0, hanging off the aft end as the code intends. The CAD
    was right and the station recorded beside it was not, so the report
    described a vehicle whose engine occupies the same space as its tank.
    Nothing compared the two.
    """
    from cadflow.packet_audit import stack_interference

    as_reported = [("nozzle", 0.0, 0.324), ("stage 1 tank", 0.0, 2.709),
                   ("interstage 1/2", 2.709, 0.208)]
    bad = failures(stack_interference(as_reported))
    assert bad and "nozzle" in bad[0].check


def test_a_correctly_placed_stack_tiles():
    """With the nozzle at its real station the vehicle closes end to end."""
    from cadflow.packet_audit import stack_interference

    real = [("nozzle", -0.324, 0.324), ("stage 1 tank", 0.0, 2.709),
            ("interstage 1/2", 2.709, 0.208), ("stage 2 tank", 2.917, 0.832)]
    assert not failures(stack_interference(real))


def test_fins_are_excluded_from_the_interference_check():
    """They attach to the outside of a tank and share its span by design.

    Including them would report every finned vehicle as broken, which is how a
    check earns being switched off.
    """
    from cadflow.packet_audit import stack_interference

    with_fins = [("stage 1 tank", 0.0, 2.709), ("fin 1", 0.167, 0.670),
                 ("fin 2", 0.167, 0.670), ("interstage 1/2", 2.709, 0.208)]
    assert not failures(stack_interference(with_fins))


def test_it_catches_a_gap_as_well_as_an_overlap():
    """A vehicle with a hole in it is as wrong as one that intersects itself."""
    from cadflow.packet_audit import stack_interference

    gapped = [("stage 1 tank", 0.0, 2.709), ("interstage 1/2", 3.200, 0.208)]
    assert failures(stack_interference(gapped))


def _derived(**over):
    """The packet's own reported values, which currently all agree."""
    base = dict(
        stages=4, split=[0.74, 0.192, 0.050, 0.018],
        achieved_km=4118.3, target_km=4000.0, error_pct=2.96,
        # Read from packet_v27, not invented. The first version of this
        # fixture guessed cp_nose_z_m and cp_fins_z_m, and the weighted-mean
        # check duly failed -- on the fixture, not on anything real. Loosening
        # the tolerance to make it pass would have disabled the check; the
        # fixture was what was wrong.
        stability={"cna_nose": 2.0, "cna_fins": 5.561450569356806,
                   "cna_total": 7.561450569356806,
                   "cp_nose_z_m": 3.7540558911437723,
                   "cp_fins_z_m": 0.3609823452793225,
                   "cp_z_m": 1.2584486487991964,
                   "static_margin_cal": 0.9},
        cg_z_m=1.8614, radius_m=0.335)
    base.update(over)
    return base


def test_the_derived_quantities_currently_agree():
    """Baseline. None of these is broken today, which is the point of adding
    them: the six that drifted were all correct until something changed one
    side and not the other."""
    from cadflow.packet_audit import derived_quantities

    assert not failures(derived_quantities(**_derived()))


def test_it_catches_a_stage_count_that_does_not_match_its_split():
    """The planner reported five stages for a three-stage split once already.

    That was fixed at the source; nothing checks it stays fixed.
    """
    from cadflow.packet_audit import derived_quantities

    assert failures(derived_quantities(**_derived(stages=5)))


def test_it_catches_an_apogee_error_computed_from_a_different_apogee():
    """A headline percentage that its own number does not support."""
    from cadflow.packet_audit import derived_quantities

    bad = failures(derived_quantities(**_derived(error_pct=0.5)))
    assert any("apogee error" in c.check for c in bad)


def test_it_catches_a_normal_force_slope_that_is_not_the_sum_of_its_parts():
    """cna_total drifting from nose plus fins means one was resized and the
    other was not -- exactly what the control trade does to fins."""
    from cadflow.packet_audit import derived_quantities

    st = dict(_derived()["stability"])
    st["cna_fins"] = 11.69          # the pre-trade value, left behind
    bad = failures(derived_quantities(**_derived(stability=st)))
    assert any("sum of its parts" in c.check for c in bad)


def test_it_catches_a_static_margin_the_trade_left_stale():
    """The fin trade moves the margin from 1.50 to 0.90.

    A reported margin still reading 1.50 while the stations say 0.90 would mean
    the trade reached the fins and not the report -- the same shape as
    struct_coeff_used reading a module default after the repair loop moved it.
    """
    from cadflow.packet_audit import derived_quantities

    st = dict(_derived()["stability"])
    st["static_margin_cal"] = 1.50
    bad = failures(derived_quantities(**_derived(stability=st)))
    assert any("static margin" in c.check for c in bad)


def test_it_catches_a_centre_of_pressure_belonging_to_different_fins():
    """cp must be the slope-weighted mean of the nose and fin contributions."""
    from cadflow.packet_audit import derived_quantities

    st = dict(_derived()["stability"])
    st["cp_z_m"] = 0.857            # the pre-trade cp, with post-trade slopes
    bad = failures(derived_quantities(**_derived(stability=st)))
    assert any("weighted mean" in c.check for c in bad)


def test_it_catches_a_design_built_from_a_different_propellant():
    """The largest defect the sweeps found, as a standing check.

    --propellant lox_lh2 produced a vehicle identical to kerosene to a tenth of
    a kilogram, because autodesign was called with no knobs and Knobs.propellant
    stayed at its default. The packet header said hydrogen and the design was
    kerosene: a document internally consistent and wholly wrong about its
    subject.
    """
    from cadflow.packet_audit import failures, input_provenance

    bad = failures(input_provenance(
        requested={"propellant": "lox_lh2"}, used={"propellant": "lox_rp1"}))
    assert bad and "lox_lh2" in bad[0].detail


def test_it_catches_a_mixture_ratio_that_two_modules_disagree_on():
    """Tanks sized at 2.56 while the chemistry burned 2.45."""
    from cadflow.packet_audit import failures, input_provenance

    assert failures(input_provenance(requested={"of_ratio": 2.45},
                                     used={"of_ratio": 2.56}))


def test_a_design_built_from_its_own_inputs_is_quiet():
    """A check that fires on correct packets gets switched off."""
    from cadflow.packet_audit import failures, input_provenance

    same = {"propellant": "lox_rp1", "payload_kg": 25.0, "chamber_bar": 55.0}
    assert not failures(input_provenance(requested=same, used=dict(same)))


def test_rounding_is_not_reported_as_a_substitution():
    """Chamber pressure round-trips through pascals and back.

    A tolerance tight enough to catch a substituted value and loose enough to
    ignore a float conversion, or the check cries wolf on every packet.
    """
    from cadflow.packet_audit import failures, input_provenance

    assert not failures(input_provenance(requested={"chamber_bar": 55.0},
                                         used={"chamber_bar": 55.00000001}))
    assert failures(input_provenance(requested={"chamber_bar": 55.0},
                                     used={"chamber_bar": 60.0}))


def test_a_repair_the_loop_reports_is_not_a_substitution():
    """Two invariants that must not be conflated.

    At 200 bar the design loop lowers chamber pressure to 153.3 to satisfy the
    throat heat flux limit. That is the loop working, and the knob trajectory
    reports it. Checking the built value against the caller's *request* flagged
    it as a substitution and drove the packet to FAILED for a design that is
    fine.

      provenance      : the loop's decision vs what was built  -- a failure
      knob trajectory : the request vs the loop's decision     -- a decision

    The check exists for the silent kind: a subsystem using a value nobody
    decided on.
    """
    from cadflow.packet_audit import failures, input_provenance

    # the loop decided 153.3 and the vehicle was built at 153.3: no failure,
    # however far that is from what the caller typed
    assert not failures(input_provenance(requested={"chamber_bar": 153.3},
                                         used={"chamber_bar": 153.3}))
    # the loop decided 153.3 and something built 200: a real disconnection
    assert failures(input_provenance(requested={"chamber_bar": 153.3},
                                     used={"chamber_bar": 200.0}))
