"""A nozzle judged against the air it actually flies in.

The defect this covers is the same shape as the six before it: two parts each
knew something and neither told the other. The trajectory integrator knew the
ambient pressure at every step, including the step each stage lit on. The
planner needed exactly that number to size a throat and, not having it, used sea
level for all four stages.

What makes it worth a module rather than a one-line fix is what the nozzle model
does with the wrong pressure. It does not fail. Asked for eps=30, 60 and 80 at
101 kPa it correctly reports all three as separated and truncates each to the
same effective ratio, so the sizing step saw one nozzle three times and returned
three identical throat areas. The planner's expansion-ratio ladder was erased by
a model behaving exactly as designed.
"""

import math

import pytest

from cadflow.nozzle_ambient import (
    MARGINAL_BELOW, VACUUM_AMBIENT_PA, check_stack, check_stage, findings,
    sizing_ambient_pa)

# lox/rp1 at 55 bar, the mission this project runs.
GAS = dict(chamber_pressure_pa=55e5, gamma=1.2, chamber_temp_k=3600.0,
           mol_mass=23.0)
SEA_LEVEL = 101325.0


def test_a_sea_level_first_stage_is_attached():
    """eps=12 against a limit near 15 -- the planner's choice is sound."""
    c = check_stage(stage=1, expansion_ratio=12.0,
                    ignition_ambient_pa=SEA_LEVEL, **GAS)
    assert c.attached
    assert c.eps_max == pytest.approx(15.0, abs=0.5)


def test_an_upper_stage_ratio_would_separate_at_sea_level():
    """Which is the whole reason sizing them there was wrong.

    eps=60 at 101 kPa is not a nozzle, it is a nozzle with the flow off the
    wall. Sizing a throat against its thrust means sizing against a nozzle the
    stage does not have.
    """
    c = check_stage(stage=3, expansion_ratio=60.0,
                    ignition_ambient_pa=SEA_LEVEL, **GAS)
    assert not c.attached
    assert c.utilisation > 3.0


def test_the_upper_stage_ratios_collapse_to_one_nozzle_at_sea_level():
    """30, 60 and 80 all truncate to the same effective ratio.

    This is the mechanism, and it is why the bug produced identical throat areas
    rather than merely wrong ones: at sea level the separation plane sets the
    effective exit, so the geometry past it is invisible. Three deliberately
    different upper stages were sized as the same engine.
    """
    limits = {check_stage(stage=i, expansion_ratio=e,
                          ignition_ambient_pa=SEA_LEVEL, **GAS).eps_max
              for i, e in enumerate((30.0, 60.0, 80.0), 2)}
    assert len(limits) == 1, "the limit is set by ambient alone"
    from generate_propulsion_trajectory_corpus import nozzle_performance
    thrusts = {round(nozzle_performance(
        chamber_pressure=55e5, chamber_temp=3600.0, expansion_ratio=e,
        throat_area=1.0, gamma=1.2, mol_mass=23.0,
        ambient_pressure=SEA_LEVEL)["thrust"], 3)
        for e in (30.0, 60.0, 80.0)}
    assert len(thrusts) == 1, (
        "the ladder had no effect on sizing thrust, which is the defect")


def test_the_same_ratios_are_attached_where_they_are_actually_lit():
    """At their real ignition altitudes every one of them flows full."""
    for i, (eps, alt_pa) in enumerate(
            [(30.0, 5_000.0), (60.0, 300.0), (80.0, 50.0)], 2):
        c = check_stage(stage=i, expansion_ratio=eps,
                        ignition_ambient_pa=alt_pa, **GAS)
        assert c.attached, c.as_dict()


def test_it_reports_the_thrust_to_weight_actually_flown():
    """The miscalibrated lever, quantified.

    A stage asked for 2.2 flies at about 2.8, because the throat was sized
    against a separated nozzle's thrust and then flown in vacuum where the real
    nozzle makes more. The trajectory was never wrong -- it integrated the true
    thrust -- but the knob the design loop pulls to hold peak axial acceleration
    down is 27% out.
    """
    c = check_stage(stage=3, expansion_ratio=60.0, ignition_ambient_pa=0.0,
                    ignition_altitude_m=60_000.0, twr_sized=2.2,
                    sized_at_ambient_pa=SEA_LEVEL, **GAS)
    assert c.twr_flown == pytest.approx(2.79, abs=0.05)
    assert c.twr_error_pct == pytest.approx(26.6, abs=2.0)


def test_a_first_stage_sized_at_sea_level_has_no_error():
    """Because sea level is where it lights. The fix must not move this one."""
    c = check_stage(stage=1, expansion_ratio=12.0,
                    ignition_ambient_pa=SEA_LEVEL, ignition_altitude_m=0.0,
                    twr_sized=4.5, sized_at_ambient_pa=SEA_LEVEL, **GAS)
    assert c.twr_error_pct == pytest.approx(0.0, abs=1e-6)


def test_near_vacuum_is_treated_as_vacuum():
    """An upper stage should not be sized against a pressure that is noise."""
    assert sizing_ambient_pa(0.5 * VACUUM_AMBIENT_PA) == 0.0
    assert sizing_ambient_pa(float("nan")) == 0.0
    assert sizing_ambient_pa(50_000.0) == 50_000.0


def test_a_marginal_nozzle_is_not_reported_as_simply_fine():
    """Summerfield is good to roughly 25%, so 1.02x the limit is not attached.

    A check with a hard threshold on an approximate criterion reports false
    confidence exactly where the answer matters most.
    """
    c = check_stage(stage=1, expansion_ratio=14.9,
                    ignition_ambient_pa=SEA_LEVEL, **GAS)
    assert c.attached and c.marginal
    assert any("marginally attached" in f for f in findings([c]))


def test_a_clean_stack_produces_no_findings():
    """The check must be quiet on a correct design or it will be switched off."""
    class S:
        def __init__(s, eps):
            s.expansion_ratio = eps
            s.chamber_pressure_pa = 55e5
            s.gamma, s.chamber_temp, s.mol_mass = 1.2, 3600.0, 23.0

    ok = check_stack([S(12.0), S(30.0), S(60.0), S(80.0)],
                     ignition_ambient_pa=[SEA_LEVEL, 5_000.0, 300.0, 50.0],
                     ignition_altitude_m=[0.0, 20_000.0, 40_000.0, 60_000.0],
                     twr_by_stage=[4.5, 3.0, 2.2, 2.0],
                     sized_at_ambient_pa=[SEA_LEVEL, 5_000.0, 0.0, 0.0])
    assert all(c.attached for c in ok)
    # Sized where they light, so the lever reads true and there is nothing to
    # say. This is the post-fix state of the planner, and a check that still
    # complained here would be reporting a defect that no longer exists.
    assert findings(ok) == []


def test_the_design_before_the_fix_is_caught():
    """The four-stage vehicle as the planner built it before this module.

    Kept as the regression case: this is the state the check has to be able to
    see, and it is no longer the state the planner produces.
    """
    class S:
        def __init__(s, eps):
            s.expansion_ratio = eps
            s.chamber_pressure_pa = 55e5
            s.gamma, s.chamber_temp, s.mol_mass = 1.2, 3600.0, 23.0

    bad = findings(check_stack(
        [S(12.0), S(30.0), S(60.0), S(80.0)],
        ignition_ambient_pa=[SEA_LEVEL, 2_000.0, 100.0, 10.0],
        ignition_altitude_m=[0.0, 26_000.0, 48_000.0, 70_000.0],
        twr_by_stage=[4.5, 3.0, 2.2, 2.0],
        # every throat sized at sea level, as the planner did before the fix
        sized_at_ambient_pa=[SEA_LEVEL] * 4))
    twr = [f for f in bad if "thrust-to-weight" in f]
    assert len(twr) == 3, "stage 1 is correct; the three above it are not"
    assert not any("stage 1 " in f for f in twr)


def test_the_limit_it_uses_is_the_one_the_trajectory_flies():
    """Not a second copy of the criterion.

    A check that re-derives the physics it is checking will agree with itself
    and drift from the model, which is how a green check comes to mean nothing.
    """
    from generate_propulsion_trajectory_corpus import separation_limited_ratio

    c = check_stage(stage=1, expansion_ratio=12.0,
                    ignition_ambient_pa=SEA_LEVEL, **GAS)
    assert c.eps_max == separation_limited_ratio(55e5, SEA_LEVEL, 1.2)


def test_vacuum_imposes_no_limit():
    c = check_stage(stage=4, expansion_ratio=200.0, ignition_ambient_pa=0.0,
                    **GAS)
    assert c.attached and math.isinf(c.eps_max) and not c.marginal


def test_a_near_vacuum_ignition_reports_vacuum_not_a_huge_number():
    """The criterion is continuous; a denormal ambient gives a limit of ~1e11.

    True, useless, and it reads as a real quantity in a report. The module
    already defines what counts as vacuum for sizing, and using a different
    definition for the limit would be the same module disagreeing with itself --
    which is the exact class of defect it was written to catch.
    """
    c = check_stage(stage=3, expansion_ratio=60.0,
                    ignition_ambient_pa=0.5 * VACUUM_AMBIENT_PA, **GAS)
    assert math.isinf(c.eps_max) and c.attached and not c.marginal


def test_just_above_the_vacuum_threshold_still_uses_the_criterion():
    """The threshold must not swallow pressures that genuinely constrain.

    A cutoff that reaches too high would report an attached nozzle for a stage
    lighting inside the atmosphere, which is the failure this module exists to
    prevent.
    """
    c = check_stage(stage=2, expansion_ratio=30.0,
                    ignition_ambient_pa=2.0 * VACUUM_AMBIENT_PA, **GAS)
    assert math.isfinite(c.eps_max)


def test_the_sized_thrust_to_weight_is_recovered_not_remembered():
    """Round-trip: build a stack at a known ladder and read it back out.

    Six defects in this project were a reported number its own subject could
    not reproduce. A thrust-to-weight list threaded through three call sites is
    that arrangement again, so it is derived from the stage instead.
    """
    from cadflow.nozzle_ambient import recover_twr_sized
    from cadflow.planner import build_stack_n

    asked = [4.5, 3.0, 2.2, 2.0]
    stack, _gross, _ = build_stack_n(
        1316.06, [0.740, 0.192, 0.050, 0.018], 25.0, 55e5, "lox_rp1",
        twr_by_stage=asked)
    got = recover_twr_sized(stack, 25.0)
    assert got == pytest.approx(asked, rel=1e-9)


def test_the_planner_now_sizes_each_stage_where_it_lights():
    """The fix, end to end: every stage flies the ratio it was asked for.

    Before, a stage asked for 2.2 flew at 2.79 because its throat was sized
    against a nozzle separated at sea level. The trajectory is what decides
    this, so it is flown rather than asserted.
    """
    from cadflow.multistage import integrate_stack
    from cadflow.nozzle_ambient import recover_twr_sized
    from cadflow.planner import build_stack_n

    asked = [4.5, 3.0, 2.2, 2.0]
    stack, _g, _ = build_stack_n(1316.06, [0.740, 0.192, 0.050, 0.018], 25.0,
                                 55e5, "lox_rp1", twr_by_stage=asked)
    flight = integrate_stack(stack, 25.0, cd=0.35,
                             ref_area_m2=math.pi * 0.335 ** 2, ref_length_m=8.0)
    checks = check_stack(stack,
                         ignition_ambient_pa=flight["ignition_ambient_pa"],
                         ignition_altitude_m=flight["ignition_altitude_m"],
                         twr_by_stage=recover_twr_sized(stack, 25.0))
    assert all(c.attached for c in checks)
    for c in checks:
        assert abs(c.twr_error_pct) < 5.0, c.as_dict()
    assert findings(checks) == []


def test_every_stage_lights_above_the_one_below_it():
    """Guards the ignition record itself.

    The altitudes and pressures are new trajectory outputs, and a check fed a
    silently wrong input reports confidence about nothing. Ignition altitude
    must rise monotonically and ambient must fall.
    """
    from cadflow.multistage import integrate_stack
    from cadflow.planner import build_stack_n

    stack, _g, _ = build_stack_n(1316.06, [0.740, 0.192, 0.050, 0.018], 25.0,
                                 55e5, "lox_rp1", twr_by_stage=[4.5, 3, 2.2, 2])
    f = integrate_stack(stack, 25.0, cd=0.35,
                        ref_area_m2=math.pi * 0.335 ** 2, ref_length_m=8.0)
    alts, ps = f["ignition_altitude_m"], f["ignition_ambient_pa"]
    assert all(math.isfinite(a) for a in alts), "a stage never lit"
    assert alts[0] == pytest.approx(0.0, abs=1.0)
    assert alts == sorted(alts) and ps == sorted(ps, reverse=True)
