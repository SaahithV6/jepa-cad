"""Axial acceleration reported by the trajectory, which is what sizes structure.

Component sizing assumed a flat 4.5 g. The integrator had the real figure at
every timestep and discarded it, so every axially loaded part was designed for
under a third of its actual load. These tests pin the quantities that fixed
that, and the per-stage split that keeps a lower stage from being sized for a
peak it never sees.
"""

import math

import pytest

from cadflow.multistage import Stage, integrate_stack

# lox/rp1-ish gas properties, enough to make a stage that flies
GAMMA, TC, MOL = 1.2, 3500.0, 22.0


def _stage(prop_kg, struct_kg, throat_m2, pc=5.5e6, eps=12.0):
    return Stage(prop_kg, struct_kg, throat_m2, pc, eps, GAMMA, TC, MOL)


def _fly(stages, payload=25.0, cd=0.42, area=0.25):
    return integrate_stack(stages, payload, cd=cd, ref_area_m2=area, dt=0.2,
                           pitchover_angle=math.radians(3.0))


def test_integrator_reports_axial_acceleration():
    res = _fly([_stage(700.0, 110.0, 0.010)])
    assert "max_axial_g" in res
    assert "max_axial_g_by_stage" in res
    assert "liftoff_thrust_n" in res
    assert res["max_axial_g"] > 0.0
    assert len(res["max_axial_g_by_stage"]) == 1


def test_peak_acceleration_exceeds_the_old_assumption():
    """The finding itself: a constant-thrust stage does not stay near 4.5 g.

    Thrust is held while mass falls, so acceleration climbs through the burn and
    peaks at burnout. Any vehicle with a decent mass ratio ends well above the
    4.5 g that sizing used to assume.
    """
    res = _fly([_stage(700.0, 110.0, 0.010)])
    assert res["max_axial_g"] > 4.5


def test_global_peak_is_the_max_of_the_per_stage_peaks():
    res = _fly([_stage(700.0, 110.0, 0.010), _stage(150.0, 25.0, 0.0025)])
    by_stage = res["max_axial_g_by_stage"]
    assert len(by_stage) == 2
    assert res["max_axial_g"] == pytest.approx(max(by_stage), rel=1e-9)


def test_upper_stage_pulls_more_than_the_lower_one():
    """The reason the split matters.

    The global peak happens at final burnout, when the lower stages have gone.
    Sizing stage 1 against it would be sizing it for a load it never sees.
    """
    res = _fly([_stage(700.0, 110.0, 0.010), _stage(150.0, 25.0, 0.0025)])
    lower, upper = res["max_axial_g_by_stage"]
    assert upper > lower
    assert lower < res["max_axial_g"]


def test_liftoff_thrust_is_the_first_burning_thrust():
    """Sea-level thrust at t=0, not gross mass times an assumed thrust-to-weight."""
    res = _fly([_stage(700.0, 110.0, 0.010)])
    thrust = res["liftoff_thrust_n"]
    gross = 700.0 + 110.0 + 25.0
    # it must at least lift the vehicle, or the mission would not start
    assert thrust > gross * 9.80665
    # and it must be consistent with the acceleration actually achieved
    assert thrust / (gross * 9.80665) < res["max_axial_g"] + 1e-6


def test_a_bigger_throat_accelerates_harder():
    small = _fly([_stage(700.0, 110.0, 0.008)])
    big = _fly([_stage(700.0, 110.0, 0.014)])
    assert big["max_axial_g"] > small["max_axial_g"]
    assert big["liftoff_thrust_n"] > small["liftoff_thrust_n"]


def test_acceleration_is_net_of_drag():
    """Reported acceleration is (thrust - drag)/m, so more drag must lower it."""
    clean = _fly([_stage(700.0, 110.0, 0.010)], cd=0.2)
    draggy = _fly([_stage(700.0, 110.0, 0.010)], cd=1.2, area=1.0)
    assert draggy["max_axial_g"] < clean["max_axial_g"]


# --- stage mass model -------------------------------------------------------

def test_stage_mass_model_includes_an_engine():
    """The omission that made the solved structural coefficient unphysical.

    The model counted tank, interstage and thrust structure and no engine at
    all. For a 48.8 kN stage an engine weighs 50 kg at T/W 100 and 125 kg at
    T/W 40, against 31 kg for everything else the model counted -- so leaving it
    out was not a rounding error, it was most of the stage.
    """
    from cadflow.structural_sizing import (
        ENGINE_THRUST_TO_WEIGHT, stage_structural_mass)

    thrust = 48834.0
    total, parts = stage_structural_mass(700.0, 0.28, thrust)
    engine = [p for p in parts if p["name"] == "engine"]
    assert engine, [p["name"] for p in parts]
    assert engine[0]["mass_kg"] == pytest.approx(
        thrust / (9.80665 * ENGINE_THRUST_TO_WEIGHT), rel=1e-9)
    assert total > engine[0]["mass_kg"]


def test_engine_mass_scales_with_thrust():
    from cadflow.structural_sizing import stage_structural_mass

    light, _ = stage_structural_mass(700.0, 0.28, 20000.0)
    heavy, _ = stage_structural_mass(700.0, 0.28, 80000.0)
    assert heavy > light


def test_solved_structural_coefficient_is_physically_plausible():
    """It converged near 0.06 with no engine; real vehicles sit at 0.10-0.25."""
    from generate_propulsion_trajectory_corpus import load_coupling
    from cadflow.planner import plan_sized

    load_coupling()
    plan, coeff, history = plan_sized(4000.0, 25.0)
    assert plan is not None
    assert 0.08 <= coeff <= 0.30, coeff
    assert history


def test_plan_sized_restores_module_state_even_when_it_fails():
    """plan_sized mutates module-level constants while it iterates.

    The restore used to sit after the loop, outside any finally, and wrote a
    hardcoded 0.14 rather than whatever had been there. On the happy path it
    looked fine -- which is precisely what let it survive -- but an exception
    anywhere in the iteration left the module permanently mis-configured, and
    every design made afterwards in the same process would silently use the last
    iterate's structural coefficient.
    """
    import cadflow.planner as planner
    import cadflow.structural_sizing as sizing
    from generate_propulsion_trajectory_corpus import load_coupling

    load_coupling()
    before = (planner.STRUCT_COEFF, planner.MAX_STAGE_MR)

    planner.plan_sized(4000.0, 25.0)
    assert (planner.STRUCT_COEFF, planner.MAX_STAGE_MR) == before

    original = sizing.stage_structural_mass

    def explode(*args, **kwargs):
        raise RuntimeError("solver blew up")

    sizing.stage_structural_mass = explode
    try:
        with pytest.raises(RuntimeError):
            planner.plan_sized(4000.0, 25.0)
    finally:
        sizing.stage_structural_mass = original

    assert (planner.STRUCT_COEFF, planner.MAX_STAGE_MR) == before


def test_plan_is_unaffected_by_a_preceding_plan_sized():
    """The observable consequence: designs must not depend on call order."""
    import cadflow.planner as planner
    from generate_propulsion_trajectory_corpus import load_coupling

    load_coupling()
    first = planner.plan(4000.0, 25.0)
    planner.plan_sized(4000.0, 25.0)
    second = planner.plan(4000.0, 25.0)
    assert first.stages == second.stages
    assert first.gross_kg == pytest.approx(second.gross_kg, rel=1e-12)


# --- plan caching -----------------------------------------------------------

def test_the_plan_cache_returns_the_same_vehicle():
    """plan() is deterministic, and the design loop asks for the same vehicle
    many times while searching. A cold call is ~2.5 s and a warm one is free."""
    import cadflow.planner as planner
    from generate_propulsion_trajectory_corpus import load_coupling

    load_coupling()
    planner.clear_plan_cache()
    first = planner.plan(4000.0, 25.0)
    second = planner.plan(4000.0, 25.0)
    assert first is second


def test_the_cache_key_carries_the_structural_coefficient():
    """The risk a cache introduces here, asserted directly.

    plan_sized mutates STRUCT_COEFF while it iterates. Keyed on the arguments
    alone, the cache would hand back a vehicle sized at whatever coefficient
    happened to be set when it was first computed -- the silent wrong answer a
    cache is best at producing.
    """
    import cadflow.planner as planner
    from generate_propulsion_trajectory_corpus import load_coupling

    load_coupling()
    planner.clear_plan_cache()
    baseline = planner.plan(4000.0, 25.0)

    saved_coeff, saved_mr = planner.STRUCT_COEFF, planner.MAX_STAGE_MR
    try:
        planner.STRUCT_COEFF = 0.25
        planner.MAX_STAGE_MR = 1.0 / 0.25 * 0.62
        heavier = planner.plan(4000.0, 25.0)
    finally:
        planner.STRUCT_COEFF, planner.MAX_STAGE_MR = saved_coeff, saved_mr

    assert heavier is not baseline
    assert heavier.gross_kg != pytest.approx(baseline.gross_kg, rel=1e-6)


def test_different_arguments_are_different_cache_entries():
    import cadflow.planner as planner
    from generate_propulsion_trajectory_corpus import load_coupling

    load_coupling()
    planner.clear_plan_cache()
    a = planner.plan(1000.0, 25.0)
    b = planner.plan(1000.0, 50.0)
    c = planner.plan(1000.0, 25.0, of_ratio=2.0)
    assert a is not b and a is not c
    assert b.gross_kg > a.gross_kg          # heavier payload, heavier vehicle
