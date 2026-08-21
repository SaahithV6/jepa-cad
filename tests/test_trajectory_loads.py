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
