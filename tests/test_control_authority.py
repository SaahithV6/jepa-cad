"""Control authority and bandwidth, the gap the packet declared twice.

A launch vehicle autopilot is squeezed from both ends. It has to be fast enough
to hold attitude against the aerodynamic moment at max-Q, and slow enough to
leave the bending modes and the slosh modes alone. Both of those frequencies
are computed elsewhere in this package now, so the window between them can be
checked instead of assumed.

Authority is the more basic question and the one a stability analysis alone will
never raise: a statically stable vehicle has to spend gimbal deflection
overcoming its own fins. Size fins for static margin without pricing that and
the result is a vehicle that is stable and cannot be steered.

The tests below pin two things this module got wrong on the first pass. The
weathercock frequency has to be derived -- a plausible-sounding guess of 0.3 Hz
was seven times off the real 2.2 Hz, and that difference decides whether a
slosh mode lands on top of it. And frequencies may only be compared across
conditions that coexist: end-of-flight slosh in vacuum against a max-Q control
requirement is two moments that never happen together.
"""

import math

import pytest

from cadflow.control_authority import (
    TYPICAL_GIMBAL_LIMIT_DEG, aero_moment_nm, bandwidth_window, check,
    rigid_body_pitch_hz)

# The packet vehicle, at max-Q.
Q, R, CNA = 87600.0, 0.335, 13.69
S = math.pi * R * R
CP, CG, IPITCH = 0.857, 1.861, 2163.1
THRUST = 1806.6 * 9.80665 * 4.5


def test_fins_cost_control_authority():
    """The finding: stable, and short of gimbal to steer.

    14.5 degrees required against the 8 a production engine typically offers.
    The vehicle is not fighting an instability -- it is fighting its own fins,
    which were sized for static margin with nothing pricing what they cost.
    """
    res = check(q_pa=Q, reference_area_m2=S, cn_alpha=CNA,
                alpha_rad=math.radians(5.0), cp_station_m=CP, cg_station_m=CG,
                thrust_n=THRUST)
    assert res.statically_stable
    assert not res.has_authority
    assert res.required_gimbal_deg == pytest.approx(14.5, abs=0.5)
    assert res.utilisation > 1.5
    assert any("fins" in n for n in res.notes)


def test_the_weathercock_frequency_is_derived_not_assumed():
    """2.23 Hz from q S CNa arm / I, not from a plausible-sounding guess.

    The first version of this module took 0.3 Hz as an input. The real value is
    seven times that, and the gap decides whether the stage-1 slosh mode is
    coincident with it or an octave away.
    """
    f = rigid_body_pitch_hz(q_pa=Q, reference_area_m2=S, cn_alpha=CNA,
                            cp_station_m=CP, cg_station_m=CG,
                            pitch_inertia_kg_m2=IPITCH)
    expected = math.sqrt(Q * S * CNA * abs(CP - CG) / IPITCH) / (2 * math.pi)
    assert f == pytest.approx(expected, rel=1e-12)
    assert f == pytest.approx(2.23, abs=0.05)


def test_slosh_lands_on_the_weathercock_mode():
    """Within 11%, at the condition where the aero moment is largest.

    This is what the integration buys. Neither the slosh model nor the
    stability model says anything alarming on its own; the two frequencies
    coinciding is the classic launch vehicle coupling problem, and it is only
    visible when both are computed for the same flight condition.
    """
    f_rigid = rigid_body_pitch_hz(q_pa=Q, reference_area_m2=S, cn_alpha=CNA,
                                  cp_station_m=CP, cg_station_m=CG,
                                  pitch_inertia_kg_m2=IPITCH)
    f_slosh = 2.48                      # stage 1 at max-Q, from cadflow.slosh
    assert 0.8 <= f_slosh / f_rigid <= 1.25, (f_slosh, f_rigid)


def test_an_unstable_vehicle_has_no_pitch_frequency():
    """It diverges rather than oscillating.

    Returning a frequency anyway would invite a comparison against slosh that
    means nothing, since there is no oscillation to couple with.
    """
    with pytest.raises(ValueError, match="unstable"):
        rigid_body_pitch_hz(q_pa=Q, reference_area_m2=S, cn_alpha=CNA,
                            cp_station_m=2.5, cg_station_m=1.861,
                            pitch_inertia_kg_m2=IPITCH)


def test_the_bandwidth_window_can_be_empty_and_says_so():
    """Reporting the two bounds separately would let a reader assume a gap.

    Needing 6.7 Hz to fly while slosh caps bandwidth at 0.5 Hz is not a tight
    design, it is an impossible one, and the module has to name that rather
    than print two numbers and leave the subtraction to the reader.
    """
    w = bandwidth_window(first_bending_hz=49.5, lowest_slosh_hz=2.48,
                         rigid_body_hz=2.23)
    assert not w["window_exists"]
    assert w["limited_by"] == "slosh"
    assert "No usable band" in w["note"]
    assert w["lower_bound_hz"] > w["upper_bound_hz"]


def test_a_workable_vehicle_reports_a_window():
    """The check must be able to pass, or it is not a check.

    A slow, stiff vehicle with well-baffled tanks has room between its
    rigid-body response and its flexible modes.
    """
    w = bandwidth_window(first_bending_hz=60.0, lowest_slosh_hz=40.0,
                         rigid_body_hz=0.4)
    assert w["window_exists"]
    assert w["upper_bound_hz"] > w["lower_bound_hz"]
    assert "must sit above" in w["note"]


def test_the_moment_sign_identifies_stability_not_just_magnitude():
    """Centre of pressure aft of the centre of gravity is the stable case.

    Both arrangements produce a moment the engine must handle; which one it is
    changes whether losing gimbal means losing accuracy or losing the vehicle.
    """
    stable_m, stable = aero_moment_nm(
        q_pa=Q, reference_area_m2=S, cn_alpha=CNA, alpha_rad=0.05,
        cp_station_m=0.857, cg_station_m=1.861)
    unstable_m, unstable = aero_moment_nm(
        q_pa=Q, reference_area_m2=S, cn_alpha=CNA, alpha_rad=0.05,
        cp_station_m=2.5, cg_station_m=1.861)
    assert stable and not unstable
    assert stable_m > 0 and unstable_m > 0


def test_a_gimbal_at_the_centre_of_gravity_is_refused():
    """It produces no moment, so it cannot steer.

    Returning an infinite deflection or silently dividing by zero would both be
    worse than saying the configuration does not work.
    """
    with pytest.raises(ValueError, match="cannot be steered"):
        check(q_pa=Q, reference_area_m2=S, cn_alpha=CNA, alpha_rad=0.05,
              cp_station_m=CP, cg_station_m=CG, thrust_n=THRUST,
              gimbal_station_m=CG)


def test_more_thrust_buys_authority():
    """Direction check on the one lever that is not a geometry change."""
    kw = dict(q_pa=Q, reference_area_m2=S, cn_alpha=CNA,
              alpha_rad=math.radians(5.0), cp_station_m=CP, cg_station_m=CG)
    weak = check(thrust_n=THRUST, **kw)
    strong = check(thrust_n=3.0 * THRUST, **kw)
    assert strong.required_gimbal_deg < weak.required_gimbal_deg
    assert strong.available_gimbal_deg == TYPICAL_GIMBAL_LIMIT_DEG


def _fake_fins(margin_cal: float) -> dict:
    """Fins whose normal-force slope falls with target margin, as real ones do.

    Smaller margin means smaller fins means less CNa and a centre of pressure
    closer to the centre of gravity. Both effects reduce the aerodynamic moment
    the engine has to overcome, which is why the trade works at all.
    """
    return {"cna_total": 2.0 + 7.8 * margin_cal,
            "cp_z_m": CG - margin_cal * 0.67,
            "span_m": 0.2 + 0.3 * margin_cal}


def test_the_loop_trades_margin_for_authority_and_keeps_what_it_can():
    """It stops at the first margin that fits, not the smallest one.

    Shrinking fins as far as possible would throw away stability the engine
    never needed. The point is to spend the minimum margin that buys
    steerability.
    """
    from cadflow.control_authority import trade_margin_for_authority

    out = trade_margin_for_authority(
        _fake_fins, q_pa=Q, reference_area_m2=S, alpha_rad=math.radians(5.0),
        cg_station_m=CG, thrust_n=THRUST, body_diameter_m=0.67)
    assert out["converged"]
    assert out["margin_cal"] < 1.5
    assert out["margin_cal"] >= 0.5
    # every step before the last must have failed, or it stopped too late
    assert all(not s["has_authority"] for s in out["steps"][:-1])
    assert out["steps"][-1]["has_authority"]


def test_it_refuses_to_make_the_vehicle_unstable_to_gain_authority():
    """Below half a caliber the trade stops, converged or not.

    A vehicle that can only be steered by giving up stability is a different
    design rather than a repaired one, and returning it as a success would hide
    that.
    """
    from cadflow.control_authority import (
        MIN_STATIC_MARGIN_CAL, trade_margin_for_authority)

    def stubborn(margin_cal: float) -> dict:
        # Fins that barely shrink, so no reachable margin ever gives authority.
        return {"cna_total": 40.0, "cp_z_m": CG - margin_cal * 0.67,
                "span_m": 1.0}

    out = trade_margin_for_authority(
        stubborn, q_pa=Q, reference_area_m2=S, alpha_rad=math.radians(5.0),
        cg_station_m=CG, thrust_n=THRUST, body_diameter_m=0.67)
    assert not out["converged"]
    assert out["fins"] is None
    assert all(s["margin_cal"] >= MIN_STATIC_MARGIN_CAL - 1e-9
               for s in out["steps"])
    assert "larger gimbal" in out["note"] or "more thrust" in out["note"]


def test_a_vehicle_that_already_steers_is_left_alone():
    """No trade when none is needed; the first margin tried is the answer."""
    from cadflow.control_authority import trade_margin_for_authority

    out = trade_margin_for_authority(
        lambda m: {"cna_total": 1.5, "cp_z_m": CG - m * 0.67, "span_m": 0.1},
        q_pa=Q, reference_area_m2=S, alpha_rad=math.radians(5.0),
        cg_station_m=CG, thrust_n=THRUST, body_diameter_m=0.67)
    assert out["converged"]
    assert out["margin_cal"] == pytest.approx(1.5)
    assert len(out["steps"]) == 1


def test_a_verdict_is_labelled_by_whether_it_survives_the_heuristics():
    """"Fails under our rule" and "fails under every rule" are different claims.

    Both bounds of the bandwidth window come from rules of thumb. A conclusion
    that flips when a defensible factor changes is a prompt for coupled
    analysis; one that holds across the whole range is a property of the
    vehicle. Reporting them identically would let a soft finding read as a hard
    one, which is the same class of error as a yield margin standing in for a
    buckling margin.
    """
    hard = bandwidth_window(first_bending_hz=49.5, lowest_slosh_hz=2.48,
                            rigid_body_hz=1.54)
    soft = bandwidth_window(first_bending_hz=49.5, lowest_slosh_hz=2.48,
                            rigid_body_hz=0.62)
    assert not hard["window_exists"] and hard["robust_to_heuristics"]
    assert not soft["window_exists"] and not soft["robust_to_heuristics"]
    assert "property of the vehicle" in hard["robustness_note"]
    assert "coupled analysis" in soft["robustness_note"]


def test_the_real_obstacle_is_how_close_the_modes_are():
    """1.61, and that is the number that decides it.

    No separation factor opens a window when the lowest flexible mode sits at
    1.6 times the rigid-body mode: any bandwidth that dominates one is near the
    other. Recording the ratio makes the physics visible instead of leaving the
    reader to infer it from two rejected bounds.
    """
    w = bandwidth_window(first_bending_hz=49.5, lowest_slosh_hz=2.48,
                         rigid_body_hz=1.54)
    assert w["flexible_over_rigid_ratio"] == pytest.approx(1.61, abs=0.02)
    assert w["limited_by"] == "slosh"


def test_a_wide_separation_passes_robustly():
    """The label must attach to good news as well as bad."""
    w = bandwidth_window(first_bending_hz=60.0, lowest_slosh_hz=40.0,
                         rigid_body_hz=0.4)
    assert w["window_exists"] and w["robust_to_heuristics"]


def test_modes_below_crossover_cannot_be_notched():
    """The distinction the bandwidth check was missing.

    A mode above crossover is gain stabilised -- notch it and the loop never
    excites it. A mode *below* crossover sits where the loop needs gain to fly
    the vehicle, so notching it removes the control authority along with the
    mode. It has to be phase stabilised instead: modelled in the controller and
    closed around with the right phase.

    Reporting both as "no usable bandwidth window" was arithmetically right and
    engineeringly misleading. Real launch vehicles fly with slosh near the
    control frequencies routinely; Saturn V phase stabilised its slosh modes.
    """
    from cadflow.control_authority import mode_disposition

    d = mode_disposition(crossover_hz=3.85,
                         modes={"slosh": 2.48, "first bending": 49.5})
    assert d["modes"]["slosh"]["stabilisation"] == "phase"
    assert d["modes"]["first bending"]["stabilisation"] == "gain"
    assert d["requires_phase_stabilisation"] == ["slosh"]
    assert "cannot be notched" in d["modes"]["slosh"]["note"]


def test_a_mode_just_above_crossover_is_flagged_as_tight():
    """Notching within a factor of three still costs phase margin.

    Above crossover is not automatically comfortable. A notch close to the
    frequency the loop is working at eats the margin that keeps it stable, so
    the two cases are reported differently.
    """
    from cadflow.control_authority import mode_disposition

    d = mode_disposition(crossover_hz=2.0, modes={"slosh": 3.0})
    assert d["modes"]["slosh"]["stabilisation"] == "gain, tight"
    assert "phase margin" in d["modes"]["slosh"]["note"]


def test_a_vehicle_with_everything_above_crossover_needs_no_special_treatment():
    """The check has to be able to report the easy case as easy."""
    from cadflow.control_authority import mode_disposition

    d = mode_disposition(crossover_hz=1.0,
                         modes={"slosh": 12.0, "first bending": 60.0})
    assert not d["requires_phase_stabilisation"]
    assert "conventional autopilot" in d["verdict"]


def test_the_verdict_reads_correctly_for_one_mode():
    """"slosh sits", not "slosh sit" -- the packet is read by people."""
    from cadflow.control_authority import mode_disposition

    d = mode_disposition(crossover_hz=3.85, modes={"slosh": 2.48})
    assert "slosh sits below crossover" in d["verdict"]
    assert " sit below" not in d["verdict"]
