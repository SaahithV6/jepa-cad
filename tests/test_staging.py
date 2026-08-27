"""Stage separation: does the spent stage clear before the next engine lights?

The trajectory already stages -- it drops spent structure at burnout and lights
the next engine, and the mass bookkeeping is right. It never asked whether those
two events can happen in that order without the two bodies occupying the same
space. Separation takes a second or two at a metre or two per second, and a
vacuum plume spreads far wider than the nozzle that produced it.

For the packet vehicle every separation clears in well under a second, so this
check passes. It is worth having anyway: a check that can only ever pass is not
a check, and the tests below pin that this one fails when it should.
"""

import math

import pytest

from cadflow.staging import (
    DEFAULT_IMPULSE_PER_KG, PLUME_CLEARANCE_DIAMETERS, check_separation,
    coast_for_clearance_s, relative_velocity_m_s)


def test_relative_velocity_uses_both_masses():
    """J (1/m1 + 1/m2), not J/m.

    The impulse acts on both bodies. Dividing by one mass understates the
    closing rate, and understating it reports a recontact that would not
    happen -- a false alarm that would push the design toward a longer coast
    and cost real velocity.
    """
    v = relative_velocity_m_s(400.0, 344.5, 488.2)
    assert v == pytest.approx(400.0 * (1 / 344.5 + 1 / 488.2), rel=1e-12)
    assert v > 400.0 / 344.5          # strictly more than either alone


def test_a_lighter_spent_stage_separates_faster():
    """Same impulse, less mass to move.

    Direction check on the term that matters most for upper stages, where the
    jettisoned structure is small.
    """
    heavy = relative_velocity_m_s(400.0, 800.0, 500.0)
    light = relative_velocity_m_s(400.0, 200.0, 500.0)
    assert light > heavy


def test_the_packet_vehicle_clears_every_separation():
    """Under 0.6 s of coast against 1.0 m of required clearance.

    Pinned so a design change that shrinks the separation system or grows the
    vehicle cannot quietly stop clearing.
    """
    for spent, upper in ((344.5, 488.2), (89.6, 145.4), (23.3, 56.3)):
        need = coast_for_clearance_s(spent_mass_kg=spent, upper_mass_kg=upper,
                                     body_diameter_m=0.670)
        assert need < 1.0, (spent, upper, need)
        r = check_separation(stage_index=1, spent_mass_kg=spent,
                             upper_mass_kg=upper, body_diameter_m=0.670,
                             coast_s=1.0)
        assert r.clears


def test_a_short_coast_on_a_fat_vehicle_fails():
    """The check must be able to say no.

    Clearance scales with diameter while separation velocity does not, so a
    wide vehicle on a brief coast is exactly where recontact lives.
    """
    r = check_separation(stage_index=1, spent_mass_kg=2000.0,
                         upper_mass_kg=8000.0, body_diameter_m=5.0,
                         coast_s=0.5)
    assert not r.clears
    assert r.clearance_m < r.required_clearance_m
    assert any("gap reaches" in n for n in r.notes)


def test_required_clearance_is_plume_not_nozzle():
    """One and a half diameters, because a vacuum plume is not the nozzle exit.

    Sizing this from the nozzle would clear the hardware and not the exhaust.
    """
    r = check_separation(stage_index=1, spent_mass_kg=100.0, upper_mass_kg=300.0,
                         body_diameter_m=2.0, coast_s=3.0)
    assert r.required_clearance_m == pytest.approx(
        PLUME_CLEARANCE_DIAMETERS * 2.0)
    assert PLUME_CLEARANCE_DIAMETERS >= 1.0


def test_differential_drag_is_not_credited_by_default():
    """Most separations are too high for air to help, so none is assumed.

    Crediting help that is not there is the wrong direction to be wrong in.
    """
    plain = check_separation(stage_index=1, spent_mass_kg=100.0,
                             upper_mass_kg=300.0, body_diameter_m=1.0,
                             coast_s=2.0)
    helped = check_separation(stage_index=1, spent_mass_kg=100.0,
                              upper_mass_kg=300.0, body_diameter_m=1.0,
                              coast_s=2.0, differential_decel_m_s2=3.0)
    assert helped.clearance_m > plain.clearance_m
    assert any("No differential drag credited" in n for n in plain.notes)


def test_the_coast_needed_inverts_the_check():
    """The sizer and the checker must agree, or one of them is decoration."""
    kw = dict(spent_mass_kg=344.5, upper_mass_kg=488.2, body_diameter_m=0.670)
    t = coast_for_clearance_s(**kw)
    r = check_separation(stage_index=1, coast_s=t, **kw)
    assert r.clearance_m == pytest.approx(r.required_clearance_m, rel=1e-9)


def test_the_unmodelled_failure_modes_are_declared():
    """Tip-off and plume impingement are real and absent.

    A clearance check that implied it covered separation would be the same
    error as a yield margin standing in for buckling.
    """
    r = check_separation(stage_index=1, spent_mass_kg=100.0, upper_mass_kg=300.0,
                         body_diameter_m=1.0, coast_s=2.0)
    joined = " ".join(r.notes)
    assert "Tip-off" in joined and "impingement" in joined


def test_degenerate_inputs_are_refused():
    with pytest.raises(ValueError):
        relative_velocity_m_s(400.0, 0.0, 500.0)
    with pytest.raises(ValueError):
        check_separation(stage_index=1, spent_mass_kg=100.0, upper_mass_kg=300.0,
                         body_diameter_m=0.0, coast_s=1.0)
