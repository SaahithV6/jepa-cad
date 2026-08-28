"""Tank pressure, and what it costs.

This subsystem was absent entirely. The barrel was sized for axial load and
bending and checked against buckling; the mass budget carried structure and
propellant and nothing else. Both omissions point the same way and cancel
nothing: the vehicle was missing the helium and its bottle, and its wall had
never been asked to carry hoop.

The mass relation is checked against its own derivation rather than against a
table, because it has one. A thin sphere at t = pR/(2 sigma) weighs
(3/2)(rho/sigma) p V regardless of radius, and a test that only compared against
a remembered number could not tell a correct implementation from a plausible
one.
"""

import math

import pytest

from cadflow.pressurization import (
    BOTTLE_STORAGE_PA, COLLAPSE_FACTOR, PROPELLANTS, R_HELIUM, TANK_FOS,
    feed_system_verdict, helium_mass_kg, hoop_stress_pa,
    pressure_fed_tank_pressure_pa, pressure_vessel_mass_kg,
    stage_pressurisation, tank_pressure)

AL = dict(density_kg_m3=2700.0, allowable_pa=280e6)


def test_vessel_mass_is_independent_of_radius():
    """The whole content of the relation.

    t = pR/(2 sigma) makes the wall thicker as the sphere grows, and the surface
    grows too, but the volume grows faster in exactly the way that cancels. A
    number that moved with radius would mean the derivation was not implemented.
    """
    a = pressure_vessel_mass_kg(300e5, 0.05, **AL)
    b = pressure_vessel_mass_kg(300e5, 0.05, **AL)
    assert a == b
    # doubling volume doubles mass, at any pressure
    assert pressure_vessel_mass_kg(300e5, 0.10, **AL) == pytest.approx(2 * a)


def test_vessel_mass_matches_a_hand_built_sphere():
    """Built the long way: pick a radius, get t, weigh the shell.

    Checked against first principles rather than against the closed form the
    module uses, so an algebra slip in the closed form cannot hide.
    """
    p, R, sigma, rho = 300e5, 0.20, 280e6, 2700.0
    t = p * R / (2 * sigma)
    m_long_way = rho * 4 * math.pi * R ** 2 * t
    V = (4.0 / 3.0) * math.pi * R ** 3
    assert pressure_vessel_mass_kg(p, V, density_kg_m3=rho,
                                   allowable_pa=sigma) == pytest.approx(
        m_long_way, rel=1e-12)


def test_a_cylinder_costs_exactly_double_a_sphere():
    """Twice the membrane stress for the same wall, so twice the mass.

    This is why pressurant bottles are spherical wherever packaging allows, and
    the factor being exactly 2 is a check on both branches at once.
    """
    v = 0.08
    sph = pressure_vessel_mass_kg(300e5, v, spherical=True, **AL)
    cyl = pressure_vessel_mass_kg(300e5, v, spherical=False, **AL)
    assert cyl == pytest.approx(2.0 * sph, rel=1e-12)


def test_a_cryogen_needs_real_ullage_and_kerosene_needs_almost_none():
    """Vapour pressure is the difference, and it is a large one.

    LOX is stored at its boiling point, so its vapour pressure is an atmosphere
    and the pump needs suction head on top of that. RP-1 at room temperature has
    essentially none. A model that treated both alike would put unnecessary
    helium on one tank and cavitate the other.
    """
    lox = tank_pressure("lox")
    rp1 = tank_pressure("rp1")
    assert lox.vapour_pa > 100 * rp1.vapour_pa
    assert lox.ullage_pa > rp1.ullage_pa


def test_ullage_is_vapour_pressure_plus_suction_head():
    """Recomputed from the propellant, not remembered."""
    t = tank_pressure("lox", npsh_m=20.0)
    rho = PROPELLANTS["lox"]["density"]
    assert t.npsh_pa == pytest.approx(rho * 9.80665 * 20.0)
    assert t.ullage_pa == pytest.approx(101325.0 + t.npsh_pa)
    assert t.design_pa == pytest.approx(t.ullage_pa * TANK_FOS)


def test_acceleration_supplies_head_the_helium_then_need_not():
    """A burning stage pressurises its own inlet.

    Four g with two metres of propellant above the outlet is tens of kPa for
    free. A design that ignores it carries helium it does not need, which is
    mass charged against the payload for nothing.
    """
    still = tank_pressure("lox", npsh_m=20.0)
    under_thrust = tank_pressure("lox", npsh_m=20.0, acceleration_g=4.0,
                                 head_height_m=2.0)
    assert under_thrust.ullage_pa < still.ullage_pa
    assert "acceleration supplies" in under_thrust.note


def test_enough_acceleration_covers_the_whole_suction_head():
    """And the ullage falls to vapour pressure, not below it.

    Below vapour pressure the propellant boils in the tank. A model that let
    acceleration drive ullage negative would be describing a tank of gas.
    """
    t = tank_pressure("lox", npsh_m=20.0, acceleration_g=6.0,
                      head_height_m=5.0)
    assert t.ullage_pa == pytest.approx(PROPELLANTS["lox"]["vapour_pa"])
    assert "covers the" in t.note


def test_helium_mass_is_the_gas_law_with_the_collapse_factor():
    m = helium_mass_kg(300_000.0, 1.5, temperature_k=250.0)
    assert m == pytest.approx(
        300_000.0 * 1.5 / (R_HELIUM * 250.0) * COLLAPSE_FACTOR)


def test_storage_pressure_barely_changes_bottle_mass():
    """It sets the bottle's size, not its weight.

    p*V is fixed by the helium the tank needs, and membrane mass goes as p*V, so
    a higher-pressure bottle is smaller rather than lighter. Worth a test because
    the intuition runs the other way, and a reader who expected 700 bar to save
    mass would go looking for a bug that is not there.
    """
    def bottle_for(storage_pa):
        he = 2.0
        v = he * R_HELIUM * 293.0 / storage_pa
        return pressure_vessel_mass_kg(storage_pa * TANK_FOS, v, **AL)

    assert bottle_for(700e5) == pytest.approx(bottle_for(350e5), rel=1e-9)


def test_the_tank_pressure_reaches_the_wall_as_hoop():
    """p r / t, and it is tension where everything else on this wall is not."""
    assert hoop_stress_pa(500_000.0, 0.335, 0.0008) == pytest.approx(
        500_000.0 * 0.335 / 0.0008)


def test_a_stage_is_pressurised_and_the_cost_is_a_real_fraction():
    """The flight vehicle in this repo: 974 kg of propellant at 335 mm.

    The point of the number is that it is neither negligible nor absurd. A
    pressurisation system that came out at 0.01% would mean the model was not
    doing anything; one at 20% would mean it was wrong.
    """
    r = stage_pressurisation(stage=1, propellant_mass_kg=973.89,
                             radius_m=0.335, wall_m=0.0008,
                             acceleration_g=1.0, head_height_m=2.0)
    assert r.tank_volume_m3 == pytest.approx(0.955, abs=0.05)
    frac = r.total_kg / 973.89
    assert 0.002 < frac < 0.06, r.as_dict()
    assert r.helium_kg > 0 and r.bottle_kg > r.helium_kg


def test_the_pressurisation_says_what_it_assumed():
    """The suction head is not derived and the report has to say so.

    Every other number in this module comes out of an equation. This one comes
    out of flown practice for a pump nobody here designs, and a reader cannot
    weigh the result without knowing which is which.
    """
    r = stage_pressurisation(stage=1, propellant_mass_kg=973.89,
                             radius_m=0.335, wall_m=0.0008)
    assert any("ASSUMED" in n for n in r.notes)


def test_the_tank_decides_the_feed_architecture():
    """55 bar chamber makes pressure-fed tanks absurd, and says by how much.

    Not a preference and not an assertion: the membrane relation weighs both
    options for the same volume and the ratio settles it.
    """
    v = feed_system_verdict(55e5, 0.955, **AL)
    assert v["verdict"] == "pump-fed"
    assert v["pressure_fed_tank_pa"] == pytest.approx(55e5 * 1.3)
    assert v["ratio"] > 10.0
    assert "decided by the tank" in v["note"]


def test_a_low_chamber_pressure_can_go_pressure_fed():
    """The verdict must be able to come out the other way.

    A check that always returns the same answer establishes nothing about the
    thing it checks. Small pressure-fed stages are real hardware.
    """
    v = feed_system_verdict(4e5, 0.05, **AL)
    assert v["ratio"] < 3.0


def test_an_unknown_propellant_is_refused_not_guessed():
    with pytest.raises(KeyError, match="unknown propellant"):
        tank_pressure("unobtainium")


def test_zero_pressure_costs_nothing_and_zero_wall_is_refused():
    assert pressure_vessel_mass_kg(0.0, 1.0, **AL) == 0.0
    assert helium_mass_kg(0.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        hoop_stress_pa(1e5, 0.3, 0.0)


def test_the_axial_term_is_exactly_half_the_hoop():
    """pr/2t against pr/t, which is why cylinders split lengthways.

    A slip here would be invisible in a margin check -- both numbers are the
    right order -- and would change whether the wall reads as in tension.
    """
    from cadflow.pressurization import wall_load_state

    w = wall_load_state(pressure_pa=454_000.0, radius_m=0.335, wall_m=0.0008)
    assert w.axial_pressure_pa == pytest.approx(0.5 * w.hoop_pa, rel=1e-12)


def test_a_pressurised_tank_is_not_in_compression_at_flight_load():
    """The finding: this wall may have no buckling mode to be sized against.

    The packet thickens the wall for buckling under combined axial and bending
    load. At the flight stresses it reports -- 36 MPa axial -- and the ullage
    this module derives, the pressure term alone puts the wall in net tension.
    A shell in tension does not go unstable.

    This does not say the repair was wrong. It says the question was never
    asked, because nothing computed the pressure.
    """
    from cadflow.pressurization import wall_load_state

    w = wall_load_state(pressure_pa=454_000.0, radius_m=0.335, wall_m=0.0008,
                        axial_flight_pa=-35.6e6, bending_pa=0.0)
    assert not w.in_compression
    assert w.net_axial_pa > 0


def test_enough_bending_still_puts_it_into_compression():
    """The relief is not unconditional and the check must be able to say no.

    A test that only ever confirmed tension would make this a claim rather than
    a check. Bending at max-Q is one-sided and can exceed the pressure term.
    """
    from cadflow.pressurization import wall_load_state

    w = wall_load_state(pressure_pa=454_000.0, radius_m=0.335, wall_m=0.0008,
                        axial_flight_pa=-35.6e6, bending_pa=-200e6)
    assert w.in_compression


def test_an_unpressurised_shell_is_in_compression_under_any_axial_load():
    """The interstage. It carries the same load with no pressure to relieve it.

    Which is the other half of the finding: relief belongs to the tanks, and the
    dry structure between them has none.
    """
    from cadflow.pressurization import wall_load_state

    w = wall_load_state(pressure_pa=0.0, radius_m=0.335, wall_m=0.0008,
                        axial_flight_pa=-35.6e6)
    assert w.in_compression and w.hoop_pa == 0.0


def test_von_mises_uses_both_membrane_stresses():
    """Hoop tension with axial tension is not the same state as hoop alone."""
    from cadflow.pressurization import wall_load_state

    w = wall_load_state(pressure_pa=454_000.0, radius_m=0.335, wall_m=0.0008)
    s1, s2 = w.hoop_pa, w.net_axial_pa
    assert w.von_mises_pa == pytest.approx(math.sqrt(s1*s1 - s1*s2 + s2*s2))
    assert w.von_mises_pa < w.hoop_pa   # biaxial tension is less severe


def test_pressurisation_is_charged_to_the_mass_closure():
    """Reported and charged, not reported instead of charged.

    The packet computes this mass and says the budget does not carry it. If the
    closure arithmetic beside that sentence then omits the same number, the two
    halves of one page disagree -- which is the shape of every disconnection
    defect this project has found.
    """
    from cadflow.assembly import VehicleAssembly, mass_closure

    asm = VehicleAssembly()
    without = mass_closure(asm, 100.0, liftoff_thrust_n=0.0)
    with_ = mass_closure(asm, 100.0, liftoff_thrust_n=0.0,
                         pressurisation_kg=14.1)
    assert with_["pressurisation_kg"] == pytest.approx(14.1)
    assert with_["accounted_kg"] == pytest.approx(without["accounted_kg"] + 14.1)
    assert with_["slack_kg"] == pytest.approx(without["slack_kg"] - 14.1)


def test_pressurisation_can_be_what_breaks_the_closure():
    """A charge that cannot change the verdict is not a charge.

    If 14 kg of helium and bottles can never turn a closing budget into a
    failing one, then adding it to the arithmetic was decoration.
    """
    from cadflow.assembly import VehicleAssembly, mass_closure

    asm = VehicleAssembly()
    assert mass_closure(asm, 10.0, pressurisation_kg=5.0)["closes"]
    assert not mass_closure(asm, 10.0, pressurisation_kg=15.0)["closes"]


def _press_stack():
    """The four-stage vehicle, tanks sized at each stage's own radius."""
    from cadflow.pressurization import stage_pressurisation

    radii = [0.338, 0.311, 0.286, 0.263]
    props = [1006.6, 261.7, 68.0, 23.9]
    return [stage_pressurisation(stage=i + 1, propellant_mass_kg=m,
                                 radius_m=r, wall_m=0.0008,
                                 acceleration_g=1.0, head_height_m=1.1)
            for i, (m, r) in enumerate(zip(props, radii))]


def test_the_domes_are_charged_because_their_relief_is_credited():
    """The loop this module opened, closed.

    wall_load_state credits pr/2t of axial tension, and that credit is what
    puts the wall in net tension with no compressive buckling mode. It exists
    only because the tank has ends -- and the packet's own mass closure lists
    tank domes among the things the geometry does not draw. Taking the credit
    without weighing the part is the overclaim direction.
    """
    r = _press_stack()[0]
    assert r.dome_kg > 0
    assert r.total_kg == pytest.approx(r.helium_kg + r.bottle_kg + r.dome_kg)


def test_the_domes_are_set_by_gauge_not_by_pressure():
    """Which is the whole reason they weigh what they do.

    At this radius and ullage the membrane requirement is about a tenth of a
    millimetre. Nobody welds that. The mass is set by what can be built, and a
    model that reported the membrane number would understate the tank ends
    ninefold.
    """
    from cadflow.pressurization import DOME_MIN_GAUGE_M, dome_mass_kg

    d = dome_mass_kg(pressure_pa=454e3, radius_m=0.338,
                     density_kg_m3=8190.0, allowable_pa=700e6)
    assert d["gauge_limited"]
    assert d["thickness_membrane_m"] < 0.2e-3
    assert d["thickness_used_m"] == pytest.approx(DOME_MIN_GAUGE_M)


def test_a_high_pressure_dome_is_membrane_limited_instead():
    """The gauge floor must not swallow cases where pressure genuinely governs.

    A floor that always wins is not a floor, it is a constant, and it would
    make dome mass independent of the pressure it holds.
    """
    from cadflow.pressurization import dome_mass_kg

    d = dome_mass_kg(pressure_pa=200e5, radius_m=0.338,
                     density_kg_m3=8190.0, allowable_pa=700e6)
    assert not d["gauge_limited"]
    assert d["thickness_used_m"] == pytest.approx(d["thickness_membrane_m"])


def test_the_smallest_stage_cannot_afford_its_own_tankage():
    """An architecture the mass fractions permit and physics does not.

    A structural coefficient is a fraction, so structure scales with
    propellant. Minimum gauge scales with nothing. Below some size they cross,
    and this vehicle's fourth stage needs 175% of its whole structural
    allowance for tankage alone -- which is why flown structural coefficients
    get worse for small stages rather than staying flat.
    """
    from cadflow.pressurization import stage_feasibility

    class S:
        def __init__(s, pm, sm):
            s.prop_mass_kg, s.struct_mass_kg = pm, sm

    stack = [S(1006.6, 349.5), S(261.7, 90.9), S(68.0, 23.6), S(23.9, 8.3)]
    rows = stage_feasibility(stack, _press_stack())
    assert [r["feasible"] for r in rows] == [True, True, True, False]
    # and the trend is monotone, which is the physical claim
    fracs = [r["fraction_of_allowance"] for r in rows]
    assert fracs == sorted(fracs)


def test_the_verdict_says_what_would_overturn_it():
    """It rests on an assumed gauge, so it has to report the sensitivity.

    A finding that depends on one constant and does not say what that constant
    would have to be is reporting the assumption rather than the vehicle. Stage
    4 needs 0.57 mm domes to fit; stage 3 clears the assumed 1.0 mm by only 36%.
    """
    from cadflow.pressurization import stage_feasibility

    class S:
        def __init__(s, pm, sm):
            s.prop_mass_kg, s.struct_mass_kg = pm, sm

    stack = [S(1006.6, 349.5), S(261.7, 90.9), S(68.0, 23.6), S(23.9, 8.3)]
    rows = stage_feasibility(stack, _press_stack())
    assert rows[3]["break_even_gauge_m"] == pytest.approx(0.57e-3, abs=0.05e-3)
    assert rows[2]["break_even_gauge_m"] == pytest.approx(1.36e-3, abs=0.1e-3)
    # a feasible stage's break-even is above the assumed gauge, by definition
    from cadflow.pressurization import DOME_MIN_GAUGE_M
    for r in rows:
        if r["feasible"]:
            assert r["break_even_gauge_m"] > DOME_MIN_GAUGE_M


def test_a_stage_that_cannot_fit_even_a_zero_thickness_dome_reports_zero():
    """The degenerate end, where helium and bottle alone exceed the allowance.

    Inverting the linear relation would otherwise return a negative thickness,
    which is not a gauge anybody can be asked to weld.
    """
    from cadflow.pressurization import stage_feasibility

    class S:
        def __init__(s, pm, sm):
            s.prop_mass_kg, s.struct_mass_kg = pm, sm

    # Below the helium and bottle themselves (0.24 kg for this stage), so no
    # dome thickness at all makes it fit. 0.5 kg was the first attempt and is
    # not degenerate -- it leaves 0.26 kg of room and yields a real, if absurd,
    # 0.018 mm gauge. The fixture was what was wrong.
    tiny = [S(23.9, 0.1)]
    rows = stage_feasibility(tiny, [_press_stack()[3]])
    assert rows[0]["break_even_gauge_m"] == 0.0
    assert not rows[0]["feasible"]


def test_the_dome_gauge_is_the_projects_gauge_not_a_second_one():
    """One manufacturing limit, one number.

    This was an independent 1.0 mm for one revision while structural_sizing used
    0.8 mm for "spun/welded shells" -- and the packet's own wall driver reads
    "minimum gauge", so both constants described the same limit on the same
    vehicle in the same alloy and disagreed by 25%. A dome is a spun and welded
    shell.
    """
    from cadflow.pressurization import DOME_MIN_GAUGE_M
    from cadflow.structural_sizing import T_MIN_M

    assert DOME_MIN_GAUGE_M == T_MIN_M


def test_the_affordability_verdict_survives_the_gauge_correction():
    """The finding must not have been an artefact of the wrong constant.

    Dome mass is linear in gauge, so 0.8 against 1.0 takes 20% off the tank
    ends. Stage 4 still cannot pay for its own: its break-even gauge is 0.57 mm,
    below both candidates, which is exactly what the break-even column was added
    to make checkable.
    """
    from cadflow.pressurization import stage_feasibility

    class S:
        def __init__(s, pm, sm):
            s.prop_mass_kg, s.struct_mass_kg = pm, sm

    stack = [S(1006.6, 349.5), S(261.7, 90.9), S(68.0, 23.6), S(23.9, 8.3)]
    rows = stage_feasibility(stack, _press_stack())
    assert not rows[3]["feasible"]
    assert rows[3]["break_even_gauge_m"] < 0.8e-3
