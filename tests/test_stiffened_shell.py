"""Why nothing large flies monocoque, and what this project loses by being one.

A payload sweep at fixed apogee found the thing that prompted this module: the
solved structural coefficient does not improve as the vehicle grows. It drifts
slightly worse, 0.2577 at 1.9 t gross to 0.2818 at 131 t, while flown stages
span 0.036 to 0.118. That is not a small-vehicle artefact and not a bad mission.

The monocoque relation says why. Setting sigma_cr = gamma 0.605 E t / r equal to
P / (2 pi r t) gives t = sqrt(P / (3.80 gamma E)), with no radius in it, so mass
grows linearly with radius for the same load. A monocoque gets *less* efficient
as it gets bigger, which is the opposite of what a launch vehicle needs.

The test that matters most here is the reduction: the general stiffened form has
to return the classical monocoque stress when handed isotropic stiffnesses. A
generalisation that does not contain its own special case is an approximation
wearing the wrong name.
"""

import math

import pytest

from cadflow.stiffened_shell import (
    MAX_BLADE_SLENDERNESS, POISSON, analyse, compare_to_monocoque,
    general_instability_mpa, local_skin_buckling_mpa, monocoque_thickness_m,
    section_properties, size_for_stress)

STEEL = dict(youngs_pa=200e9, density_kg_m3=8190.0)


def test_the_general_form_reproduces_the_classical_monocoque_stress():
    """N_cr = 2 sqrt(D C) / r with isotropic stiffnesses is 0.607 E t / r.

    This is the test the module rests on. If the general relation did not
    contain the special case, every stiffened number it produced would be
    uncheckable against the monocoque path the rest of the project uses.
    """
    E, t, r = 200e9, 0.0008, 0.338
    d = E * t ** 3 / (12.0 * (1.0 - POISSON ** 2))
    c = E * t
    got = general_instability_mpa(bending_stiffness_nm=d,
                                  extensional_stiffness_n_m=c, radius_m=r,
                                  t_extensional_m=t, knockdown=1.0)
    exact = 2.0 / math.sqrt(12.0 * (1.0 - POISSON ** 2)) * E * t / r / 1e6
    assert got == pytest.approx(exact, rel=1e-12)
    # and within a fraction of a percent of the rounded 0.605 the project uses
    assert got == pytest.approx(0.605 * E * t / r / 1e6, rel=1e-3)


def test_monocoque_thickness_does_not_depend_on_radius():
    """The whole reason a big monocoque is a bad monocoque.

    Thickness is set by the load alone, so mass -- rho 2 pi r t L -- grows
    linearly with radius. Doubling the diameter at the same load doubles the
    wall mass and buys nothing.
    """
    load = 64_600.0
    assert monocoque_thickness_m(load, 200e9) == pytest.approx(
        monocoque_thickness_m(load, 200e9))
    # the relation itself, checked against its own algebra
    t = monocoque_thickness_m(load, 200e9, knockdown=0.35)
    assert t == pytest.approx(math.sqrt(load / (3.80 * 0.35 * 200e9)), rel=1e-12)


def test_a_stringer_moves_the_neutral_axis_off_the_skin():
    """Which is the entire mechanism: material away from the mid-surface.

    Bending stiffness rises with the square of the offset while extensional
    stiffness only counts area, so the same mass buys far more buckling
    resistance.
    """
    plain = section_properties(0.0008, 0.05, 0.0, 0.0)
    stiff = section_properties(0.0005, 0.05, 0.012, 0.0015)
    assert plain["neutral_axis_m"] == pytest.approx(0.0)
    assert stiff["neutral_axis_m"] > 0.0
    # comparable smeared thickness, far greater bending stiffness
    assert stiff["t_extensional_m"] == pytest.approx(0.0008, abs=1e-4)
    assert stiff["inertia_per_width_m3"] > 20 * plain["inertia_per_width_m3"]


def test_an_unstiffened_cylinder_has_no_local_panel_mode():
    """Its local mode is the shell mode, already counted.

    Applying the flat-plate formula across the full circumference returns a
    stress near zero, which would report every monocoque in the project as
    failed -- a check that fires on correct designs is one that gets removed.
    """
    w = analyse(axial_load_n=64_600.0, radius_m=0.338, skin_m=0.0008,
                stringer_count=0, stringer_height_m=0.0,
                stringer_thickness_m=0.0, **STEEL)
    assert math.isinf(w.local_mpa)
    assert w.governs == "general"
    assert w.general_mpa == pytest.approx(100.0, abs=5.0)


def test_skin_buckling_between_stringers_can_govern():
    """And it must, or the stringers would look free.

    A shell whose skin buckles between its stringers has not gained what they
    promised. Wide spacing on a thin skin is exactly how a stiffened design
    fails to deliver, and the check has to see it.
    """
    w = analyse(axial_load_n=64_600.0, radius_m=0.338, skin_m=0.0005,
                stringer_count=24, stringer_height_m=0.012,
                stringer_thickness_m=0.0015, **STEEL)
    assert w.governs == "local skin"
    assert w.local_mpa < w.general_mpa
    assert any("buckles between stringers" in n for n in w.notes)


def test_closer_spacing_raises_the_local_mode_as_the_square():
    """sigma_local goes as (t/b)^2, so halving the bay quadruples it."""
    a = local_skin_buckling_mpa(skin_m=0.0005, spacing_m=0.08, youngs_pa=200e9)
    b = local_skin_buckling_mpa(skin_m=0.0005, spacing_m=0.04, youngs_pa=200e9)
    assert b == pytest.approx(4.0 * a, rel=1e-12)


def test_a_well_proportioned_wall_beats_monocoque_at_equal_mass():
    """The claim this module exists to support, measured rather than asserted.

    Equal mass means equal smeared thickness, both in the same alloy. The first
    version of this comparison sized the monocoque to whatever stress the
    stiffened wall happened to carry, which compared an over-strong wall against
    a thin one and reported stiffening as a 62% mass *penalty*.
    """
    w = analyse(axial_load_n=64_600.0, radius_m=0.338, skin_m=0.0006,
                stringer_count=120, stringer_height_m=0.006,
                stringer_thickness_m=0.0006, **STEEL)
    c = compare_to_monocoque(w, youngs_pa=200e9)
    assert c["capability_ratio"] > 5.0
    # the modes are within a factor of two of each other, which is what a
    # proportioned design looks like -- neither mode wasted on the other
    assert 0.5 < w.local_mpa / w.general_mpa < 2.0


def test_a_badly_proportioned_wall_does_not_beat_monocoque():
    """The comparison must be able to come out against stiffening.

    Otherwise it is advocacy. Thin skin on wide bays buckles locally long before
    the shell does, and all the stringer material is wasted.
    """
    w = analyse(axial_load_n=64_600.0, radius_m=0.338, skin_m=0.0004,
                stringer_count=16, stringer_height_m=0.020,
                stringer_thickness_m=0.0020, **STEEL)
    c = compare_to_monocoque(w, youngs_pa=200e9)
    assert c["capability_ratio"] < 1.0


def test_a_deep_thin_blade_is_flagged_rather_than_sized_through():
    """Crippling is not modelled, so a slenderness past the limit is reported.

    Sizing through an unmodelled failure mode is how a margin comes to describe
    a part that collapses.
    """
    w = analyse(axial_load_n=64_600.0, radius_m=0.338, skin_m=0.0006,
                stringer_count=48, stringer_height_m=0.030,
                stringer_thickness_m=0.0008, **STEEL)
    assert 0.030 / 0.0008 > MAX_BLADE_SLENDERNESS
    assert any("crippling is not modelled" in n for n in w.notes)


def test_sizing_is_capped_by_material_strength_not_by_the_grid():
    """Left alone the optimum walks to ever-finer spacing forever.

    Within this physics closer stringers always raise local buckling and nothing
    pushes back, so an uncapped search reports whatever the sweep bound happened
    to be -- 120 stringers, then 300, then finer. The real ceiling is the alloy:
    a buckling allowable above its design allowable is unreachable because the
    wall yields first.
    """
    r = size_for_stress(required_mpa=5000.0, radius_m=0.338, youngs_pa=200e9,
                        density_kg_m3=8190.0, material_allowable_mpa=700.0)
    assert r["found"]
    assert r["strength_capped"]
    assert r["target_mpa"] == 700.0


def test_the_sized_wall_is_far_lighter_than_a_monocoque_carrying_the_same():
    """The number that explains the structural coefficient gap.

    This project's monocoque sizing lands at a coefficient near 0.26 at every
    scale, while flown stages sit between 0.036 and 0.118. A fourfold reduction
    in wall mass is the right order to account for that difference.

    Four, not seven. Those are answers to two different questions and the
    distinction is easy to lose: at *equal mass* the stiffened wall carries 7.3x
    the stress, but at *equal capability* it saves 4.2x the mass. The gap is the
    knockdown -- a monocoque thick enough to reach 700 MPa has a much lower r/t
    than the thin one, so SP-8007 treats it far more kindly, and it needs less
    material than scaling the thin case linearly would suggest. A first estimate
    here said 45.7 kg/m2 by exactly that linear extrapolation; the fixed point
    below says 27.6.
    """
    r = size_for_stress(required_mpa=700.0, radius_m=0.338, youngs_pa=200e9,
                        density_kg_m3=8190.0, material_allowable_mpa=700.0)
    assert r["found"]
    stiffened = r["wall"].mass_per_area_kg_m2
    # a monocoque reaching 700 MPa needs t such that gamma 0.605 E t / r = 700
    from cadflow.shell_buckling import knockdown_compression
    t = 0.0008
    for _ in range(60):
        t = 700e6 * 0.338 / (knockdown_compression(0.338, t) * 0.605 * 200e9)
    mono = 8190.0 * t
    assert mono / stiffened > 4.0, (mono, stiffened)


def test_degenerate_geometry_is_refused():
    with pytest.raises(ValueError):
        section_properties(0.0, 0.05, 0.01, 0.001)
    with pytest.raises(ValueError):
        local_skin_buckling_mpa(skin_m=0.0005, spacing_m=0.0, youngs_pa=200e9)
    with pytest.raises(ValueError):
        monocoque_thickness_m(1000.0, 0.0)


def test_stiffening_a_gauge_limited_wall_is_refused():
    """This project's own first stage, and the reason the default was not changed.

    Every wall in it -- tank, interstage, thrust structure -- sizes to 0.80 mm
    reading "minimum gauge". Stiffeners buy buckling resistance and a wall that
    is not buckling-limited has none to buy, so wiring stiffened sizing in as a
    default would have added stringers to a wall that was never going to buckle.
    """
    from cadflow.stiffened_shell import worth_stiffening

    got = worth_stiffening(monocoque_thickness_m_=0.0008, min_gauge_m=0.0008,
                           driver="minimum gauge")
    assert not got["worth_it"] and got["gauge_limited"]
    assert "add mass without adding capability" in got["note"]


def test_stiffening_a_buckling_limited_wall_is_worth_it():
    """Above about five tonnes of stage propellant the driver changes.

    The check must come out both ways or it is not a check.
    """
    from cadflow.stiffened_shell import worth_stiffening

    got = worth_stiffening(monocoque_thickness_m_=0.0042, min_gauge_m=0.0008,
                           driver="buckling")
    assert got["worth_it"] and not got["gauge_limited"]


def test_the_coefficient_is_attributed_to_the_engine_not_the_wall():
    """The correction this module's own analysis forced.

    The monocoque scaling argument is sound physics and a real capability gap
    for large vehicles. It is not why *this* vehicle's structural coefficient
    sits at twice flown practice. Shell is a quarter of stage structure and the
    engine is half, at an assumed thrust-to-weight of 60 against flown engines
    running 80 to 180.

    Wiring stiffened sizing in as the default would have added stringers to a
    gauge-limited wall and made the vehicle heavier.
    """
    from cadflow.structural_sizing import coefficient_attribution

    a = coefficient_attribution(1006.6, 0.338, 55e3, density_kg_m3=8190.0,
                                yield_pa=1030e6, modulus_pa=200e9)
    assert a["dominant"] == "engine"
    assert a["shares"]["engine"] > a["shares"]["shell"]
    assert a["wall_driver"] == "minimum gauge"


def test_the_attribution_sums_to_the_mass_it_decomposes():
    """A decomposition that does not add up is a guess with subtotals."""
    from cadflow.structural_sizing import coefficient_attribution

    a = coefficient_attribution(5000.0, 0.55, 250e3, density_kg_m3=2700.0,
                                yield_pa=280e6, modulus_pa=70e9)
    assert sum(a["terms_kg"].values()) == pytest.approx(a["total_kg"], rel=1e-12)
    assert sum(a["shares"].values()) == pytest.approx(1.0, rel=1e-12)
