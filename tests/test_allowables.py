"""The structural gate must know what the part is made of, and say so.

The gate was `ALLOWABLE_MPA = 200.0`: one constant, no material, no basis, no
factor of safety. Two consequences, both real.

Every component was verified in Al 6061 at E = 70 GPa regardless of the alloy
`autodesign` selected. When the loop picked Inconel 718 for the skin, the
vehicle was charged 8,190 kg/m3 in the mass budget and simultaneously forbidden
from carrying more than 200 MPa -- under a third of what the alloy is good for.
The design paid for the material and was not allowed to use it.

And 276 MPa for 6061-T6 is a *typical* strength, the middle of a distribution
that roughly half of real coupons fall below. Design allowables are lower
tolerance bounds on that distribution. This project has no population
statistics, so these tests pin that it declines to claim one rather than
quietly treating a typical value as an allowable.
"""

import pytest

from cadflow.allowables import (
    FOS_YIELD, TYPICAL_TO_DESIGN_KNOCKDOWN, design_allowable, elastic_properties)


def test_the_allowable_follows_the_material():
    """The whole point: a stronger alloy raises the gate.

    Inconel 718 yields at 1,030 MPa against 6061-T6's 276. Under the old fixed
    constant both were gated at 200, so selecting the superalloy bought nothing
    structurally while costing three times the density.
    """
    al = design_allowable("al-6061-t6")
    inco = design_allowable("inconel-718")
    assert inco.allowable_mpa > 3.0 * al.allowable_mpa
    assert al.allowable_mpa == pytest.approx(
        276.0 * TYPICAL_TO_DESIGN_KNOCKDOWN / FOS_YIELD, rel=1e-9)


def test_a_factor_of_safety_is_applied_and_recorded():
    """A margin nobody can trace is not a margin.

    The number must be lower than the material's strength, and the record must
    carry both the strength it came from and the factor that reduced it, so a
    reader can reconstruct it without reading the source.
    """
    a = design_allowable("ti-6al-4v")
    assert a.allowable_mpa < a.source_strength_mpa
    assert a.factor_of_safety == FOS_YIELD >= 1.25
    assert a.knockdown < 1.0
    assert a.as_dict()["source_strength_mpa"] == 880.0


def test_it_refuses_to_claim_a_statistical_basis_it_does_not_have():
    """Typical is not A-basis, and calling it one would be the whole failure.

    `certifiable` is computed from the basis string rather than stored, so
    supplying real A- or B-basis data is what flips it -- not an edit to a
    flag that someone could set without changing the data underneath.
    """
    for mid in ("al-6061-t6", "inconel-718", "ti-6al-4v"):
        a = design_allowable(mid)
        assert a.strength_basis == "typical (non-statistical)"
        assert not a.certifiable
        assert any("not an A- or B-basis" in c for c in a.caveats)


def test_hot_use_of_a_room_temperature_number_is_flagged():
    """No material here carries a yield-versus-temperature curve.

    The catalogue gates on max_service_temp_k, which bounds how wrong a
    room-temperature allowable can be but does not make it right: 6061-T6 has
    lost most of its strength long before its service limit. The caveat is the
    only thing that says so.
    """
    cold = design_allowable("al-6061-t6", temperature_k=295.0)
    hot = design_allowable("al-6061-t6", temperature_k=600.0)
    assert not any("room-temperature value applied" in c for c in cold.caveats)
    assert any("room-temperature value applied" in c for c in hot.caveats)
    assert any("exceeds the catalogue limit" in c for c in hot.caveats)


def test_an_unusable_material_raises_rather_than_defaulting():
    """A silent default is how the fixed 200 MPa survived.

    It applied to everything, so nothing ever looked wrong. Anything that
    cannot be sized must fail loudly at the point of use.
    """
    with pytest.raises(KeyError):
        design_allowable("unobtanium")
    with pytest.raises(KeyError):
        elastic_properties("unobtanium")


def test_elastic_properties_track_the_alloy_too():
    """Stiffness was hard-coded alongside strength.

    Running an Inconel part at aluminium's 70 GPa understates its stiffness
    nearly threefold, which inflates displacement and trips a deflection gate
    on a part that would have passed.
    """
    e_al, nu_al = elastic_properties("al-6061-t6")
    e_in, _ = elastic_properties("inconel-718")
    assert e_in > 2.5 * e_al
    assert 0.25 <= nu_al <= 0.35


def test_the_old_constant_is_gone_from_the_verification_path():
    """It must fail loudly, not resolve to a plausible default.

    Leaving the name bound to something reasonable is how a fixed gate comes
    back: an import keeps working and the number silently stops tracking the
    design.
    """
    import scripts.plan_and_verify as pv

    assert not hasattr(pv, "ALLOWABLE_MPA")


def test_the_hot_caveat_is_useless_unless_the_temperature_is_passed():
    """The caveat existed for a whole session and never once fired.

    plan_and_verify computed its allowable before the trajectory existed, so it
    defaulted to room temperature, and the warning below could not trigger --
    while the thermal section two pages later said the skin reaches 863 K and
    that "every allowable in the component table below is a room-temperature
    value". Two parts of the packet each knew, and neither told the other.

    This pins the mechanism rather than the call site: a default-temperature
    allowable is silent, and a hot one is not.
    """
    cold = design_allowable("inconel-718")
    hot = design_allowable("inconel-718", temperature_k=863.0)
    assert len(hot.caveats) > len(cold.caveats)
    assert any("863 K" in c for c in hot.caveats)
    # and the number itself is unchanged, because no data supports knocking it
    # down -- an invented reduction would look computed
    assert hot.allowable_mpa == pytest.approx(cold.allowable_mpa, rel=1e-12)


def test_every_catalogue_material_resolves_a_density():
    """The packet weighs parts in whatever alloy the loop selected.

    That lookup used to be a hardcoded 2,700 for aluminium, which understated
    every component mass on an Inconel vehicle by 3.03x. The replacement I first
    wrote guessed from whether the material id began with "al-", which is right
    for the two alloys this loop currently picks and 85% wrong for titanium --
    a material the catalogue contains, whose yield-over-density is the best in
    it, and which best_material_for returns at 500 K.

    Nothing had exercised the titanium path, so the guess would have survived
    until a mission happened to land in that temperature band. This checks the
    class rather than waiting for the case.
    """
    from cadflow.space_materials import iter_materials

    for m in iter_materials():
        if not m.yield_mpa:
            continue
        assert m.density_kg_m3 and m.density_kg_m3 > 0, m.material_id
        # and the "al-" heuristic would be badly wrong for a real selection
        if m.category == "titanium":
            assert abs(8190.0 - m.density_kg_m3) / m.density_kg_m3 > 0.5, (
                "titanium must be far enough from the superalloy default that "
                "guessing is visibly wrong")


def test_the_upgrade_path_can_reach_more_than_two_alloys():
    """best_material_for spans the catalogue, not just aluminium and Inconel.

    Every design this project has produced chose one of two alloys, which is
    why the material-dependent paths went unexercised for so long. The selector
    itself reaches further, so those paths are reachable and have to be right.
    """
    from cadflow.autodesign import best_material_for

    picked = {best_material_for(float(t)).material_id
              for t in (300, 500, 700, 900)
              if best_material_for(float(t)) is not None}
    assert len(picked) >= 3, picked
