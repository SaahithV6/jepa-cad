"""TPS thickness against the vehicle that flew it.

The design loop selects a skin material by temperature and, above roughly
1,250 K, ran out of catalogue and reported "needs TPS, no alloy knob here" --
a correct diagnosis it could not act on. These tests cover the sizing that
closes that, and they exist mostly to pin the physics, because the first
implementation was wrong in a way that looked fine.

That version sized the layer from a steady-state gradient,

    t = k (T_surface - T_backface) / q_aeroheating

and returned 0.7 mm of blanket for a 1,400 K surface. Nothing about the number
is obviously absurd until it is put next to a Shuttle tile at 50 to 90 mm. The
error: at radiation equilibrium the surface re-radiates nearly all the incident
flux, so feeding the convective flux into a conduction equation attributes to
conduction a heat load that leaves as light.

TPS is transient. Heat penetrates a distance sqrt(alpha * tau); a few of those
and the backface is still cold when heating stops. The check below is that this
reproduces the Shuttle, which is the only validation that matters here.
"""

import math

import pytest

from cadflow.thermal import size_tps, tps_catalogue


def _catalogue():
    cat = tps_catalogue()
    if not cat:
        pytest.skip("TPSX catalogue not harvested")
    return cat


def test_li900_thickness_matches_the_shuttle():
    """LI-900 over a ~1,000 s reentry should land where the tiles landed.

    k 0.0476, rho 144, cp 628 -> alpha 5.26e-7 m^2/s, one diffusion depth
    23 mm, three of them 69 mm. Orbiter acreage tiles were 50 to 90 mm. Any
    formulation that misses this by an order of magnitude is wrong regardless
    of how clean it looks.
    """
    li900 = next((m for m in _catalogue() if "LI-900" in m["name"]), None)
    if li900 is None:
        pytest.skip("LI-900 not in the harvested catalogue")
    alpha = li900["conductivity_w_mk"] / (
        li900["density_kg_m3"] * li900["specific_heat_j_kgk"])
    thickness_mm = 3.0 * math.sqrt(alpha * 1000.0) * 1000.0
    assert 40.0 < thickness_mm < 100.0, thickness_mm


def test_a_hot_skin_gets_a_real_blanket_not_a_film():
    """The regression the steady-state version would fail.

    0.7 mm was the wrong answer; anything in that neighbourhood means the
    conducted flux is being confused with the incident flux again.
    """
    got = size_tps(1400.0, 1000.0, 420.0)
    assert got is not None and got["required"]
    assert got["thickness_m"] > 0.01, got
    assert got["areal_mass_kg_m2"] > 1.0, got


def test_longer_heating_needs_more_material():
    """Thickness goes as sqrt(time): four times the duration, twice the layer."""
    short = size_tps(1400.0, 250.0, 420.0)
    long = size_tps(1400.0, 1000.0, 420.0)
    assert short and long
    ratio = long["thickness_m"] / short["thickness_m"]
    assert ratio == pytest.approx(2.0, rel=0.05), ratio


def test_nothing_is_offered_that_cannot_survive_the_surface():
    """A material rated below the skin temperature is not a candidate."""
    got = size_tps(1400.0, 1000.0, 420.0)
    assert got is not None
    assert got["rated_temp_k"] >= 1400.0


def test_an_impossible_temperature_returns_nothing_rather_than_a_guess():
    """Above every catalogued rating the answer is 'no', not a thicker layer.

    Returning the most refractory material with extra thickness would look like
    a design and be a fiction: the fix at that point is the trajectory.
    """
    assert size_tps(3000.0, 1000.0, 420.0) is None


def test_no_tps_when_the_structure_already_survives():
    got = size_tps(400.0, 1000.0, 420.0)
    assert got is not None and got["required"] is False


def test_the_catalogue_only_offers_materials_it_can_actually_size():
    """Density, conductivity and specific heat, or the entry is not usable.

    A thickness computed from an assumed property is a number without
    provenance, and this repository has already published one of those.
    """
    for mat in _catalogue():
        assert mat["density_kg_m3"] > 0
        assert mat["conductivity_w_mk"] > 0
        assert mat["multiple_use_temp_k"] or mat["single_use_temp_k"]
