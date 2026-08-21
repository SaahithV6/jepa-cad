"""Combustion chemistry: stoichiometry exactly, equilibrium properly.

Propellants used to be a five-row lookup of (gamma, chamber temperature,
molecular mass) with no mixture-ratio dependence at all, so the central design
variable of a liquid engine could not be represented, let alone optimised.

The tests come in two kinds. Stoichiometry and product composition are pure atom
balance and are checked against exact answers -- LOX/LH2 burns to nothing but
water at a mass ratio of 7.936, and no test needs to be told that. Chamber
conditions come from a real equilibrium solve and are checked against published
values and against a physical property no fit was made to: the mixture ratio a
vehicle actually flies must lie between the one that maximises performance and
the one that maximises performance per unit tank volume.
"""

import math

import pytest

from cadflow.combustion import (
    COMBINATIONS,
    FUELS,
    OXIDISERS,
    DENSITIES,
    REFERENCE,
    bulk_density,
    chamber_equilibrium,
    characteristic_velocity,
    combustion_products,
    density_impulse,
    mean_molecular_mass,
    mixture_gamma,
    stoichiometric_of_ratio,
)

cantera = pytest.importorskip("cantera")

#: What each combination is actually flown at, independent of anything here.
FLOWN_OF = {"lox_rp1": 2.3, "lox_lh2": 5.5, "lox_ch4": 3.4, "n2o4_mmh": 1.9}

#: Published chamber conditions, which the equilibrium solve must reproduce.
PUBLISHED = {
    "lox_rp1": (3600.0, 23.0, 1.20),
    "lox_lh2": (3300.0, 12.0, 1.24),
    "lox_ch4": (3500.0, 21.0, 1.20),
    "n2o4_mmh": (3200.0, 22.0, 1.24),
}


# --- exact: atom balance ----------------------------------------------------

def test_hydrogen_oxygen_stoichiometry_is_exact():
    """2 H2 + O2 -> 2 H2O. The mass ratio is arithmetic, not a correlation."""
    got = stoichiometric_of_ratio("lox", "lh2")
    assert got == pytest.approx(15.999 / 2.016, rel=1e-12)
    assert got == pytest.approx(7.936, abs=0.001)


def test_methane_stoichiometry_is_exact():
    """CH4 + 2 O2 -> CO2 + 2 H2O, so O/F = 64/16 = 4."""
    assert stoichiometric_of_ratio("lox", "ch4") == pytest.approx(3.989, abs=0.002)


@pytest.mark.parametrize("combo,expected", [
    ("lox_rp1", 3.41), ("lox_lh2", 7.94),
    ("lox_ch4", 3.99), ("n2o4_mmh", 2.50),
])
def test_stoichiometry_matches_known_values(combo, expected):
    ox, fu = COMBINATIONS[combo]
    assert stoichiometric_of_ratio(ox, fu) == pytest.approx(expected, abs=0.02)


def test_hydrogen_burns_to_pure_water_at_stoichiometric():
    ox, fu = COMBINATIONS["lox_lh2"]
    of = stoichiometric_of_ratio(ox, fu)
    prods = combustion_products(ox, fu, of)
    assert prods["H2O"] == pytest.approx(1.0, rel=1e-6)
    for other in ("CO2", "CO", "H2", "O2", "C"):
        assert prods.get(other, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert mean_molecular_mass(ox, fu, of) == pytest.approx(18.015, abs=0.01)


@pytest.mark.parametrize("of", [0.5, 1.0, 2.0, 2.45, 3.4, 5.0])
def test_carbon_is_conserved_at_every_mixture_ratio(of):
    """It was not. Below the sooting limit the carbon vanished from the
    products entirely, because oxygen was spent on hydrogen first and there was
    none left to make even CO."""
    ox, fu = COMBINATIONS["lox_rp1"]
    prods = combustion_products(ox, fu, of)
    carbon_out = prods["CO2"] + prods["CO"] + prods.get("C", 0.0)
    assert carbon_out == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("of", [1.5, 2.45, 3.4, 5.0])
def test_hydrogen_and_oxygen_are_conserved(of):
    ox, fu = COMBINATIONS["lox_rp1"]
    prods = combustion_products(ox, fu, of)
    h_out = 2 * prods["H2O"] + 2 * prods["H2"]
    assert h_out == pytest.approx(1.95, rel=1e-9)
    o_out = 2 * prods["CO2"] + prods["H2O"] + prods["CO"] + 2 * prods["O2"]
    o_in = of * FUELS["rp1"].molar_mass / OXIDISERS["lox"].molar_mass * 2.0
    assert o_out == pytest.approx(o_in, rel=1e-9)


def test_richer_mixtures_are_lighter():
    """The whole reason engines run rich: molecular mass falls with O/F."""
    ox, fu = COMBINATIONS["lox_rp1"]
    masses = [mean_molecular_mass(ox, fu, of) for of in (1.5, 2.0, 2.5, 3.0)]
    assert all(a < b for a, b in zip(masses, masses[1:])), masses


def test_gamma_is_derived_and_lands_where_chambers_sit():
    for combo in COMBINATIONS:
        ox, fu = COMBINATIONS[combo]
        g = mixture_gamma(ox, fu, REFERENCE[combo][0])
        assert 1.15 < g < 1.30, (combo, g)


def test_characteristic_velocity_is_hot_and_light():
    fast = characteristic_velocity(3600.0, 12.0, 1.2)
    slow = characteristic_velocity(3600.0, 24.0, 1.2)
    assert fast > slow
    assert fast / slow == pytest.approx(math.sqrt(2.0), rel=1e-9)


# --- equilibrium ------------------------------------------------------------

@pytest.mark.parametrize("combo", list(COMBINATIONS))
def test_equilibrium_reproduces_published_chamber_conditions(combo):
    """Solved from thermodynamic data, not read from a table."""
    of, _tc, _m = REFERENCE[combo]
    state = chamber_equilibrium(combo, of)
    tc_pub, m_pub, gamma_pub = PUBLISHED[combo]
    assert state.chamber_temp_k == pytest.approx(tc_pub, rel=0.08)
    assert state.molecular_mass == pytest.approx(m_pub, rel=0.08)
    assert state.gamma == pytest.approx(gamma_pub, rel=0.05)


def test_equilibrium_contains_the_dissociated_species():
    """What the simplified model could not represent, and why it failed.

    A real chamber at 3,600 K is several percent OH and atomic hydrogen. Those
    species both absorb the energy that caps chamber temperature and lighten the
    products, and no model without them can find the right mixture ratio.
    """
    state = chamber_equilibrium("lox_ch4", 3.4)
    assert "OH" in state.major_species
    assert state.major_species["OH"] > 0.01
    assert state.major_species.get("H", 0.0) > 0.005


def test_flown_mixture_ratio_lies_between_the_two_optima():
    """The check nothing was fitted to.

    Characteristic velocity wants one mixture ratio and tank volume wants
    another, and every real engine sits between them. For hydrogen the two are
    far apart -- 2.4 against 8.8 -- because its density penalty is extreme, and
    for dense propellants they nearly coincide.
    """
    for combo, flown in FLOWN_OF.items():
        ratios = [0.8 + 0.2 * i for i in range(41)]
        best_c = max(ratios, key=lambda o: chamber_equilibrium(combo, o).c_star_m_s)
        best_d = max(ratios, key=lambda o: density_impulse(combo, o))
        lo, hi = min(best_c, best_d), max(best_c, best_d)
        assert lo - 0.35 <= flown <= hi + 0.35, (combo, lo, flown, hi)


def test_bulk_density_is_between_the_two_propellants():
    for combo in COMBINATIONS:
        ox, fu = COMBINATIONS[combo]
        rho = bulk_density(combo, REFERENCE[combo][0])
        assert min(DENSITIES[ox], DENSITIES[fu]) < rho < max(DENSITIES[ox], DENSITIES[fu])


def test_hydrogen_is_the_least_dense_load_by_far():
    """Which is the entire reason its flown mixture ratio is so far off its
    performance optimum."""
    rho = {c: bulk_density(c, REFERENCE[c][0]) for c in COMBINATIONS}
    assert rho["lox_lh2"] == min(rho.values())
    assert rho["lox_lh2"] < 0.5 * rho["lox_rp1"]


def test_leaner_mixtures_are_denser():
    for combo in COMBINATIONS:
        d = [bulk_density(combo, of) for of in (1.5, 2.5, 3.5, 4.5)]
        assert all(a < b for a, b in zip(d, d[1:])), (combo, d)


def test_equilibrium_rejects_nonsense():
    with pytest.raises(ValueError):
        chamber_equilibrium("lox_rp1", 0.0)
    with pytest.raises(ValueError):
        chamber_equilibrium("not_a_propellant", 2.0)
