"""Chamber equilibrium against NASA CEA's own published answers.

The combustion module builds MMH and N2O4 as custom species from shifted NASA
polynomials, because Cantera's gri30 does not carry them. That construction was
checked for internal consistency -- atom balance, sensible product ordering --
and never against an authoritative external answer. Internal consistency is
what a wrong constant looks like from the inside.

NASA CEA ships worked examples with their reference output published in
RP-1311 Part 2, Appendix G. Example 12 is MMH/N2O4 at O/F 2.5 and Pc 1000 psia:
the same propellant pair, mixture ratio and pressure this project computes, from
an implementation that shares no code and no thermodynamic data file with it.

    CEA:  chamber T 3386.57 K, Pc 68.947 bar
    here: chamber T 3416.23 K, Pc 68.948 bar   (+0.88%)

Sources
-------
Deck:  https://github.com/nasa/cea  samples/rp1311_examples.inp  (Apache-2.0)
Ref:   NASA RP-1311 Part 2, Appendix G (McBride & Gordon, 1996)

One trap worth recording, because it produces a plausible wrong answer rather
than an obvious one: chamber pressure is quoted at different reference stations
across the literature -- injector face, throat stagnation, nozzle stagnation.
The F-1 appears as 960, 982, 983 and 1126 psia in four NASA documents, all
correct at their own station. A c* compared against a pressure from a different
station chases a ~15% error that is not in anyone's code.
"""

import pytest

pytest.importorskip("cantera")

from cadflow.combustion import chamber_equilibrium  # noqa: E402

#: NASA RP-1311 Part 2, Appendix G, Example 12.
CEA_EXAMPLE_12 = {
    "combination": "n2o4_mmh",
    "of_ratio": 2.5,
    "pressure_psia": 1000.0,
    "chamber_temp_k": 3386.57,
    "pressure_bar": 68.947,
}

PSI_TO_PA = 6894.757


def test_mmh_n2o4_chamber_temperature_matches_cea():
    """The number this project's custom species exist to produce.

    A 1% band is the honest tolerance: CEA and Cantera use different
    thermodynamic data (thermo.inp vs gri30 plus shifted polynomials for the
    two species gri30 lacks) and different equilibrium solvers, so exact
    agreement would suggest one is copying the other rather than that both are
    right.
    """
    state = chamber_equilibrium(CEA_EXAMPLE_12["combination"],
                                CEA_EXAMPLE_12["of_ratio"],
                                CEA_EXAMPLE_12["pressure_psia"] * PSI_TO_PA)
    assert state.chamber_temp_k == pytest.approx(
        CEA_EXAMPLE_12["chamber_temp_k"], rel=0.01), (
        f"{state.chamber_temp_k:.2f} K against CEA {CEA_EXAMPLE_12['chamber_temp_k']} K")


def test_chamber_pressure_is_the_pressure_it_was_given():
    """Trivial, and it is here because pressure station is the classic error.

    If this ever fails, the solve moved the pressure -- which would mean every
    c* comparison against published data is being made at a different station
    from the one the published number was reduced at.
    """
    state = chamber_equilibrium(CEA_EXAMPLE_12["combination"],
                                CEA_EXAMPLE_12["of_ratio"],
                                CEA_EXAMPLE_12["pressure_psia"] * PSI_TO_PA)
    assert state.pressure_pa / 1e5 == pytest.approx(
        CEA_EXAMPLE_12["pressure_bar"], rel=1e-4)


def test_major_species_are_the_ones_this_reaction_makes():
    """Composition, not just temperature.

    A wrong species set can still land on a plausible temperature, so the
    products are checked too: MMH/N2O4 slightly fuel-rich of stoichiometric
    burns to water and nitrogen with carbon split between CO and CO2, and
    dissociation shows up as OH and H2 at a few percent.
    """
    state = chamber_equilibrium(CEA_EXAMPLE_12["combination"],
                                CEA_EXAMPLE_12["of_ratio"],
                                CEA_EXAMPLE_12["pressure_psia"] * PSI_TO_PA)
    frac = state.major_species
    assert frac["H2O"] > 0.30, frac
    assert frac["N2"] > 0.25, frac
    assert frac["CO"] + frac["CO2"] > 0.10, frac
    # dissociation at 3400 K is real and must not be absent
    assert frac.get("OH", 0.0) > 0.01, frac
    # nothing exotic should dominate
    assert max(frac.values()) < 0.5, frac


def test_gamma_and_molecular_mass_are_physical():
    """Both feed c*, so a wrong one propagates straight into engine sizing."""
    state = chamber_equilibrium(CEA_EXAMPLE_12["combination"],
                                CEA_EXAMPLE_12["of_ratio"],
                                CEA_EXAMPLE_12["pressure_psia"] * PSI_TO_PA)
    assert 1.1 < state.gamma < 1.3, state.gamma
    assert 18.0 < state.molecular_mass < 28.0, state.molecular_mass
    # c* for storables sits near 1700 m/s; far outside that is a unit error
    assert 1400.0 < state.c_star_m_s < 1900.0, state.c_star_m_s
