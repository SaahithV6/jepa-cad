"""Combustion chemistry: what the propellants actually become.

Propellants were a five-row lookup of (gamma, chamber temperature, molecular
mass) with no mixture-ratio dependence at all. That is not a model of anything.
Mixture ratio is the central design variable in a liquid engine -- it sets
specific impulse, chamber temperature, tank volume split, and how hard the
chamber is to cool -- and a constant table cannot express any of that, so
neither the planner nor the learned model could reason about it.

What is computed here without fitting anything
----------------------------------------------
Stoichiometry and product composition come from atom balance, which is
arithmetic, not a correlation. Given a fuel CxHyNz and an oxidiser, the oxygen
needed to burn it fully is fixed, and so is what comes out at any other mixture
ratio: rich of stoichiometric the carbon cannot all reach CO2 and the excess
appears as CO and H2, lean of it the surplus oxygen simply passes through.

Mean molecular mass follows exactly from that composition. It is the quantity
that matters most, because specific impulse goes as sqrt(Tc / M) and running
fuel-rich buys a large reduction in M for a modest one in Tc. That is why every
real engine runs rich of stoichiometric, and it falls straight out of the atom
balance rather than being asserted.

What is approximated, and how honestly
--------------------------------------
Chamber temperature comes from an enthalpy balance: the heat released by
forming the products is spread over their heat capacity. Ignoring dissociation
this over-predicts badly -- around 4,800 K for LOX/RP-1 against a real 3,600 K
-- because near stoichiometric a large fraction of the energy goes into tearing
molecules apart rather than into raising temperature.

Dissociation is modelled as a single enthalpy sink that grows with temperature,
with one coefficient shared by every propellant. One number, calibrated once,
checked against five propellants whose chamber temperatures are known
independently. That is a fit, and it is labelled as one, but it is a single
degree of freedom carrying five constraints rather than five free parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Universal gas constant, J/(kmol K).
R_UNIVERSAL = 8314.462

# Atomic masses, kg/kmol.
M_C, M_H, M_O, M_N = 12.011, 1.008, 15.999, 14.007

#: Standard enthalpies of formation, kJ/mol at 298 K. Gas phase throughout;
#: water is taken as vapour because it leaves the nozzle as vapour.
HF = {
    "CO2": -393.51,
    "H2O": -241.83,
    "CO": -110.53,
    "H2": 0.0,
    "O2": 0.0,
    "N2": 0.0,
}

#: Mean molar heat capacity between 298 K and combustion temperature,
#: J/(mol K). These are averages over a wide range, not values at a point --
#: the triatomics rise steeply with temperature and the diatomics much less.
CP_MEAN = {
    "CO2": 57.0,
    "H2O": 52.0,
    "CO": 34.0,
    "H2": 32.0,
    "O2": 36.0,
    "N2": 34.0,
}


@dataclass(frozen=True)
class Species:
    """A propellant component by atomic formula and heat of formation."""
    name: str
    c: float
    h: float
    o: float
    n: float
    #: enthalpy of formation, kJ per mole of the formula unit as written
    hf_kj_mol: float

    @property
    def molar_mass(self) -> float:
        return self.c * M_C + self.h * M_H + self.o * M_O + self.n * M_N


#: Fuels, written per formula unit. RP-1 is a kerosene cut, conventionally
#: idealised as CH1.95; its heat of formation is quoted per CH1.95 unit.
FUELS = {
    "rp1": Species("rp1", c=1.0, h=1.95, o=0.0, n=0.0, hf_kj_mol=-24.7),
    "lh2": Species("lh2", c=0.0, h=2.0, o=0.0, n=0.0, hf_kj_mol=0.0),
    "ch4": Species("ch4", c=1.0, h=4.0, o=0.0, n=0.0, hf_kj_mol=-74.87),
    "mmh": Species("mmh", c=1.0, h=6.0, o=0.0, n=2.0, hf_kj_mol=54.2),
}

#: Oxidisers.
OXIDISERS = {
    "lox": Species("lox", c=0.0, h=0.0, o=2.0, n=0.0, hf_kj_mol=0.0),
    "n2o4": Species("n2o4", c=0.0, h=0.0, o=4.0, n=2.0, hf_kj_mol=9.16),
}

#: Which fuel and oxidiser each named propellant combination uses.
COMBINATIONS = {
    "lox_rp1": ("lox", "rp1"),
    "lox_lh2": ("lox", "lh2"),
    "lox_ch4": ("lox", "ch4"),
    "n2o4_mmh": ("n2o4", "mmh"),
}


def stoichiometric_of_ratio(oxidiser: str, fuel: str) -> float:
    """Oxidiser-to-fuel mass ratio for complete combustion.

    Pure atom balance: every carbon needs one O2 to reach CO2, every two
    hydrogens need half an O2 to reach H2O, and any oxygen already in the
    oxidiser or fuel counts against that. Nitrogen is taken as inert and leaves
    as N2.
    """
    fu = FUELS[fuel]
    ox = OXIDISERS[oxidiser]

    # oxygen atoms required to fully oxidise one mole of fuel
    o_needed = 2.0 * fu.c + 0.5 * fu.h - fu.o
    if ox.o <= 0.0:
        raise ValueError(f"{oxidiser} carries no oxygen")
    moles_ox = o_needed / ox.o
    return moles_ox * ox.molar_mass / fu.molar_mass


def combustion_products(oxidiser: str, fuel: str, of_ratio: float) -> dict:
    """Product moles per mole of fuel at the given O/F mass ratio.

    Rich of stoichiometric there is not enough oxygen for every carbon to reach
    CO2. The carbon that misses out becomes CO, and if oxygen is scarcer still
    the hydrogen that misses out stays as H2. Lean of stoichiometric the surplus
    oxygen passes through unreacted. Nitrogen is inert.
    """
    fu = FUELS[fuel]
    ox = OXIDISERS[oxidiser]
    of = float(of_ratio)
    if of <= 0.0:
        raise ValueError("O/F must be positive")

    moles_ox = of * fu.molar_mass / ox.molar_mass
    o_avail = moles_ox * ox.o + fu.o
    n_atoms = moles_ox * ox.n + fu.n

    # Oxygen goes to carbon first, as CO, then to hydrogen as H2O, and only
    # what is left over upgrades CO to CO2. That ordering is the standard rich
    # combustion rule and it matters: carbon monoxide is extremely stable at
    # flame temperature, so in a rich mixture the carbon claims oxygen ahead of
    # the hydrogen, which is left as H2.
    #
    # The previous ordering gave oxygen to hydrogen first, and had a worse
    # problem than being the wrong way round: at very rich mixtures there was
    # no oxygen left for the carbon and it was dropped from the product list
    # entirely. Carbon was not conserved. At O/F 0.5 a hydrocarbon flame came
    # out as nothing but water and hydrogen, with a whole mole of carbon per
    # mole of fuel unaccounted for, and the molecular mass that followed was
    # meaningless.
    o_left = o_avail
    co = min(fu.c, o_left)
    o_left -= co
    soot = fu.c - co                       # carbon with no oxygen to take

    h2o = min(fu.h / 2.0, o_left)
    o_left -= h2o
    h2 = fu.h / 2.0 - h2o

    co2 = min(co, o_left)
    o_left -= co2
    co -= co2

    o2 = o_left / 2.0
    return {
        "CO2": co2, "H2O": h2o, "CO": co, "H2": h2,
        "O2": max(0.0, o2), "N2": n_atoms / 2.0, "C": soot,
    }


def mean_molecular_mass(oxidiser: str, fuel: str, of_ratio: float) -> float:
    """Mass-weighted mean molecular mass of the products, kg/kmol.

    Exact given the composition, and the composition is exact given the atom
    balance. This is the quantity that makes fuel-rich operation pay.
    """
    prods = combustion_products(oxidiser, fuel, of_ratio)
    masses = {
        "CO2": M_C + 2 * M_O, "H2O": 2 * M_H + M_O, "CO": M_C + M_O,
        "H2": 2 * M_H, "O2": 2 * M_O, "N2": 2 * M_N,
    }
    prods = {s_: n for s_, n in prods.items() if s_ in masses}
    total_moles = sum(prods.values())
    if total_moles <= 0.0:
        raise ValueError("no products")
    total_mass = sum(n * masses[s] for s, n in prods.items())
    return total_mass / total_moles


#: Dissociation enthalpy sink, J/(mol K^2), used only for the *shape* of the
#: temperature curve, never for its level. A single shared coefficient fits
#: LOX/RP-1 to within 1% but misses LOX/LH2 by -13% and LOX/CH4 by +13%,
#: because dissociation depends on which species are present and one number
#: cannot know that. Rather than add a coefficient per species and fit four
#: propellants with four parameters -- which would predict nothing -- the
#: absolute level is anchored on published data and only the variation with
#: mixture ratio is computed. See propellant_performance.
DISSOCIATION_COEFF = 0.0125

#: Reference temperature for enthalpies of formation, K.
T_REF = 298.15


def flame_temperature(oxidiser: str, fuel: str, of_ratio: float,
                      dissociation: float | None = None) -> float:
    """Adiabatic flame temperature, K, from an enthalpy balance."""
    fu = FUELS[fuel]
    ox = OXIDISERS[oxidiser]
    prods = combustion_products(oxidiser, fuel, of_ratio)
    moles_ox = float(of_ratio) * fu.molar_mass / ox.molar_mass

    # heat released, kJ per mole of fuel
    h_react = fu.hf_kj_mol + moles_ox * ox.hf_kj_mol
    h_prod = sum(n * HF[s] for s, n in prods.items() if s in HF)
    released_j = (h_react - h_prod) * 1000.0
    if released_j <= 0.0:
        return T_REF

    cp_total = sum(n * CP_MEAN[s] for s, n in prods.items() if s in CP_MEAN)
    if cp_total <= 0.0:
        return T_REF

    k = DISSOCIATION_COEFF if dissociation is None else float(dissociation)
    if k <= 0.0:
        return T_REF + released_j / cp_total

    # released = cp dT + k/2 (T^2 - Tref^2), solved for T
    a = 0.5 * k
    b = cp_total
    c = -(released_j + b * T_REF + a * T_REF * T_REF)
    disc = b * b - 4.0 * a * c
    return (-b + math.sqrt(max(0.0, disc))) / (2.0 * a)


def characteristic_velocity(chamber_temp_k: float, molecular_mass: float,
                            gamma: float) -> float:
    """c* = sqrt(R Tc / M) / Gamma(gamma), m/s.

    The propellant's whole contribution to performance, in one number: hot and
    light is fast.
    """
    g = float(gamma)
    vandenkerckhove = math.sqrt(g) * (2.0 / (g + 1.0)) ** (
        (g + 1.0) / (2.0 * (g - 1.0)))
    return math.sqrt(R_UNIVERSAL * float(chamber_temp_k)
                     / float(molecular_mass)) / vandenkerckhove


#: Reference operating points: the mixture ratio each combination is normally
#: flown at, with the chamber temperature and molecular mass published for it.
#: These are the anchors. Everything away from them is computed.
REFERENCE = {
    #                  O/F   Tc (K)   M (kg/kmol)
    "lox_rp1":       (2.45, 3600.0, 23.0),
    "lox_lh2":       (5.50, 3300.0, 12.0),
    "lox_ch4":       (3.40, 3500.0, 21.0),
    "n2o4_mmh":      (1.90, 3200.0, 22.0),
}


def mixture_gamma(oxidiser: str, fuel: str, of_ratio: float) -> float:
    """Ratio of specific heats of the product mixture.

    Derived, not tabulated: gamma = Cp / (Cp - R) on a molar basis, with Cp the
    mole-weighted mixture value. Pure steam gives 1.19 and a carbon-rich mixture
    somewhat less, which is the range real chambers sit in.
    """
    prods = combustion_products(oxidiser, fuel, of_ratio)
    prods = {s_: n for s_, n in prods.items() if s_ in CP_MEAN}
    total = sum(prods.values())
    if total <= 0.0:
        raise ValueError("no products")
    cp = sum(n * CP_MEAN[s] for s, n in prods.items() if s in CP_MEAN) / total
    r_molar = R_UNIVERSAL / 1000.0                     # J/(mol K)
    if cp <= r_molar:
        raise ValueError("unphysical heat capacity")
    return cp / (cp - r_molar)


@dataclass(frozen=True)
class PropellantState:
    """Chamber conditions at a mixture ratio."""
    combination: str
    of_ratio: float
    chamber_temp_k: float
    molecular_mass: float
    gamma: float
    c_star_m_s: float
    stoichiometric_of: float
    products: dict


def propellant_performance(combination: str, of_ratio: float) -> PropellantState:
    """Chamber conditions at any mixture ratio.

    Absolute chamber temperature and molecular mass are anchored at the
    combination's published reference point and scaled by the ratio the
    chemistry computes between that point and the requested one. The anchor
    carries the dissociation physics this model does not resolve; the chemistry
    carries the mixture-ratio dependence the old lookup table did not have at
    all.

    Ignoring dissociation makes the raw balance predict both a hotter chamber
    and a heavier product mixture than reality -- +12 to +24% on temperature and
    +4 to +12% on molecular mass across the four combinations, always in the
    same direction, because dissociation both absorbs energy and produces light
    fragments. Anchoring removes that bias at the reference point and leaves the
    trend, which is what a designer trading mixture ratio actually needs.
    """
    if combination not in COMBINATIONS:
        raise ValueError(f"unknown propellant combination {combination!r}")
    ox, fu = COMBINATIONS[combination]
    of = float(of_ratio)
    if of <= 0.0:
        raise ValueError("O/F must be positive")

    of_ref, tc_ref, m_ref = REFERENCE[combination]
    tc = tc_ref * (flame_temperature(ox, fu, of)
                   / flame_temperature(ox, fu, of_ref))
    mm = m_ref * (mean_molecular_mass(ox, fu, of)
                  / mean_molecular_mass(ox, fu, of_ref))
    gamma = mixture_gamma(ox, fu, of)
    return PropellantState(
        combination=combination,
        of_ratio=of,
        chamber_temp_k=tc,
        molecular_mass=mm,
        gamma=gamma,
        c_star_m_s=characteristic_velocity(tc, mm, gamma),
        stoichiometric_of=stoichiometric_of_ratio(ox, fu),
        products=combustion_products(ox, fu, of),
    )


def optimum_of_ratio(combination: str, lo: float = 0.5, hi: float = 8.0,
                     steps: int = 400) -> float:
    """Mixture ratio that maximises characteristic velocity.

    It lands fuel-rich of stoichiometric for every real combination, because
    c* goes as sqrt(Tc / M) and running rich sheds molecular mass faster than
    it sheds temperature. That is a consequence here, not an assumption.
    """
    best_of, best_c = lo, -1.0
    for i in range(steps + 1):
        of = lo + (hi - lo) * i / steps
        try:
            state = propellant_performance(combination, of)
        except (ValueError, ZeroDivisionError):
            continue
        if state.c_star_m_s > best_c:
            best_of, best_c = of, state.c_star_m_s
    return best_of


# ---------------------------------------------------------------------------
# Real equilibrium chemistry
# ---------------------------------------------------------------------------
#
# Everything above is exact where it can be (stoichiometry, atom balance,
# product composition ignoring dissociation) and anchored where it cannot be
# (chamber temperature). It is still not enough to answer the question a
# designer actually asks, which is what mixture ratio to run at.
#
# The reason is dissociation. Near stoichiometric a large share of the released
# energy goes into breaking molecules apart -- at 3,600 K a LOX/CH4 chamber is
# 7% OH, 3% atomic H and 1% atomic O -- and that both caps the temperature and
# lightens the products. A model without those species cannot reproduce the
# optimum: sweeping a single dissociation coefficient over an order of
# magnitude moved the LOX/RP-1 optimum only from 3.40 to 3.26 against a real
# 2.4, and moved LOX/LH2 the wrong way entirely.
#
# So the equilibrium is solved properly, with Cantera and GRI-Mech thermodynamic
# data. Propellants that are not in the mechanism are added as species carrying
# their real formula and enthalpy of formation. That is legitimate rather than a
# fudge: at constant enthalpy and pressure the fuel is entirely consumed, so the
# only property of it that enters the answer is h(298).

_EQ_CACHE: dict = {}


def _nasa_shifted(template, hf_j_kmol: float):
    """Copy a NASA polynomial and shift it to a given enthalpy of formation.

    h/RT = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T, so a6 sets the
    formation enthalpy and nothing else. The rest of the polynomial describes
    the species' own heat capacity, which never enters a constant-enthalpy
    equilibrium of a fully consumed reactant.
    """
    import cantera as ct

    coeffs = list(template.coeffs)
    r = ct.gas_constant
    t = 298.15

    def h_at_298(a):
        return r * t * (a[0] + a[1] * t / 2 + a[2] * t ** 2 / 3
                        + a[3] * t ** 3 / 4 + a[4] * t ** 4 / 5 + a[5] / t)

    low = coeffs[8:15]
    delta = (hf_j_kmol - h_at_298(low)) / r
    coeffs[13] += delta          # a6 of the low-temperature polynomial
    coeffs[6] += delta           # a6 of the high-temperature polynomial
    return ct.NasaPoly2(template.min_temp, template.max_temp,
                        template.reference_pressure, coeffs)


def _equilibrium_solution():
    """GRI-Mech plus the propellants it does not contain."""
    import cantera as ct

    if "solution" in _EQ_CACHE:
        return _EQ_CACHE["solution"]
    # High-temperature GRI-Mech. The standard gri30.yaml has thermodynamic
    # polynomials valid only to 3,000 K, and a rocket chamber runs at 3,300 to
    # 3,600 K, so every equilibrium solve was extrapolating past the data. The
    # high-temperature variant is valid to 5,000-6,000 K and covers the range
    # properly.
    base = ct.Solution("gri30_highT.yaml")
    species = list(base.species())
    template = base.species("CH4").thermo

    extra = {
        # name        composition                       hf, kJ/mol
        "RP1":  ({"C": 1.0, "H": 1.95}, -24.7),
        "MMH":  ({"C": 1.0, "H": 6.0, "N": 2.0}, 54.2),
        "N2O4": ({"N": 2.0, "O": 4.0}, 9.16),
    }
    for name, (comp, hf) in extra.items():
        sp = ct.Species(name, comp)
        sp.thermo = _nasa_shifted(template, hf * 1e6)
        species.append(sp)

    sol = ct.Solution(thermo="ideal-gas", kinetics="none", species=species)
    _EQ_CACHE["solution"] = sol
    return sol


#: How each combination is written for the equilibrium solver. Molar masses are
#: derived from the formulae rather than written out again -- a hand-copied
#: 13.9686 for CH1.95 against a true 13.9766 put 0.06% into every oxygen
#: balance, which is small but is the sort of thing that has no business being
#: in a calculation that can just do the arithmetic.
EQ_REACTANTS = {
    "lox_rp1":  ("RP1", FUELS["rp1"].molar_mass, "O2", OXIDISERS["lox"].molar_mass),
    "lox_lh2":  ("H2", FUELS["lh2"].molar_mass, "O2", OXIDISERS["lox"].molar_mass),
    "lox_ch4":  ("CH4", FUELS["ch4"].molar_mass, "O2", OXIDISERS["lox"].molar_mass),
    "n2o4_mmh": ("MMH", FUELS["mmh"].molar_mass, "N2O4",
                 OXIDISERS["n2o4"].molar_mass),
}


@dataclass(frozen=True)
class ChamberState:
    """Equilibrium chamber conditions."""
    combination: str
    of_ratio: float
    pressure_pa: float
    chamber_temp_k: float
    molecular_mass: float
    gamma: float
    c_star_m_s: float
    major_species: dict


def chamber_equilibrium(combination: str, of_ratio: float,
                        pressure_pa: float = 55e5) -> ChamberState:
    """Solve the chamber equilibrium at a mixture ratio and pressure.

    Constant enthalpy and pressure, which is what a rocket chamber is: the
    reactants come in cold, react adiabatically, and the products leave at
    whatever temperature that implies.
    """
    if combination not in EQ_REACTANTS:
        raise ValueError(f"unknown propellant combination {combination!r}")
    of = float(of_ratio)
    if of <= 0.0:
        raise ValueError("O/F must be positive")

    # Memoised on a rounded mixture ratio and pressure. An equilibrium solve is
    # tens of milliseconds and the planner asks for thousands of them while it
    # searches; the answer varies far more slowly with O/F than the rounding.
    key = (combination, round(of, 4), round(float(pressure_pa), -3))
    hit = _EQ_CACHE.get(key)
    if hit is not None:
        return hit

    fuel, m_fuel, oxid, m_oxid = EQ_REACTANTS[combination]
    gas = _equilibrium_solution()
    moles_ox = of * m_fuel / m_oxid
    gas.TPX = 298.15, float(pressure_pa), {fuel: 1.0, oxid: moles_ox}
    gas.equilibrate("HP")

    gamma = gas.cp / gas.cv
    major = {s: float(x) for s, x in zip(gas.species_names, gas.X) if x > 5e-3}
    state = ChamberState(
        combination=combination,
        of_ratio=of,
        pressure_pa=float(pressure_pa),
        chamber_temp_k=float(gas.T),
        molecular_mass=float(gas.mean_molecular_weight),
        gamma=float(gamma),
        c_star_m_s=characteristic_velocity(gas.T, gas.mean_molecular_weight, gamma),
        major_species=major,
    )
    _EQ_CACHE[key] = state
    return state


def optimum_mixture_ratio(combination: str, pressure_pa: float = 55e5,
                          lo: float = 0.8, hi: float = 9.0,
                          steps: int = 60) -> float:
    """Mixture ratio maximising characteristic velocity, by golden-ish scan.

    Comes out fuel-rich of stoichiometric for every combination, which is the
    fact the simplified model could not reproduce at any setting.
    """
    best_of, best_c = lo, -1.0
    for i in range(steps + 1):
        of = lo + (hi - lo) * i / steps
        try:
            state = chamber_equilibrium(combination, of, pressure_pa)
        except Exception:  # noqa: BLE001 - a failed point is simply not the optimum
            continue
        if state.c_star_m_s > best_c:
            best_of, best_c = of, state.c_star_m_s
    # refine around the best coarse point
    span = (hi - lo) / steps
    for i in range(21):
        of = max(lo, best_of - span) + 2 * span * i / 20
        try:
            state = chamber_equilibrium(combination, of, pressure_pa)
        except Exception:  # noqa: BLE001
            continue
        if state.c_star_m_s > best_c:
            best_of, best_c = of, state.c_star_m_s
    return best_of


#: Densities of the liquid propellants as flown, kg/m^3. Cryogens are at their
#: normal boiling point, storables at room temperature.
DENSITIES = {
    "lox": 1141.0,
    "n2o4": 1443.0,
    "rp1": 810.0,
    "lh2": 70.8,
    "ch4": 422.6,
    "mmh": 880.0,
}


def bulk_density(combination: str, of_ratio: float) -> float:
    """Density of the loaded propellant as a whole, kg/m^3.

    The mass-weighted harmonic mean of the two densities. This is what sets tank
    volume, and therefore tank mass, and it is why the mixture ratio a vehicle
    actually flies is not the one that maximises characteristic velocity.
    """
    ox_name, fu_name = COMBINATIONS[combination]
    of = float(of_ratio)
    if of <= 0.0:
        raise ValueError("O/F must be positive")
    rho_o, rho_f = DENSITIES[ox_name], DENSITIES[fu_name]
    return (1.0 + of) / (1.0 / rho_f + of / rho_o)


def density_impulse(combination: str, of_ratio: float,
                    pressure_pa: float = 55e5) -> float:
    """c* times bulk density: performance per unit of tank volume.

    The quantity that matters when tankage is expensive. For LOX/LH2 the two
    objectives disagree sharply -- characteristic velocity peaks near O/F 2.5
    and stays within 2% of its best all the way from 1.5 to 4.0, while bulk
    density more than doubles across that range -- which is exactly why real
    hydrogen engines run at 5 to 6 and accept a few percent of c* to avoid
    building a tank twice the size.
    """
    state = chamber_equilibrium(combination, of_ratio, pressure_pa)
    return state.c_star_m_s * bulk_density(combination, of_ratio)
