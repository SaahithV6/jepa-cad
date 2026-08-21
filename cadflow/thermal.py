"""Chamber and nozzle heat transfer, and whether the engine can be cooled.

Thermal was missing entirely. Three conditioning slots existed for it --
throat heat flux, wall temperature, skin temperature -- and nothing populated
any of them, so the model had no way to learn that a chamber which cannot be
cooled is not a design regardless of how good its performance looks.

The throat is the worst place on the engine. The gas is at its densest while
still nearly at chamber temperature, and the area is at its smallest, so the
heat flux there exceeds anything else on the vehicle by an order of magnitude.
An engine is very often limited by that number rather than by structures or
performance.

What is computed rather than assumed
------------------------------------
Gas properties come from the equilibrium solution -- real viscosity, thermal
conductivity and heat capacity of the actual product mixture, giving a Prandtl
number of 0.61 for LOX/CH4 rather than a textbook 0.7 assumed for air. Reynolds
number follows from the mass flux the nozzle is actually passing.

The convective correlation, Nu = 0.026 Re^0.8 Pr^0.4, is the one empirical piece.
It is the standard turbulent form -- Dittus-Boelter uses 0.023, Bartz 0.026 for
rocket throats -- and the exponents are what make the scalings checkable: heat
flux must go as chamber pressure to the 0.8 and inversely as throat diameter to
the 0.2, and those are asserted rather than taken on trust.

The cooling check needs no correlation at all. Whether the fuel flow can absorb
the heat load without boiling is an energy balance, and energy balances are
exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Recovery factor exponent for a turbulent boundary layer. The wall does not
#: see the full stagnation temperature: some of the kinetic energy is carried
#: away rather than recovered, and for turbulent flow the recovered fraction is
#: Pr^(1/3).
_RECOVERY_EXPONENT = 1.0 / 3.0

#: Turbulent convection constant. Dittus-Boelter is 0.023 for pipe flow; Bartz
#: uses 0.026 for rocket throats, where the strong acceleration thins the
#: boundary layer.
NUSSELT_COEFF = 0.026

#: Coolant properties for regenerative cooling, per fuel: specific heat
#: J/(kg K) and the temperature it must stay below in the jacket.
COOLANTS = {
    "rp1": {"cp": 2100.0, "max_temp_k": 700.0},
    "lh2": {"cp": 14300.0, "max_temp_k": 500.0},
    "ch4": {"cp": 3500.0, "max_temp_k": 600.0},
    "mmh": {"cp": 2900.0, "max_temp_k": 550.0},
}


@dataclass(frozen=True)
class ThroatHeatLoad:
    """Heat transfer at the nozzle throat, the hottest place on the engine."""
    throat_diameter_m: float
    gas_temp_k: float
    recovery_temp_k: float
    wall_temp_k: float
    prandtl: float
    reynolds: float
    h_gas_w_m2k: float
    heat_flux_w_m2: float

    @property
    def heat_flux_mw_m2(self) -> float:
        return self.heat_flux_w_m2 / 1e6


def gas_transport(chamber_state) -> dict:
    """Viscosity, conductivity, heat capacity and Prandtl of the products.

    The equilibrium phase is built without a transport model, because the
    propellant species added to it carry no transport data. The products,
    though, are all ordinary GRI-Mech species, so the converged state is copied
    into a transport-capable phase to be interrogated.
    """
    import cantera as ct

    from cadflow.combustion import _EQ_CACHE, _equilibrium_solution

    if "transport" not in _EQ_CACHE:
        _EQ_CACHE["transport"] = ct.Solution("gri30_highT.yaml")
    tr = _EQ_CACHE["transport"]

    eq = _equilibrium_solution()
    fractions = {s: x for s, x in chamber_state.major_species.items()
                 if s in tr.species_names}
    if not fractions:
        raise ValueError("no transportable species in the product mixture")
    tr.TPX = chamber_state.chamber_temp_k, chamber_state.pressure_pa, fractions
    _ = eq
    return {
        "viscosity": float(tr.viscosity),
        "conductivity": float(tr.thermal_conductivity),
        "cp": float(tr.cp),
        "prandtl": float(tr.viscosity * tr.cp / tr.thermal_conductivity),
        "density": float(tr.density),
    }


def throat_conditions(chamber_temp_k: float, chamber_pressure_pa: float,
                      gamma: float) -> tuple[float, float]:
    """Static temperature and pressure at a choked throat, from isentropic flow."""
    g = float(gamma)
    t = float(chamber_temp_k) / (1.0 + 0.5 * (g - 1.0))
    p = float(chamber_pressure_pa) * (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    return t, p


def throat_heat_flux(chamber_state, throat_area_m2: float,
                     mass_flow_kg_s: float,
                     wall_temp_k: float = 800.0) -> ThroatHeatLoad:
    """Convective heat flux at the throat.

    Wall temperature is an input rather than a result: it is set by whatever
    cooling scheme is chosen, and the flux is what that scheme then has to
    carry. A regeneratively cooled copper throat runs near 800 K.
    """
    if throat_area_m2 <= 0.0 or mass_flow_kg_s <= 0.0:
        raise ValueError("throat area and mass flow must be positive")

    props = gas_transport(chamber_state)
    gamma = chamber_state.gamma
    t_throat, _p_throat = throat_conditions(
        chamber_state.chamber_temp_k, chamber_state.pressure_pa, gamma)

    diameter = math.sqrt(4.0 * float(throat_area_m2) / math.pi)
    mass_flux = float(mass_flow_kg_s) / float(throat_area_m2)
    reynolds = mass_flux * diameter / props["viscosity"]
    prandtl = props["prandtl"]

    nusselt = NUSSELT_COEFF * reynolds ** 0.8 * prandtl ** 0.4
    h_gas = nusselt * props["conductivity"] / diameter

    # adiabatic wall temperature at Mach 1 with a turbulent recovery factor
    recovery = prandtl ** _RECOVERY_EXPONENT
    t_aw = t_throat * (1.0 + recovery * 0.5 * (gamma - 1.0))

    flux = h_gas * max(0.0, t_aw - float(wall_temp_k))
    return ThroatHeatLoad(
        throat_diameter_m=diameter,
        gas_temp_k=t_throat,
        recovery_temp_k=t_aw,
        wall_temp_k=float(wall_temp_k),
        prandtl=prandtl,
        reynolds=reynolds,
        h_gas_w_m2k=h_gas,
        heat_flux_w_m2=flux,
    )


def chamber_heat_load(chamber_state, throat_area_m2: float,
                      mass_flow_kg_s: float, chamber_area_ratio: float = 3.0,
                      chamber_length_m: float | None = None,
                      wall_temp_k: float = 800.0) -> dict:
    """Total heat into the chamber and throat, watts.

    The throat flux is the peak; the chamber runs cooler because the gas is
    slower and the area larger. Integrating properly would need the full contour,
    so the chamber is treated as a cylinder at its own local flux and the throat
    region as a short section at the peak. That under-resolves the convergent
    section, and it is meant to size a cooling system rather than to qualify one.
    """
    throat = throat_heat_flux(chamber_state, throat_area_m2, mass_flow_kg_s,
                              wall_temp_k)
    a_chamber = float(chamber_area_ratio) * float(throat_area_m2)
    d_chamber = math.sqrt(4.0 * a_chamber / math.pi)
    length = float(chamber_length_m) if chamber_length_m else 3.0 * d_chamber

    # local flux scales with mass flux^0.8 and inversely with diameter^0.2
    scale = (float(chamber_area_ratio) ** -0.8) * (
        (d_chamber / throat.throat_diameter_m) ** -0.2)
    flux_chamber = throat.heat_flux_w_m2 * scale

    area_chamber = math.pi * d_chamber * length
    area_throat = math.pi * throat.throat_diameter_m * throat.throat_diameter_m

    q_chamber = flux_chamber * area_chamber
    q_throat = throat.heat_flux_w_m2 * area_throat
    return {
        "throat": throat,
        "chamber_flux_w_m2": flux_chamber,
        "chamber_area_m2": area_chamber,
        "throat_area_wetted_m2": area_throat,
        "q_chamber_w": q_chamber,
        "q_throat_w": q_throat,
        "q_total_w": q_chamber + q_throat,
    }


def regenerative_cooling(heat_load_w: float, fuel_flow_kg_s: float,
                         fuel: str, inlet_temp_k: float = 300.0) -> dict:
    """Can the fuel flow carry the heat load without exceeding its limit?

    Pure energy balance -- the heat has to go somewhere, and in a regenerative
    engine it goes into the fuel on its way to the injector. No correlation is
    involved and nothing is fitted: the coolant rises by Q / (mdot cp), and
    either that lands under its temperature limit or the engine needs a
    different cooling scheme.
    """
    if fuel not in COOLANTS:
        raise ValueError(f"no coolant data for {fuel!r}")
    if fuel_flow_kg_s <= 0.0:
        raise ValueError("fuel flow must be positive")

    props = COOLANTS[fuel]
    delta_t = float(heat_load_w) / (float(fuel_flow_kg_s) * props["cp"])
    outlet = float(inlet_temp_k) + delta_t
    return {
        "fuel": fuel,
        "delta_t_k": delta_t,
        "outlet_temp_k": outlet,
        "limit_temp_k": props["max_temp_k"],
        "margin_k": props["max_temp_k"] - outlet,
        "feasible": outlet <= props["max_temp_k"],
        "heat_load_w": float(heat_load_w),
    }


def exhaust_thermal_power(mass_flow_kg_s: float, chamber_state) -> float:
    """Thermal power carried by the exhaust, watts.

    The scale against which a heat load should be judged. A well-designed
    chamber rejects a small percentage of this to its walls -- if the number
    comes out at tens of percent, the heat transfer model is wrong rather than
    the engine being remarkable.
    """
    from cadflow.combustion import R_UNIVERSAL

    cp_molar = chamber_state.gamma / (chamber_state.gamma - 1.0) * R_UNIVERSAL
    cp_mass = cp_molar / chamber_state.molecular_mass
    return float(mass_flow_kg_s) * cp_mass * chamber_state.chamber_temp_k


# ---------------------------------------------------------------------------
# Aeroheating
# ---------------------------------------------------------------------------
#
# The airframe has its own thermal problem, and it was invisible: the structural
# analysis uses room-temperature aluminium allowables while the skin can run
# hundreds of degrees hotter than ambient on the way up. max_skin_temp_K was a
# conditioning slot with nothing behind it.

#: Stefan-Boltzmann constant, W/(m^2 K^4).
SIGMA_SB = 5.670374419e-8

#: Emissivity of an oxidised or painted metal skin. Bare polished aluminium is
#: far lower, around 0.05, and would run much hotter -- which is itself a design
#: decision rather than a property of the vehicle.
SKIN_EMISSIVITY = 0.8

#: Sutherland's law constants for air.
_MU_REF, _T_REF_SUTH, _S_SUTH = 1.716e-5, 273.15, 110.4

#: Air properties taken as constant across the flight envelope. Prandtl varies
#: only between about 0.68 and 0.70 over the range that matters.
AIR_PRANDTL = 0.70
AIR_GAMMA = 1.4
AIR_CP = 1005.0


def sutherland_viscosity(temp_k: float) -> float:
    """Dynamic viscosity of air, Pa s.

    Checked against Cantera's transport data over the flight range: within 1%
    below 600 K, drifting to -8% by 2000 K, which is Sutherland's known
    high-temperature behaviour. Viscosity enters the heat transfer as Re^0.8, so
    8% there is about 1.6% in the answer, and the alternative is a Cantera call
    at every timestep of every trajectory.
    """
    t = max(1.0, float(temp_k))
    return _MU_REF * (t / _T_REF_SUTH) ** 1.5 * (_T_REF_SUTH + _S_SUTH) / (t + _S_SUTH)


def recovery_temperature(ambient_temp_k: float, mach: float,
                         prandtl: float = AIR_PRANDTL,
                         gamma: float = AIR_GAMMA) -> float:
    """Temperature the skin would reach with no heat loss at all.

    Exact given the Mach number: a turbulent boundary layer recovers Pr^(1/3) of
    the kinetic energy rather than all of it. This is a hard ceiling on skin
    temperature -- nothing the vehicle does can exceed it.
    """
    r = float(prandtl) ** (1.0 / 3.0)
    return float(ambient_temp_k) * (1.0 + r * 0.5 * (gamma - 1.0) * float(mach) ** 2)


def skin_temperature(ambient_temp_k: float, ambient_density: float,
                     velocity_m_s: float, length_m: float,
                     emissivity: float = SKIN_EMISSIVITY) -> dict:
    """Radiation-equilibrium skin temperature.

    The skin heats convectively and cools by radiating to space, and settles
    where the two balance: h (T_recovery - T_wall) = eps sigma T_wall^4. Solved
    by bisection, which is safe because the left side falls monotonically with
    wall temperature and the right side rises.

    Conduction into the structure and radiation from the atmosphere are both
    neglected, which makes this an upper bound on the steady temperature at that
    condition -- and the vehicle is often not there long enough to reach it.
    """
    rho = max(0.0, float(ambient_density))
    v = max(0.0, float(velocity_m_s))
    t_inf = max(1.0, float(ambient_temp_k))
    if rho <= 0.0 or v <= 0.0:
        return {"skin_temp_k": t_inf, "recovery_temp_k": t_inf,
                "h_w_m2k": 0.0, "mach": 0.0, "flux_w_m2": 0.0}

    sound = math.sqrt(AIR_GAMMA * 287.05 * t_inf)
    mach = v / sound
    t_rec = recovery_temperature(t_inf, mach)

    mu = sutherland_viscosity(t_inf)
    reynolds = rho * v * max(0.01, float(length_m)) / mu
    # turbulent flat plate, averaged over the length
    nusselt = 0.0296 * reynolds ** 0.8 * AIR_PRANDTL ** (1.0 / 3.0)
    conductivity = mu * AIR_CP / AIR_PRANDTL
    h = nusselt * conductivity / max(0.01, float(length_m))

    eps = max(1e-6, float(emissivity))
    lo, hi = t_inf, t_rec
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h * (t_rec - mid) > eps * SIGMA_SB * mid ** 4:
            lo = mid
        else:
            hi = mid
    wall = 0.5 * (lo + hi)
    return {"skin_temp_k": wall, "recovery_temp_k": t_rec, "h_w_m2k": h,
            "mach": mach, "flux_w_m2": h * (t_rec - wall)}
