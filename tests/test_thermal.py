"""Chamber and nozzle heat transfer, and whether the engine can be cooled.

Thermal was absent entirely -- three conditioning slots with nothing populating
them -- so nothing in the system knew that a chamber which cannot be cooled is
not a design however good its performance looks. The throat is usually what
limits an engine: the gas is at its densest while still nearly at chamber
temperature and the area is at its smallest, so the flux there is an order of
magnitude above anything else on the vehicle.

The tests fall into three kinds. Transport properties are real, from the
equilibrium mixture, and are checked against the range combustion gases occupy.
The correlation's scalings are exact consequences of its exponents and are
checked to three decimals. And the cooling check is an energy balance, so it is
checked as one.
"""

import math

import pytest

from cadflow.combustion import chamber_equilibrium
from cadflow.thermal import (
    COOLANTS,
    chamber_heat_load,
    exhaust_thermal_power,
    gas_transport,
    regenerative_cooling,
    throat_conditions,
    throat_heat_flux,
)

pytest.importorskip("cantera")

THROAT_AREA = 0.0035


def _engine(propellant="lox_rp1", of=2.45, pc_bar=55.0, throat_area=THROAT_AREA):
    state = chamber_equilibrium(propellant, of, pc_bar * 1e5)
    mdot = state.pressure_pa * throat_area / state.c_star_m_s
    return state, mdot, throat_area


# --- real gas properties ----------------------------------------------------

def test_transport_properties_are_physical():
    """From the actual product mixture, not a textbook value for air."""
    state, _m, _a = _engine()
    props = gas_transport(state)
    assert 0.4 < props["prandtl"] < 0.9, props["prandtl"]
    assert 5e-5 < props["viscosity"] < 5e-4, props["viscosity"]
    assert 0.1 < props["conductivity"] < 1.5, props["conductivity"]
    assert 1500.0 < props["cp"] < 6000.0, props["cp"]


def test_throat_is_choked_and_cooler_than_the_chamber():
    """Isentropic flow to Mach 1: T/Tc = 2/(gamma+1)."""
    tc, pc, gamma = 3600.0, 55e5, 1.2
    t, p = throat_conditions(tc, pc, gamma)
    assert t == pytest.approx(tc * 2.0 / (gamma + 1.0), rel=1e-12)
    assert t < tc
    assert p < pc
    assert p / pc == pytest.approx((2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0)))


def test_flow_at_the_throat_is_turbulent():
    """The correlation is a turbulent one, so it had better be turbulent."""
    state, mdot, area = _engine()
    load = throat_heat_flux(state, area, mdot)
    assert load.reynolds > 1e5, load.reynolds


def test_wall_sees_less_than_the_chamber_temperature():
    """Recovery: a turbulent boundary layer recovers Pr^(1/3) of the kinetic
    energy, not all of it, so the adiabatic wall sits below stagnation."""
    state, mdot, area = _engine()
    load = throat_heat_flux(state, area, mdot)
    assert load.gas_temp_k < state.chamber_temp_k
    assert load.gas_temp_k < load.recovery_temp_k < state.chamber_temp_k


# --- the scalings the correlation implies -----------------------------------

def test_heat_flux_scales_with_throat_diameter_to_the_minus_fifth():
    """Nu ~ Re^0.8 with Re ~ G D and h = Nu k / D gives q ~ D^-0.2 exactly."""
    state, _m, _a = _engine()
    points = []
    for area in (0.001, 0.004, 0.016, 0.064):
        mdot = state.pressure_pa * area / state.c_star_m_s
        load = throat_heat_flux(state, area, mdot)
        points.append((load.throat_diameter_m, load.heat_flux_w_m2))
    for (d0, q0), (d1, q1) in zip(points, points[1:]):
        exponent = math.log(q1 / q0) / math.log(d1 / d0)
        assert exponent == pytest.approx(-0.2, abs=0.005), exponent


def test_heat_flux_rises_with_chamber_pressure_near_the_point_eight_power():
    """The reason high-pressure engines are hard to cool."""
    points = []
    for pc in (20.0, 40.0, 80.0, 160.0):
        state, mdot, area = _engine(pc_bar=pc)
        points.append((pc, throat_heat_flux(state, area, mdot).heat_flux_w_m2))
    for (p0, q0), (p1, q1) in zip(points, points[1:]):
        exponent = math.log(q1 / q0) / math.log(p1 / p0)
        assert 0.75 < exponent < 0.9, exponent


def test_a_colder_wall_draws_more_heat():
    state, mdot, area = _engine()
    hot = throat_heat_flux(state, area, mdot, wall_temp_k=1200.0)
    cold = throat_heat_flux(state, area, mdot, wall_temp_k=500.0)
    assert cold.heat_flux_w_m2 > hot.heat_flux_w_m2


def test_a_wall_at_the_recovery_temperature_takes_no_heat():
    """The definition of adiabatic wall temperature, asserted."""
    state, mdot, area = _engine()
    load = throat_heat_flux(state, area, mdot)
    none = throat_heat_flux(state, area, mdot, wall_temp_k=load.recovery_temp_k)
    assert none.heat_flux_w_m2 == pytest.approx(0.0, abs=1e-6)


# --- the scale check --------------------------------------------------------

def test_heat_rejected_is_a_small_share_of_the_exhaust_power():
    """The check that catches a heat transfer model being wrong by an order of
    magnitude. A real chamber puts one to three percent of its thermal power
    into the walls; anything near double figures means the model is broken
    rather than the engine remarkable."""
    state, mdot, area = _engine()
    load = chamber_heat_load(state, area, mdot)
    power = exhaust_thermal_power(mdot, state)
    fraction = load["q_total_w"] / power
    assert 0.002 < fraction < 0.06, fraction


def test_the_throat_is_the_worst_place_on_the_engine():
    state, mdot, area = _engine()
    load = chamber_heat_load(state, area, mdot)
    assert load["throat"].heat_flux_w_m2 > load["chamber_flux_w_m2"]


# --- cooling, which is an energy balance ------------------------------------

def test_coolant_temperature_rise_is_exactly_the_energy_balance():
    """Q = mdot cp dT, with nothing else in it."""
    cool = regenerative_cooling(2.0e6, 3.0, "rp1", inlet_temp_k=300.0)
    assert cool["delta_t_k"] == pytest.approx(
        2.0e6 / (3.0 * COOLANTS["rp1"]["cp"]), rel=1e-12)
    assert cool["outlet_temp_k"] == pytest.approx(300.0 + cool["delta_t_k"])


def test_more_coolant_flow_means_a_smaller_rise():
    lean = regenerative_cooling(2.0e6, 1.0, "rp1")
    rich = regenerative_cooling(2.0e6, 4.0, "rp1")
    assert rich["delta_t_k"] < lean["delta_t_k"]
    assert rich["margin_k"] > lean["margin_k"]


def test_hydrogen_is_an_extraordinary_coolant():
    """Its heat capacity is seven times kerosene's, which is why hydrogen
    engines can run chamber pressures that would destroy a kerosene one."""
    q, flow = 2.0e6, 1.0
    h2 = regenerative_cooling(q, flow, "lh2")
    rp1 = regenerative_cooling(q, flow, "rp1")
    assert h2["delta_t_k"] < rp1["delta_t_k"] / 5.0


def test_an_engine_that_cannot_be_cooled_is_reported_as_such():
    """Not an exception -- a design finding."""
    cool = regenerative_cooling(5.0e7, 0.5, "rp1")
    assert cool["feasible"] is False
    assert cool["margin_k"] < 0.0


def test_a_realistic_engine_closes_its_cooling():
    state, mdot, area = _engine()
    load = chamber_heat_load(state, area, mdot)
    fuel_flow = mdot / (1.0 + 2.45)
    cool = regenerative_cooling(load["q_total_w"], fuel_flow, "rp1")
    assert cool["feasible"], cool
    assert 0.0 < cool["delta_t_k"] < 600.0, cool


def test_cooling_rejects_nonsense():
    with pytest.raises(ValueError):
        regenerative_cooling(1.0e6, 0.0, "rp1")
    with pytest.raises(ValueError):
        regenerative_cooling(1.0e6, 1.0, "unobtainium")
    state, mdot, _a = _engine()
    with pytest.raises(ValueError):
        throat_heat_flux(state, 0.0, mdot)


# --- aeroheating ------------------------------------------------------------
# The airframe's own thermal problem, and it was invisible: the structural
# analysis uses room-temperature aluminium allowables while the skin can run
# many hundreds of degrees hotter on the way up.

def test_sutherland_matches_real_transport_data():
    """Checked against Cantera rather than trusted.

    Within 1% below 600 K and drifting to -8% by 2000 K, which is Sutherland's
    known high-temperature behaviour. Viscosity enters as Re^0.8, so that is
    about 1.6% in the answer.
    """
    ct = pytest.importorskip("cantera")
    from cadflow.thermal import sutherland_viscosity

    air = ct.Solution("air.yaml")
    for temp in (200.0, 300.0, 600.0):
        air.TPX = temp, 101325.0, {"O2": 0.21, "N2": 0.79}
        assert sutherland_viscosity(temp) == pytest.approx(air.viscosity, rel=0.02)


def test_recovery_temperature_is_the_isentropic_relation():
    """Exact given Mach: a turbulent layer recovers Pr^(1/3) of the energy."""
    from cadflow.thermal import AIR_PRANDTL, recovery_temperature

    t_inf, mach, gamma = 250.0, 5.0, 1.4
    r = AIR_PRANDTL ** (1.0 / 3.0)
    want = t_inf * (1.0 + r * 0.5 * (gamma - 1.0) * mach**2)
    assert recovery_temperature(t_inf, mach) == pytest.approx(want, rel=1e-12)
    assert recovery_temperature(t_inf, 0.0) == pytest.approx(t_inf)


def test_skin_never_exceeds_the_recovery_temperature():
    """A hard ceiling: radiation can only cool, so the skin cannot pass the
    temperature it would reach with no heat loss at all."""
    from generate_propulsion_trajectory_corpus import atmosphere
    from cadflow.thermal import skin_temperature

    for alt_km, v in ((5, 400), (15, 1100), (30, 2400), (60, 3400)):
        rho, _p, temp = atmosphere(alt_km * 1000.0)
        s = skin_temperature(temp, rho, v, 4.0)
        assert s["skin_temp_k"] <= s["recovery_temp_k"] + 1e-6, (alt_km, s)
        assert s["skin_temp_k"] >= temp - 1e-6


def test_thin_air_cannot_heat_the_skin():
    """High above the atmosphere the recovery temperature is enormous and
    entirely irrelevant, because there is nothing to carry the heat."""
    from generate_propulsion_trajectory_corpus import atmosphere
    from cadflow.thermal import skin_temperature

    rho_lo, _p, t_lo = atmosphere(80_000.0)
    rho_hi, _p2, t_hi = atmosphere(15_000.0)
    high = skin_temperature(t_lo, rho_lo, 3800.0, 4.0)
    low = skin_temperature(t_hi, rho_hi, 1100.0, 4.0)
    assert high["recovery_temp_k"] > low["recovery_temp_k"] * 3
    assert high["skin_temp_k"] < low["skin_temp_k"]


def test_a_shinier_skin_runs_hotter():
    """Emissivity is a design decision, not a property of the vehicle: polished
    metal radiates poorly and pays for it."""
    from generate_propulsion_trajectory_corpus import atmosphere
    from cadflow.thermal import skin_temperature

    rho, _p, temp = atmosphere(25_000.0)
    dull = skin_temperature(temp, rho, 2000.0, 4.0, emissivity=0.9)
    shiny = skin_temperature(temp, rho, 2000.0, 4.0, emissivity=0.05)
    assert shiny["skin_temp_k"] > dull["skin_temp_k"]


def test_stationary_vehicle_is_at_ambient():
    from cadflow.thermal import skin_temperature

    s = skin_temperature(250.0, 1.0, 0.0, 4.0)
    assert s["skin_temp_k"] == pytest.approx(250.0)
    assert s["flux_w_m2"] == pytest.approx(0.0)


def test_trajectory_reports_peak_skin_temperature():
    """The number that says whether room-temperature allowables hold."""
    from generate_propulsion_trajectory_corpus import load_coupling
    from cadflow.planner import plan

    load_coupling()
    traj = plan(4000.0, 25.0).trajectory
    assert "max_skin_temp_k" in traj
    assert traj["max_skin_temp_k"] > 300.0
    assert traj["max_skin_temp_altitude_m"] > 0.0
    # and it must peak in the atmosphere, not out of it
    assert traj["max_skin_temp_altitude_m"] < 100_000.0
