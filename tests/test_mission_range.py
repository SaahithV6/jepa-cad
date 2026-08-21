"""The design chain across very different missions.

Everything else was validated on one mission -- 25 kg to 4,000 km -- and a
pipeline tuned to a single case looks perfect right up until the second one.
These run the whole non-FEA chain (architecture, trajectory, vehicle mass
properties, stability sizing) over three missions that differ by two orders of
magnitude in payload and by more than one in altitude, and assert the things
that must hold for any of them.

FEA is deliberately excluded: it takes minutes per component and is covered
elsewhere. What is covered here is that the reasoning around it does not fall
over when the vehicle is not the one it was written against.
"""

import math

import pytest

from cadflow.planner import plan
from cadflow.profiles import nose_profile
from cadflow.vehicle import (
    flight_vehicle_properties,
    size_fins_for_margin,
    static_margin,
)
from generate_propulsion_trajectory_corpus import load_coupling

MISSIONS = [
    pytest.param(5.0, 120.0, id="5kg-120km"),
    pytest.param(25.0, 4000.0, id="25kg-4000km"),
    pytest.param(250.0, 1500.0, id="250kg-1500km"),
]


@pytest.fixture(scope="module", autouse=True)
def _coupling():
    load_coupling()


def _design(payload_kg, apogee_km):
    p = plan(apogee_km, payload_kg)
    assert p is not None, f"no architecture closes {payload_kg} kg to {apogee_km} km"
    flight_r = max(0.10, (p.gross_kg / 1000.0) ** (1 / 3) * 0.55) / 2.0
    fv = flight_vehicle_properties(p.stack, payload_kg, flight_r)
    return p, flight_r, fv


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_mission_closes_and_masses_are_consistent(payload_kg, apogee_km):
    p, _r, fv = _design(payload_kg, apogee_km)
    expected = payload_kg + sum(s.prop_mass_kg + s.struct_mass_kg for s in p.stack)
    assert fv["mass_kg"] == pytest.approx(expected, rel=1e-9)
    assert fv["mass_kg"] == pytest.approx(p.gross_kg, rel=1e-3)
    assert p.gross_kg > payload_kg


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_trajectory_reports_the_loads_that_size_structure(payload_kg, apogee_km):
    p, _r, _fv = _design(payload_kg, apogee_km)
    traj = p.trajectory
    assert traj["max_q_pa"] > 0.0
    assert traj["max_axial_g"] > 1.0, "a vehicle that never exceeds 1 g never leaves"
    by_stage = traj["max_axial_g_by_stage"]
    assert len(by_stage) == p.stages
    assert traj["max_axial_g"] == pytest.approx(max(by_stage), rel=1e-9)
    assert traj["liftoff_thrust_n"] > p.gross_kg * 9.80665


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_vehicle_is_a_plausible_shape(payload_kg, apogee_km):
    """Long and thin, centre of gravity inside it, pitch inertia dominant."""
    _p, flight_r, fv = _design(payload_kg, apogee_km)
    assert fv["length_m"] > 2.0 * (2.0 * flight_r), "shorter than two diameters"
    assert 0.0 < fv["cg_z_m"] < fv["length_m"]
    assert fv["Ixx_kg_m2"] > fv["Izz_kg_m2"]


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_fins_can_always_be_sized_to_the_margin(payload_kg, apogee_km):
    """Stability sizing must converge for any vehicle the planner produces."""
    p, flight_r, fv = _design(payload_kg, apogee_km)
    prof = nose_profile(flight_r, 4.0 * flight_r, "ogive", 2000)
    fins = size_fins_for_margin(prof, flight_r,
                                nose_tip_station_m=fv["length_m"],
                                cg_z_m=fv["cg_z_m"],
                                fin_root_le_station_m=2.0 * flight_r)
    assert fins["met"], fins["static_margin_cal"]
    assert fins["static_margin_cal"] == pytest.approx(1.5, abs=1e-3)
    assert 0.0 < fins["span_m"] < 4.0 * flight_r
    assert fins["cna_fins"] > 0.0


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_margin_holds_through_the_burn(payload_kg, apogee_km):
    """Sizing for the worst burn state must leave the others no worse."""
    p, flight_r, fv = _design(payload_kg, apogee_km)
    prof = nose_profile(flight_r, 4.0 * flight_r, "ogive", 2000)

    states = [[1.0] * p.stages]
    if p.stages > 1:
        states.append([0.0] + [1.0] * (p.stages - 1))
    else:
        states.append([0.0])
    cgs = [flight_vehicle_properties(p.stack, payload_kg, flight_r,
                                     propellant_remaining=r)["cg_z_m"]
           for r in states]

    fins = size_fins_for_margin(prof, flight_r,
                                nose_tip_station_m=fv["length_m"],
                                cg_z_m=min(cgs),
                                fin_root_le_station_m=2.0 * flight_r)
    for cg in cgs:
        margin = static_margin(cg, fins["cp_z_m"], 2.0 * flight_r)
        assert margin >= fins["static_margin_cal"] - 1e-6, (cg, margin)
        assert math.isfinite(margin)


@pytest.mark.parametrize("payload_kg,apogee_km", MISSIONS)
def test_component_specs_cover_every_stage(payload_kg, apogee_km):
    """One tank per stage, one interstage per join, and the three fixed parts."""
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from plan_and_verify import component_specs, stack_order

    p, _r, _fv = _design(payload_kg, apogee_km)
    body_r = max(20.0, min(50.0, 16.0 * (p.gross_kg / 100.0) ** (1 / 3)))
    specs = component_specs(body_r, p.stack, p.gross_kg,
                            p.trajectory["max_q_pa"], payload_kg,
                            axial_g_by_stage=p.trajectory["max_axial_g_by_stage"],
                            liftoff_thrust_n=p.trajectory["liftoff_thrust_n"])
    names = [s[0] for s in specs]
    assert len(names) == 3 + p.stages + max(0, p.stages - 1)
    assert all(load > 0.0 for _n, _w, load, _g in specs)

    order = stack_order(names)
    assert order[0] == "nose cone"
    assert order[-1] == "thrust structure"
    for i in range(1, p.stages):
        assert order.index(f"stage {i+1} tank") < order.index(f"interstage {i}/{i+1}")
        assert order.index(f"interstage {i}/{i+1}") < order.index(f"stage {i} tank")
