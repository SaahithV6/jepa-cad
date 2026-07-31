"""Unit tests for legacy OpenFOAM family routing."""
from __future__ import annotations

from cadflow.legacy_cfd_routes import recipe_for_part


def _part(label: str, family: str | None = None) -> dict:
    props = {"name": label}
    if family:
        props["family"] = family
    return {"id": "part:test", "type": "Part", "label": label, "properties": props}


def test_cubesat_bus_skips_cfd():
    r = recipe_for_part(_part("sweep-spacecraft_bus-birds4-cubesat-frame", "spacecraft_bus"))
    assert r.application is None
    assert r.recipe_id == "skip_exoatmospheric"
    assert r.primary_modality == "fea"


def test_nozzle_gets_rho_central():
    r = recipe_for_part(
        _part(
            "sweep-combustion_chamber-osrengines-nozzle-bell-throat-internal-flow",
            "combustion_chamber",
        )
    )
    assert r.recipe_id == "nozzle_compressible"
    assert r.application == "rhoCentralFoam"


def test_injector_gets_simplefoam_orifice():
    r = recipe_for_part(
        _part(
            "sweep-combustion_chamber-osrengines-injectors-goxethanol-internal-flow",
            "combustion_chamber",
        )
    )
    assert r.recipe_id == "injector_orifice"
    assert r.application == "simpleFoam"


def test_fin_stays_external_aero():
    r = recipe_for_part(_part("sweep-fin-mojave-fin-bracket", "fin"))
    assert r.recipe_id == "external_aero"
    assert r.application == "simpleFoam"


def test_wall_stress_chamber_still_gets_cfd():
    """wall/stress path is FEA-primary structurally, but still duct-CFD'd."""
    r = recipe_for_part(
        _part(
            "sweep-combustion_chamber-osrengines-chamber-wall-stress-v01",
            "combustion_chamber",
        )
    )
    assert r.application == "rhoSimpleFoam"
    assert r.recipe_id == "chamber_internal"
    assert r.primary_modality == "both"


def test_tank_gets_pressure_flow():
    r = recipe_for_part(_part("sweep-tank-meop-vessel-ullage", "tank"))
    assert r.recipe_id == "tank_pressure_flow"
    assert r.application == "simpleFoam"


def test_turbopump_gets_internal():
    r = recipe_for_part(
        _part("sweep-combustion_chamber-turbopump-impeller-inducer", "combustion_chamber")
    )
    assert r.recipe_id == "turbopump_internal"
    assert r.application == "simpleFoam"
