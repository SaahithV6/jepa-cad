"""Realistic per-family physics targets for the TAO graph.

Each engineering family gets targets drawn from realistic ranges so the JEPA
model learns proper parameter windows (e.g. nose cone Cd, tank MEOP, nozzle
chamber pressure/expansion ratio) instead of a single generic stress value.
Solver setups carry realistic boundary conditions (OpenFOAM freestream,
CalculiX load cases) so downstream simulation regeneration is credible.

Values are deterministic per part fingerprint: the same part always gets the
same target set, which keeps the graph stable across rebuilds.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

# ---------------------------------------------------------------------------
# Family physics catalogs — realistic engineering windows.
# Each spec: (name, unit, lo, hi, log_scale)
# ---------------------------------------------------------------------------

FAMILY_PHYSICS: dict[str, dict[str, Any]] = {
    "nose_cone": {
        "solver": "openfoam",
        "sim_kind": "external_aero",
        "targets": [
            ("Cd", "-", 0.15, 0.55, False),
            ("fineness_ratio", "-", 3.0, 6.0, False),
            ("max_skin_temp_K", "K", 350.0, 700.0, False),
        ],
        "boundary_conditions": {
            "freestream_velocity_mps": (68.0, 340.0),   # Mach 0.2–1.0 at sea level
            "air_density_kgm3": 1.225,
            "kinematic_viscosity_m2s": 1.48e-5,
            "turbulence_model": "kOmegaSST",
            "reference_area_mode": "frontal",
        },
    },
    "fin": {
        "solver": "openfoam",
        "sim_kind": "external_aero",
        "targets": [
            ("CL_alpha_per_rad", "1/rad", 2.0, 6.5, False),
            ("max_root_stress_mpa", "MPa", 40.0, 250.0, False),
            ("flutter_margin", "-", 1.15, 2.0, False),
        ],
        "boundary_conditions": {
            "freestream_velocity_mps": (68.0, 272.0),
            "air_density_kgm3": 1.225,
            "kinematic_viscosity_m2s": 1.48e-5,
            "turbulence_model": "kOmegaSST",
            "angle_of_attack_deg": (0.0, 8.0),
        },
    },
    "fairing": {
        "solver": "openfoam",
        "sim_kind": "external_aero",
        "targets": [
            ("Cd", "-", 0.2, 0.6, False),
            ("max_dynamic_pressure_kpa", "kPa", 20.0, 60.0, False),
            ("acoustic_transmission_db", "dB", -12.0, -3.0, False),
        ],
        "boundary_conditions": {
            "freestream_velocity_mps": (200.0, 500.0),
            "air_density_kgm3": 0.7,   # ~5 km altitude, max-Q regime
            "kinematic_viscosity_m2s": 1.8e-5,
            "turbulence_model": "kOmegaSST",
        },
    },
    "combustion_chamber": {
        "solver": "fea",
        "sim_kind": "thermo_structural",
        "targets": [
            ("chamber_pressure_bar", "bar", 10.0, 100.0, True),
            ("wall_temp_max_K", "K", 600.0, 1200.0, False),
            ("hoop_stress_margin", "-", 1.25, 2.0, False),
            ("heat_flux_MWm2", "MW/m^2", 1.0, 20.0, True),
        ],
        "boundary_conditions": {
            "internal_pressure_mode": "chamber_pressure",
            "wall_material_default": "copper_alloy",
            "cooling": "regenerative_or_heatsink",
            "load_case": "proof_1p5x",
        },
    },
    "nozzle": {
        "solver": "openfoam",
        "sim_kind": "internal_flow",
        "targets": [
            ("expansion_ratio", "-", 4.0, 80.0, True),
            ("thrust_kN", "kN", 0.5, 50.0, True),
            ("isp_vac_s", "s", 250.0, 380.0, False),
            ("throat_heat_flux_MWm2", "MW/m^2", 5.0, 60.0, True),
        ],
        "boundary_conditions": {
            "inlet_total_pressure_bar": (10.0, 100.0),
            "inlet_total_temp_K": (2800.0, 3600.0),
            "gas_gamma": 1.2,
            "gas_R_JkgK": 340.0,
            "outlet": "supersonic_outflow",
        },
    },
    "injector": {
        "solver": "openfoam",
        "sim_kind": "internal_flow",
        "targets": [
            ("pressure_drop_bar", "bar", 3.0, 15.0, False),
            ("discharge_coefficient", "-", 0.6, 0.85, False),
            ("mixture_ratio", "-", 1.2, 3.0, False),
            ("mass_flow_kgps", "kg/s", 0.05, 5.0, True),
        ],
        "boundary_conditions": {
            "manifold_pressure_bar": (15.0, 120.0),
            "propellant_pair": "LOX/RP1_or_LOX_IPA",
            "cavitation_check": True,
        },
    },
    "tank": {
        "solver": "fea",
        "sim_kind": "pressure_vessel",
        "targets": [
            ("meop_bar", "bar", 20.0, 60.0, False),
            ("hoop_stress_margin", "-", 1.5, 2.5, False),
            ("mass_fraction", "-", 0.85, 0.95, False),
            ("proof_pressure_bar", "bar", 30.0, 90.0, False),
        ],
        "boundary_conditions": {
            "internal_pressure_mode": "meop",
            "load_case": "proof_1p5x_burst_2x",
            "boundary_fixture": "aft_ring",
        },
    },
    "valve": {
        "solver": "mbd",
        "sim_kind": "mechanism_flow",
        "targets": [
            ("cv_flow_coefficient", "-", 0.5, 25.0, True),
            ("actuation_time_ms", "ms", 20.0, 500.0, True),
            ("seat_leak_sccm", "sccm", 0.0, 10.0, False),
            ("rated_pressure_bar", "bar", 20.0, 400.0, True),
        ],
        "boundary_conditions": {
            "working_fluid": "LOX_GN2_or_RP1",
            "duty_cycle": "pulse_and_hold",
        },
    },
    "feed_system": {
        "solver": "openfoam",
        "sim_kind": "internal_flow",
        "targets": [
            ("pressure_drop_bar", "bar", 0.2, 5.0, True),
            ("flow_velocity_mps", "m/s", 2.0, 15.0, False),
            ("water_hammer_margin", "-", 1.5, 3.0, False),
        ],
        "boundary_conditions": {
            "inlet_pressure_bar": (5.0, 80.0),
            "working_fluid_density_kgm3": 1140.0,  # LOX
            "pipe_roughness_um": 3.2,
        },
    },
    "structure": {
        "solver": "fea",
        "sim_kind": "static_structural",
        "targets": [
            ("max_stress_mpa", "MPa", 50.0, 400.0, False),
            ("safety_factor", "-", 1.4, 3.0, False),
            ("first_mode_hz", "Hz", 25.0, 400.0, True),
            ("max_displacement_mm", "mm", 0.1, 5.0, True),
        ],
        "boundary_conditions": {
            "load_case": "axial_6g_lateral_2g",
            "boundary_fixture": "base_ring",
        },
    },
    "fastener": {
        "solver": "fea",
        "sim_kind": "static_structural",
        "targets": [
            ("preload_kN", "kN", 1.0, 60.0, True),
            ("max_shear_stress_mpa", "MPa", 100.0, 600.0, False),
            ("safety_factor", "-", 2.0, 4.0, False),
        ],
        "boundary_conditions": {
            "load_case": "tension_shear_combined",
        },
    },
    "mechanism": {
        "solver": "mbd",
        "sim_kind": "kinematic",
        "targets": [
            ("actuation_torque_nm", "N*m", 0.5, 50.0, True),
            ("deploy_time_s", "s", 0.1, 30.0, True),
            ("backlash_deg", "deg", 0.01, 1.0, True),
            ("cycle_life", "cycles", 1000.0, 100000.0, True),
        ],
        "boundary_conditions": {
            "gravity": "0g_and_1g_test",
            "friction_model": "coulomb_stribeck",
        },
    },
    "deployable": {
        "solver": "mbd",
        "sim_kind": "deployment",
        "targets": [
            ("deploy_time_s", "s", 1.0, 60.0, True),
            ("latch_force_n", "N", 5.0, 200.0, True),
            ("deployed_frequency_hz", "Hz", 0.5, 10.0, False),
            ("shock_load_g", "g", 5.0, 50.0, False),
        ],
        "boundary_conditions": {
            "gravity": "0g",
            "hinge_friction_nm": (0.01, 0.5),
        },
    },
    "antenna": {
        "solver": "fea",
        "sim_kind": "static_structural",
        "targets": [
            ("surface_rms_error_mm", "mm", 0.05, 1.0, True),
            ("first_mode_hz", "Hz", 10.0, 100.0, False),
            ("pointing_error_deg", "deg", 0.01, 0.5, True),
        ],
        "boundary_conditions": {
            "load_case": "thermal_gradient_orbit",
            "temp_range_K": (173.0, 373.0),
        },
    },
    "spacecraft_bus": {
        "solver": "fea",
        "sim_kind": "static_structural",
        "targets": [
            ("first_mode_hz", "Hz", 35.0, 120.0, False),
            ("max_stress_mpa", "MPa", 50.0, 300.0, False),
            ("mass_budget_kg", "kg", 1.0, 500.0, True),
        ],
        "boundary_conditions": {
            "load_case": "launch_quasi_static_6g",
            "boundary_fixture": "separation_ring",
        },
    },
    "generic": {
        "solver": "fea",
        "sim_kind": "static_structural",
        "targets": [
            ("max_stress_mpa", "MPa", 60.0, 300.0, False),
            ("safety_factor", "-", 1.5, 3.0, False),
        ],
        "boundary_conditions": {
            "load_case": "axial_6g",
        },
    },
}

# Alias mapping for families that exist in the graph but share physics
FAMILY_ALIASES = {
    "turbopump": "injector",
    "sensor": "structure",
    "cubesat": "spacecraft_bus",
    "assembly": "structure",
    "reference_shape": "generic",
    # OpenRocket / TPS hardware families → nearest physics window
    "body_tube": "structure",
    "transition": "fairing",
    "engine_mount": "structure",
    "tps_tile": "fairing",
    "blanket": "deployable",
    "solar_panel": "deployable",
    "ring_frame": "structure",
    "bulkhead": "structure",
    "strut": "structure",
}


def _unit_hash(fingerprint: str, salt: str) -> float:
    """Deterministic uniform [0,1) from part fingerprint + target name."""
    h = hashlib.sha256(f"{fingerprint}:{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _sample_target(fingerprint: str, name: str, lo: float, hi: float, log_scale: bool) -> float:
    u = _unit_hash(fingerprint, name)
    if log_scale and lo > 0:
        return round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))), 4)
    return round(lo + u * (hi - lo), 4)


def resolve_family(family: str | None) -> str:
    fam = (family or "generic").lower()
    fam = FAMILY_ALIASES.get(fam, fam)
    return fam if fam in FAMILY_PHYSICS else "generic"


def physics_targets_for(fingerprint: str, family: str | None) -> dict[str, Any]:
    """Generate realistic deterministic physics targets for one part."""
    fam = resolve_family(family)
    spec = FAMILY_PHYSICS[fam]
    targets: dict[str, float] = {}
    units: dict[str, str] = {}
    for name, unit, lo, hi, log_scale in spec["targets"]:
        targets[name] = _sample_target(fingerprint, name, lo, hi, log_scale)
        units[name] = unit

    # Materialize any ranged boundary conditions deterministically too
    bcs: dict[str, Any] = {}
    for key, value in spec["boundary_conditions"].items():
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            bcs[key] = _sample_target(fingerprint, f"bc:{key}", float(value[0]), float(value[1]), False)
        else:
            bcs[key] = value

    return {
        "family": fam,
        "solver": spec["solver"],
        "sim_kind": spec["sim_kind"],
        "targets": targets,
        "units": units,
        "boundary_conditions": bcs,
    }


# Canonical ordering of physics quantities for the conditioning vector.
# Missing values are 0. Scales normalize to roughly [0, 1].
CONDITIONING_QUANTITIES: tuple[tuple[str, float], ...] = (
    ("Cd", 1.0),
    ("CL_alpha_per_rad", 1.0 / 7.0),
    ("chamber_pressure_bar", 1.0 / 120.0),
    ("expansion_ratio", 1.0 / 100.0),
    ("thrust_kN", 1.0 / 60.0),
    ("isp_vac_s", 1.0 / 400.0),
    ("meop_bar", 1.0 / 100.0),
    ("hoop_stress_margin", 1.0 / 3.0),
    ("max_stress_mpa", 1.0 / 600.0),
    ("mean_stress_mpa", 1.0 / 400.0),
    ("max_displacement_mm", 1.0 / 20.0),
    ("safety_factor", 1.0 / 4.0),
    ("first_mode_hz", 1.0 / 400.0),
    ("pressure_drop_bar", 1.0 / 20.0),
    ("mass_flow_kgps", 1.0 / 6.0),
    ("deploy_time_s", 1.0 / 60.0),
    ("actuation_torque_nm", 1.0 / 60.0),
    ("wall_temp_max_K", 1.0 / 1500.0),
    ("max_skin_temp_K", 1.0 / 1500.0),
    ("fineness_ratio", 1.0 / 10.0),
    ("flutter_margin", 1.0 / 3.0),
    ("max_dynamic_pressure_kpa", 1.0 / 80.0),
    ("heat_flux_MWm2", 1.0 / 40.0),
    ("throat_heat_flux_MWm2", 1.0 / 80.0),
    ("acoustic_transmission_db", 1.0 / 20.0),
    ("discharge_coefficient", 1.0),
    ("mixture_ratio", 1.0 / 5.0),
    ("proof_pressure_bar", 1.0 / 150.0),
    ("mass_fraction", 1.0),
    # Part mass / mass-distribution (spaceflight-critical; from STL×ρ on TAO)
    ("mass_kg", 1.0 / 50.0),
    ("Ixx_kg_m2", 1.0 / 5.0),
    ("Iyy_kg_m2", 1.0 / 5.0),
    ("Izz_kg_m2", 1.0 / 5.0),
    # Airflow / bodyfit CFD fields (hybrid aero conditioning)
    ("U_mag_max", 1.0 / 50.0),
    ("p_mean", 1.0 / 5.0),
    ("p_delta", 1.0 / 5.0),
)


def conditioning_values(targets: dict[str, Any]) -> list[float]:
    """Project a targets dict onto the canonical conditioning slots."""
    out: list[float] = []
    for name, scale in CONDITIONING_QUANTITIES:
        value = targets.get(name)
        out.append(round(float(value) * scale, 6) if isinstance(value, (int, float)) else 0.0)
    return out
