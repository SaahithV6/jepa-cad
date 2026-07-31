"""Tests for LatticeZero material-property eval framework."""
from __future__ import annotations

from cadflow.material_eval import (
    MaterialEvalCase,
    default_golden_suite,
    enriched_properties,
    evaluate_case,
    run_suite,
    seed_app_materials,
)
from cadflow.space_materials import MATERIALS_BY_ID


def test_enriched_properties_include_allowables() -> None:
    props = enriched_properties(MATERIALS_BY_ID["al-6061-t6"])
    assert props["poisson_ratio"] > 0
    assert props["allowable_stress_mpa"] is not None
    assert props["shear_modulus_gpa"] is not None


def test_stress_gate_pass_and_fail() -> None:
    ok = evaluate_case(
        MaterialEvalCase("ok", "al-6061-t6", "fin", 1e-5, applied_stress_mpa=100, service_temp_k=300)
    )
    bad = evaluate_case(
        MaterialEvalCase("bad", "al-6061-t6", "fin", 1e-5, applied_stress_mpa=500, service_temp_k=300)
    )
    assert ok.passed
    assert not bad.passed


def test_golden_suite_runs() -> None:
    suite = run_suite(default_golden_suite())
    assert suite["cases"] == 10
    assert suite["failed"] >= 2  # overtemp + overstress
    assert suite["passed"] >= 6


def test_seed_app_materials(tmp_path) -> None:
    stats = seed_app_materials(tmp_path)
    assert stats["materials"] >= 30
    assert (tmp_path / "materials" / "materials_catalog.json").exists()
    assert (tmp_path / "evals" / "materials" / "material_eval_latest.json").exists()
    assert (tmp_path / "evals" / "materials" / "material_conditioning_vectors.json").exists()
