"""Material-property evaluation harness for LatticeZero → JEPA flywheel feedback.

LatticeZero needs more than material *names*: closed-form / handbook property
checks (mass, allowables, thermal limits, CTE mismatch, TPS rating) that can run
locally without a lab campaign, plus hooks to attach FEA stress results when
available. Results are written as structured eval records for promotion /
conditioning feedback into the generative JEPA loop.

This is intentionally an open, testable framework — not a substitute for
coupon/test-campaign data, but the on-device gate LatticeZero ships with.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cadflow.space_materials import MATERIALS_BY_ID, SpaceMaterial, catalog_as_dicts, iter_materials


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MaterialEvalCase:
    case_id: str
    material_id: str
    family: str
    volume_m3: float
    applied_stress_mpa: float | None = None
    service_temp_k: float | None = None
    join_material_id: str | None = None
    fea_max_stress_mpa: float | None = None
    notes: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # info | warn | fail
    metric: float | None
    limit: float | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialEvalReport:
    case_id: str
    material_id: str
    material_name: str
    passed: bool
    checks: list[CheckResult]
    mass_kg: float | None
    allowable_mpa: float | None
    properties: dict[str, Any]
    recorded_at: str = field(default_factory=_utc)
    framework: str = "latticezero-material-eval/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "material_id": self.material_id,
            "material_name": self.material_name,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "mass_kg": self.mass_kg,
            "allowable_mpa": self.allowable_mpa,
            "properties": self.properties,
            "recorded_at": self.recorded_at,
            "framework": self.framework,
        }


def _poisson_default(category: str) -> float:
    return {
        "aluminum": 0.33,
        "titanium": 0.34,
        "steel": 0.29,
        "superalloy": 0.29,
        "copper": 0.34,
        "composite": 0.30,
        "polymer": 0.40,
        "tps": 0.20,
        "ceramic": 0.22,
    }.get(category, 0.30)


def enriched_properties(mat: SpaceMaterial) -> dict[str, Any]:
    """Handbook-style property pack used by LatticeZero evals + JEPA conditioning."""
    props = mat.to_dict()
    poisson = _poisson_default(mat.category)
    # G = E / (2(1+ν))
    shear = mat.youngs_modulus_gpa / (2.0 * (1.0 + poisson)) if mat.youngs_modulus_gpa > 0 else None
    if mat.yield_mpa is not None:
        allowable = mat.yield_mpa / 1.5
    elif mat.ultimate_mpa is not None:
        allowable = mat.ultimate_mpa / 2.0
    else:
        allowable = None
    props.update(
        {
            "poisson_ratio": poisson,
            "shear_modulus_gpa": None if shear is None else round(shear, 3),
            "allowable_stress_mpa": None if allowable is None else round(allowable, 2),
            "safety_factor_yield": 1.5,
            "safety_factor_ultimate": 2.0,
            "property_source": "handbook_typical_v1",
            "property_disclaimer": (
                "Typical published handbook ranges for design screening — "
                "not a certified allowables basis. Replace with program allowables for flight."
            ),
        }
    )
    return props


def evaluate_case(case: MaterialEvalCase) -> MaterialEvalReport:
    mat = MATERIALS_BY_ID.get(case.material_id)
    if mat is None:
        return MaterialEvalReport(
            case_id=case.case_id,
            material_id=case.material_id,
            material_name="UNKNOWN",
            passed=False,
            checks=[
                CheckResult(
                    name="catalog_lookup",
                    passed=False,
                    severity="fail",
                    metric=None,
                    limit=None,
                    message=f"Unknown material_id={case.material_id}",
                )
            ],
            mass_kg=None,
            allowable_mpa=None,
            properties={},
        )

    props = enriched_properties(mat)
    checks: list[CheckResult] = []
    mass = case.volume_m3 * mat.density_kg_m3
    checks.append(
        CheckResult(
            name="mass_estimate",
            passed=True,
            severity="info",
            metric=mass,
            limit=None,
            message=f"Estimated mass {mass:.4f} kg from ρ={mat.density_kg_m3} kg/m³",
            details={"volume_m3": case.volume_m3, "density_kg_m3": mat.density_kg_m3},
        )
    )

    allowable = props.get("allowable_stress_mpa")
    stress = case.fea_max_stress_mpa if case.fea_max_stress_mpa is not None else case.applied_stress_mpa
    if stress is not None and allowable is not None:
        margin = (allowable - stress) / allowable
        ok = stress <= allowable
        checks.append(
            CheckResult(
                name="stress_allowable",
                passed=ok,
                severity="fail" if not ok else "info",
                metric=stress,
                limit=allowable,
                message=(
                    f"Stress {stress:.1f} MPa vs allowable {allowable:.1f} MPa "
                    f"(margin {margin*100:.1f}%)"
                ),
                details={"source": "fea" if case.fea_max_stress_mpa is not None else "applied"},
            )
        )
    elif stress is not None and allowable is None:
        checks.append(
            CheckResult(
                name="stress_allowable",
                passed=False,
                severity="warn",
                metric=stress,
                limit=None,
                message="Stress provided but material has no yield/ultimate for allowables",
            )
        )

    if case.service_temp_k is not None:
        ok = case.service_temp_k <= mat.max_service_temp_k
        checks.append(
            CheckResult(
                name="service_temperature",
                passed=ok,
                severity="fail" if not ok else "info",
                metric=case.service_temp_k,
                limit=mat.max_service_temp_k,
                message=(
                    f"Service T={case.service_temp_k:.0f} K vs max {mat.max_service_temp_k:.0f} K"
                ),
            )
        )

    if case.join_material_id and case.join_material_id in MATERIALS_BY_ID:
        other = MATERIALS_BY_ID[case.join_material_id]
        if mat.cte_1e6_k is not None and other.cte_1e6_k is not None:
            delta = abs(mat.cte_1e6_k - other.cte_1e6_k)
            # Soft warn above ~8 ppm/K mismatch for bonded joints
            ok = delta <= 8.0
            checks.append(
                CheckResult(
                    name="cte_mismatch",
                    passed=ok,
                    severity="warn" if not ok else "info",
                    metric=delta,
                    limit=8.0,
                    message=(
                        f"CTE Δ={delta:.1f} ppm/K between {mat.name} and {other.name}"
                    ),
                    details={"a": mat.cte_1e6_k, "b": other.cte_1e6_k},
                )
            )

    if mat.category in {"tps", "ceramic"} and case.service_temp_k is not None:
        # TPS rating uses max service temp; ablators tolerate higher spikes
        headroom = mat.max_service_temp_k - case.service_temp_k
        checks.append(
            CheckResult(
                name="tps_thermal_rating",
                passed=headroom >= 0,
                severity="fail" if headroom < 0 else "info",
                metric=headroom,
                limit=0.0,
                message=f"TPS thermal headroom {headroom:.0f} K",
            )
        )

    # Stiffness sanity for JEPA conditioning
    if mat.youngs_modulus_gpa <= 0:
        checks.append(
            CheckResult(
                name="stiffness_defined",
                passed=False,
                severity="fail",
                metric=mat.youngs_modulus_gpa,
                limit=0.0,
                message="Young's modulus missing/non-positive",
            )
        )
    else:
        checks.append(
            CheckResult(
                name="stiffness_defined",
                passed=True,
                severity="info",
                metric=mat.youngs_modulus_gpa,
                limit=None,
                message=f"E={mat.youngs_modulus_gpa} GPa, ν={props['poisson_ratio']}",
            )
        )

    hard_fail = any(c.severity == "fail" and not c.passed for c in checks)
    return MaterialEvalReport(
        case_id=case.case_id,
        material_id=mat.material_id,
        material_name=mat.name,
        passed=not hard_fail,
        checks=checks,
        mass_kg=mass,
        allowable_mpa=allowable,
        properties=props,
    )


def default_golden_suite() -> list[MaterialEvalCase]:
    """Built-in regression suite shipped inside LatticeZero AppImage."""
    return [
        MaterialEvalCase("al6061_bracket", "al-6061-t6", "structure", 2.5e-5, applied_stress_mpa=120, service_temp_k=300),
        MaterialEvalCase("ti64_mount", "ti-6al-4v", "engine_mount", 1.2e-5, applied_stress_mpa=400, service_temp_k=350),
        MaterialEvalCase("inconel_nozzle", "inconel-625", "nozzle", 8.0e-5, applied_stress_mpa=200, service_temp_k=900),
        MaterialEvalCase("cfrp_fairing", "cfrp-epoxy", "fairing", 1.5e-3, applied_stress_mpa=150, service_temp_k=320),
        MaterialEvalCase("li900_tile", "li-900", "tps_tile", 3.0e-4, service_temp_k=1200),
        MaterialEvalCase("pica_heatshield", "pica", "tps_tile", 2.0e-3, service_temp_k=2500),
        MaterialEvalCase("al_ti_joint", "al-6061-t6", "structure", 1.0e-5, applied_stress_mpa=80, service_temp_k=300, join_material_id="ti-6al-4v"),
        MaterialEvalCase("peek_insulator", "peek", "structure", 5.0e-6, applied_stress_mpa=40, service_temp_k=400),
        # Expected fail: aluminum past service temp
        MaterialEvalCase("al_overtemp", "al-6061-t6", "structure", 1.0e-5, applied_stress_mpa=50, service_temp_k=700),
        # Expected fail: stress over allowable
        MaterialEvalCase("al_overstress", "al-6061-t6", "fin", 1.0e-5, applied_stress_mpa=500, service_temp_k=300),
    ]


def run_suite(cases: list[MaterialEvalCase] | None = None) -> dict[str, Any]:
    cases = cases or default_golden_suite()
    reports = [evaluate_case(c) for c in cases]
    passed = sum(1 for r in reports if r.passed)
    return {
        "framework": "latticezero-material-eval/v1",
        "recorded_at": _utc(),
        "cases": len(reports),
        "passed": passed,
        "failed": len(reports) - passed,
        "pass_rate": passed / max(len(reports), 1),
        "materials_catalog_size": len(MATERIALS_BY_ID),
        "reports": [r.to_dict() for r in reports],
    }


def write_suite_report(out_dir: Path, suite: dict[str, Any] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = suite or run_suite()
    path = out_dir / f"material_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    (out_dir / "material_eval_latest.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    # conditioning vector shard for JEPA flywheel feedback
    vectors = []
    for rep in suite["reports"]:
        props = rep.get("properties") or {}
        vectors.append(
            {
                "case_id": rep["case_id"],
                "material_id": rep["material_id"],
                "passed": rep["passed"],
                "conditioning": {
                    "density_kg_m3": props.get("density_kg_m3"),
                    "youngs_modulus_gpa": props.get("youngs_modulus_gpa"),
                    "poisson_ratio": props.get("poisson_ratio"),
                    "allowable_stress_mpa": props.get("allowable_stress_mpa"),
                    "max_service_temp_k": props.get("max_service_temp_k"),
                    "cte_1e6_k": props.get("cte_1e6_k"),
                    "thermal_conductivity_w_mk": props.get("thermal_conductivity_w_mk"),
                    "category_onehot": props.get("category"),
                },
            }
        )
    (out_dir / "material_conditioning_vectors.json").write_text(
        json.dumps({"vectors": vectors, "recorded_at": _utc()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def seed_app_materials(app_root: Path) -> dict[str, Any]:
    """Install catalog + run golden suite into LatticeZero user data dir.

    If the user data dir is not writable (e.g. root-owned leftover), still return
    in-memory catalog + suite results so the desktop UI stays usable.
    """
    enriched = [enriched_properties(m) for m in iter_materials()]
    suite = run_suite()
    catalog_path: Path | None = None
    report_path: Path | None = None
    write_error: str | None = None
    try:
        mat_dir = app_root / "materials"
        eval_dir = app_root / "evals" / "materials"
        mat_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = mat_dir / "materials_catalog.json"
        catalog_path.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
        report_path = write_suite_report(eval_dir, suite)
    except OSError as exc:
        write_error = str(exc)
    return {
        "catalog_path": str(catalog_path) if catalog_path else None,
        "materials": len(enriched),
        "eval_report": str(report_path) if report_path else None,
        "suite_passed": suite["passed"],
        "suite_failed": suite["failed"],
        "pass_rate": suite["pass_rate"],
        **({"error": write_error} if write_error else {}),
    }
