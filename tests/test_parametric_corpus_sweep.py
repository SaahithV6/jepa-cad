from __future__ import annotations

from pathlib import Path

from cadflow.backends import MockCadBackend
from cadflow.corpus_sweep import (
    SourceGeometryProfile,
    build_sweep_cases,
    build_variant_geometry,
    classify_geometry_family,
    run_parametric_corpus_sweep,
)


def _write_ascii_stl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "solid part",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 1 1 0",
                "    endloop",
                "  endfacet",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 1 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                "endsolid part",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_classify_geometry_family_handles_subassembly_parts() -> None:
    assert classify_geometry_family("rocket/NoseCone.step") == "nose_cone"
    assert classify_geometry_family("engine/EngineChamber.stl") == "combustion_chamber"
    assert classify_geometry_family("structures/FinCan3fins.stl") == "fin"
    assert classify_geometry_family("thermal/SpaceTug_Exterior.stl") == "fairing"
    assert classify_geometry_family("mechanisms/retainer.step") == "mechanism"
    assert classify_geometry_family("moon/Moon (near side).stl") == "reference_shape"


def test_build_variant_geometry_changes_nose_cone_profile() -> None:
    profile = SourceGeometryProfile(
        path=Path("/tmp/NoseCone.step"),
        family="nose_cone",
        priority=3,
        solver_suite=(("openfoam", "external-flow", {"targets": {"Cd": 0.18}}),),
        extents=(2.0, 1.0, 0.8),
        notes="nose cone test",
    )
    a = build_variant_geometry(profile, 0, 3)
    b = build_variant_geometry(profile, 2, 3)
    assert a["kind"] == "extrude"
    assert b["kind"] == "extrude"
    assert a["profile"] != b["profile"]
    assert a["features"] != b["features"]
    assert a["height"] != b["height"]


def test_parametric_corpus_sweep_runs_parallel_and_promotes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "nose").mkdir(parents=True)
    (raw / "engine").mkdir(parents=True)
    (raw / "fin").mkdir(parents=True)
    (raw / "body").mkdir(parents=True)
    (raw / "reference").mkdir(parents=True)
    _write_ascii_stl(raw / "nose" / "nose_cone.stl")
    _write_ascii_stl(raw / "engine" / "engine_chamber.stl")
    _write_ascii_stl(raw / "fin" / "fincan.step")
    _write_ascii_stl(raw / "body" / "structural_bracket.stl")
    _write_ascii_stl(raw / "reference" / "moon.stl")

    out = tmp_path / "sweep"
    result = run_parametric_corpus_sweep(
        [raw],
        out,
        flywheel_path=tmp_path / "flywheel.jsonl",
        variants_per_source=1,
        max_sources=5,
        include_reference=False,
        max_workers=2,
        backend=MockCadBackend(),
        prefer_real_cad=False,
        allow_solver_fallback=True,
        num_points=32,
        num_fields=3,
        fmt="npz",
        promote_limit=20,
    )

    assert result.discovered_sources == 5
    assert result.sweep_cases >= 4
    assert result.run_ok == result.sweep_cases
    assert result.verified >= 1
    assert result.promoted >= 1
    assert Path(result.source_report_path).exists()
    assert Path(result.sweep_report_path).exists()
    assert Path(result.curated_manifest_path).exists()
    assert Path(result.master_graph_path).exists()
    assert result.neo4j_report.exit_code == 0
    assert result.ok is True
    assert result.pipeline_results
    assert any(row["family"] == "nose_cone" for row in result.pipeline_results)
    assert any(row["suite_name"] == "external-flow" for row in result.pipeline_results)
    assert any(row["suite_name"] == "wall-stress" or row["suite_name"] == "root-stress" for row in result.pipeline_results)
