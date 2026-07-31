"""User-facing CLI for CAD/CAE orchestration and flywheel promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadflow.corpus_sweep import run_parametric_corpus_sweep
from cadflow.cloud import build_cloud_training_plan
from cadflow.backends import get_backend
from cadflow.design_loop import run_design_loop
from cadflow.doctor import build_doctor_report, render_doctor_report
from cadflow.e2e import run_end_to_end
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest
from cadflow.pipeline import run_pipeline
from cadflow.project import intake_project
from cadflow.promotion import promote_verified_to_dataset
from cadflow.runtime import resolve_solver_runtime
from cadflow.corpus_graph import build_processed_corpus_graph, render_corpus_graph_report
from cadflow.evaluation_graph import build_flywheel_evaluation_graph, render_evaluation_graph_report
from cadflow.local_data_graph import build_local_data_graph, render_local_data_graph_report
from cadflow.graph_schema import build_source_registry_graph, build_spaceflight_graph_schema, render_graph_document, render_graph_schema
from cadflow.neo4j_store import Neo4jImportReport, import_graph_to_neo4j, render_neo4j_import_report, write_neo4j_bundle
from cadflow.source_validation import render_validation_report, validate_source_registry
from data.ingest import ingest_sources


def _load_manifest(path: Path) -> JobManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return JobManifest.from_dict(payload)


def _runtime_from_args(args: argparse.Namespace):
    return resolve_solver_runtime(
        root=getattr(args, "solver_root", None),
        bin_dirs=getattr(args, "solver_bin_dir", None),
        lib_dirs=getattr(args, "solver_lib_dir", None),
    )


def _parse_target_overrides(items: list[str] | None) -> dict[str, object]:
    import yaml

    targets: dict[str, object] = {}
    for expr in items or []:
        if "=" not in expr:
            raise SystemExit(f"target overrides must use key=value syntax, got {expr!r}")
        key, rhs = expr.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"target override is missing a key: {expr!r}")
        targets[key] = yaml.safe_load(rhs.strip())
    return targets


def cmd_run(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    flywheel = DataFlywheel(args.flywheel) if args.flywheel else None
    backend = get_backend(prefer_real=not args.mock_cad)
    runtime = _runtime_from_args(args)
    result = run_pipeline(
        manifest,
        backend=backend,
        workdir=args.workdir,
        flywheel=flywheel,
        solver_kind=args.solver,
        prefer_real_cad=not args.mock_cad,
        allow_solver_fallback=not args.require_native_solver,
        promote_to=args.promote_to,
        promote_limit=args.promote_limit,
        runtime=runtime,
    )
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    (out / "verification.txt").write_text(result.report_text + "\n", encoding="utf-8")
    print(result.report_text)
    print(f"status={result.run.status} ok={result.ok}")
    print(f"wrote {out / 'result.json'}")
    return 0 if result.ok else 2


def cmd_promote(args: argparse.Namespace) -> int:
    flywheel = DataFlywheel(args.flywheel)
    result = promote_verified_to_dataset(
        flywheel,
        args.out_dir,
        limit=args.limit,
        num_points=args.num_points,
        num_fields=args.num_fields,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.promoted else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    if not args.raw_dir and not args.flywheel:
        raise SystemExit("at least one --raw-dir or --flywheel is required")
    result = ingest_sources(
        args.raw_dir,
        args.out_dir,
        flywheel_path=args.flywheel,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        recursive=not args.non_recursive,
        limit=args.limit,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ingested else 1


def cmd_corpus_sweep(args: argparse.Namespace) -> int:
    raw_dirs = args.raw_dir or ["data/raw_downloads"]
    backend = get_backend(prefer_real=not args.mock_cad)
    result = run_parametric_corpus_sweep(
        raw_dirs,
        args.out_dir,
        flywheel_path=args.flywheel,
        seed_flywheel=args.seed_flywheel,
        data_root=args.data_root,
        variants_per_source=args.variants_per_source,
        include_reference=args.include_reference,
        max_sources=args.max_sources,
        recursive=not args.non_recursive,
        max_workers=args.max_workers,
        backend=backend,
        prefer_real_cad=not args.mock_cad,
        allow_solver_fallback=not args.require_native_solver,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        promote_limit=args.promote_limit,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
        print(f"sweep_cases={result.sweep_cases} run_ok={result.run_ok} verified={result.verified} promoted={result.promoted}")
    return 0 if result.ok else 2


def cmd_project_intake(args: argparse.Namespace) -> int:
    result = intake_project(
        args.project_root,
        goal=args.goal,
        family=args.family,
        solver=args.solver,
        material=args.material,
        targets=_parse_target_overrides(args.target),
        notes=args.notes,
        out_dir=args.out_dir,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(json.dumps(result.to_dict(), indent=2))
    if result.questions:
        print("\nNext questions:")
        for question in result.questions:
            print(f"- {question}")
    return 0


def cmd_cloud_plan(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    plan = build_cloud_training_plan(
        manifest,
        family=args.family,
        provider_preference=args.provider,
        max_dataset_sources=args.max_dataset_sources,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    print(f"primary_provider={plan.primary_provider}")
    if plan.secondary_provider:
        print(f"secondary_provider={plan.secondary_provider}")
    print(f"family={plan.family}")
    print(f"project_manifest={plan.project_manifest}")
    if plan.dataset_sources:
        print("dataset_sources:")
        for source in plan.dataset_sources:
            print(f"- {source.key}: {source.title} -> {source.url}")
    print("preprocessing_steps:")
    for step in plan.preprocessing_steps:
        print(f"- {step}")
    print("training_steps:")
    for step in plan.training_steps:
        print(f"- {step}")
    print("evaluation_steps:")
    for step in plan.evaluation_steps:
        print(f"- {step}")
    print("notes:")
    for note in plan.notes:
        print(f"- {note}")
    return 0


def cmd_design_loop(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest)) if args.manifest else None
    result = run_design_loop(
        manifest=manifest,
        project_root=args.project_root,
        goal=args.goal,
        family=args.family,
        solver=args.solver,
        material=args.material,
        targets=_parse_target_overrides(args.target),
        out_dir=args.out_dir,
        repeat=args.repeat,
        tolerance=args.tolerance,
        notes=args.notes,
        allow_solver_fallback=not args.require_native_solver,
        prefer_real_cad=not args.mock_cad,
        solver_payload_factory=None,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


def cmd_e2e(args: argparse.Namespace) -> int:
    if not args.raw_dir and not args.flywheel:
        raise SystemExit("at least one --raw-dir or --flywheel is required")
    result = run_end_to_end(
        args.raw_dir,
        args.out_dir,
        flywheel_path=args.flywheel,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        recursive=not args.non_recursive,
        limit=args.limit,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        family=args.family,
        data_source=args.data_source,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        extra_overrides=args.set or [],
    )
    if result.train_stdout:
        print(result.train_stdout, end="")
    if result.train_stderr:
        print(result.train_stderr, file=sys.stderr, end="")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


def cmd_loop(args: argparse.Namespace) -> int:
    from cadflow.loop_controller import run_loop_controller

    if not args.raw_dir and not args.flywheel:
        raise SystemExit("at least one --raw-dir or --flywheel is required")
    result = run_loop_controller(
        args.raw_dir,
        args.out_dir,
        repeat=args.repeat,
        interval_seconds=args.interval_seconds,
        stop_file=args.stop_file,
        flywheel_path=args.flywheel,
        family=args.family,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        recursive=not args.non_recursive,
        limit=args.limit,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        data_source=args.data_source,
        probe_data_source=args.probe_data_source,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        extra_overrides=args.set or [],
        promote_limit=args.promote_limit,
        baseline_checkpoint=args.baseline_checkpoint,
        improvement_threshold=args.improvement_threshold,
    )
    if result.results:
        last = result.results[-1]
        if last.train_stdout:
            print(last.train_stdout, end="")
        if last.train_stderr:
            print(last.train_stderr, file=sys.stderr, end="")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


def cmd_autopilot(args: argparse.Namespace) -> int:
    from cadflow.autopilot import run_autopilot

    result = run_autopilot(
        args.raw_dir,
        args.out_dir,
        flywheel_path=args.flywheel,
        family=args.family,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        recursive=not args.non_recursive,
        limit=args.limit,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        data_source=args.data_source,
        probe_data_source=args.probe_data_source,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        extra_overrides=args.set or [],
        promote_limit=args.promote_limit,
        baseline_checkpoint=args.baseline_checkpoint,
        improvement_threshold=args.improvement_threshold,
        skip_tests=args.skip_tests,
        repair_env=not args.no_repair_env,
    )
    if result.pytest_stdout:
        print(result.pytest_stdout, end="")
    if result.pytest_stderr:
        print(result.pytest_stderr, file=sys.stderr, end="")
    if result.loop is not None:
        if result.loop.train_stdout:
            print(result.loop.train_stdout, end="")
        if result.loop.train_stderr:
            print(result.loop.train_stderr, file=sys.stderr, end="")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2



def cmd_preflight(args: argparse.Namespace) -> int:
    from cadflow.preflight import run_pretraining_preflight, render_pretraining_preflight

    result = run_pretraining_preflight(
        project_root=args.project_root,
        goal=args.goal,
        family=args.family,
        material=args.material,
        out_dir=args.out_dir,
        data_root=args.data_root,
        raw_dirs=args.raw_dir,
        config=args.config,
        data_source=args.data_source,
        max_steps=args.max_steps,
        run_smoke=args.run_smoke,
        smoke_num_points=args.num_points,
        smoke_num_fields=args.num_fields,
        smoke_format=args.format,
    )
    print(render_pretraining_preflight(result, as_json=args.json), end="")
    if not args.json:
        print()
    return 0 if result.ok else 1



def cmd_modal_train(args: argparse.Namespace) -> int:
    # Ensure venv site-packages is ahead of system paths
    import sys
    venv_site = Path(__file__).parent.parent / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if str(venv_site) not in sys.path:
        sys.path.insert(0, str(venv_site))
    
    from cadflow.modal_training import launch_modal_training

    if not args.raw_dir and not args.flywheel and not args.graph_path:
        raise SystemExit("at least one --raw-dir, --flywheel, or --graph-path is required")
    result = launch_modal_training(
        project_root=args.project_root,
        goal=args.goal,
        raw_dirs=args.raw_dir,
        out_dir=args.out_dir,
        family=args.family,
        material=args.material,
        flywheel_path=args.flywheel,
        config=args.config,
        data_source=args.data_source,
        probe_data_source=args.probe_data_source,
        graph_path=args.graph_path,
        num_points=args.num_points,
        num_fields=args.num_fields,
        fmt=args.format,
        recursive=not args.non_recursive,
        limit=args.limit,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        extra_overrides=args.set or [],
        promote_limit=args.promote_limit,
        baseline_checkpoint=args.baseline_checkpoint,
        improvement_threshold=args.improvement_threshold,
        sync_to_latticezero=not args.no_sync_latticezero,
        latticezero_root=args.latticezero_root,
    )
    remote = result.remote_result
    if remote.get("train_stdout"):
        print(remote["train_stdout"], end="")
    if remote.get("train_stderr"):
        print(remote["train_stderr"], file=sys.stderr, end="")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    report = build_doctor_report(runtime)
    print(render_doctor_report(report, as_json=args.json))
    return 0 if report.get("native_ready") else 1



def cmd_space_eval(args: argparse.Namespace) -> int:
    from eval.probe import load_config
    from eval.space_eval import compare_checkpoints

    cfg = load_config(args.config)
    if args.family is not None:
        from utils.config import load_yaml_with_family

        cfg = load_yaml_with_family(args.config, family=args.family)
    result = compare_checkpoints(
        cfg,
        args.candidate,
        args.baseline,
        args.data_source,
        threshold=args.threshold,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"candidate={result.candidate} baseline={result.baseline} data_source={result.data_source} "
            f"score_name={result.score_name} candidate_score={result.candidate_score:.6f} "
            f"baseline_score={result.baseline_score if result.baseline_score is not None else 'nan'} "
            f"improved={result.improved} improvement={result.improvement if result.improvement is not None else 'nan'}"
        )
    return 0 if result.improved or result.baseline is None else 1


def cmd_validate_sources(args: argparse.Namespace) -> int:
    report = validate_source_registry(limit=args.limit, domains=args.domain, keys=args.key)
    print(render_validation_report(report, as_json=args.json), end="")
    return 0 if report.counts.get("blocked", 0) == 0 else 2


def cmd_graph_schema(args: argparse.Namespace) -> int:
    catalog = build_spaceflight_graph_schema()
    print(render_graph_schema(catalog, as_json=args.json), end="")
    return 0


def cmd_graph_export(args: argparse.Namespace) -> int:
    graph = build_source_registry_graph()
    print(render_graph_document(graph, as_json=args.json), end="")
    return 0


def cmd_neo4j_import(args: argparse.Namespace) -> int:
    graph = build_source_registry_graph()
    bundle = write_neo4j_bundle(graph, args.out_dir, database=args.database)
    report = Neo4jImportReport(
        bundle=bundle,
        database=args.database,
        cypher_shell="bundle-only",
        exit_code=0,
        stdout_path=bundle.log_path,
        stderr_path=bundle.log_path,
        node_count=bundle.node_count,
        edge_count=bundle.edge_count,
        status="bundle_written",
        notes=("bundle written without live Neo4j import",),
    )
    print(render_neo4j_import_report(report, as_json=args.json), end="")
    return 0


def _bundle_only_report(bundle, database: str, notes: tuple[str, ...] = ()) -> Neo4jImportReport:
    return Neo4jImportReport(
        bundle=bundle,
        database=database,
        cypher_shell="bundle-only",
        exit_code=0,
        stdout_path=bundle.log_path,
        stderr_path=bundle.log_path,
        node_count=bundle.node_count,
        edge_count=bundle.edge_count,
        status="bundle_written",
        notes=notes or ("bundle written without live Neo4j import",),
    )


def cmd_neo4j_import_corpus(args: argparse.Namespace) -> int:
    report = build_processed_corpus_graph(
        args.manifest,
        args.processed_dir,
        source_key=args.source_key,
        include_raw_assets=not args.no_raw_assets,
    )
    if args.dump_only:
        print(render_corpus_graph_report(report, as_json=args.json), end="")
        return 0
    bundle = write_neo4j_bundle(report.graph, args.out_dir, database=args.database)
    import_report = _bundle_only_report(bundle, args.database, notes=(f"loaded from {report.manifest_path}",))
    print(render_corpus_graph_report(report, as_json=args.json), end="")
    print(render_neo4j_import_report(import_report, as_json=args.json), end="")
    return 0


def cmd_neo4j_import_local_data(args: argparse.Namespace) -> int:
    report = build_local_data_graph(args.data_root)
    if args.dump_only:
        print(render_local_data_graph_report(report, as_json=args.json), end="")
        return 0
    bundle = write_neo4j_bundle(report.graph, args.out_dir, database=args.database)
    import_report = _bundle_only_report(bundle, args.database, notes=(f"loaded from {report.data_root}",))
    print(render_local_data_graph_report(report, as_json=args.json), end="")
    print(render_neo4j_import_report(import_report, as_json=args.json), end="")
    return 0


def cmd_neo4j_import_evaluation(args: argparse.Namespace) -> int:
    report = build_flywheel_evaluation_graph(args.flywheel)
    if args.dump_only:
        print(render_evaluation_graph_report(report, as_json=args.json), end="")
        return 0
    bundle = write_neo4j_bundle(report.graph, args.out_dir, database=args.database)
    import_report = _bundle_only_report(bundle, args.database, notes=(f"loaded from {report.flywheel_path}",))
    print(render_evaluation_graph_report(report, as_json=args.json), end="")
    print(render_neo4j_import_report(import_report, as_json=args.json), end="")
    return 0


def cmd_graph_enrich(args: argparse.Namespace) -> int:
    from cadflow.graph_enrichment import build_enrichment_graph, merge_graphs, render_enrichment_report
    from cadflow.graph_schema import GraphDocument, GraphNode, GraphEdge

    raw_dirs = [Path(d) for d in (args.raw_dir or [])]
    if not raw_dirs:
        # default: all raw source dirs
        raw_base = Path("data/raw_downloads/external")
        if raw_base.exists():
            raw_dirs = [d for d in raw_base.iterdir() if d.is_dir()]

    base_path = Path(args.base_graph)
    existing_doc: GraphDocument | None = None
    if base_path.exists():
        existing_data = json.loads(base_path.read_bytes())
        existing_doc = GraphDocument(
            name=existing_data["name"],
            generated_at=existing_data["generated_at"],
            nodes=tuple(GraphNode(id=n["id"], type=n["type"], label=n["label"], properties=n.get("properties", {})) for n in existing_data["nodes"]),
            edges=tuple(GraphEdge(id=e["id"], type=e["type"], source=e["source"], target=e["target"], properties=e.get("properties", {})) for e in existing_data["edges"]),
            metadata=existing_data.get("metadata", {}),
        )

    raw_dirs_typed: list[Path | str] = list(raw_dirs)
    enrichment_doc, report = build_enrichment_graph(raw_dirs_typed, existing_graph=existing_doc)

    if existing_doc is not None:
        merged = merge_graphs(existing_doc, enrichment_doc, name="enriched-spaceflight-graph")
    else:
        merged = enrichment_doc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "enrichment-graph.json").write_bytes(json.dumps(enrichment_doc.to_dict(), indent=2, default=str).encode())
    (out_dir / "merged-graph.json").write_bytes(json.dumps(merged.to_dict(), indent=2, default=str).encode())

    bundle = write_neo4j_bundle(merged, str(out_dir), database=args.database)
    report_str = render_enrichment_report(report, as_json=args.json)
    print(report_str, end="" if args.json else "\n")
    summary = f"merged_nodes={len(merged.nodes)} merged_edges={len(merged.edges)} cypher={Path(bundle.cypher_path).stat().st_size//1024}KB"
    if not args.json:
        print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadflow", description="JEPA-CAD CAD/CAE orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run manifest -> geometry -> solver -> verify -> flywheel")
    run.add_argument("--manifest", required=True, help="Path to JobManifest JSON")
    run.add_argument("--workdir", default="artifacts/runs")
    run.add_argument("--outdir", default="artifacts/results")
    run.add_argument("--flywheel", default="artifacts/flywheel.jsonl")
    run.add_argument("--solver", default=None, help="Override solver kind (fea|openfoam|mbd)")
    run.add_argument("--mock-cad", action="store_true", help="Force mock CAD backend")
    run.add_argument("--require-native-solver", action="store_true", help="Fail if solver binary missing")
    run.add_argument("--promote-to", default=None, help="Optional curated dataset output dir")
    run.add_argument("--promote-limit", type=int, default=5)
    run.add_argument("--solver-root", default=None, help="Native solver root directory")
    run.add_argument("--solver-bin-dir", action="append", default=None, help="Additional native solver bin dir")
    run.add_argument("--solver-lib-dir", action="append", default=None, help="Additional native solver library dir")
    run.set_defaults(func=cmd_run)

    promote = sub.add_parser("promote", help="Promote verified flywheel runs to curated shards")
    promote.add_argument("--flywheel", required=True)
    promote.add_argument("--out-dir", required=True)
    promote.add_argument("--limit", type=int, default=50)
    promote.add_argument("--num-points", type=int, default=1024)
    promote.add_argument("--num-fields", type=int, default=3)
    promote.set_defaults(func=cmd_promote)

    ingest = sub.add_parser("ingest", help="Ingest raw files and verified flywheel runs into training shards")
    ingest.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    ingest.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    ingest.add_argument("--out-dir", required=True, help="Curated shard output directory")
    ingest.add_argument("--num-points", type=int, default=1024)
    ingest.add_argument("--num-fields", type=int, default=3)
    ingest.add_argument("--format", choices=["npz", "pt"], default="npz")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    ingest.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow unsupported raw files to fall back to synthetic samples",
    )
    ingest.set_defaults(func=cmd_ingest)

    project = sub.add_parser("project", help="Intake an existing project into a manifest")
    project.add_argument("--project-root", required=True, help="Existing CAD/CAE project root")
    project.add_argument("--goal", required=True, help="Design goal or optimization objective")
    project.add_argument("--family", default="space", help="Config family, e.g. space")
    project.add_argument("--solver", choices=["fea", "openfoam", "mbd"], default=None)
    project.add_argument("--material", default=None, help="Primary material family")
    project.add_argument("--target", action="append", default=None, help="Target override, e.g. max_stress_mpa=180")
    project.add_argument("--notes", default=None, help="Optional extra notes")
    project.add_argument("--json", action="store_true", help="Emit JSON only")
    project.add_argument("--out-dir", default=None, help="Where to write the generated manifest")
    project.set_defaults(func=cmd_project_intake)

    design = sub.add_parser("design-loop", help="Iterate a manifest/project against simulation results")
    design.add_argument("--manifest", default=None, help="Path to a JobManifest JSON")
    design.add_argument("--project-root", default=None, help="Existing CAD/CAE project root")
    design.add_argument("--goal", default=None, help="Goal for intake if no manifest is supplied")
    design.add_argument("--family", default="space", help="Config family, e.g. space")
    design.add_argument("--solver", choices=["fea", "openfoam", "mbd"], default=None)
    design.add_argument("--material", default=None, help="Primary material family")
    design.add_argument("--target", action="append", default=None, help="Target override, e.g. max_stress_mpa=180")
    design.add_argument("--notes", default=None, help="Optional extra notes")
    design.add_argument("--out-dir", required=True, help="Design-loop output directory")
    design.add_argument("--repeat", type=int, default=3)
    design.add_argument("--tolerance", type=float, default=0.05)
    design.add_argument("--mock-cad", action="store_true", help="Force mock CAD backend")
    design.add_argument("--require-native-solver", action="store_true", help="Fail if solver binary missing")
    design.set_defaults(func=cmd_design_loop)

    cloud_plan = sub.add_parser("cloud-plan", help="Generate a Modal/Fireworks training plan from a manifest")
    cloud_plan.add_argument("--manifest", required=True, help="Path to a JobManifest JSON")
    cloud_plan.add_argument("--family", default="space", help="Config family, e.g. space")
    cloud_plan.add_argument("--provider", default=None, choices=["Modal", "Fireworks", "modal", "fireworks"], help="Preferred primary provider")
    cloud_plan.add_argument("--max-dataset-sources", type=int, default=6)
    cloud_plan.add_argument("--json", action="store_true", help="Emit JSON output")
    cloud_plan.set_defaults(func=cmd_cloud_plan)

    e2e = sub.add_parser("e2e", help="Ingest data and run a short JEPA training job")
    e2e.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    e2e.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    e2e.add_argument("--out-dir", required=True, help="Curated shard output directory")
    e2e.add_argument("--config", default="configs/base.yaml", help="Training config path")
    e2e.add_argument("--family", default=None, help="Optional config family overlay, e.g. space")
    e2e.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    e2e.add_argument("--num-points", type=int, default=1024)
    e2e.add_argument("--num-fields", type=int, default=3)
    e2e.add_argument("--format", choices=["npz", "pt"], default="npz")
    e2e.add_argument("--limit", type=int, default=None)
    e2e.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    e2e.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow unsupported raw files to fall back to synthetic samples",
    )
    e2e.add_argument("--max-steps", type=int, default=1, help="Training steps for the smoke run")
    e2e.add_argument("--grad-accum-steps", type=int, default=None)
    e2e.add_argument(
        "--set",
        type=str,
        action="append",
        default=None,
        help="Extra training overrides, e.g. --set model.embed_dim=256",
    )
    e2e.set_defaults(func=cmd_e2e)

    corpus_sweep = sub.add_parser("corpus-sweep", help="Run parallel parametric sweeps across geometry-bearing source directories")
    corpus_sweep.add_argument("--raw-dir", action="append", default=[], help="Geometry root (repeatable); defaults to data/raw_downloads if omitted")
    corpus_sweep.add_argument("--flywheel", default=None, help="Flywheel JSONL path")
    corpus_sweep.add_argument("--seed-flywheel", default=None, help="Optional existing flywheel JSONL to seed from")
    corpus_sweep.add_argument("--data-root", default="data/processed", help="Processed data root for corpus graph building")
    corpus_sweep.add_argument("--out-dir", required=True, help="Output directory for the sweep artifacts")
    corpus_sweep.add_argument("--variants-per-source", type=int, default=2, help="Number of parametric variants per source")
    corpus_sweep.add_argument("--include-reference", action="store_true", help="Include low-priority reference shapes in the sweep")
    corpus_sweep.add_argument("--max-sources", type=int, default=None, help="Optional cap on discovered geometry sources")
    corpus_sweep.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    corpus_sweep.add_argument("--max-workers", type=int, default=8, help="Parallel worker count")
    corpus_sweep.add_argument("--num-points", type=int, default=1024)
    corpus_sweep.add_argument("--num-fields", type=int, default=6)
    corpus_sweep.add_argument("--format", choices=["npz", "pt"], default="npz")
    corpus_sweep.add_argument("--promote-limit", type=int, default=10000)
    corpus_sweep.add_argument("--mock-cad", action="store_true", help="Use mock CAD backend")
    corpus_sweep.add_argument("--require-native-solver", action="store_true", help="Fail if native solver unavailable")
    corpus_sweep.add_argument("--json", action="store_true", help="Emit JSON output")
    corpus_sweep.set_defaults(func=cmd_corpus_sweep)

    loop = sub.add_parser("loop", help="Run the verified-data flywheel: ingest -> promote -> train -> probe -> promote")
    loop.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    loop.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    loop.add_argument("--out-dir", required=True, help="Loop output directory")
    loop.add_argument("--config", default="configs/base.yaml", help="Training config path")
    loop.add_argument("--family", default=None, help="Optional config family overlay, e.g. space")
    loop.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    loop.add_argument("--probe-data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    loop.add_argument("--num-points", type=int, default=1024)
    loop.add_argument("--num-fields", type=int, default=3)
    loop.add_argument("--format", choices=["npz", "pt"], default="npz")
    loop.add_argument("--limit", type=int, default=None)
    loop.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    loop.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow unsupported raw files to fall back to synthetic samples",
    )
    loop.add_argument("--max-steps", type=int, default=1, help="Training steps for the loop run")
    loop.add_argument("--grad-accum-steps", type=int, default=None)
    loop.add_argument("--promote-limit", type=int, default=50)
    loop.add_argument("--baseline-checkpoint", default=None, help="Optional prior checkpoint to compare against")
    loop.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Required fractional improvement over the baseline probe score",
    )
    loop.add_argument("--repeat", type=int, default=1, help="Number of loop cycles to run; 0 means run until stopped")
    loop.add_argument(
        "--interval-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep between loop cycles",
    )
    loop.add_argument(
        "--stop-file",
        default=None,
        help="Optional file path; if it exists, the loop stops before the next cycle",
    )
    loop.add_argument(
        "--set",
        type=str,
        action="append",
        default=None,
        help="Extra training overrides, e.g. --set model.embed_dim=256",
    )
    loop.set_defaults(func=cmd_loop)

    autopilot = sub.add_parser("autopilot", help="Run env repair, pytest, and the recursive improvement loop")
    autopilot.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    autopilot.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    autopilot.add_argument("--out-dir", required=True, help="Autopilot report / loop output directory")
    autopilot.add_argument("--config", default="configs/base.yaml", help="Training config path")
    autopilot.add_argument("--family", default=None, help="Optional config family overlay, e.g. space")
    autopilot.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    autopilot.add_argument("--probe-data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    autopilot.add_argument("--num-points", type=int, default=1024)
    autopilot.add_argument("--num-fields", type=int, default=3)
    autopilot.add_argument("--format", choices=["npz", "pt"], default="npz")
    autopilot.add_argument("--limit", type=int, default=None)
    autopilot.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    autopilot.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow unsupported raw files to fall back to synthetic samples",
    )
    autopilot.add_argument("--max-steps", type=int, default=1, help="Training steps for the loop run")
    autopilot.add_argument("--grad-accum-steps", type=int, default=None)
    autopilot.add_argument("--promote-limit", type=int, default=50)
    autopilot.add_argument("--baseline-checkpoint", default=None, help="Optional prior checkpoint to compare against")
    autopilot.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Required fractional improvement over the baseline probe score",
    )
    autopilot.add_argument("--skip-tests", action="store_true", help="Skip the pytest gate")
    autopilot.add_argument("--no-repair-env", action="store_true", help="Do not auto-install missing requirements")
    autopilot.add_argument(
        "--set",
        type=str,
        action="append",
        default=None,
        help="Extra training overrides, e.g. --set model.embed_dim=256",
    )
    autopilot.set_defaults(func=cmd_autopilot)

    preflight = sub.add_parser("preflight", help="Run the pre-training checklist and optional ingest->train smoke")
    preflight.add_argument("--project-root", required=True, help="Existing project root to intake")
    preflight.add_argument("--goal", required=True, help="Training goal / first wedge")
    preflight.add_argument("--family", default="space", help="Domain family, e.g. space")
    preflight.add_argument("--material", default=None, help="Optional primary material")
    preflight.add_argument("--out-dir", required=True, help="Directory for the preflight report and smoke artifacts")
    preflight.add_argument("--data-root", default="data", help="Local data root to graph and inspect")
    preflight.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable) for smoke runs")
    preflight.add_argument("--config", default="configs/base.yaml", help="Training config path for smoke runs")
    preflight.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    preflight.add_argument("--num-points", type=int, default=1024)
    preflight.add_argument("--num-fields", type=int, default=6)
    preflight.add_argument("--format", choices=["npz", "pt"], default="npz")
    preflight.add_argument("--max-steps", type=int, default=1, help="Training steps for the smoke run")
    preflight.add_argument("--run-smoke", action="store_true", help="Run ingest -> train and validate the graph-backed dataset path")
    preflight.add_argument("--json", action="store_true", help="Emit JSON output")
    preflight.set_defaults(func=cmd_preflight)

    modal_train = sub.add_parser("modal-train", help="Run the verified-data flywheel on Modal and sync the promoted registry to LatticeZero")
    modal_train.add_argument("--project-root", required=True, help="Existing project root to intake")
    modal_train.add_argument("--goal", required=True, help="Training goal / first wedge")
    modal_train.add_argument("--family", default="space", help="Domain family, e.g. space")
    modal_train.add_argument("--material", default=None, help="Optional primary material")
    modal_train.add_argument("--out-dir", required=True, help="Directory for the cloud run report and synced artifacts")
    modal_train.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    modal_train.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    modal_train.add_argument("--config", default="configs/base.yaml", help="Training config path")
    modal_train.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    modal_train.add_argument("--probe-data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    modal_train.add_argument("--graph-path", default=None, help="Graph export JSON to stage remotely for data-source=graph (TAO conditioning)")
    modal_train.add_argument("--num-points", type=int, default=1024)
    modal_train.add_argument("--num-fields", type=int, default=6)
    modal_train.add_argument("--format", choices=["npz", "pt"], default="npz")
    modal_train.add_argument("--limit", type=int, default=None)
    modal_train.add_argument("--non-recursive", action="store_true", help="Only scan the top level of raw dirs")
    modal_train.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow unsupported raw files to fall back to synthetic samples",
    )
    modal_train.add_argument("--max-steps", type=int, default=1, help="Training steps for the cloud run")
    modal_train.add_argument("--grad-accum-steps", type=int, default=None)
    modal_train.add_argument("--promote-limit", type=int, default=50)
    modal_train.add_argument("--baseline-checkpoint", default=None, help="Optional prior checkpoint to compare against")
    modal_train.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Required fractional improvement over the baseline probe score",
    )
    modal_train.add_argument("--no-sync-latticezero", action="store_true", help="Skip syncing Modal registry artifacts back to the local LatticeZero state")
    modal_train.add_argument("--latticezero-root", default=None, help="Override the local LatticeZero data root")
    modal_train.add_argument("--json", action="store_true", help="Emit JSON output")
    modal_train.add_argument(
        "--set",
        type=str,
        action="append",
        default=None,
        help="Extra training overrides, e.g. --set model.embed_dim=256",
    )
    modal_train.set_defaults(func=cmd_modal_train)

    doctor = sub.add_parser("doctor", help="Inspect native solver readiness")
    doctor.add_argument("--json", action="store_true", help="Emit JSON diagnostic output")
    doctor.add_argument("--solver-root", default=None, help="Native solver root directory")
    doctor.add_argument("--solver-bin-dir", action="append", default=None, help="Additional native solver bin dir")
    doctor.add_argument("--solver-lib-dir", action="append", default=None, help="Additional native solver library dir")
    doctor.set_defaults(func=cmd_doctor)

    space_eval = sub.add_parser("space-eval", help="Compare a space-family candidate checkpoint against a baseline")
    space_eval.add_argument("--config", default="configs/base.yaml", help="Training config path")
    space_eval.add_argument("--family", default=None, help="Optional config family overlay, e.g. space")
    space_eval.add_argument("--candidate", required=True, help="Path to candidate checkpoint")
    space_eval.add_argument("--baseline", default=None, help="Path to baseline checkpoint")
    space_eval.add_argument("--data-source", choices=["real", "synthetic", "mixed", "graph"], default="real")
    space_eval.add_argument("--threshold", type=float, default=0.0, help="Required fractional improvement over baseline")
    space_eval.add_argument("--seed", type=int, default=None)
    space_eval.add_argument("--data-dir", default=None, help="Override data.data_dir for probing")
    space_eval.add_argument("--json", action="store_true", help="Emit JSON output")
    space_eval.set_defaults(func=cmd_space_eval)

    validate_sources = sub.add_parser("validate-sources", help="Validate and classify known public source registry entries")
    validate_sources.add_argument("--json", action="store_true", help="Emit JSON output")
    validate_sources.add_argument("--limit", type=int, default=None, help="Limit the number of sources checked")
    validate_sources.add_argument("--domain", action="append", default=None, help="Restrict validation to one or more domains")
    validate_sources.add_argument("--key", action="append", default=None, help="Restrict validation to one or more source keys")
    validate_sources.set_defaults(func=cmd_validate_sources)

    graph_schema = sub.add_parser("graph-schema", help="Render the adaptable spaceflight graph schema")
    graph_schema.add_argument("--json", action="store_true", help="Emit JSON output")
    graph_schema.set_defaults(func=cmd_graph_schema)

    graph_export = sub.add_parser("graph-export", help="Export the current source registry as a graph document")
    graph_export.add_argument("--json", action="store_true", help="Emit JSON output")
    graph_export.set_defaults(func=cmd_graph_export)

    neo4j_import = sub.add_parser("neo4j-import", help="Load the current source registry into local Neo4j")
    neo4j_import.add_argument("--out-dir", default="artifacts/neo4j-import", help="Directory for the export bundle and logs")
    neo4j_import.add_argument("--database", default="neo4j", help="Neo4j database name")
    neo4j_import.add_argument("--json", action="store_true", help="Emit JSON output")
    neo4j_import.set_defaults(func=cmd_neo4j_import)

    neo4j_import_corpus = sub.add_parser("neo4j-import-corpus", help="Load a processed corpus manifest into local Neo4j")
    neo4j_import_corpus.add_argument("--manifest", default="data/processed/nasa3d/ingestion_manifest.json", help="Processed corpus manifest")
    neo4j_import_corpus.add_argument("--processed-dir", default="data/processed/nasa3d", help="Processed corpus directory")
    neo4j_import_corpus.add_argument("--source-key", default=None, help="Optional source registry key for provenance")
    neo4j_import_corpus.add_argument("--no-raw-assets", action="store_true", help="Skip raw asset nodes in the export")
    neo4j_import_corpus.add_argument("--out-dir", default="artifacts/neo4j-import-corpus", help="Directory for the export bundle and logs")
    neo4j_import_corpus.add_argument("--database", default="neo4j", help="Neo4j database name")
    neo4j_import_corpus.add_argument("--dump-only", action="store_true", help="Only render the graph report, no Neo4j bundle")
    neo4j_import_corpus.add_argument("--json", action="store_true", help="Emit JSON output")
    neo4j_import_corpus.set_defaults(func=cmd_neo4j_import_corpus)

    neo4j_import_local_data = sub.add_parser("neo4j-import-local-data", help="Load the complete local data tree into local Neo4j")
    neo4j_import_local_data.add_argument("--data-root", default="data", help="Local data root to scan")
    neo4j_import_local_data.add_argument("--out-dir", default="artifacts/neo4j-import-local-data", help="Directory for the export bundle and logs")
    neo4j_import_local_data.add_argument("--database", default="neo4j", help="Neo4j database name")
    neo4j_import_local_data.add_argument("--dump-only", action="store_true", help="Only render the graph report, no Neo4j bundle")
    neo4j_import_local_data.add_argument("--json", action="store_true", help="Emit JSON output")
    neo4j_import_local_data.set_defaults(func=cmd_neo4j_import_local_data)

    neo4j_import_eval = sub.add_parser("neo4j-import-evaluation", help="Load the flywheel evaluation graph into local Neo4j")
    neo4j_import_eval.add_argument("--flywheel", default="artifacts/flywheel.jsonl", help="Flywheel JSONL path")
    neo4j_import_eval.add_argument("--out-dir", default="artifacts/neo4j-import-evaluation", help="Directory for the export bundle and logs")
    neo4j_import_eval.add_argument("--database", default="neo4j", help="Neo4j database name")
    neo4j_import_eval.add_argument("--dump-only", action="store_true", help="Only render the graph report, no Neo4j bundle")
    neo4j_import_eval.add_argument("--json", action="store_true", help="Emit JSON output")
    neo4j_import_eval.set_defaults(func=cmd_neo4j_import_evaluation)

    graph_enrich = sub.add_parser("graph-enrich", help="Enrich master graph with RealPart, Material, Document, Assembly, and ExperimentalResult nodes from raw source dirs")
    graph_enrich.add_argument("--raw-dir", action="append", default=[], help="Raw source directory to enrich from (repeatable)")
    graph_enrich.add_argument("--base-graph", default="artifacts/corpus-sweep-run/sweep/neo4j/spaceflight-graph.json", help="Existing graph JSON to enrich on top of")
    graph_enrich.add_argument("--out-dir", default="artifacts/enriched-graph", help="Output directory for enriched graph bundle")
    graph_enrich.add_argument("--database", default="spaceflight-enriched", help="Neo4j database name")
    graph_enrich.add_argument("--json", action="store_true", help="Emit JSON report")
    graph_enrich.set_defaults(func=cmd_graph_enrich)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

