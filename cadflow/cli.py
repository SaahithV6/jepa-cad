"""User-facing CLI for CAD/CAE orchestration and flywheel promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadflow.autopilot import run_autopilot
from cadflow.backends import get_backend
from cadflow.doctor import build_doctor_report, render_doctor_report
from cadflow.e2e import run_end_to_end
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest
from cadflow.pipeline import run_pipeline
from cadflow.loop_controller import run_loop_controller
from cadflow.promotion import promote_verified_to_dataset
from cadflow.runtime import resolve_solver_runtime
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
        config=args.config,
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
    if not args.raw_dir and not args.flywheel:
        raise SystemExit("at least one --raw-dir or --flywheel is required")
    result = run_loop_controller(
        args.raw_dir,
        args.out_dir,
        repeat=args.repeat,
        interval_seconds=args.interval_seconds,
        stop_file=args.stop_file,
        flywheel_path=args.flywheel,
        config=args.config,
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
    result = run_autopilot(
        args.raw_dir,
        args.out_dir,
        flywheel_path=args.flywheel,
        config=args.config,
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


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    report = build_doctor_report(runtime)
    print(render_doctor_report(report, as_json=args.json))
    return 0 if report.get("native_ready") else 1


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

    e2e = sub.add_parser("e2e", help="Ingest data and run a short JEPA training job")
    e2e.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    e2e.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    e2e.add_argument("--out-dir", required=True, help="Curated shard output directory")
    e2e.add_argument("--config", default="configs/base.yaml", help="Training config path")
    e2e.add_argument("--data-source", choices=["real", "synthetic", "mixed"], default="real")
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

    loop = sub.add_parser("loop", help="Run the verified-data flywheel: ingest -> promote -> train -> probe -> promote")
    loop.add_argument("--raw-dir", action="append", default=[], help="Raw input directory (repeatable)")
    loop.add_argument("--flywheel", default=None, help="Optional flywheel JSONL path")
    loop.add_argument("--out-dir", required=True, help="Loop output directory")
    loop.add_argument("--config", default="configs/base.yaml", help="Training config path")
    loop.add_argument("--data-source", choices=["real", "synthetic", "mixed"], default="real")
    loop.add_argument("--probe-data-source", choices=["real", "synthetic", "mixed"], default="real")
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
    autopilot.add_argument("--data-source", choices=["real", "synthetic", "mixed"], default="real")
    autopilot.add_argument("--probe-data-source", choices=["real", "synthetic", "mixed"], default="real")
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

    doctor = sub.add_parser("doctor", help="Inspect native solver readiness")
    doctor.add_argument("--json", action="store_true", help="Emit JSON diagnostic output")
    doctor.add_argument("--solver-root", default=None, help="Native solver root directory")
    doctor.add_argument("--solver-bin-dir", action="append", default=None, help="Additional native solver bin dir")
    doctor.add_argument("--solver-lib-dir", action="append", default=None, help="Additional native solver library dir")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

