"""User-facing CLI for CAD/CAE orchestration and flywheel promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadflow.backends import get_backend
from cadflow.doctor import build_doctor_report, render_doctor_report
from cadflow.flywheel import DataFlywheel
from cadflow.manifest import JobManifest
from cadflow.pipeline import run_pipeline
from cadflow.promotion import promote_verified_to_dataset
from cadflow.runtime import resolve_solver_runtime


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

