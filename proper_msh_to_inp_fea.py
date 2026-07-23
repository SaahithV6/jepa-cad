#!/usr/bin/env python3.12
"""Convert Gmsh MSH2 meshes to solid-only CalculiX decks and run batch FEA.

Uses direct MSH2 parsing (not gmsh.write()) to avoid 1D/2D surface elements
that cause CalculiX constraint allocation failures.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cadflow.msh_to_calculix import (
    DEFAULT_CCX,
    case_has_valid_frd,
    convert_and_run_case,
    generate_fea_case_inp,
    ingest_fea_results_to_graph,
    run_calculix_case,
)

FEA_DIR = Path("artifacts/fea_final")
GRAPH_FILE = Path("artifacts/jepa-train-bundle/graph.json")
INDEX_FILE = Path("artifacts/jepa-train-bundle/physics_verified_index.json")
SUMMARY_FILE = Path("data/physics_verified_summary.json")


def _case_dirs(*, limit: int | None = None, resume: bool = False, force: bool = False) -> list[Path]:
    dirs = sorted(d for d in FEA_DIR.glob("*") if d.is_dir())
    if resume and not force:
        dirs = [d for d in dirs if not case_has_valid_frd(d)]
    return dirs[:limit] if limit is not None else dirs


def _convert_case(case_dir: Path) -> bool:
    try:
        generate_fea_case_inp(case_dir)
        return True
    except (OSError, ValueError, FileNotFoundError):
        return False


def _run_case(case_dir: Path, ccx_binary: Path, timeout: int) -> bool:
    try:
        # Always regenerate solid deck — other scripts may overwrite case.inp.
        generate_fea_case_inp(case_dir)
        result = run_calculix_case(case_dir, ccx_binary=ccx_binary, timeout=timeout)
        return result.converged and case_has_valid_frd(case_dir)
    except (OSError, ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _update_graph() -> dict:
    if not GRAPH_FILE.exists():
        print(f"Graph missing: {GRAPH_FILE}", file=sys.stderr)
        return {}
    stats = ingest_fea_results_to_graph(
        GRAPH_FILE,
        FEA_DIR,
        index_path=INDEX_FILE,
    )
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(
        json.dumps(
            {
                **stats,
                "pipeline": "msh2-solid-only-calculix",
                "fea_dir": str(FEA_DIR),
                "graph": str(GRAPH_FILE),
                "index": str(INDEX_FILE),
                "units": {
                    "mesh": "meters",
                    "youngs_modulus": "Pa",
                    "load": "N",
                    "reported_stress": "MPa",
                    "reported_displacement": "mm",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Run solid-only CalculiX FEA batch")
    parser.add_argument("--limit", type=int, default=None, help="Process only N cases")
    parser.add_argument("--workers", type=int, default=8, help="Parallel CalculiX workers")
    parser.add_argument("--timeout", type=int, default=600, help="Per-case timeout (seconds)")
    parser.add_argument("--ccx", type=Path, default=DEFAULT_CCX, help="CalculiX binary path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only process cases missing a valid case.frd",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all cases even if case.frd already exists",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Convert meshes and write case.inp without running CalculiX",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only parse existing FRDs and update the TAO graph",
    )
    parser.add_argument(
        "--single",
        type=Path,
        default=None,
        help="Run a single case directory (for debugging)",
    )
    args = parser.parse_args()

    if args.ingest_only:
        stats = _update_graph()
        print(json.dumps(stats, indent=2))
        return 0 if stats.get("parts_linked", 0) > 0 else 1

    if args.single is not None:
        setup, result = convert_and_run_case(
            args.single, ccx_binary=args.ccx, timeout=args.timeout
        )
        print(
            f"case={setup.case_dir.name} nodes={setup.node_count} "
            f"tets={setup.element_count} frd_bytes={result.frd_bytes}"
        )
        return 0 if result.converged and case_has_valid_frd(args.single) else 1

    case_dirs = _case_dirs(limit=args.limit, resume=args.resume, force=args.force)
    print("=" * 80)
    print("MSH2 SOLID-ONLY CONVERSION + CALCULIX FEA")
    print("=" * 80)
    print(f"Cases: {len(case_dirs)} (resume={args.resume} force={args.force})")

    print("\n[PHASE 1] Converting MSH2 -> mesh_solid.inp + case.inp ...")
    converted = 0
    for idx, case_dir in enumerate(case_dirs, start=1):
        if _convert_case(case_dir):
            converted += 1
        if idx % 200 == 0 or idx == len(case_dirs):
            print(f"  [{idx}/{len(case_dirs)}] converted", flush=True)

    print(f"Converted: {converted}/{len(case_dirs)}", flush=True)

    if args.setup_only:
        return 0

    print(f"\n[PHASE 2] Running CalculiX ({args.workers} workers) ...", flush=True)
    successful = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_case, case_dir, args.ccx, args.timeout): case_dir
            for case_dir in case_dirs
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            if future.result():
                successful += 1
            if idx % 50 == 0 or idx == len(case_dirs):
                print(f"  [{idx}/{len(case_dirs)}] complete ok={successful}", flush=True)

    print(f"FEA success (this run): {successful}/{len(case_dirs)}", flush=True)

    print("\n[PHASE 3] Ingesting FRD results into TAO graph ...", flush=True)
    stats = _update_graph()
    print(
        f"Graph: {stats.get('parts_linked', 0)}/{stats.get('parts_total', 0)} "
        f"parts physics-verified from {stats.get('fea_cases_with_frd', 0)} FRDs",
        flush=True,
    )

    print("=" * 80)
    return 0 if stats.get("parts_linked", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
