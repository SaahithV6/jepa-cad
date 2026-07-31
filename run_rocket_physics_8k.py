#!/usr/bin/env python3.12
"""Batch CalculiX + OpenFOAM physics for the OpenRocket 8k/10.5k hardware corpus.

Adds Part nodes to the TAO graph and attaches real FEA/CFD annotations.
Does not modify artifacts/fea_final or artifacts/cfd_final.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadflow.rocket_cfd_curate import ROCKET_CFD_FAMILIES, is_degenerate_box
from cadflow.rocket_physics_suite import (
    DEFAULT_CFD_ROOT,
    DEFAULT_CORPUS,
    DEFAULT_FEA_ROOT,
    DEFAULT_GRAPH,
    DEFAULT_SUMMARY,
    ensure_parts_in_graph,
    filter_fea_skip_safe_dupes,
    ingest_cfd_to_graph,
    ingest_fea_to_graph,
    load_manifest,
    run_batch_cfd,
    run_batch_fea,
    select_entries,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--fea-root", type=Path, default=DEFAULT_FEA_ROOT)
    ap.add_argument("--cfd-root", type=Path, default=DEFAULT_CFD_ROOT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Limit to N parts (0 = all matching)",
    )
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fea-only", action="store_true")
    ap.add_argument("--cfd-only", action="store_true")
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument(
        "--no-ingest",
        action="store_true",
        help="Run solvers but skip TAO graph updates (safer for parallel FEA/CFD)",
    )
    ap.add_argument("--register-only", action="store_true", help="Only add Part nodes to graph")
    ap.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip Part registration (use when graph already has rocket parts)",
    )
    ap.add_argument(
        "--families",
        type=str,
        default="",
        help="Comma-separated families (e.g. nose_cone,fin,nozzle)",
    )
    ap.add_argument(
        "--curated-rocket",
        action="store_true",
        help="FEA/CFD only classic rocket families; skip degenerate low-face boxes; prefer high-face parts",
    )
    ap.add_argument(
        "--cl-max-mm",
        type=float,
        default=4.0,
        help="Fixed Gmsh cl_max (mm); ignored when --target-tets > 0",
    )
    ap.add_argument(
        "--target-tets",
        type=int,
        default=15_000,
        help="Adaptive mesh sizing toward this tet count (0 = use --cl-max-mm). "
        "Default 15000 aims for ~10k–25k tets across part sizes.",
    )
    ap.add_argument(
        "--mesh-timeout",
        type=int,
        default=180,
        help="Gmsh volume-mesh timeout per part (seconds)",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus missing: {args.corpus}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.corpus)
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    if args.curated_rocket and not families:
        families = sorted(ROCKET_CFD_FAMILIES)
    entries = select_entries(
        manifest, limit=args.pilot, families=families, offset=args.offset
    )
    if args.curated_rocket:
        before = len(entries)
        entries = [e for e in entries if not is_degenerate_box(e)]
        # Skip ultra-dense STLs that hang gmsh; prefer easier meshes first.
        entries = [e for e in entries if int(e.get("faces") or 0) <= 12_000]
        # Prefer aero airframe before heavy tanks/nozzles (tanks blow up tet count).
        family_rank = {
            "fin": 0,
            "bulkhead": 1,
            "engine_mount": 2,
            "body_tube": 3,
            "nose_cone": 4,
            "transition": 5,
            "fairing": 6,
            "nozzle": 7,
            "tank": 8,
        }
        entries.sort(
            key=lambda e: (
                family_rank.get(str(e.get("family")), 9),
                int(e.get("faces") or 0),
                e["part_id"],
            )
        )
        print(
            f"curated-rocket filter: {before} -> {len(entries)} "
            f"(dropped degenerate/huge; airframe-first)",
            flush=True,
        )

    print("=" * 72)
    print("ROCKET HARDWARE PHYSICS SUITE (CalculiX + OpenFOAM)")
    print("=" * 72)
    print(
        f"corpus={args.corpus} entries={len(entries)}/{len(manifest)} "
        f"pilot={args.pilot} families={families}",
        flush=True,
    )

    if args.skip_register:
        print("graph register: skipped", flush=True)
    else:
        reg = ensure_parts_in_graph(
            args.graph,
            manifest if args.pilot == 0 and not families else entries,
            corpus_dir=args.corpus,
        )
        print(f"graph register: {reg}", flush=True)
    if args.register_only:
        return 0

    do_fea = not args.cfd_only
    do_cfd = not args.fea_only
    fea_results: list[dict] = []
    cfd_results: list[dict] = []

    if args.ingest_only:
        if do_fea:
            linked = ingest_fea_to_graph(args.graph, args.fea_root)
            print(f"FEA ingest linked={linked}", flush=True)
        if do_cfd:
            # rebuild results from disk
            from run_cfd_5k_proper import summarize_fields

            for case_dir in sorted(args.cfd_root.iterdir()) if args.cfd_root.exists() else []:
                if not case_dir.is_dir():
                    continue
                metrics = summarize_fields(case_dir)
                if not metrics:
                    continue
                cfd_results.append(
                    {"part_id": case_dir.name, "success": True, "metrics": metrics}
                )
            linked = ingest_cfd_to_graph(args.graph, cfd_results)
            print(f"CFD ingest linked={linked}", flush=True)
        return 0

    if do_fea:
        print(f"\n[FEA] CalculiX solid-only ({args.workers} workers) ...", flush=True)
        from cadflow.msh_to_calculix import case_has_valid_frd

        def _fast_frd_ok(part_id: str) -> bool:
            """Size gate only — full DISP/STRESS check happens in the worker."""
            frd = args.fea_root / part_id / "case.frd"
            try:
                return frd.is_file() and frd.stat().st_size >= 50_000
            except OSError:
                return False

        if not args.force:
            before = len(entries)
            entries_fea = [e for e in entries if not _fast_frd_ok(e["part_id"])]
            print(
                f"FEA queue: {len(entries_fea)}/{before} missing FRD "
                f"(skipping cached size≥50KB)",
                flush=True,
            )
        else:
            entries_fea = entries
        entries_fea = filter_fea_skip_safe_dupes(entries_fea, args.fea_root)
        if args.target_tets > 0:
            print(
                f"FEA mesh: adaptive target_tets={args.target_tets} "
                f"(mesh_timeout={args.mesh_timeout}s)",
                flush=True,
            )
        else:
            print(
                f"FEA mesh: fixed cl_max_mm={args.cl_max_mm} "
                f"(mesh_timeout={args.mesh_timeout}s)",
                flush=True,
            )
        fea_results = run_batch_fea(
            entries_fea,
            args.corpus,
            args.fea_root,
            workers=args.workers,
            force=args.force,
            timeout=args.timeout,
            cl_max_mm=args.cl_max_mm,
            target_tets=args.target_tets,
            mesh_timeout_s=args.mesh_timeout,
        )
        fea_ok = sum(1 for e in entries if _fast_frd_ok(e["part_id"]))
        print(f"FEA ok≈{fea_ok}/{len(entries)} (ran {len(fea_results)} this pass)", flush=True)
        if not args.no_ingest:
            linked = ingest_fea_to_graph(args.graph, args.fea_root, fea_results)
            print(f"FEA graph linked={linked}", flush=True)
        else:
            print("FEA graph ingest skipped (--no-ingest)", flush=True)

    if do_cfd:
        print(f"\n[CFD] OpenFOAM simpleFoam ({args.workers} workers) ...", flush=True)
        cfd_results = run_batch_cfd(
            entries,
            args.corpus,
            args.cfd_root,
            workers=args.workers,
            force=args.force,
            timeout=args.timeout,
        )
        cfd_ok = sum(1 for r in cfd_results if r["success"])
        print(f"CFD ok={cfd_ok}/{len(cfd_results)}", flush=True)
        if not args.no_ingest:
            linked = ingest_cfd_to_graph(args.graph, cfd_results)
            print(f"CFD graph linked={linked}", flush=True)
        else:
            print("CFD graph ingest skipped (--no-ingest)", flush=True)

    summary = {
        "corpus": str(args.corpus),
        "entries_requested": len(entries),
        "fea_total": len(fea_results),
        "fea_ok": sum(1 for r in fea_results if r["success"]),
        "cfd_total": len(cfd_results),
        "cfd_ok": sum(1 for r in cfd_results if r["success"]),
        "fea_root": str(args.fea_root),
        "cfd_root": str(args.cfd_root),
        "graph": str(args.graph),
        "pipeline": "rocket-stl-volume-msh2-calculix-simplefoam",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("=" * 72)
    ok = True
    if do_fea and summary["fea_ok"] == 0 and entries:
        ok = False
    if do_cfd and summary["cfd_ok"] == 0 and entries:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
