#!/usr/bin/env python3
"""Batch-extract real physics-field training shards from solver output.

Writes compact ``.npz`` shards (real nodal von Mises / displacement / stress
sampled on the geometry) under ``artifacts/physics_shards/`` plus a JSONL
manifest. Graph registration happens elsewhere (single writer) so this can run
alongside the solvers/ingest loop without racing the graph.

    python run_physics_shards.py --source rocket --workers 3
    python run_physics_shards.py --source legacy_alt --workers 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from cadflow.build_physics_shards import (
    FEA_SHARD_DIR,
    FEA_MANIFEST,
    ROOT,
    fea_case_to_shard,
    write_shard,
)

SOURCES = {
    # name: (fea_root, graph id prefix, frd filename)
    "rocket": (ROOT / "artifacts/rocket_fea_8k", "part:rocket:", "case.frd"),
    "legacy_alt": (ROOT / "artifacts/fea_alt", "part:", "case_alt.frd"),
    "legacy_final": (ROOT / "artifacts/fea_final", "part:", "case.frd"),
}


def _one(frd_str: str, prefix: str, num_points: int) -> dict | None:
    frd = Path(frd_str)
    payload = fea_case_to_shard(frd, num_points=num_points)
    if payload is None:
        return None
    write_shard(payload, FEA_SHARD_DIR)
    return {
        "part_id": f"{prefix}{frd.parent.name}",
        "case_id": frd.parent.name,
        "kind": "fea",
        "shard_path": str((FEA_SHARD_DIR / f"{frd.parent.name}.npz").relative_to(ROOT)),
        "metrics": payload["metrics"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), default="rocket")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--num-points", type=int, default=2048)
    ap.add_argument("--min-bytes", type=int, default=50_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fea_root, prefix, frd_name = SOURCES[args.source]
    FEA_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    FEA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if FEA_MANIFEST.exists() and not args.force:
        for line in FEA_MANIFEST.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["part_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    frds: list[str] = []
    for case in sorted(fea_root.iterdir()):
        if not case.is_dir():
            continue
        frd = case / frd_name
        try:
            if not (frd.is_file() and frd.stat().st_size >= args.min_bytes):
                continue
        except OSError:
            continue
        if f"{prefix}{case.name}" in done:
            continue
        frds.append(str(frd))
        if args.limit and len(frds) >= args.limit:
            break

    print(f"source={args.source} root={fea_root} frd={frd_name} pending={len(frds)} workers={args.workers}", flush=True)
    ok = fail = 0
    with FEA_MANIFEST.open("a", encoding="utf-8") as mf:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(_one, f, prefix, args.num_points): f for f in frds}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception:  # noqa: BLE001
                    rec = None
                if rec is None:
                    fail += 1
                    continue
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                ok += 1
                if (ok + fail) % 100 == 0:
                    print(f"[shards] ok={ok} fail={fail} / {len(frds)}", flush=True)

    print(json.dumps({"source": args.source, "ok": ok, "fail": fail, "manifest": str(FEA_MANIFEST)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
