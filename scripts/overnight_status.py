#!/usr/bin/env python3
"""Compute overnight finish gate: FEA/CFD coverage + TAO readiness."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frd_ok(fea: Path, pid: str) -> bool:
    frd = fea / pid / "case.frd"
    try:
        return frd.is_file() and frd.stat().st_size >= 50_000
    except OSError:
        return False


def compute() -> dict:
    from cadflow.rocket_cfd_curate import ROCKET_CFD_FAMILIES, is_degenerate_box
    from cadflow.rocket_physics_suite import load_manifest

    man = load_manifest(ROOT / "data/openrocket_hardware_8k")
    fea = ROOT / "artifacts/rocket_fea_8k"
    cfd = ROOT / "artifacts/rocket_cfd_bodyfit"
    skip: set[str] = set()
    sp = ROOT / "artifacts/fea_skip_parts.json"
    if sp.is_file():
        try:
            skip = set(json.loads(sp.read_text()).get("part_ids") or [])
        except Exception:
            skip = set()

    elig = [
        e
        for e in man
        if e.get("family") in ROCKET_CFD_FAMILIES
        and not is_degenerate_box(e)
        and int(e.get("faces") or 0) <= 12_000
        and e["part_id"] not in skip
    ]
    fea_ok = sum(1 for e in elig if _frd_ok(fea, e["part_id"]))
    fea_miss_by = Counter()
    for e in elig:
        if not _frd_ok(fea, e["part_id"]):
            fea_miss_by[e.get("family") or "?"] += 1

    curated_path = ROOT / "artifacts/rocket_cfd_curated.json"
    curated = []
    if curated_path.is_file():
        curated = json.loads(curated_path.read_text()).get("entries") or []
    cfd_ok = sum(1 for e in curated if (cfd / e["part_id"] / "meta.json").is_file())

    body = ROOT / "artifacts/cfd_bodyfit"
    internal = ROOT / "artifacts/cfd_internal"
    legacy_body = (
        sum(1 for d in body.iterdir() if d.is_dir() and (d / "meta.json").is_file())
        if body.is_dir()
        else 0
    )
    legacy_int = (
        sum(1 for d in internal.iterdir() if d.is_dir() and (d / "meta.json").is_file())
        if internal.is_dir()
        else 0
    )

    gpath = ROOT / "artifacts/jepa-train-bundle/graph.json"
    graph_stats: dict = {}
    if gpath.is_file():
        g = json.loads(gpath.read_text())
        rocket = legacy = r_fea = r_cfd = l_fea = l_cfd = 0
        types = Counter()
        for n in g["nodes"]:
            types[n.get("type")] += 1
            if n.get("type") != "Part":
                continue
            pid = str(n.get("id") or "")
            is_r = pid.startswith("part:rocket:")
            hf = bool(n.get("has_fea") or n.get("simulation_results_fea"))
            hc = bool(n.get("has_cfd") or n.get("simulation_results_cfd"))
            if is_r:
                rocket += 1
                r_fea += int(hf)
                r_cfd += int(hc)
            else:
                legacy += 1
                l_fea += int(hf)
                l_cfd += int(hc)
        graph_stats = {
            "rocket_parts": rocket,
            "legacy_parts": legacy,
            "rocket_fea": r_fea,
            "rocket_cfd": r_cfd,
            "legacy_fea": l_fea,
            "legacy_cfd": l_cfd,
            "Document": types.get("Document", 0),
            "Source": types.get("Source", 0),
            "TensorShard": types.get("TensorShard", 0),
            "Sample": types.get("Sample", 0),
            "Material": types.get("Material", 0),
            "graph_mb": round(gpath.stat().st_size / 1e6, 1),
        }

    # Gate: curated CFD done; FEA remaining only fairings (or <50 non-fairing);
    # graph has rocket CFD close to curated + FEA mostly linked.
    non_fairing_miss = sum(v for k, v in fea_miss_by.items() if k != "fairing")
    fairing_miss = fea_miss_by.get("fairing", 0)
    cfd_done = len(curated) > 0 and cfd_ok >= len(curated)
    fea_core_done = non_fairing_miss <= 30
    # Full gate waits for fairings too, but training gate can open earlier.
    solvers_drained = fea_core_done and cfd_done and fairing_miss <= 50
    train_gate = fea_core_done and cfd_done and graph_stats.get("rocket_fea", 0) >= 5500

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rocket_fea_eligible_ok": fea_ok,
        "rocket_fea_eligible_total": len(elig),
        "rocket_fea_missing_by_family": dict(fea_miss_by),
        "rocket_fea_non_fairing_missing": non_fairing_miss,
        "rocket_fea_fairing_missing": fairing_miss,
        "rocket_cfd_curated_ok": cfd_ok,
        "rocket_cfd_curated_total": len(curated),
        "legacy_bodyfit_metas": legacy_body,
        "legacy_internal_metas": legacy_int,
        "fea_skips": len(skip),
        "graph": graph_stats,
        "gates": {
            "cfd_done": cfd_done,
            "fea_core_done": fea_core_done,
            "solvers_drained": solvers_drained,
            "train_gate": train_gate,
        },
    }
    return out


def main() -> int:
    status = compute()
    out = ROOT / "artifacts/overnight_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
