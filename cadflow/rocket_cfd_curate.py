"""Curate OpenRocket corpus entries for body-fitted external CFD.

Bodyfit CFD = STL as a wall in freestream (snappyHexMesh + simpleFoam).
This is geometry-aware external flow — not an empty channel box.

We only keep classic rocket hardware with enough mesh complexity to be a
real part (not a flat 8-face rectangle), and we dedupe exact STL copies.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

# External-aero / vehicle hardware — not TPS tiles, blankets, solar panels, antennas.
ROCKET_CFD_FAMILIES = frozenset(
    {
        "nose_cone",
        "body_tube",
        "fin",
        "nozzle",
        "transition",
        "fairing",
        "tank",
        "engine_mount",
        "bulkhead",
    }
)

# Fins are thin by nature; reject only obviously degenerate facet counts.
MIN_FACES = {
    "fin": 80,
    "default": 48,
}


def stl_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def is_degenerate_box(entry: dict[str, Any]) -> bool:
    """True for near-empty faceted boxes / paper plates with almost no mesh."""
    faces = int(entry.get("faces") or 0)
    ext = [float(x) for x in (entry.get("extents_mm") or [0, 0, 0])]
    if len(ext) != 3:
        return True
    thin, mid, long = sorted(ext)
    if long < 8.0:
        return True
    fam = str(entry.get("family", ""))
    min_faces = MIN_FACES.get(fam, MIN_FACES["default"])
    if faces < min_faces:
        return True
    # Non-fin: reject paper-thin slabs with very few faces (weird rectangles)
    if fam != "fin" and thin < 2.0 and faces < 200:
        return True
    return False


def curate_rocket_cfd_entries(
    manifest: list[dict[str, Any]],
    corpus_dir: Path,
    *,
    families: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deduped rocket-worthy entries + audit stats."""
    fams = families or set(ROCKET_CFD_FAMILIES)
    corpus_dir = Path(corpus_dir)
    kept: list[dict[str, Any]] = []
    seen_hash: set[str] = set()
    stats: dict[str, Any] = {
        "input": len(manifest),
        "family_excluded": 0,
        "degenerate": 0,
        "missing_stl": 0,
        "exact_stl_dupes_skipped": 0,
        "kept": 0,
        "kept_by_family": {},
        "dupe_canonical": {},  # discarded_id -> kept_id
    }

    # Stable order
    ordered = sorted(
        (e for e in manifest if str(e.get("family", "")).lower() in fams),
        key=lambda e: e["part_id"],
    )
    stats["family_excluded"] = len(manifest) - len(
        [e for e in manifest if str(e.get("family", "")).lower() in fams]
    )

    for e in ordered:
        if is_degenerate_box(e):
            stats["degenerate"] += 1
            continue
        stl = corpus_dir / e["stl"]
        if not stl.is_file():
            stats["missing_stl"] += 1
            continue
        digest = stl_sha1(stl)
        if digest in seen_hash:
            stats["exact_stl_dupes_skipped"] += 1
            # record first keeper for this hash
            canon = next(k["part_id"] for k in kept if k.get("_stl_sha1") == digest)
            stats["dupe_canonical"][e["part_id"]] = canon
            continue
        seen_hash.add(digest)
        row = dict(e)
        row["_stl_sha1"] = digest
        kept.append(row)

    stats["kept"] = len(kept)
    stats["kept_by_family"] = dict(Counter(e["family"] for e in kept))
    # Compact printable copy (full dupe map stays on disk via write_curated_manifest)
    stats["dupe_canonical_count"] = len(stats["dupe_canonical"])
    return kept, stats


def write_curated_manifest(
    entries: list[dict[str, Any]],
    out_path: Path,
    stats: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # strip internal keys for the on-disk list used by runners
    clean = []
    for e in entries:
        row = {k: v for k, v in e.items() if not k.startswith("_")}
        row["stl_sha1"] = e.get("_stl_sha1")
        clean.append(row)
    payload = {"stats": stats, "entries": clean}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def demote_duplicate_and_non_curated_cfd(
    graph_path: Path,
    curated_ids: set[str],
    dupe_canonical: dict[str, str],
) -> dict[str, int]:
    """Keep bodyfit CFD only on curated canonical parts; demote the rest."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    demoted = 0
    kept = 0
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        pid = str(n.get("id", ""))
        if not pid.startswith("part:rocket:"):
            continue
        short = pid.split("part:rocket:", 1)[-1]
        cfd = n.get("simulation_results_cfd")
        is_bodyfit = isinstance(cfd, dict) and (
            cfd.get("mesh") == "snappyHexMesh_external" or n.get("cfd_mesh") == "snappyHexMesh_external"
        )
        if short in curated_ids and is_bodyfit:
            kept += 1
            continue
        if not (n.get("has_cfd") or is_bodyfit):
            continue
        # Demote non-curated or duplicate bodyfit / leftover CFD
        if isinstance(cfd, dict):
            n["simulation_results_cfd_demoted"] = cfd
        n.pop("simulation_results_cfd", None)
        n["has_cfd"] = False
        if short in dupe_canonical:
            n["cfd_status"] = "demoted_duplicate"
            n["cfd_canonical_part"] = dupe_canonical[short]
        else:
            n["cfd_status"] = "demoted_not_curated_rocket"
        n["cfd_mesh"] = n.get("cfd_mesh") or "demoted"
        pd = n.get("physics_data") if isinstance(n.get("physics_data"), dict) else {}
        pd["cfd"] = False
        pd["cfd_bodyfit"] = False
        pd["fea"] = bool(n.get("has_fea"))
        pd["verified"] = bool(n.get("has_fea"))
        n["physics_data"] = pd
        if not n.get("has_fea"):
            n["physics_verified"] = False
        demoted += 1
    graph_path.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    return {"kept_bodyfit": kept, "demoted": demoted}
