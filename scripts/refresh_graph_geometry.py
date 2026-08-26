"""Recompute the graph's cached geometry for parts whose mesh has changed.

The generated corpus was written as empty STLs, and the geometric metrics were
computed from those empty files and cached in the graph:

    {"volume": 1e-09, "log_volume": -9.0, "aspect_ratio_xy": 0.0,
     "bbox_x": 0.0, "bbox_y": 0.0, "bbox_z": 0.0, "face_count": 0}

Those are not labels, they are inputs -- `_walk_associations` feeds them into
`payload["geometry"]`, which becomes part of the sample's `graph_metadata`. So
every one of these records has been telling the model that a nozzle is a
zero-volume point with no faces, and regenerating the meshes does not change
that on its own: the numbers live in the graph, not in the file.

This recomputes them from whatever the mesh on disk actually is now, and leaves
every other node untouched. A part whose file is still missing or unreadable
keeps its existing entry rather than being silently zeroed -- that is how the
bad values got here in the first place.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metrics_from_mesh(path: Path) -> dict | None:
    """Volume, extents and shape ratios in metres, or None if unreadable."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception:  # noqa: BLE001
        return None
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=float)
    if v.size == 0:
        return None

    # the corpus stores these in metres; the generated meshes are in millimetres
    ext_mm = v.max(axis=0) - v.min(axis=0)
    ext_m = ext_mm / 1000.0
    volume_m3 = abs(float(mesh.volume)) / 1e9
    if volume_m3 <= 0.0:
        return None

    bx, by, bz = (float(e) for e in ext_m)
    bbox_vol = max(bx * by * bz, 1e-15)
    return {
        "volume": volume_m3,
        "log_volume": math.log10(volume_m3),
        "aspect_ratio_xy": float(bx / by) if by > 0 else 0.0,
        "aspect_ratio_xz": float(bx / bz) if bz > 0 else 0.0,
        "compactness": float(min(1.0, volume_m3 / bbox_vol)),
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_z": bz,
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--match", type=str, default="generated_spaceflight_cad",
                    help="only refresh parts whose path contains this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())
    nodes = graph["nodes"]
    by_id = {n["id"]: n for n in nodes}

    # map part id -> file path, from the file/Part nodes that carry one
    part_path: dict[str, Path] = {}
    for n in nodes:
        props = n.get("properties") or {}
        p = props.get("path") or props.get("source_path")
        if not isinstance(p, str) or args.match not in p:
            continue
        pid = props.get("part_id") or n.get("id")
        if isinstance(pid, str):
            part_path.setdefault(pid, ROOT / p)

    updated, missing, unchanged = 0, 0, 0
    for n in nodes:
        if n.get("type") != "GeometricMetric":
            continue
        nid = n.get("id", "")
        if not nid.startswith("geometric_metric:assoc:"):
            continue
        pid = nid.split("geometric_metric:assoc:", 1)[1]
        path = part_path.get(pid)
        if path is None:
            continue
        if not path.exists():
            missing += 1
            continue
        fresh = metrics_from_mesh(path)
        if fresh is None:
            missing += 1
            continue
        props = n.setdefault("properties", {})
        before = props.get("volume")
        if before is not None and abs(float(before) - fresh["volume"]) < 1e-15:
            unchanged += 1
            continue
        props.update(fresh)
        updated += 1

    print(f"{len(part_path)} parts matched {args.match!r}")
    print(f"  {updated} geometry nodes refreshed")
    print(f"  {unchanged} already current")
    print(f"  {missing} left alone (file missing or unreadable)")

    if args.dry_run:
        print("\ndry run: graph not written")
        return 0
    if updated:
        backup = args.graph.with_suffix(".json.stale_geometry_backup")
        if not backup.exists():
            backup.write_text(args.graph.read_text())
            print(f"  original preserved at {backup.name}")
        args.graph.write_text(json.dumps(graph))
        print(f"  wrote {args.graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
