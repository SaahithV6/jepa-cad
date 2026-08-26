"""What data is registered, what is on disk, and what is on disk but unusable.

Three different numbers get conflated when people say a corpus is "big": how
many sources are known, how many were actually fetched, and how many fetched
files the loader can read. This prints all three, because the gaps between them
are where the cheapest data lives -- a SolidWorks part already downloaded and
skipped costs nothing to convert and everything to re-acquire.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READABLE = {".stl", ".step", ".stp", ".obj", ".npz", ".pt", ".igs", ".iges", ".ply"}


def main() -> int:
    graph_path = ROOT / "artifacts/jepa-train-bundle/graph.json"
    registered: dict[str, dict] = {}
    if graph_path.exists():
        graph = json.loads(graph_path.read_text())
        for node in graph.get("nodes", []):
            if node.get("type") != "Source":
                continue
            props = node.get("properties") or {}
            key = props.get("key") or node.get("id")
            registered[str(key)] = {
                "domain": props.get("domain"),
                "blocked": bool(props.get("blocked")),
                "url": props.get("url") or props.get("repo_url") or "",
                "license": props.get("license"),
            }

    print(f"registered sources: {len(registered)}")
    by_domain = collections.Counter(v["domain"] for v in registered.values())
    print(f"  by domain: {dict(by_domain)}")
    print(f"  marked blocked: {sum(1 for v in registered.values() if v['blocked'])}")

    fetched = set()
    ext_root = ROOT / "data/raw_downloads/external"
    if ext_root.exists():
        fetched = {p.name for p in ext_root.iterdir() if p.is_dir()}
    for extra in ("nasa3d", "openrocket_hardware_8k", "generated_spaceflight_cad"):
        if (ROOT / "data" / extra).exists() or (ROOT / "data/raw_downloads" / extra).exists():
            fetched.add(extra)
    print(f"\nfetched directories: {len(fetched)}")

    unfetched = [k for k, v in registered.items()
                 if not v["blocked"] and k not in fetched]
    print(f"registered but not fetched: {len(unfetched)}")
    for key in sorted(unfetched)[:25]:
        info = registered[key]
        print(f"   {key:38} {str(info['url'])[:60]}")
    if len(unfetched) > 25:
        print(f"   ... and {len(unfetched) - 25} more")

    print("\non disk by extension:")
    counts: collections.Counter = collections.Counter()
    sizes: collections.Counter = collections.Counter()
    for path in (ROOT / "data").rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        counts[ext] += 1
        try:
            sizes[ext] += path.stat().st_size
        except OSError:
            pass
    usable = unusable = 0
    for ext, n in counts.most_common(18):
        mark = "readable" if ext in READABLE else "SKIPPED"
        if ext in READABLE:
            usable += n
        elif ext in (".sldprt", ".sldasm", ".ipt", ".iam", ".catpart", ".prt", ".f3d"):
            unusable += n
        print(f"   {ext or '(none)':10} {n:7}  {sizes[ext] / 1e6:9.1f} MB  {mark}")
    print(f"\nreadable geometry files: {usable}")
    print(f"proprietary CAD on disk the loader cannot open: {unusable}")
    if unusable:
        print("   -- already downloaded, already licensed, currently worth nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
