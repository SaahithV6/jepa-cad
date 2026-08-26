"""Fetch the registered sources that were never actually downloaded.

The corpus graph knows about 153 sources; roughly 25 have a directory on disk.
The rest were catalogued and never pulled, so the project has been training on a
fraction of what it already decided it wanted.

Sources split into two kinds and only one of them is worth bandwidth here:
repositories that contain geometry, and papers that contain numbers a human has
to read. This clones the first kind and only records the second, because a PDF
in data/ is not training data -- it is a 400 MB reminder that someone should
read it, and this corpus already carries 380 of them.

Nothing is counted as fetched until it is on disk with geometry in it. A clone
that succeeds and contains no meshes is reported as empty rather than as a
source, which is the distinction the earlier corpus lost when 1,961 files that
existed but held nothing were counted as data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_EXT = {".stl", ".step", ".stp", ".obj", ".iges", ".igs", ".ply", ".ork",
                ".3mf", ".dxf", ".scad", ".f3d", ".sldprt", ".sldasm"}

#: Hosts we can clone from directly. Anything else is recorded for a human.
CLONEABLE = ("github.com", "gitlab.com", "codeberg.org", "git.sr.ht")


def classify(url: str) -> str:
    if not url:
        return "no-url"
    low = url.lower()
    if any(h in low for h in CLONEABLE):
        return "repo"
    if low.endswith(".pdf") or "/pdf/" in low or "doi.org" in low:
        return "paper"
    if "kaggle.com" in low or "zenodo.org" in low or "figshare" in low:
        return "dataset-portal"
    return "web"


#: An LFS pointer is ~130 bytes of text naming a hash. Nothing real is that
#: small, so size is a cheap pre-filter before reading.
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def geometry_count(path: Path) -> tuple[int, int]:
    """(real geometry files, LFS pointers) under this path.

    Counting by extension alone reported 16,847 geometry files for one clone
    whose every last .step was a 130-byte pointer -- a repo cloned without
    git-lfs looks identical to a repo full of parts until something opens one.
    The whole point of this script is to distinguish a source from a directory,
    so it has to read, not count.
    """
    real = stubs = 0
    for item in path.rglob("*"):
        try:
            if not item.is_file() or item.suffix.lower() not in GEOMETRY_EXT:
                continue
            if item.stat().st_size <= 2048 and \
                    item.open("rb").read(45).startswith(_LFS_MAGIC):
                stubs += 1
            else:
                real += 1
        except OSError:
            continue
    return real, stubs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/raw_downloads/external")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())
    sources = []
    for node in graph.get("nodes", []):
        if node.get("type") != "Source":
            continue
        props = node.get("properties") or {}
        if props.get("blocked"):
            continue
        key = str(props.get("key") or node.get("id"))
        url = str(props.get("url") or props.get("repo_url") or "")
        sources.append({"key": key, "url": url, "kind": classify(url),
                        "license": props.get("license"),
                        "domain": props.get("domain")})

    existing = {p.name for p in args.out.iterdir() if p.is_dir()} if args.out.exists() else set()
    todo = [s for s in sources if s["key"] not in existing]
    repos = [s for s in todo if s["kind"] == "repo"]
    others = [s for s in todo if s["kind"] != "repo"]

    print(f"{len(sources)} registered, {len(existing)} already on disk")
    print(f"{len(repos)} cloneable repositories not yet fetched")
    print(f"{len(others)} non-repository sources (papers/portals) -- recorded, not cloned")
    from collections import Counter
    print(f"   breakdown: {dict(Counter(s['kind'] for s in others))}")

    if args.list_only:
        for s in repos[:40]:
            print(f"   {s['key']:36} {s['url'][:64]}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    if args.limit:
        repos = repos[:args.limit]

    cloned = empty = failed = 0
    report = []
    for src in repos:
        dest = args.out / src["key"]
        if dest.exists():
            continue
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", src["url"], str(dest)],
                capture_output=True, text=True, timeout=args.timeout,
                env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"})
            if proc.returncode != 0:
                failed += 1
                report.append({**src, "result": "clone failed",
                               "detail": proc.stderr.strip()[-120:]})
                print(f"   FAILED {src['key']}: {proc.stderr.strip()[-70:]}")
                continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            report.append({**src, "result": "clone error", "detail": str(exc)[:120]})
            print(f"   ERROR  {src['key']}: {str(exc)[:70]}")
            continue

        n, stubs = geometry_count(dest)
        note = f" (+{stubs} LFS pointers needing fetch)" if stubs else ""
        if n == 0:
            empty += 1
            report.append({**src, "result": "no geometry", "geometry_files": 0,
                           "lfs_pointers": stubs})
            print(f"   empty  {src['key']}: cloned, no real geometry{note}")
        else:
            cloned += 1
            report.append({**src, "result": "ok", "geometry_files": n,
                           "lfs_pointers": stubs})
            print(f"   ok     {src['key']}: {n} real geometry files{note}")

    summary = args.out.parent / "registry_fetch_report.json"
    summary.write_text(json.dumps(report, indent=2))
    print(f"\n{cloned} sources with geometry, {empty} cloned but empty, {failed} failed")
    print(f"report: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
