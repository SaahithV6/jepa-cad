"""Download the real content behind Git LFS pointer files already in the corpus.

154 files in `data/` are not files. They are 129-byte LFS pointers -- a repo was
cloned without git-lfs, so every large asset came down as a stub naming a hash
it does not contain. Among them are 42 STLs, 9 OpenRocket designs and 89
SolidWorks parts, and the loader has been faithfully reporting them as corrupt
geometry ever since ("no vertices in mesh: .../janus.stl", which is 30 MB of
real mesh on the server).

git-lfs is not installed here, but it does not need to be: GitHub serves LFS
content over plain HTTPS at media.githubusercontent.com, so the pointer's own
path is enough to fetch the object.

Each download is verified against the size the pointer declares before it
replaces the stub, and anything that comes back as HTML -- a rate limit, a
404, a login wall -- is refused rather than written. A pointer replaced by an
error page is worse than a pointer, because it stops looking like a stub.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def read_pointer(path: Path) -> dict | None:
    """Parse an LFS pointer, or None if this is a real file."""
    try:
        head = path.read_bytes()[:400]
    except OSError:
        return None
    if not head.startswith(POINTER_MAGIC):
        return None
    info: dict = {}
    for line in head.decode("utf-8", errors="replace").splitlines():
        if line.startswith("oid sha256:"):
            info["oid"] = line.split("sha256:", 1)[1].strip()
        elif line.startswith("size "):
            try:
                info["size"] = int(line.split(None, 1)[1])
            except (ValueError, IndexError):
                pass
    return info if "oid" in info and "size" in info else None


def repo_info(path: Path, verbose: bool = False) -> tuple[str, str, str, str] | None:
    """(host, owner/repo, branch, path-within-repo) for a file in a git clone.

    `path` is resolved first: git reports an absolute toplevel, so a relative
    search path made `relative_to` raise, the bare except returned None, and
    every file was reported as "not a github clone" -- including the GitLab
    ones this function had just been taught to handle. The failure looked like
    a routing decision and was a path-shape bug.
    """
    path = path.resolve()
    try:
        top = subprocess.run(["git", "-C", str(path.parent), "rev-parse",
                              "--show-toplevel"], capture_output=True,
                             text=True, timeout=30)
        if top.returncode != 0:
            return None
        root = Path(top.stdout.strip())
        remote = subprocess.run(["git", "-C", str(root), "remote", "get-url",
                                 "origin"], capture_output=True, text=True,
                                timeout=30)
        branch = subprocess.run(["git", "-C", str(root), "rev-parse",
                                 "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True, timeout=30)
        if remote.returncode != 0:
            return None
        url = remote.stdout.strip()
        host = next((h for h in ("github.com", "gitlab.com") if h in url), None)
        if host is None:
            return None
        slug = url.split(host, 1)[1].lstrip(":/").removesuffix(".git")
        rel = path.relative_to(root).as_posix()
        return host, slug, (branch.stdout.strip() or "main"), rel
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"   repo_info failed for {path.name}: {exc}")
        return None


def lfs_url(meta, info) -> str | None:
    """Where to fetch this object from, per host.

    GitHub serves LFS content as plain files under media.githubusercontent.com,
    so the path is enough. GitLab does not -- it implements the standard LFS
    batch API, which has to be asked for a signed href per object. Public repos
    need no credentials either way; the difference is purely protocol, and
    assuming the GitHub shape for both silently skipped every GitLab object.
    """
    host, slug, branch, rel = meta
    if host == "github.com":
        return f"https://media.githubusercontent.com/media/{slug}/{branch}/{rel}"
    if host == "gitlab.com":
        import json as _json

        body = _json.dumps({
            "operation": "download", "transfers": ["basic"],
            "objects": [{"oid": info["oid"], "size": info["size"]}]})
        try:
            proc = subprocess.run(
                ["curl", "-sS", "--max-time", "60", "-X", "POST",
                 "-H", "Content-Type: application/vnd.git-lfs+json",
                 "-H", "Accept: application/vnd.git-lfs+json",
                 "-d", body,
                 f"https://gitlab.com/{slug}.git/info/lfs/objects/batch"],
                capture_output=True, text=True, timeout=90)
            payload = _json.loads(proc.stdout)
            return payload["objects"][0]["actions"]["download"]["href"]
        except Exception:  # noqa: BLE001
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stubs = []
    for path in args.search.rglob("*"):
        # Broken symlinks and unreadable entries raise from stat(), which
        # aborted the whole scan with no output and no indication of which file
        # did it. A corpus this size always has a few.
        try:
            if not path.is_file() or path.stat().st_size > 2048:
                continue
        except OSError:
            continue
        info = read_pointer(path)
        if info:
            stubs.append((path, info))
    if args.limit:
        stubs = stubs[:args.limit]

    print(f"{len(stubs)} LFS pointer stubs")
    total_mb = sum(i["size"] for _p, i in stubs) / 1e6
    print(f"declared content behind them: {total_mb:.1f} MB")
    if args.dry_run:
        return 0

    fetched = failed = 0
    recovered_bytes = 0
    for path, info in stubs:
        meta = repo_info(path, verbose=True)
        if meta is None:
            print(f"   skip (no git remote resolved): {path.name}")
            failed += 1
            continue
        url = lfs_url(meta, info)
        if url is None:
            print(f"   skip (no LFS route): {path.name}")
            failed += 1
            continue
        tmp = path.with_suffix(path.suffix + ".lfstmp")
        try:
            proc = subprocess.run(
                ["curl", "-sSL", "--max-time", str(args.timeout), "-o", str(tmp), url],
                capture_output=True, timeout=args.timeout + 30)
            if proc.returncode != 0 or not tmp.exists():
                raise RuntimeError("curl failed")
            got = tmp.stat().st_size
            head = tmp.read_bytes()[:200].lstrip().lower()
            # An error page is still a 200 with a body. Writing one over a
            # pointer destroys the only record of what the file should be.
            if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
                raise RuntimeError("server returned HTML, not the object")
            if abs(got - info["size"]) > max(64, 0.01 * info["size"]):
                raise RuntimeError(f"size {got} != declared {info['size']}")
            tmp.replace(path)
            fetched += 1
            recovered_bytes += got
            print(f"   {path.name}: {got / 1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            failed += 1
            print(f"   FAILED {path.name}: {str(exc)[:70]}")

    print(f"\n{fetched} recovered ({recovered_bytes / 1e6:.1f} MB), {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
