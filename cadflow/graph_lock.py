"""Cross-process serialization for TAO graph read-modify-write cycles.

Several independent pipelines mutate ``artifacts/jepa-train-bundle/graph.json``:
``scripts/tao_ingest_loop.sh`` (FEA/CFD ingest, mass sidecar, shard registration)
and ``scripts/jepa_train_supervisor.sh`` (its own ``ingest_and_mass``). The graph is
~250MB, so a non-atomic rewrite leaves a torn file visible for seconds at a time —
long enough for a concurrent reader to hit ``JSONDecodeError`` mid-document, and for
two writers to clobber each other's nodes.

Guarding at the Python layer rather than in the shell scripts is deliberate: the bash
loops keep an already-parsed loop body in memory, so editing them requires a restart,
while these modules are re-imported on every subprocess launch.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def lock_path_for(graph_path: Path | str) -> Path:
    p = Path(graph_path)
    return p.with_name(p.name + ".lock")


# flock is tied to the open file description, so a second open() in the same process
# would block against the first. Ingest entry points nest (FEA ingest re-applies the
# mass sidecar), so track depth and make re-entry a no-op.
_held: dict[str, int] = {}


@contextmanager
def graph_lock(graph_path: Path | str, *, timeout_s: float = 900.0) -> Iterator[None]:
    """Hold an exclusive flock for the duration of a graph read-modify-write.

    Falls through without locking if the lock file cannot be created (read-only
    mount, unusual filesystem) — a missing lock must never block ingestion.
    """
    lock_file = lock_path_for(graph_path)
    key = str(lock_file.resolve() if lock_file.exists() else lock_file)
    if _held.get(key):
        _held[key] += 1
        try:
            yield
        finally:
            _held[key] -= 1
        return

    fh = None
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_file, "a+")
    except OSError:
        yield
        return

    acquired = False
    try:
        # flock has no native timeout; poll so a wedged holder can't hang a solver run.
        import time

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.5)
        _held[key] = _held.get(key, 0) + 1
        yield
    finally:
        _held[key] = max(0, _held.get(key, 1) - 1)
        if fh is not None:
            if acquired:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()


def read_graph(graph_path: Path | str, *, retries: int = 4, delay_s: float = 5.0) -> dict[str, Any]:
    """Read the graph, retrying past a torn read from an unlocked concurrent writer."""
    import time

    path = Path(graph_path)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(delay_s)
    raise last if last is not None else RuntimeError("graph read failed")


def write_graph_atomic(graph_path: Path | str, graph: dict[str, Any]) -> None:
    """Rewrite the graph via tmp + ``os.replace`` so readers never see a partial file."""
    path = Path(graph_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
