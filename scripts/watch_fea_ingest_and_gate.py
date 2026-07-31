#!/usr/bin/env python3.12
"""Periodically ingest FRDs into the TAO graph until LatticeZero training gate.

Purpose: keep physics_verified Part annotations fresh while CalculiX resume runs,
so LatticeZero / JEPA can launch as soon as parts_linked >= GATE.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "physics_verified_summary.json"
FEA_DIR = ROOT / "artifacts" / "fea_final"
LOG = ROOT / "artifacts" / "ingest_watch.log"
GATE = 1500
INTERVAL_S = 120


def _good_frd_count() -> int:
    n = 0
    for d in FEA_DIR.iterdir():
        if not d.is_dir():
            continue
        frd = d / "case.frd"
        if frd.exists() and frd.stat().st_size > 100_000:
            n += 1
    return n


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _ingest() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "proper_msh_to_inp_fea.py"), "--ingest-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {"error": proc.stderr[-500:], "returncode": proc.returncode}


def main() -> int:
    last_frd = -1
    _log(f"watch start gate={GATE} interval={INTERVAL_S}s")
    while True:
        frd = _good_frd_count()
        stats = _ingest()
        linked = int(stats.get("parts_linked") or 0)
        _log(
            f"frd={frd} (Δ{frd - last_frd if last_frd >= 0 else 0}) "
            f"linked={linked}/{stats.get('parts_total')} "
            f"index_frd={stats.get('fea_cases_with_frd')} "
            f"gate_ok={linked >= GATE}"
        )
        last_frd = frd
        if linked >= GATE:
            ready = ROOT / "artifacts" / "TRAINING_GATE_READY"
            ready.write_text(
                json.dumps({"parts_linked": linked, "fea_frd": frd, "gate": GATE}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            _log(
                f"GATE MET — wrote {ready}; DO NOT launch Modal/24B until user confirms"
            )
            return 0
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
