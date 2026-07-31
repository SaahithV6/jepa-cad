#!/usr/bin/env python3.12
"""Deprecated stub — use run_rocket_physics_8k.py --cfd-only instead."""
from __future__ import annotations

import sys

from run_rocket_physics_8k import main

if __name__ == "__main__":
    if "--cfd-only" not in sys.argv:
        sys.argv.append("--cfd-only")
    raise SystemExit(main())
