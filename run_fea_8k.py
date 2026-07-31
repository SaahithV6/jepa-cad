#!/usr/bin/env python3.12
"""Deprecated stub — use run_rocket_physics_8k.py --fea-only instead."""
from __future__ import annotations

import sys

from run_rocket_physics_8k import main

if __name__ == "__main__":
    # Preserve legacy entrypoint; forward to the real suite in FEA-only mode.
    if "--fea-only" not in sys.argv:
        sys.argv.append("--fea-only")
    raise SystemExit(main())
