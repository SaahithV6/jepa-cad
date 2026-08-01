#!/usr/bin/env bash
# thin wrappers so hooks.json stays simple
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/goal_stop.py"
