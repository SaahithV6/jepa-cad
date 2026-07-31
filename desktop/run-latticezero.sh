#!/usr/bin/env bash
# Launch LatticeZero AppImage with writable dirs + Linux sandbox fallback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPIMAGE="${LATTICEZERO_APPIMAGE:-$ROOT/release/LatticeZero-0.1.0-x86_64.AppImage}"
UID_NUM="$(id -u)"
export LATTICEZERO_DATA_DIR="${LATTICEZERO_DATA_DIR:-$HOME/.local/share/latticezero-user}"
mkdir -p "$LATTICEZERO_DATA_DIR"

if [[ ! -x "$APPIMAGE" ]]; then
  echo "Missing AppImage: $APPIMAGE" >&2
  echo "Build with: cd desktop && npm run build" >&2
  exit 1
fi

# Do not auto-elevate; if chrome-sandbox isn't setuid the app itself enables --no-sandbox.
exec "$APPIMAGE" --no-sandbox "$@"
