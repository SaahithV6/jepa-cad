#!/usr/bin/env bash
# Autonomous TAO growth supervisor.
#
# Keeps the physics solvers producing and the TAO graph consistent while the
# operator is away. Safe to run alongside the other agent's rocket bodyfit CFD
# (we NEVER touch artifacts/rocket_cfd_bodyfit.pid).
#
# Responsibilities:
#   * Restart rocket FEA mesh-retry if it dies (OMP pinned, meshable families).
#   * Restart legacy internal duct CFD if it dies; then sweep remaining recipes.
#   * Periodically re-apply the mass-properties sidecar so FEA/CFD graph writes
#     never leave mass_kg wiped.
#   * Log a heartbeat with coverage counts.
#
# Env budget: pauses launching new work when MemAvailable < 2 GB.

set -u
ROOT="/home/best/jepa-cad"
cd "$ROOT" || exit 1
PY=".venv/bin/python"
LOG_DIR="$ROOT/artifacts/logs"
mkdir -p "$LOG_DIR"
HB="$LOG_DIR/tao_autobuild.log"

log() { echo "$(date -Is) $*" >>"$HB"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

# Liveness is detected by matching the command line in /proc (shared across the
# sandbox PID namespace) rather than kill -0, which fails cross-namespace and
# caused duplicate launches. Exclude the cursor sandbox wrapper lines.
proc_running() {
  local pat="$1"
  pgrep -f "$pat" 2>/dev/null | while read -r p; do
    grep -qa cursorsandbox "/proc/$p/cmdline" 2>/dev/null || echo "$p"
  done | grep -q .
}
fea_running() { proc_running 'run_rocket_physics_8k.py .*--fea-only'; }
duct_running() { proc_running 'run_legacy_cfd_internal.py'; }
bodyfit_running() { proc_running 'run_rocket_cfd_bodyfit.py'; }

start_fea() {
  if fea_running; then return; fi
  [ "$(avail_mb)" -lt 2048 ] && { log "skip FEA start (low mem $(avail_mb)MB)"; return; }
  log "starting rocket FEA mesh-retry"
  nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" -u run_rocket_physics_8k.py \
    --fea-only --workers 2 --skip-register --no-ingest \
    --families tank,nose_cone,nozzle,fin,transition,body_tube,bulkhead,engine_mount \
    --target-tets 12000 --mesh-timeout 150 --timeout 300 \
    >>"$LOG_DIR/fea_mesh_retry.log" 2>&1 &
  echo $! >"$ROOT/artifacts/rocket_fea.pid"
}

# Legacy internal duct CFD, recipe sweep. Advances through recipes as each drains.
LEGACY_RECIPES=(component_duct tank_pressure_flow nozzle_compressible chamber_internal injector_orifice valve_feed)
start_legacy_cfd() {
  if duct_running; then return; fi
  [ "$(avail_mb)" -lt 2048 ] && { log "skip legacy CFD start (low mem $(avail_mb)MB)"; return; }
  for r in "${LEGACY_RECIPES[@]}"; do
    log "starting legacy internal CFD recipe=$r"
    nohup "$PY" -u run_legacy_cfd_internal.py \
      --recipe "$r" --workers 1 \
      --timeout-mesh 600 --timeout-solve 480 \
      >>"$LOG_DIR/legacy_cfd_duct_retry.log" 2>&1 &
    echo $! >"$ROOT/artifacts/legacy_cfd_internal_retry.pid"
    return
  done
}

reapply_mass() {
  "$PY" - <<'PY' >>"$HB" 2>&1
try:
    from cadflow.enrich_tao_mass_properties import apply_sidecar_to_graph
    n = apply_sidecar_to_graph()
    print("mass_sidecar_reapplied", n)
except Exception as exc:  # noqa: BLE001
    print("mass_sidecar_error", exc)
PY
}

coverage() {
  "$PY" - <<'PY' >>"$HB" 2>&1
import json
from pathlib import Path
try:
    g = json.loads(Path("artifacts/jepa-train-bundle/graph.json").read_text())
    parts = [n for n in g["nodes"] if n.get("type") == "Part"]
    def cnt(pred):
        return sum(1 for p in parts if pred(p))
    print("coverage parts=%d fea=%d cfd=%d mass=%d" % (
        len(parts),
        cnt(lambda p: p.get("has_fea") or (p.get("properties") or {}).get("has_fea")),
        cnt(lambda p: p.get("has_cfd") or (p.get("properties") or {}).get("has_cfd")),
        cnt(lambda p: p.get("mass_kg") is not None),
    ))
except Exception as exc:  # noqa: BLE001
    print("coverage_error", exc)
PY
}

log "=== supervisor start (mem $(avail_mb)MB) ==="
i=0
while true; do
  i=$((i + 1))
  start_fea
  start_legacy_cfd
  if [ $((i % 3)) -eq 0 ]; then reapply_mass; fi
  if [ $((i % 3)) -eq 0 ]; then coverage; fi
  # heartbeat every loop
  log "loop=$i mem=$(avail_mb)MB fea=$(fea_running && echo up || echo down) duct=$(duct_running && echo up || echo down) bodyfit=$(bodyfit_running && echo up || echo down)"
  sleep 120
done
