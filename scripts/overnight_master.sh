#!/usr/bin/env bash
# Overnight master: finish FEA/CFD → TAO finalize → Modal 24B full train.
#
# Safe to re-run. Coordinates with coast_supervisor (keeps solvers alive) but
# owns the finish gate, fairing FEA push, TAO reconnect, and Modal launch.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBER_OF_CPUS=1
source "$ROOT/scripts/pylibs_env.sh" 2>/dev/null || true

LOG="$ROOT/artifacts/logs/overnight_master.log"
STATE="$ROOT/artifacts/overnight_master.state"
mkdir -p "$ROOT/artifacts/logs"
echo $$ >"$ROOT/artifacts/overnight_master.pid"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
phase() { echo "phase=$1" >"$STATE"; log "PHASE $1 — $2"; }

alive_pidfile() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pid; pid="$(tr -d ' \n' <"$f" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

ensure_coast() {
  if alive_pidfile "$ROOT/artifacts/coast_supervisor.pid"; then
    return 0
  fi
  log "restarting coast_supervisor"
  nohup bash "$ROOT/scripts/coast_supervisor.sh" >>"$ROOT/artifacts/logs/coast_supervisor.log" 2>&1 &
  echo $! >"$ROOT/artifacts/coast_supervisor.pid"
  sleep 3
}

# Mark chronically failing FEA parts so we don't thrash forever.
update_fea_skips() {
  "$PY" - <<'PY' >>"$LOG" 2>&1 || true
import json, time
from pathlib import Path
from cadflow.rocket_cfd_curate import ROCKET_CFD_FAMILIES, is_degenerate_box
from cadflow.rocket_physics_suite import load_manifest

fea = Path("artifacts/rocket_fea_8k")
man = load_manifest(Path("data/openrocket_hardware_8k"))
skip_path = Path("artifacts/fea_skip_parts.json")
skip = set()
if skip_path.is_file():
    try:
        skip = set(json.loads(skip_path.read_text()).get("part_ids") or [])
    except Exception:
        skip = set()
added = []
now = time.time()
for e in man:
    if e.get("family") not in ROCKET_CFD_FAMILIES:
        continue
    if is_degenerate_box(e) or int(e.get("faces") or 0) > 12000:
        continue
    pid = e["part_id"]
    if pid in skip:
        continue
    frd = fea / pid / "case.frd"
    if frd.is_file() and frd.stat().st_size >= 50000:
        continue
    # Skip if case dir is old (>3h) and still has no valid FRD after retries
    d = fea / pid
    if not d.is_dir():
        continue
    age = now - d.stat().st_mtime
    meta = d / "meta.json"
    err = ""
    if meta.is_file():
        try:
            err = str(json.loads(meta.read_text()).get("error") or "")
        except Exception:
            err = ""
    # Fairings / tanks that keep failing partitioning after long churn
    if age > 3 * 3600 and (
        "partition" in err.lower()
        or "mesh_" in err.lower()
        or e.get("family") == "fairing"
    ):
        # only auto-skip fairings that have been touched and still fail
        if e.get("family") == "fairing" and age > 6 * 3600:
            skip.add(pid)
            added.append(pid)
skip_path.write_text(json.dumps({"part_ids": sorted(skip), "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "added_last": added}, indent=2) + "\n")
print("fea_skips", len(skip), "added", len(added))
PY
}

# Dedicated fairing FEA push (coarser, hull-friendly) when core is done.
start_fairing_fea() {
  local workers="${1:-6}"
  if alive_pidfile "$ROOT/artifacts/rocket_fea.pid"; then
    # If already running fairing-only, leave it
    local args
    args="$(ps -p "$(tr -d ' \n' <"$ROOT/artifacts/rocket_fea.pid")" -o args= 2>/dev/null || true)"
    if [[ "$args" == *fairing* ]]; then
      return 0
    fi
  fi
  log "starting fairing FEA workers=$workers"
  # Let coast own the generic FEA; we temporarily override with fairing focus
  if alive_pidfile "$ROOT/artifacts/rocket_fea.pid"; then
    kill "$(tr -d ' \n' <"$ROOT/artifacts/rocket_fea.pid")" 2>/dev/null || true
    sleep 2
  fi
  nohup "$PY" -u "$ROOT/run_rocket_physics_8k.py" \
    --fea-only --workers "$workers" --skip-register --no-ingest \
    --families fairing,nose_cone,tank \
    --target-tets 6000 --mesh-timeout 240 --timeout 360 \
    >>"$ROOT/artifacts/logs/fea_fairing_overnight.log" 2>&1 &
  echo $! >"$ROOT/artifacts/rocket_fea.pid"
}

launch_modal_full() {
  if [[ -f "$ROOT/artifacts/OVERNIGHT_MODAL_LAUNCHED" ]]; then
    log "Modal already launched — skip"
    return 0
  fi
  log "Launching Modal 24B FULL advanced training"
  nohup bash "$ROOT/scripts/launch_jepa24b_modal.sh" full-advanced \
    >>"$ROOT/artifacts/logs/modal_24b_full_overnight.log" 2>&1 &
  echo $! >"$ROOT/artifacts/modal_24b_full.pid"
  date -Is >"$ROOT/artifacts/OVERNIGHT_MODAL_LAUNCHED"
  log "Modal launch pid=$(cat "$ROOT/artifacts/modal_24b_full.pid")"
}

# ---------- boot ----------
phase solvers "ensure coast + drive queues to drain"
ensure_coast
log "overnight_master start pid=$$"

idle_core=0
idle_full=0
finalize_done=0

while true; do
  ensure_coast
  update_fea_skips
  if ! "$PY" -u "$ROOT/scripts/overnight_status.py" >"$ROOT/artifacts/overnight_status.json.tmp" 2>>"$LOG"; then
    log "status script failed — retry next tick"
    sleep 60
    continue
  fi
  mv "$ROOT/artifacts/overnight_status.json.tmp" "$ROOT/artifacts/overnight_status.json"

  core_miss="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json"))["rocket_fea_non_fairing_missing"])')"
  fair_miss="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json"))["rocket_fea_fairing_missing"])')"
  cfd_done="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json"))["gates"]["cfd_done"])')"
  train_gate="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json"))["gates"]["train_gate"])')"
  drained="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json"))["gates"]["solvers_drained"])')"
  avail="$(awk '/MemAvailable/{printf "%.1f",$2/1024/1024}' /proc/meminfo)"
  log "status core_miss=${core_miss} fair_miss=${fair_miss} cfd_done=${cfd_done} train_gate=${train_gate} drained=${drained} mem=${avail}G"

  # When non-fairing FEA is nearly done, focus workers on fairing/nose/tank leftovers
  if [[ "${core_miss}" -le 40 && "${fair_miss}" -gt 20 ]]; then
    start_fairing_fea 8
  fi

  if [[ "${core_miss}" -le 30 && "${cfd_done}" == "True" ]]; then
    idle_core=$((idle_core + 1))
  else
    idle_core=0
  fi

  # After core FEA+CFD stable, finalize TAO (can re-run; idempotent-ish)
  if [[ "${idle_core}" -ge 2 && "${finalize_done}" -eq 0 ]]; then
    phase tao "ingest + associate + enrich + validate"
    "$PY" -u "$ROOT/scripts/overnight_tao_finalize.py" >>"$LOG" 2>&1 \
      && finalize_done=1 \
      || log "tao_finalize exit=$?"
    # Extract remaining physics shards aggressively once solvers quiet
    nohup "$PY" -u run_physics_shards.py --source rocket --workers 2 --num-points 2048 \
      >>"$ROOT/artifacts/logs/physics_shards_overnight.log" 2>&1 &
    nohup "$PY" -u run_physics_shards.py --source legacy --workers 1 --num-points 2048 \
      >>"$ROOT/artifacts/logs/physics_shards_legacy_overnight.log" 2>&1 &
  fi

  if [[ "${drained}" == "True" ]]; then
    idle_full=$((idle_full + 1))
  else
    idle_full=0
  fi

  # Training gate: core done + CFD done + TAO finalized at least once
  if [[ "${train_gate}" == "True" && "${finalize_done}" -eq 1 ]]; then
    phase train "Modal 24B full-advanced"
    # One more finalize pass to pick up late FRDs
    "$PY" -u "$ROOT/scripts/overnight_tao_finalize.py" >>"$LOG" 2>&1 || true
    # Flip physics_shards_only when shards are dense enough
    shards="$("$PY" -c 'import json;print(json.load(open("artifacts/overnight_status.json")).get("graph",{}).get("TensorShard",0))')"
    if [[ "${shards}" -ge 5000 ]]; then
      log "enabling physics_shards_only (shards=${shards})"
      "$PY" - <<'PY'
from pathlib import Path
p = Path("configs/families/space_24b.yaml")
t = p.read_text()
t2 = t.replace("physics_shards_only: false", "physics_shards_only: true")
if t2 != t:
    p.write_text(t2)
    print("flipped physics_shards_only=true")
else:
    print("physics_shards_only already set or missing")
PY
    fi
    launch_modal_full
    phase done "solvers draining / Modal launched — master stays for fairing mop-up"
  fi

  # Exit only when fully drained AND Modal launched (or fairings skipped away)
  if [[ "${idle_full}" -ge 3 && -f "$ROOT/artifacts/OVERNIGHT_MODAL_LAUNCHED" ]]; then
    # Final TAO pass
    "$PY" -u "$ROOT/scripts/overnight_tao_finalize.py" >>"$LOG" 2>&1 || true
    phase complete "overnight finished"
    date -Is >"$ROOT/artifacts/OVERNIGHT_COMPLETE"
    log "OVERNIGHT COMPLETE"
    exit 0
  fi

  sleep 300
done
