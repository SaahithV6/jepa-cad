#!/usr/bin/env bash
# Coast supervisor: keep rocket FEA + bodyfit CFD + TAO ingest + shard extract
# alive forever, while holding ~4GB MemAvailable at all times.
#
# Progress > peak throughput. Shed CFD before FEA; restart as soon as headroom
# returns. Replaces the old mem_watch that only killed and never restarted.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBER_OF_CPUS=1

FEA_PID_F="$ROOT/artifacts/rocket_fea.pid"
CFD_PID_F="$ROOT/artifacts/rocket_cfd_bodyfit.pid"
LEGACY_PID_F="$ROOT/artifacts/legacy_cfd_internal.pid"
INGEST_PID_F="$ROOT/artifacts/tao_ingest_loop.pid"
SHARD_PID_F="$ROOT/artifacts/fin_shard_watch.pid"
STATE_F="$ROOT/artifacts/coast_supervisor.state"
LOG="$ROOT/artifacts/logs/coast_supervisor.log"
FEA_LOG="$ROOT/artifacts/logs/fea_coast.log"
CFD_LOG="$ROOT/artifacts/logs/bodyfit_coast.log"
LEGACY_LOG="$ROOT/artifacts/logs/legacy_cfd_parallel.log"
mkdir -p "$ROOT/artifacts/logs"

# Memory policy (MB) — burn ~22GB of the ~25GB box; leave ~2.5GB free.
TARGET_FREE_MB="${TARGET_FREE_MB:-2560}"   # ~22GB usable on a 24.8GB host
CFD_START_MB="${CFD_START_MB:-3200}"
FEA_START_MB="${FEA_START_MB:-2800}"
INGEST_START_MB="${INGEST_START_MB:-2600}"
SHED_CFD_MB="${SHED_CFD_MB:-2800}"         # shed CFD first under pressure
SHED_FEA_MB="${SHED_FEA_MB:-2200}"         # last resort stop FEA

log() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

avail_mb() { awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo; }

alive_pidfile() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pid
  pid="$(tr -d ' \n' <"$f" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

stop_pidfile() {
  local f="$1" label="${2:-proc}"
  alive_pidfile "$f" || return 0
  local pid
  pid="$(tr -d ' \n' <"$f")"
  log "STOP $label pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  sleep 2
  pkill -P "$pid" 2>/dev/null || true
  sleep 1
  kill -KILL "$pid" 2>/dev/null || true
  # stray children matching the job
  return 0
}

write_state() {
  printf 'fea_workers=%s\ncfd_workers=%s\nupdated=%s\n' \
    "${1:-0}" "${2:-0}" "$(date -Is)" >"$STATE_F"
}

# Choose worker counts from headroom above TARGET — push hard toward ~22GB used.
# Returns: FEA_W CFD_W LEGACY_W
pick_workers() {
  local mb="$1"
  local head=$((mb - TARGET_FREE_MB))
  if (( mb < SHED_FEA_MB )); then
    echo "0 0 0"
  elif (( mb < SHED_CFD_MB )); then
    echo "3 0 0"
  elif (( head < 2000 )); then
    echo "4 2 1"
  elif (( head < 5000 )); then
    echo "6 3 1"
  elif (( head < 9000 )); then
    echo "8 4 2"
  elif (( head < 14000 )); then
    echo "10 5 2"
  else
    echo "10 6 2"
  fi
}

cmdline_workers() {
  # Extract --workers N from a process cmdline; echo 0 if missing.
  local pid="$1"
  local args
  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$args" =~ --workers[[:space:]]+([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

fin_fea_remaining() {
  "$PY" - <<'PY' 2>/dev/null || echo 999
from pathlib import Path
import json
from cadflow.rocket_physics_suite import load_manifest, DEFAULT_FEA_ROOT
man = load_manifest(Path("data/openrocket_hardware_8k"))
fea = DEFAULT_FEA_ROOT
skip = set()
sp = Path("artifacts/fea_skip_parts.json")
if sp.is_file():
    try:
        skip = set(json.loads(sp.read_text()).get("part_ids") or [])
    except Exception:
        skip = set()
n = 0
for e in man:
    if e.get("family") != "fin":
        continue
    pid = e["part_id"]
    if pid in skip:
        continue
    frd = fea / pid / "case.frd"
    if not (frd.is_file() and frd.stat().st_size >= 50_000):
        n += 1
print(n)
PY
}

cfd_queue_remaining() {
  "$PY" - <<'PY' 2>/dev/null || echo 0
import json
from pathlib import Path
cur = Path("artifacts/rocket_cfd_curated.json")
root = Path("artifacts/rocket_cfd_bodyfit")
if not cur.is_file():
    print(0); raise SystemExit
entries = json.loads(cur.read_text()).get("entries") or []
miss = sum(1 for e in entries if not (root / e["part_id"] / "meta.json").is_file())
print(miss)
PY
}

fea_queue_remaining() {
  "$PY" - <<'PY' 2>/dev/null || echo 999
import json
from pathlib import Path
from cadflow.rocket_cfd_curate import ROCKET_CFD_FAMILIES, is_degenerate_box
from cadflow.rocket_physics_suite import load_manifest, DEFAULT_FEA_ROOT
man = load_manifest(Path("data/openrocket_hardware_8k"))
fea = DEFAULT_FEA_ROOT
skip = set()
sp = Path("artifacts/fea_skip_parts.json")
if sp.is_file():
    try:
        skip = set(json.loads(sp.read_text()).get("part_ids") or [])
    except Exception:
        pass
n = 0
for e in man:
    if e.get("family") not in ROCKET_CFD_FAMILIES:
        continue
    if is_degenerate_box(e) or int(e.get("faces") or 0) > 12000:
        continue
    pid = e["part_id"]
    if pid in skip:
        continue
    frd = fea / pid / "case.frd"
    if not (frd.is_file() and frd.stat().st_size >= 50_000):
        n += 1
print(n)
PY
}

start_fea() {
  local workers="$1"
  (( workers > 0 )) || return 0
  local mb
  mb="$(avail_mb)"
  if (( mb < FEA_START_MB )); then
    log "skip FEA start mem=${mb}MB < ${FEA_START_MB}"
    return 0
  fi
  local remain
  remain="$(fea_queue_remaining)"
  if [[ "${remain}" -eq 0 ]]; then
    log "skip FEA start — queue empty"
    return 0
  fi
  local fins_left
  fins_left="$(fin_fea_remaining)"
  local fams
  local tets=12000
  if [[ "${fins_left}" -gt 0 ]]; then
    fams="fin"
    log "FEA prioritize fins remaining=${fins_left}"
  else
    # Fins done — grind holes; fairings get coarser target via prepare_fea_case.
    fams="tank,nozzle,fairing,nose_cone,body_tube,transition,engine_mount,bulkhead"
    log "FEA fins cleared → tank/nozzle/fairing/… holes remain=${remain}"
  fi
  stop_pidfile "$FEA_PID_F" "old_fea"
  pkill -f 'run_rocket_physics_8k.py --fea-only --workers' 2>/dev/null || true
  sleep 1
  nohup "$PY" -u "$ROOT/run_rocket_physics_8k.py" \
    --fea-only --workers "$workers" --skip-register --no-ingest \
    --families "$fams" \
    --target-tets "$tets" --mesh-timeout 180 --timeout 300 \
    >>"$FEA_LOG" 2>&1 &
  echo $! >"$FEA_PID_F"
  log "START FEA pid=$(cat "$FEA_PID_F") workers=$workers families=$fams remain=${remain} mem=${mb}MB"
}

start_cfd() {
  local workers="$1"
  (( workers > 0 )) || return 0
  local mb
  mb="$(avail_mb)"
  if (( mb < CFD_START_MB )); then
    log "skip CFD start mem=${mb}MB < ${CFD_START_MB}"
    return 0
  fi
  local remain
  remain="$(cfd_queue_remaining)"
  if [[ "${remain}" -eq 0 ]]; then
    log "skip CFD start — curated queue empty"
    return 0
  fi
  stop_pidfile "$CFD_PID_F" "old_cfd"
  # don't pkill all simpleFoam globally — stop via parent tree
  nohup "$PY" -u "$ROOT/run_rocket_cfd_bodyfit.py" \
    --curated --skip-done --pilot 0 --workers "$workers" --no-ingest \
    >>"$CFD_LOG" 2>&1 &
  echo $! >"$CFD_PID_F"
  log "START CFD pid=$(cat "$CFD_PID_F") workers=$workers remain=${remain} mem=${mb}MB"
}

# Legacy internal CFD (duct / nozzle / chamber…) — runs in parallel with rocket.
# Sweeps recipes; skip-done is inherent (cached metas). Also re-ingests disk metas.
LEGACY_RECIPES=(component_duct nozzle_compressible chamber_internal tank_pressure_flow injector_orifice valve_feed)
start_legacy() {
  local workers="$1"
  (( workers > 0 )) || return 0
  local mb
  mb="$(avail_mb)"
  if (( mb < CFD_START_MB )); then
    log "skip legacy start mem=${mb}MB < ${CFD_START_MB}"
    return 0
  fi
  if alive_pidfile "$LEGACY_PID_F"; then
    return 0
  fi
  # One-shot ingest of any disk metas not yet on the graph, then keep solving.
  nohup bash -c "
    cd '$ROOT' || exit 1
    '$PY' -u run_legacy_cfd_internal.py --ingest-only >>'$LEGACY_LOG' 2>&1 || true
    for r in ${LEGACY_RECIPES[*]}; do
      echo \"\$(date -Is) legacy recipe=\$r workers=$workers\" >>'$LEGACY_LOG'
      '$PY' -u run_legacy_cfd_internal.py --recipe \"\$r\" --workers $workers \
        --timeout-mesh 600 --timeout-solve 480 \
        >>'$LEGACY_LOG' 2>&1 || true
    done
    '$PY' -u run_legacy_cfd_internal.py --ingest-only >>'$LEGACY_LOG' 2>&1 || true
    echo \"\$(date -Is) legacy sweep done\" >>'$LEGACY_LOG'
  " >/dev/null 2>&1 &
  echo $! >"$LEGACY_PID_F"
  log "START legacy pid=$(cat "$LEGACY_PID_F") workers=$workers mem=${mb}MB"
}

start_ingest() {
  local mb
  mb="$(avail_mb)"
  if (( mb < INGEST_START_MB )); then
    log "skip ingest start mem=${mb}MB < ${INGEST_START_MB}"
    return 0
  fi
  if alive_pidfile "$INGEST_PID_F"; then
    return 0
  fi
  nohup bash "$ROOT/scripts/tao_ingest_loop.sh" >>"$ROOT/artifacts/logs/tao_ingest_loop.log" 2>&1 &
  echo $! >"$INGEST_PID_F"
  log "START ingest pid=$(cat "$INGEST_PID_F") mem=${mb}MB"
}

start_shard_watch() {
  if alive_pidfile "$SHARD_PID_F"; then
    return 0
  fi
  nohup bash -c '
    while true; do
      sleep 120
      cd /home/best/jepa-cad || exit 1
      mb=$(awk "/MemAvailable/{printf \"%d\", \$2/1024}" /proc/meminfo)
      if [ "$mb" -lt 4200 ]; then continue; fi
      .venv/bin/python -u run_physics_shards.py --source rocket --workers 1 --num-points 2048 \
        >> artifacts/logs/physics_shards_rocket.log 2>&1
    done
  ' >/dev/null 2>&1 &
  echo $! >"$SHARD_PID_F"
  log "START shard_watch pid=$(cat "$SHARD_PID_F")"
}

# Retire old kill-only mem watches (they don't restart; conflict with coast policy).
retire_old_mem_watch() {
  if [[ -f "$ROOT/artifacts/mem_watch.pid" ]]; then
    stop_pidfile "$ROOT/artifacts/mem_watch.pid" "old_mem_watch"
  fi
  # best-effort: only the known one-liner loops
  pkill -f 'avail_mb=.*rocket_cfd_bodyfit.pid.*rocket_fea.pid' 2>/dev/null || true
}

# ---- boot ----
log "coast_supervisor start pid=$$ TARGET_FREE_MB=${TARGET_FREE_MB}"
retire_old_mem_watch
echo $$ >"$ROOT/artifacts/coast_supervisor.pid"

read -r WANT_FEA WANT_CFD WANT_LEG <<<"$(pick_workers "$(avail_mb)")"
alive_pidfile "$FEA_PID_F" || start_fea "$WANT_FEA"
alive_pidfile "$CFD_PID_F" || start_cfd "$WANT_CFD"
alive_pidfile "$LEGACY_PID_F" || start_legacy "$WANT_LEG"
start_ingest
start_shard_watch
write_state "$WANT_FEA" "$WANT_CFD"

# ---- loop ----
while true; do
  mb="$(avail_mb)"
  read -r want_fea want_cfd want_leg <<<"$(pick_workers "$mb")"
  fea_up=0; cfd_up=0; leg_up=0; ing_up=0; sh_up=0
  alive_pidfile "$FEA_PID_F" && fea_up=1
  alive_pidfile "$CFD_PID_F" && cfd_up=1
  alive_pidfile "$LEGACY_PID_F" && leg_up=1
  alive_pidfile "$INGEST_PID_F" && ing_up=1
  alive_pidfile "$SHARD_PID_F" && sh_up=1

  # Shed load to protect free-RAM floor (legacy+rocket CFD first, then FEA)
  if (( mb < SHED_FEA_MB )); then
    (( leg_up )) && stop_pidfile "$LEGACY_PID_F" "legacy_shed"
    (( cfd_up )) && stop_pidfile "$CFD_PID_F" "cfd_shed"
    (( fea_up )) && stop_pidfile "$FEA_PID_F" "fea_shed"
    leg_up=0; cfd_up=0; fea_up=0
    log "HARD shed mem=${mb}MB (floor ${TARGET_FREE_MB})"
  elif (( mb < SHED_CFD_MB )); then
    (( leg_up )) && stop_pidfile "$LEGACY_PID_F" "legacy_shed"
    (( cfd_up )) && stop_pidfile "$CFD_PID_F" "cfd_shed"
    leg_up=0; cfd_up=0
    log "SOFT shed CFD+legacy mem=${mb}MB"
  fi

  # Restart if down and headroom allows
  if (( fea_up == 0 && want_fea > 0 )); then
    start_fea "$want_fea"
  fi
  if (( cfd_up == 0 && want_cfd > 0 )); then
    start_cfd "$want_cfd"
  fi
  if (( leg_up == 0 && want_leg > 0 )); then
    start_legacy "$want_leg"
  fi
  if (( ing_up == 0 )); then
    start_ingest
  fi
  if (( sh_up == 0 )); then
    start_shard_watch
  fi

  # Scale FEA/CFD workers up or down to match policy (don't leave free RAM idle).
  if (( fea_up == 1 && want_fea > 0 )); then
    cur_w="$(cmdline_workers "$(tr -d ' \n' <"$FEA_PID_F")")"
    if (( cur_w > 0 && cur_w != want_fea )); then
      log "rescale FEA ${cur_w}→${want_fea} mem=${mb}MB"
      start_fea "$want_fea"
    fi
  fi
  if (( cfd_up == 1 && want_cfd > 0 )); then
    cur_w="$(cmdline_workers "$(tr -d ' \n' <"$CFD_PID_F")")"
    if (( cur_w > 0 && cur_w != want_cfd )); then
      log "rescale CFD ${cur_w}→${want_cfd} mem=${mb}MB"
      start_cfd "$want_cfd"
    fi
  fi

  write_state "$want_fea" "$want_cfd"
  log "tick mem=${mb}MB want_fea=${want_fea} want_cfd=${want_cfd} want_leg=${want_leg} fea=$(alive_pidfile "$FEA_PID_F" && echo up || echo down) cfd=$(alive_pidfile "$CFD_PID_F" && echo up || echo down) legacy=$(alive_pidfile "$LEGACY_PID_F" && echo up || echo down) ingest=$(alive_pidfile "$INGEST_PID_F" && echo up || echo down) shards=$(alive_pidfile "$SHARD_PID_F" && echo up || echo down)"

  sleep 45
done
