#!/usr/bin/env bash
# Keep rocket FEA + bodyfit alive; ingest TAO; always re-apply mass sidecar after ingest.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

FEA_PID_F="$ROOT/artifacts/rocket_fea.pid"
CFD_PID_F="$ROOT/artifacts/rocket_cfd_bodyfit.pid"
LEG_PID_F="$ROOT/artifacts/legacy_cfd_bodyfit.pid"
LOG="$ROOT/artifacts/logs/jepa_train_supervisor.log"
mkdir -p "$ROOT/artifacts/logs"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

alive() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pid; pid="$(tr -d ' \n' <"$f" || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_fea() {
  nohup "$PY" -u "$ROOT/run_rocket_physics_8k.py" \
    --fea-only --workers 4 --skip-register --no-ingest \
    --curated-rocket --target-tets 15000 --mesh-timeout 120 --timeout 240 \
    >>"$ROOT/artifacts/logs/fea_target15k.log" 2>&1 &
  echo $! >"$FEA_PID_F"
  log "started FEA pid=$(cat "$FEA_PID_F")"
}

start_cfd() {
  nohup "$PY" -u "$ROOT/run_rocket_cfd_bodyfit.py" \
    --curated --skip-done --pilot 0 --workers 6 --no-ingest \
    >>"$ROOT/artifacts/logs/bodyfit_train.log" 2>&1 &
  echo $! >"$CFD_PID_F"
  log "started bodyfit pid=$(cat "$CFD_PID_F")"
}

start_legacy_bodyfit() {
  if [[ -f "$ROOT/run_legacy_cfd_bodyfit.py" ]]; then
    nohup "$PY" -u "$ROOT/run_legacy_cfd_bodyfit.py" --workers 3 --no-ingest \
      >>"$ROOT/artifacts/logs/legacy_bodyfit.log" 2>&1 &
    echo $! >"$LEG_PID_F"
    log "started legacy_bodyfit pid=$(cat "$LEG_PID_F")"
  fi
}

ingest_and_mass() {
  log "ingest FEA+bodyfit + re-apply mass sidecar"
  "$PY" -u "$ROOT/run_rocket_physics_8k.py" --fea-only --ingest-only --skip-register --curated-rocket \
    >>"$ROOT/artifacts/logs/tao_ingest_loop.log" 2>&1 || true
  "$PY" -u -c "
import json
from pathlib import Path
from cadflow.rocket_cfd_bodyfit import ingest_bodyfit_to_graph
r=Path('artifacts/rocket_cfd_bodyfit'); res=[]
for d in r.iterdir():
  if d.is_dir() and (d/'meta.json').is_file():
    try:
      m=json.loads((d/'meta.json').read_text())
      res.append({'part_id':d.name,'success':True,'metrics':m.get('metrics') or m})
    except Exception:
      pass
print('cfd_linked', ingest_bodyfit_to_graph(Path('artifacts/jepa-train-bundle/graph.json'), res), flush=True)
" >>"$ROOT/artifacts/logs/tao_ingest_loop.log" 2>&1 || true
  "$PY" -u -m cadflow.enrich_tao_mass_properties --apply-sidecar \
    >>"$ROOT/artifacts/logs/tao_ingest_loop.log" 2>&1 || true
}

alive "$FEA_PID_F" || start_fea
alive "$CFD_PID_F" || start_cfd
alive "$LEG_PID_F" || start_legacy_bodyfit

while true; do
  avail="$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo)"
  log "heartbeat mem_avail_g=${avail} fea=$(alive "$FEA_PID_F" && echo 1 || echo 0) cfd=$(alive "$CFD_PID_F" && echo 1 || echo 0) leg=$(alive "$LEG_PID_F" && echo 1 || echo 0)"
  if ! alive "$FEA_PID_F"; then start_fea; fi
  if ! alive "$CFD_PID_F"; then start_cfd; fi
  if ! alive "$LEG_PID_F"; then start_legacy_bodyfit; fi
  # bail if RAM critically low
  avail_i="$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)"
  if [[ "$avail_i" -lt 1500 ]]; then
    log "LOW_MEM pausing bodyfit"
    kill "$(tr -d ' \n' <"$CFD_PID_F")" 2>/dev/null || true
    sleep 120
    continue
  fi
  ingest_and_mass
  sleep 600
done
