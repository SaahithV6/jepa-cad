#!/usr/bin/env bash
# Keep rocket FEA + bodyfit alive until queues drain, then ingest TAO.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

FEA_PID_F="$ROOT/artifacts/rocket_fea.pid"
CFD_PID_F="$ROOT/artifacts/rocket_cfd_bodyfit.pid"
LOG="$ROOT/artifacts/logs/finish_supervisor.log"
mkdir -p "$ROOT/artifacts/logs"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

alive() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pid
  pid="$(tr -d ' \n' <"$f" || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_fea() {
  local out="$ROOT/artifacts/logs/fea_target15k.log"
  nohup "$PY" -u "$ROOT/run_rocket_physics_8k.py" \
    --fea-only --workers 6 --skip-register --no-ingest \
    --curated-rocket --target-tets 15000 --mesh-timeout 180 --timeout 300 \
    >>"$out" 2>&1 &
  echo $! >"$FEA_PID_F"
  log "started FEA pid=$(cat "$FEA_PID_F") workers=6"
}

start_cfd() {
  local out="$ROOT/artifacts/logs/bodyfit_continue.log"
  nohup "$PY" -u "$ROOT/run_rocket_cfd_bodyfit.py" \
    --curated --skip-done --pilot 0 --workers 5 --no-ingest \
    >>"$out" 2>&1 &
  echo $! >"$CFD_PID_F"
  log "started bodyfit pid=$(cat "$CFD_PID_F") workers=5"
}

queue_counts() {
  "$PY" - <<'PY'
import json
from pathlib import Path
from cadflow.rocket_cfd_curate import ROCKET_CFD_FAMILIES, is_degenerate_box
from cadflow.rocket_physics_suite import load_manifest

man = load_manifest(Path("data/openrocket_hardware_8k"))
fea_root = Path("artifacts/rocket_fea_8k")
cfd_root = Path("artifacts/rocket_cfd_bodyfit")

def frd_ok(pid: str) -> bool:
    frd = fea_root / pid / "case.frd"
    try:
        return frd.is_file() and frd.stat().st_size >= 50_000
    except OSError:
        return False

fea_eligible = [
    e for e in man
    if e.get("family") in ROCKET_CFD_FAMILIES
    and not is_degenerate_box(e)
    and int(e.get("faces") or 0) <= 12_000
]
fea_missing = sum(1 for e in fea_eligible if not frd_ok(e["part_id"]))

curated = json.loads(Path("artifacts/rocket_cfd_curated.json").read_text())
entries = curated.get("entries") or []
cfd_missing = sum(
    1
    for e in entries
    if not (cfd_root / e["part_id"] / "meta.json").is_file()
)
print(f"{fea_missing} {cfd_missing} {len(fea_eligible)} {len(entries)}")
PY
}

ingest_all() {
  log "ingesting FEA+bodyfit into TAO"
  "$PY" -u "$ROOT/run_rocket_physics_8k.py" \
    --fea-only --ingest-only --skip-register --curated-rocket \
    >>"$ROOT/artifacts/logs/fea_ingest_final.log" 2>&1 || log "FEA ingest exit=$?"
  "$PY" -u - <<'PY' >>"$ROOT/artifacts/logs/bodyfit_ingest_final.log" 2>&1
import json
from pathlib import Path
from cadflow.rocket_cfd_bodyfit import ingest_bodyfit_to_graph

root = Path("artifacts/rocket_cfd_bodyfit")
results = []
for d in root.iterdir():
    if not d.is_dir():
        continue
    meta = d / "meta.json"
    if not meta.is_file():
        continue
    try:
        m = json.loads(meta.read_text())
    except Exception:
        continue
    metrics = m.get("metrics") or m
    results.append({"part_id": d.name, "success": True, "metrics": metrics})
n = ingest_bodyfit_to_graph(Path("artifacts/jepa-train-bundle/graph.json"), results)
print(f"bodyfit_ingest linked={n} from={len(results)}")
PY
  log "ingest done"
}

# boot
alive "$FEA_PID_F" || start_fea
alive "$CFD_PID_F" || start_cfd

idle_rounds=0
while true; do
  read -r fea_miss cfd_miss fea_tot cfd_tot <<<"$(queue_counts)"
  avail="$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo)"
  log "queue fea_missing=${fea_miss}/${fea_tot} cfd_missing=${cfd_miss}/${cfd_tot} mem_avail_g=${avail} fea_alive=$(alive "$FEA_PID_F" && echo 1 || echo 0) cfd_alive=$(alive "$CFD_PID_F" && echo 1 || echo 0)"

  if ! alive "$FEA_PID_F"; then
    if [[ "${fea_miss}" -gt 0 ]]; then
      start_fea
    fi
  fi
  if ! alive "$CFD_PID_F"; then
    if [[ "${cfd_miss}" -gt 0 ]]; then
      start_cfd
    fi
  fi

  if [[ "${fea_miss}" -eq 0 && "${cfd_miss}" -eq 0 ]]; then
    idle_rounds=$((idle_rounds + 1))
  else
    idle_rounds=0
  fi

  # two consecutive empty polls → finished
  if [[ "${idle_rounds}" -ge 2 ]]; then
    ingest_all
    log "FINISHED rocket FEA+bodyfit"
    exit 0
  fi

  # When FEA queue is empty, kill bodyfit and restart with more workers once.
  boost_flag="$ROOT/artifacts/logs/bodyfit_boosted.flag"
  if [[ "${fea_miss}" -eq 0 && ! -f "$boost_flag" && "${cfd_miss}" -gt 0 ]]; then
    log "FEA done — boosting bodyfit to 8 workers"
    if alive "$CFD_PID_F"; then
      kill "$(tr -d ' \n' <"$CFD_PID_F")" 2>/dev/null || true
      sleep 3
    fi
    out="$ROOT/artifacts/logs/bodyfit_continue.log"
    nohup "$PY" -u "$ROOT/run_rocket_cfd_bodyfit.py" \
      --curated --skip-done --pilot 0 --workers 8 --no-ingest \
      >>"$out" 2>&1 &
    echo $! >"$CFD_PID_F"
    touch "$boost_flag"
    log "boosted bodyfit pid=$(cat "$CFD_PID_F")"
  fi

  sleep 180
done
