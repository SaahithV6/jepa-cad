#!/usr/bin/env bash
set -euo pipefail
cd /home/best/jepa-cad
# Restart fins FEA so purged ellipticals re-enter the queue
if [[ -f artifacts/rocket_fea.pid ]]; then
  old=$(cat artifacts/rocket_fea.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    kill -TERM "${old}" 2>/dev/null || true
    sleep 2
    pkill -P "${old}" 2>/dev/null || true
    sleep 1
    kill -KILL "${old}" 2>/dev/null || true
  fi
fi
pkill -f 'run_rocket_physics_8k.py --fea-only' 2>/dev/null || true
sleep 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBER_OF_CPUS=1
echo "$(date -Is) restart after elliptical purge" >> artifacts/logs/fea_fins.log
nohup .venv/bin/python -u run_rocket_physics_8k.py \
  --fea-only --workers 3 --skip-register --no-ingest \
  --families fin \
  --target-tets 12000 --mesh-timeout 180 --timeout 300 \
  >> artifacts/logs/fea_fins.log 2>&1 &
echo $! > artifacts/rocket_fea.pid
sleep 8
ps -p "$(cat artifacts/rocket_fea.pid)" -o pid,etime,cmd
tail -15 artifacts/logs/fea_fins.log
echo "fin_frds=$(find artifacts/rocket_fea_8k -path '*/fin_*/case.frd' -size +50k | wc -l)"
echo "fin_shards=$(ls artifacts/physics_shards/fea/fin_*.npz 2>/dev/null | wc -l)"
# graph counts
.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
g='artifacts/jepa-train-bundle/graph.json'
print('fin_parts', subprocess.getoutput(f"rg -o 'part:rocket:fin_[0-9]+' {g} | sort -u | wc -l"))
print('fin_npz_refs', subprocess.getoutput(f"rg -c 'fea/fin_' {g} || echo 0"))
print('mass_kg_hits', subprocess.getoutput(f"rg -c '\"mass_kg\"' {g} || echo 0"))
PY
