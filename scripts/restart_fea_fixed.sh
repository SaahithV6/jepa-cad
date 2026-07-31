#!/usr/bin/env bash
set -euo pipefail
cd /home/best/jepa-cad
LOG=artifacts/logs/fea_mesh_retry.log
pkill -f 'run_rocket_physics_8k.py --fea-only --workers' 2>/dev/null || true
sleep 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBER_OF_CPUS=1
echo "$(date -Is) restart FEA with realistic fins" >>"$LOG"
nohup .venv/bin/python -u run_rocket_physics_8k.py \
  --fea-only --workers 2 --skip-register --no-ingest \
  --families fin,tank,nozzle,nose_cone,transition,engine_mount,bulkhead \
  --target-tets 12000 --mesh-timeout 180 --timeout 300 \
  >>"$LOG" 2>&1 &
echo $! | tee artifacts/rocket_fea.pid
sleep 15
ps -p "$(cat artifacts/rocket_fea.pid)" -o pid,etime,cmd
tail -12 "$LOG"
