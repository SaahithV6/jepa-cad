#!/usr/bin/env bash
set -euo pipefail
cd /home/best/jepa-cad
mkdir -p artifacts/logs

# Restart ingest only if down
need=1
if [[ -f artifacts/tao_ingest_loop.pid ]]; then
  pid=$(cat artifacts/tao_ingest_loop.pid || true)
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    need=0
    echo "ingest already up pid=${pid}"
  fi
fi
if [[ "${need}" -eq 1 ]]; then
  nohup bash scripts/tao_ingest_loop.sh >> artifacts/logs/tao_ingest_loop.log 2>&1 &
  echo $! > artifacts/tao_ingest_loop.pid
  echo "started ingest pid=$(cat artifacts/tao_ingest_loop.pid)"
fi

# Status dump
sleep 2
{
  echo "=== FEA ==="
  ps -p "$(cat artifacts/rocket_fea.pid)" -o pid,etime,pcpu,cmd || echo fea_down
  tail -8 artifacts/logs/fea_fins.log
  echo "fin_frds=$(find artifacts/rocket_fea_8k -path '*/fin_*/case.frd' -size +50k | wc -l)"
  echo "=== SHARDS ==="
  echo "fin_shards=$(ls artifacts/physics_shards/fea/fin_*.npz 2>/dev/null | wc -l)"
  echo "=== INGEST ==="
  ps -p "$(cat artifacts/tao_ingest_loop.pid)" -o pid,etime,cmd || echo ingest_down
  tail -5 artifacts/logs/tao_ingest_loop.log
  echo "=== WORKERS ==="
  ps --ppid "$(cat artifacts/rocket_fea.pid)" -o pid,etime,pcpu,rss,cmd 2>/dev/null | head || true
  pgrep -a ccx | head -5 || true
} | tee artifacts/logs/fin_pipeline_status.txt
