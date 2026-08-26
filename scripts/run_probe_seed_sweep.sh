#!/usr/bin/env bash
# Probe the checkpoints once per seed, sequentially.
#
# Sequentially and not in parallel on purpose. Five of these at once put the
# machine at load 16 with nothing finishing after thirteen minutes, and an
# earlier detached batch orphaned and had to be killed. One at a time, niced,
# each bounded by its own timeout, is slower in wall-clock and finishes.
#
# Every run writes JSON into the repository. The results of four completed
# sweeps were lost once because they existed only as stdout under /tmp and the
# machine restarted; a run whose output is not in artifacts/ has not happened.
#
# Already-finished seeds are skipped, so this is safe to re-run after an
# interruption without repeating an hour of work.
set -uo pipefail

cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1

OUT=artifacts/verification/probe_seeds
mkdir -p "$OUT"

CKPTS=(checkpoints/from_killed_run_step200.pt
       checkpoints/step_001000.pt
       checkpoints/step_001500.pt)

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 7)

for SEED in "${SEEDS[@]}"; do
  if [ -f "$OUT/seed_$SEED.json" ]; then
    echo "[seed $SEED] already complete, skipping"
    continue
  fi
  echo "[seed $SEED] starting $(date +%H:%M:%S)"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  timeout 5400 nice -n 19 python -u scripts/probe_representation.py \
    --ckpt "${CKPTS[@]}" \
    --target max_stress --samples 1600 --draws 3 --seed "$SEED" \
    --group-by content \
    --out "$OUT/seed_$SEED.json" \
    > "$OUT/seed_$SEED.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    # 124 is timeout's own signal that it killed the child. Reported rather
    # than retried: a seed that needs more than 90 minutes needs a different
    # budget, not another 90 minutes.
    echo "[seed $SEED] FAILED rc=$rc $( [ $rc -eq 124 ] && echo '(timed out)' )"
  else
    echo "[seed $SEED] done $(date +%H:%M:%S): $(grep -c '\.pt ' "$OUT/seed_$SEED.log") checkpoints"
  fi
done

echo
echo "=== aggregate ==="
python scripts/aggregate_probe_seeds.py "$OUT"/seed_*.json
