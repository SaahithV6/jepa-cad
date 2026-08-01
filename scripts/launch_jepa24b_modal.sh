#!/usr/bin/env bash
# Launch JEPA space-24b training on Modal from the live TAO bundle.
#
#   ./scripts/launch_jepa24b_modal.sh pilot          # ~30 optimizer steps on T4 (<$1)
#   ./scripts/launch_jepa24b_modal.sh full           # 100k steps on A100-40GB
#   ./scripts/launch_jepa24b_modal.sh full-advanced  # 150k steps, larger effective batch
#
# Prereqs (already satisfied):
#   * ~/.modal.toml token
#   * ./pylibs shim (typing_extensions/pyparsing/multidict/yarl/protobuf)
#   * artifacts/jepa-train-bundle/graph.json = live TAO graph
#   * Prefer artifacts/portable-train-package/ after:
#       ./scripts/prepare_portable_train_package.sh
#     so physics_shards resolve (Modal staging also mounts them now).
#   * graph_metadata_dim must match data.graph_dataset.GRAPH_METADATA_DIM

set -euo pipefail
cd /home/best/jepa-cad
source scripts/pylibs_env.sh

MODE="${1:-pilot}"
STAMP="$(date +%Y%m%d-%H%M%S)"

case "$MODE" in
  pilot)
    GPU="T4"
    MAX_STEPS=30
    BATCH=2
    ACCUM=2
    POINTS=2048
    OUT="artifacts/modal-24b-pilot-$STAMP"
    ;;
  full)
    GPU="A100-40GB"
    MAX_STEPS=100000
    BATCH=4
    ACCUM=8
    POINTS=2048
    OUT="artifacts/modal-24b-full-$STAMP"
    ;;
  full-advanced)
    # Prompt/params → physics-verified CAD: larger effective batch, denser points,
    # longer schedule. space_24b.yaml already has warmup-cosine + EMA + high mix.
    GPU="A100-40GB"
    MAX_STEPS=150000
    BATCH=4
    ACCUM=16
    POINTS=4096
    OUT="artifacts/modal-24b-full-adv-$STAMP"
    ;;
  *)
    echo "usage: $0 [pilot|full|full-advanced]" >&2
    exit 2
    ;;
esac

echo "mode=$MODE gpu=$GPU max_steps=$MAX_STEPS batch=$BATCH accum=$ACCUM points=$POINTS out=$OUT"

PYTHONUNBUFFERED=1 JEPA_MODAL_GPU="$GPU" \
  .venv/bin/python -m cadflow.cli modal-train \
  --project-root . \
  --goal "physics-verified spaceflight CAD from prompts+params (payload, size, weight, material) ($MODE)" \
  --family space \
  --config configs/families/space_24b.yaml \
  --data-source graph --probe-data-source graph \
  --graph-path artifacts/jepa-train-bundle/graph.json \
  --num-points "$POINTS" --num-fields 8 \
  --max-steps "$MAX_STEPS" \
  --set "train.batch_size=$BATCH" \
  --set "train.grad_accum_steps=$ACCUM" \
  --set "data.num_points=$POINTS" \
  --set "data.mix_ratio=0.95" \
  --set "data.prefer_physics_shards=true" \
  --out-dir "$OUT"
