# JEPA-CAD

Self-supervised pretraining for **CAD geometry paired with CFD/FEA simulation annotations**, using a [JEPA](https://arxiv.org/abs/2301.08243) (Joint Embedding Predictive Architecture) objective.

The repo now also declares a real CAD/geometry toolchain for downstream modeling work:
- `cadquery` for solid modeling / export paths
- `trimesh`, `scipy`, and `shapely` for geometry processing and validation utilities

This repository is still intentionally modular and test-driven: the goal is a correct single-machine training and tooling loop that can be scaled up once the architecture is validated.

## CAD/CAE orchestration (`cadflow/`)

Deterministic CAD/CAE loop where the LLM is planner/orchestrator only:

```
manifest → CadQuery/mock geometry → STEP/STL export → solver wrap/probe
        → verification → append-only flywheel (verified runs only)
```

| Module | Role |
|--------|------|
| `cadflow/backends.py` | Parametric + sculpt primitives, metadata, STEP/STL export |
| `cadflow/solver.py` | Normalized `SolverResult`, binary probes, subprocess wrap, fallbacks |
| `cadflow/verification.py` | Volume / bbox / validity / watertight checks + text reports |
| `cadflow/manifest.py` | Fingerprinted jobs, provenance, run records |
| `cadflow/flywheel.py` | Append-only JSONL history + verified ranking/promotion |
| `cadflow/pipeline.py` | Thin end-to-end orchestration |

```bash
pytest tests/test_cadflow_*.py -q
```

## Project scope

### In scope (v1)

- Point-cloud canonical representation with per-point simulation fields (pressure, temperature, stress)
- JEPA block masking on spatial regions (I-JEPA / V-JEPA style)
- Context encoder + EMA target encoder + latent predictor (no point-level reconstruction loss)
- Synthetic parametric data generator for smoke tests and mix-ratio experiments
- Real-data shard pipeline (`prepare_data.py`) with `--dry-run`
- Single-device training with AdamW, warmup + cosine LR, logging, checkpoints
- Collapse detection (embedding std warning)
- Mixed real/synthetic batches via `mix_ratio` config
- Linear probe eval on frozen encoder (`eval/probe.py`)
- Unit tests for masking logic and EMA updates

### Out of scope (explicit TODOs)

- Distributed / multi-node training (marked TODO in `train.py`)
- Physically accurate synthetic physics
- Full STL/VTK/STEP parsers (placeholder in `prepare_data.py`)
- Voxel grids and mesh-based encoders (tradeoffs documented in `data/dataset.py`)
- Production-scale datasets and hyperparameter sweeps

## Repository layout

```
jepa-cad/
├── cadflow/                # CAD/CAE orchestration (backends, solvers, verify, flywheel)
├── configs/base.yaml       # all hyperparameters
├── data/
│   ├── dataset.py          # lazy shard loading + synthetic flag
│   ├── transforms.py       # JEPA block masking
│   ├── prepare_data.py     # raw → .npz/.pt shards
│   └── synthetic.py        # parametric mock generator
├── models/
│   ├── encoder.py            # point transformer encoder
│   ├── predictor.py          # latent predictor
│   └── jepa.py               # JEPA wrapper + EMA
├── train.py                  # main training loop
├── eval/probe.py             # linear probe sanity check
├── utils/                    # logging, checkpoints
└── tests/                    # JEPA + cadflow tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (definition of done)
python train.py --max-steps 50 --data-source synthetic

# Unit tests
pytest tests/ -q

# Linear probe after training
python eval/probe.py --checkpoint checkpoints/latest.pt --data-source synthetic

# Inspect real shards (sanity checks)
python -m data.stats --data-dir data/processed --limit 32
```

## Data representation

Each sample is a dict of tensors:

| Key | Shape | Description |
|-----|-------|-------------|
| `points` | `(N, 3)` | Point cloud geometry |
| `fields` | `(N, F)` | Per-point simulation scalars |
| `max_stress` | scalar | Proxy label for linear probe |

Shards are stored as `.npz` (default) or `.pt` under `data/processed/`.

### Prepare real data

```bash
python -m data.prepare_data --raw-dir /path/to/raw --out-dir data/processed --dry-run
python -m data.prepare_data --raw-dir /path/to/raw --out-dir data/processed
```

`--dry-run` processes 5 samples and prints shapes/stats without writing shards.

## Training

```bash
python train.py --config configs/base.yaml --data-source synthetic
python train.py --config configs/base.yaml --data-source real
python train.py --config configs/base.yaml --data-source mixed
python train.py --resume checkpoints/latest.pt --data-source synthetic
python train.py --max-steps 500 --data-source synthetic
```

CLI flags:

- `--config` — YAML config path
- `--resume` — checkpoint to resume
- `--data-source {real,synthetic,mixed}`
- `--max-steps` — stop early for smoke tests

Mixed batches use `data.mix_ratio` in config (e.g. `0.7` = 70% real, 30% synthetic per batch).

## Configuration

All hyperparameters live in `configs/base.yaml`: model dims, masking ratios, batch size, learning rate, EMA decay, checkpoint frequency, collapse threshold, etc. Scale the model by editing config — not code.

## JEPA training objective

1. Mask spatial blocks into **context** (visible) and **target** (hidden) regions.
2. Encode context with the trainable context encoder.
3. Encode full geometry with the EMA target encoder (no gradients).
4. Predict target block embeddings from context + target block positions.
5. Minimize smooth L1 (or cosine) distance in embedding space — **not** point reconstruction.

After each optimizer step, target encoder weights are updated:

`θ_target ← decay · θ_target + (1 − decay) · θ_context`

## Monitoring

Training logs to `runs/<experiment_name>/metrics.jsonl` and stdout:

- loss, learning rate, gradient norm
- embedding norm and std (collapse detection)
- samples/sec

If target embedding std drops below `train.collapse_std_threshold`, a warning is printed.

## Scaling up

Once smoke tests pass:

1. Prepare real shards with `prepare_data.py`
2. Increase `model.embed_dim`, encoder layers, and `data.num_points` in config
3. Train with `--data-source mixed` and tune `mix_ratio`
4. Run `eval/probe.py` before committing to long runs

Distributed training is left as a TODO in `train.py`.

## License

TBD — add before public release.
