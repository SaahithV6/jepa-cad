# TAO oneshot rocket training data path

**Goal:** 24B JEPA learns to map **text/params** (family + diameter/length/shape/material)
→ **physics-verified rocket Part fields** (real FEA von Mises / displacement / stress
on the geometry), so it can oneshot design-spaceflight CAD with a physics loop.

## Why the old bundle was “only 2.6 GB”

The live TAO graph (`artifacts/jepa-train-bundle/graph.json`, ~239 MB JSON +
~2.4 GB geometry shards) stored **scalar summaries** (`max_stress`, `U_mag`) while
**~150 GB** of raw solver output stayed on disk unused:

| Disk | Size | Role |
|------|------|------|
| `artifacts/rocket_fea_8k/` | ~50 GB | CalculiX FRD nodal fields |
| `artifacts/fea_alt/` + `fea_final/` | ~55 GB | Legacy FEA |
| `artifacts/cfd_*` + `rocket_cfd_bodyfit/` | ~25 GB | OpenFOAM U/p |
| `data/raw_downloads/` | ~18 GB | Source CAD/docs |

Placeholder `.npz` shards had `fields ∈ [0,1]` and `max_stress=1.0` (no real physics).

## What we feed TAO now

1. **Solvers keep growing coverage** (FEA mesh-retry, bodyfit CFD, legacy ducts).
2. **`run_physics_shards.py`** extracts real FRD fields → compact `.npz` under
   `artifacts/physics_shards/fea/` (channels: von_mises, disp_mag, ux,uy,uz, sxx,syy,szz)
   + `fea_manifest.jsonl`.
3. **`register_manifest_to_graph`** (via ingest loop) adds `TensorShard` nodes +
   `HAS_SAMPLE` edges to Parts, stamps `has_field_shard`, mirrors Part params onto shards.
4. **Training config** (`configs/families/space_24b.yaml`):
   - `prefer_physics_shards: true` — real FEA shards ranked first in the dataset
   - `physics_shards_only: false` until coverage is high; flip to `true` for pure oneshot
   - 89-d conditioning = family + physics + geometry + material + **params**
     (all 10.5k rocket Parts already have params like `diameter_mm`, `length_mm`, `shape`)

## Commands

```bash
# Extract (resume-safe)
python -u run_physics_shards.py --source rocket --workers 3
python -u run_physics_shards.py --source legacy_alt --workers 2

# Register into live TAO graph (also done by scripts/tao_ingest_loop.sh)
python -c "from pathlib import Path; from cadflow.build_physics_shards import register_manifest_to_graph; print(register_manifest_to_graph(Path('artifacts/jepa-train-bundle/graph.json')))"

# Train later (NOT launched until you say so)
./scripts/launch_jepa24b_modal.sh pilot
# then flip physics_shards_only=true in space_24b.yaml for oneshot-focused run
```

## Oneshot loop (post-train)

Params/spec → JEPA latent CAD/fields → CalculiX/OpenFOAM verify → accept or regenerate.
The TAO graph is the bridge: provenance + conditioning + physics-verified targets.
