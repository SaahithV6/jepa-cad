# Plan: full corpus → TAO → training data (no train yet)

Date: 2026-07-31  
Owner: agent under `/goal`  
Constraint: **do not launch Modal or local training**

## Problem

We generated/gathered a large spaceflight corpus (OpenRocket STLs, FEA/CFD, docs, NASA3D, raw CAD). The TAO graph and JEPA dataset only see a thin slice: ~12k physics NPZ + OpenRocket Parts + partial Document text. Unused trees (`generated_spaceflight_cad`, most `raw_downloads`, ORKs, many PDFs) never become Samples/conditioning. Modal staging drops `physics_shards` entirely.

## Approach

1. **Audit** disk vs graph vs dataset (machine-readable JSON).
2. **Widen the training contract** so PhysicsTargets, mesh stats, and params already on Parts are not zeroed.
3. **Densify ingest** to create Parts/Samples/Documents for leftover mesh + text.
4. **Package** physics shards beside the train bundle and teach path resolution / Modal staging.
5. **Stop before train**; leave an explicit gate checklist.

## Implementation checklist

- [x] Audit (explore) → roadmap
- [x] `scripts/corpus_coverage_audit.py`
- [x] Expand `CONDITIONING_QUANTITIES` + `PARAM_QUANTITIES` + configs (141-d)
- [x] `graph_dataset` extra roots + mesh/structural ingest + categorical params→text
- [x] `scripts/corpus_full_coverage.py` (ran; see report)
- [x] `scripts/prepare_portable_train_package.sh` + Modal staging
- [x] Run audit + coverage densify; write reports
- [x] Dataset dim smoke (no optimizer steps) — GRAPH_METADATA_DIM=141

## Post-densify snapshot (2026-07-31)

- Parts: 28,800 (spec_prompt on all); generated CAD Parts: 1,961
- Documents: 3,581 with text on 3,533
- Samples: 20,335; TensorShards unchanged at 12,444 physics-path
- Portable package: 9,029 FEA + 3,264 CFD NPZ linked

## Still open (not blocking this /goal)

- ~4.2k rocket CFD metas without volume NPZ (need re-solve or accept scalar-only)
- Fairing/TPS FEA holes (train around)
- Semantic text / CAD decoder / Neo4j (architecture gate)
- **Do not train until user OK**
