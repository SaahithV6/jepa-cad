# Corpus full-coverage roadmap

**Goal:** make ingestion + training-data plumbing consume every useful mesh / structural / text signal from the generated and gathered corpus. **Do not train / do not launch Modal** until packaging + densify gates pass.

**Status:** in progress (2026-07-31). Audit numbers from live `artifacts/jepa-train-bundle/graph.json` + disk inventory.

---

## Snapshot (post densify_tao_hybrid)

| Layer | Count / size | Notes |
|---|---|---|
| Disk solvers + CAD | ~160GB | FRD/OpenFOAM trees + `data/` |
| TAO graph | 241k nodes / 368k edges / ~283MB | Index + conditioning, not raw fields |
| Physics NPZ | 9,029 FEA + 3,264 CFD | Real train field signal |
| Resolvable train records | ~29.2k | 12.3k physics + 16.9k geometry |
| Documents with text | 731 / 1,843 | Rest title-only or missing extract |
| Rocket Parts | 10,500 | OpenRocket STLs fully as Parts |
| Unused generated STL | ~1,961 | `data/generated_spaceflight_cad/` → **0 Parts** |
| Rocket CFD metas without volume NPZ | ~4.2k | Fields cleaned before extract |

---

## Phases

### P0 — Make every on-disk useful file reachable from training code (no cloud run)

1. **Portable package:** stage `artifacts/physics_shards/` with the train bundle (`prepare_portable_train_package.sh` + Modal staging).
2. **Path resolution:** `graph_dataset` searches `extra_search_roots` + `data/processed` so `nasa3d/*.npz` and `artifacts/physics_shards/...` resolve.
3. **Widen conditioning:** add dropped PhysicsTarget / param / mesh structural slots; bump `graph_metadata_dim`.
4. **Corpus densify:** `scripts/corpus_full_coverage.py` ingests leftover STL/STEP, densifies remaining docs, ORK text, rewrites nasa3d paths, re-associates Samples.

### P1 — Recover / backfill physics signal

1. Re-extract CFD volume shards where OpenFOAM fields still exist; otherwise mark metas as scalar-only with honest flags.
2. Finish non-fairing FEA holes (tank/nose); **skip** hopeless fairing/TPS hulls (train around).
3. Register all manifests → TensorShard + `HAS_SAMPLE`.

### P2 — Text + CAD depth

1. Extract remaining PDFs under `raw_downloads/` + bundle `files/`.
2. ORK → Document summaries linked to Parts.
3. Convert SLDPRT/ASM/GLB → STEP/STL offline where needed (`data/parsers.py` stays STL/OBJ/PLY/STEP).
4. Ingest `extracted_geometries/` + `spaceflight_components/` STEP as Parts/Samples.

### P3 — Architecture (after corpus dense; still no train without user OK)

1. Semantic text encoder (replace 32-d MD5 bag).
2. Modality-correct FEA vs CFD channels (drop shared max-norm lie).
3. Graph encoder / CAD decoder; honest scale naming (not “24B”).
4. Neo4j bulk import + Cypher parity (train still JSON-first).

---

## Done means (verification)

- [x] `artifacts/corpus_coverage_audit.json` exists with disk↔graph deltas
- [x] `artifacts/corpus_full_coverage_report.json` shows new Parts/Samples/docs/text
- [x] `configs/*` `graph_metadata_dim` matches `GRAPH_METADATA_DIM` (141)
- [x] Dataset smoke: resolvable physics + nasa paths; 141-d metadata; assoc/text signal
- [x] `scripts/prepare_portable_train_package.sh` creates a package listing that includes physics_shards
- [x] **No** Modal/train process started

---

## Entry points

| Script | Role |
|---|---|
| `scripts/corpus_full_coverage.py` | Full densify pass (graph-locked) |
| `scripts/densify_tao_hybrid.py` | Prior hybrid densify (still valid) |
| `scripts/prepare_portable_train_package.sh` | Bundle + shards layout for Modal |
| `scripts/corpus_coverage_audit.py` | Read-only coverage audit JSON |
| `data/graph_dataset.py` | Training consumer |
| `cadflow/modal_training.py` | Staging (shards + extra roots) |

See also: `docs/plans/2026-07-31-corpus-full-coverage.md`, `AGENT_CROSSTALK.md` architecture gate.
