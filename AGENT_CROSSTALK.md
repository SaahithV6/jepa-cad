# Agent Crosstalk — LatticeZero / JEPA spaceflight

**Purpose:** Physics-verified spaceflight CAD in the TAO graph. Modal/24B: user gave explicit OK 2026-07-24 (~14:15 PDT) — pilot launched.

**Updated:** 2026-07-25 ~09:45 PDT

## GENERATIVE HEAD (2026-08-01) — local evidence

- `models/text_encoder.py` SemanticTextEncoder + `models/cad_decoder.py` CadAssemblyDecoder
- `model.enable_generative: true` path in `JEPAModel`
- Local train: `artifacts/text_cad_local_train/TRAIN_METRICS.json` (30 steps CPU)
- Infer+solid verify: `artifacts/text_cad_infer/INFER_REPORT.json`
- Offline export: `artifacts/offline-export-*.tar.zst` + `jepa-export-core.tar.zst`
- Full Modal/TAO-scale train still optional; do not launch without GPU budget confirm

## MODAL GATE (2026-08-01) — physics-confirmed unlock

User expects **params → physics-confirmed designs** (native CalculiX). Modal **not allowed**
until `artifacts/train_gate` exists.

Unlock (`train_gate`) only when ALL hold:
1. `scripts/doctor_native_fea.py` exits 0 (`solver_mode=native`)
2. `scripts/params_to_physics_confirmed.py` exits 0 with `CONFIRMED_REPORT.json` `solver_mode=native`
3. `scripts/request_rocket_assembly.py --no-fallback` fails closed unless native-confirmed
4. Local train on real graph + generative head: `artifacts/text_cad_confirmed_train/TRAIN_METRICS.json`
5. Offline export refreshed including confirmed report

Out of scope for unlock: fake 24B scale, freeform mate graphs, Modal launch itself
(stop at train_gate unless user explicitly launches).

## ARCHITECTURE GATE (2026-07-25 ~09:50 PDT) — DO NOT MODAL-TRAIN

Confirmed by architecture audit + local counts:

- Model: **~76.5M trainable / ~127M total** (not 24B)
- Objective: latent JEPA only — **no CAD decoder**
- Text: 32-d MD5 bag — **no semantic encoder**
- Graph: flatten to conditioning vector — **no GNN**; Neo4j offline / unused at train time
- Modal staging: **fixed 2026-07-31** to also stage `artifacts/physics_shards/` + `data/processed/nasa3d` via `extra_search_roots` (still verify portable package before launch)
- FEA/CFD share one 8-ch tensor with per-shard [0,1] norm (magnitudes discarded)
- Corpus coverage work: see `docs/corpus-full-coverage-roadmap.md` + `scripts/corpus_full_coverage.py`

Before any cloud train: portable dataset package (bundle + physics_shards), Neo4j bulk load + Cypher parity,
semantic text + graph encoders, modality-correct fields, CAD decoder, honest scale ladder.

## RESUMED — TAO densify for hybrid text+CAD (2026-07-25 AM)


User correctly flagged: graph (~277MB) was thin vs ~140GB solvers; Documents had
**no text**; **Cd=0**; rocket CFD volume fields mostly cleaned after metas.

**Done (do not re-freeze without asking):**
- `scripts/densify_tao_hybrid.py` → `artifacts/tao_densify_report.json`
- Documents with text: **722** (was 0); new docs linked; DESCRIBES/MENTIONS edges
- Parts with Cd_proxy + airflow: **7457**; spec_prompt on **all 12659** Parts
- SimulationCase links: **+24k**; TensorShards **12444**; graph now **241k nodes / 368k edges / ~283MB**
- Dataset conditioning: **124-d** (was 89) = physics airflow slots + 32-d hashed text bag
- Existing physics shards: FEA **9029** npz + CFD **~3.2k** npz (rocket bodyfit U/p mostly deleted post-meta — cannot resurrect without re-solve)

**Still true / honest limits:**
- Graph JSON will never equal 140GB of FRD/OpenFOAM; training signal is shards + conditioning
- Rocket CFD field shards capped by what was extracted before cleanup (~3k)
- Modal **not** launched yet — verify dataset smoke, then detached Modal

## PAUSED / TAO FROZEN (2026-07-25 ~06:33 PDT)

User shutting machine off. **Do not mutate graph / do not launch Modal** until resume.

- Marker: `artifacts/TAO_FROZEN`
- Handoff: `artifacts/RESUME_WHEN_BACK.md`
- On resume: **verify TAO first**, then ingest gaps, then detached Modal 24B (fairings skip OK)

## OVERNIGHT FINISH (this agent, 2026-07-25)

**User ask:** finish ALL FEA/CFD (legacy + rocket) → ingest TAO → reconnect/docs →
configure+launch Modal 24B advanced for prompt/params→CAD. Running till morning.

**Owns:**
- `scripts/overnight_master.sh` (pid in `artifacts/overnight_master.pid`)
- `scripts/coast_supervisor.sh` (queue-aware; no CFD thrash when curated empty)
- Fairing FEA hull recovery in `cadflow/rocket_physics_suite.py`
- TAO finalize: `scripts/overnight_tao_finalize.py`
- Status gate: `artifacts/overnight_status.json`
- Modal: `./scripts/launch_jepa24b_modal.sh full-advanced` when `train_gate`

**Do not:** spawn a second coast/overnight master; kill mid-ingest.

Logs: `artifacts/logs/overnight_master.log`, `coast_supervisor.log`, `fea_fairing_overnight.log`, `modal_24b_full_overnight.log`

---

## Real physics-field shards → oneshot rocket (2026-07-24 ~18:00 PDT)

**Priority (user):** keep solvers + field shards healthy for training; material-faithful FEA/density channel is fine later.

**CFD fix:** bodyfit now extracts U/p volume shards via `append_cfd_shard_manifest` *before* cleanup. FEA shards fully caught up + 10min watcher.


Why TAO looked "only 2.6GB": graph stored scalar summaries; ~150GB FRD/OpenFOAM sat unused.
Real FEA fields now extract into `artifacts/physics_shards/fea/` as TensorShard nodes.

- Extractors: `run_physics_shards.py` (rocket `case.frd`, legacy_alt `case_alt.frd`)
- Dataset prefers them (`prefer_physics_shards=true` in `space_24b.yaml`)
- Docs: `docs/tao-oneshot-rocket.md`
- **Do not train** until user says; keep feeding shards + solvers.

### Registered coverage (verified 17:05 PDT, graph 255MB / 209,470 nodes)

| | Parts with field shards | conditioning |
|---|---|---|
| rocket | 3,833 / 10,500 (extracting) | family + params + material/geometry (29/89 slots) |
| legacy | 2,031 / 2,159 (**extraction done**, 0 fail) | family=generic + material/geometry (21/89 slots) |
| **TensorShard total** | **6,014** | 8/8 field channels real, `is_synthetic=0` |

Dataset ordering verified: physics shards sort ahead of all geometry-only records, so
batches hit real solver signal first. Sampled shards carry real von Mises values
(e.g. 60.7 MPa), not normalized placeholders.

Legacy Parts have no `params` and `family="generic"`, so they contribute physics-field
diversity but no parametric spec signal. Rocket Parts carry the text+params → CAD
conditioning. Shard edges use `HAS_SAMPLE`, which is in `_ASSOC_EDGE_TYPES`, so the
association walk still reaches the Part for material/geometry on both sets.

### Three bugs fixed here — read before restarting anything

1. **Registration was frozen at 225.** `scripts/tao_ingest_loop.sh` gained the
   `register_manifest_to_graph` step *after* the loop process had started, and bash
   keeps an already-parsed loop body in memory. **Any edit to that script requires
   restarting the loop** — wait for an `ingest end` line first, never kill mid-cycle.
   The log now appends instead of truncating, so restarts keep history.
2. **Legacy shards linked to nothing.** Legacy Parts are keyed `part:<part_hash>` but
   case dirs use a different hash stored in `fea_case_id`. `register_manifest_to_graph`
   now builds a case→Part index; legacy stamping went 0 → 2,031.
3. **Two concurrent graph writers → torn reads.** `jepa_train_supervisor.sh`
   (`ingest_and_mass`) writes `graph.json` at the same time as `tao_ingest_loop.sh`,
   and three writers used non-atomic `write_text` on a 250MB file. Readers hit
   `JSONDecodeError` mid-document and writers could clobber each other.

   Fixed in **`cadflow/graph_lock.py`**: an flock mutex plus atomic tmp+`os.replace`
   writes, applied to `ingest_fea_to_graph`, `ingest_cfd_to_graph`,
   `ingest_bodyfit_to_graph`, `apply_sidecar_to_graph`, and
   `register_manifest_to_graph`. Guarding in Python (not the shell) was deliberate:
   these modules are re-imported per subprocess, so no supervisor restart was needed.
   The lock is re-entrant by design — FEA ingest re-applies the mass sidecar, and
   flock is per-file-description, so a naive nested acquire would self-deadlock.

   **If you add a graph mutation, wrap it in `graph_lock(graph_path)` and write via
   `write_graph_atomic`.**



---

## TRAINING PIPELINE — UNBLOCKED (2026-07-24 PM)

The "Modal installation is broken" saga was **five missing pure-Python deps** in
the uv-built venv (no pip): `typing_extensions`, `pyparsing`, `multidict`,
`yarl`, `protobuf`. They now live in **`./pylibs`** (shim; `source
scripts/pylibs_env.sh`). Modal 1.5.2 imports, token configured.

Fixes that keep the 24B run from crashing:

- **`graph_metadata_dim` was wrong (82) vs emitted (89)** — physics grew 16→20,
  geometry 8→11. Fixed in `configs/{base,families/space_24b}.yaml` and made
  `data/graph_dataset.py::GRAPH_METADATA_DIM` computed, not hardcoded.
- CPU smoke train (tiny model, real TAO graph, 3 steps): **loss decreases, no
  collapse, checkpoint saved**. STL (trimesh), STEP (cadquery), NPZ all parse.
- Dataset from live graph: **16,907 records** (10,719 STL / 4,216 NPZ / 1,972 STEP).

Launch (one command):

```bash
./scripts/launch_jepa24b_modal.sh pilot   # T4, 30 steps, <$1  (RUNNING — artifacts/logs/modal_24b_pilot.log)
./scripts/launch_jepa24b_modal.sh full    # A100-40GB, 100k steps
```

## TAO coverage snapshot (2026-07-24 14:40 PDT)

| | Parts | FEA | CFD | mass_kg |
|---|---|---|---|---|
| rocket | 10,500 | 4,282↑ | 3,099↑ | 10,500 ✅ |
| legacy | 2,159 | 2,031 | 1,458↑ | 2,063 |
| **total** | **12,659** | **6,313** | **4,557** | **12,563** |

FEA+CFD both: 3,528. Solvers still running (one each, deduped):
`rocket_fea.pid`=FEA mesh-retry · `legacy_cfd_internal_retry.pid`=duct retry ·
`rocket_cfd_bodyfit.pid`=bodyfit (other agent — do not touch).

**Process hygiene warning:** the Cursor sandbox has its own PID namespace —
`kill -0 <pid>` lies about host processes and caused duplicate solver spawns.
Use `pgrep -f` + `grep -v cursorsandbox /proc/$p/cmdline` for liveness, and run
kills outside the sandbox. `scripts/tao_autobuild_supervisor.sh` is fixed but
intentionally NOT running (solver batches outlive the hour anyway).

---

## OWNERSHIP NOW (do not collide)

| Who | Owns | Artifacts | Status |
|-----|------|-----------|--------|
| **physics-8k agent** | OpenRocket **10.5k** corpus mesh+FEA+**bodyfit CFD** | `artifacts/rocket_fea_8k/`, `artifacts/rocket_cfd_bodyfit/` | ACTIVE bodyfit (5w); FEA helped |
| **This agent (legacy + help)** | Legacy CFD/FEA + FEA mesh-retry help | `artifacts/cfd_internal/`, `artifacts/cfd_bodyfit/` | duct retry running |

### Help landed for physics-8k (2026-07-24)

- **Mass race fixed:** enrich finished → `mass_kg` on **10500 rocket + 2063 legacy**. Sidecar at `artifacts/mass_properties_sidecar.json`. FEA/CFD ingest now re-applies sidecar after graph write (`rocket_physics_suite.ingest_*`, `ingest_rocket_fea_fast`).
- **FEA was stuck at ok=0** on curated remain queue (mostly open fairings / hollow meshes / PaStiX+OpenMP under bodyfit load). Restarted as PID in `artifacts/rocket_fea.pid` with:
  - families **excluding fairing** (0/1000 volume-meshable so far)
  - workers=2, `OMP_NUM_THREADS=1`, target_tets=12000
  - progressive CL/angle mesh retries in `prepare_fea_case`
  - log: `artifacts/logs/fea_mesh_retry.log` — already producing FRDs again
- **Do not kill** `rocket_cfd_bodyfit.pid` (bodyfit ~3k+ cases).

### Legacy CFD gaps (why not every part)

Of **701** legacy Parts without `has_cfd`:

| Reason | ~N | Action |
|--------|----|--------|
| `skip_exoatmospheric` | 448 | Bus/deployable/antenna — vacuum; FEA/thermal only by design |
| `component_duct` unsolved | 182 | Retry running (`legacy_cfd_duct_retry.log`, longer timeouts) |
| `external_aero` no STL | 42 | Need geometry recovery |
| tank/nozzle/chamber/valve leftovers | ~29 | Same internal runner |

Lifelike CFD = bodyfit external aero + recipe internal ducts — **not** freestream CubeSats.

### Legacy real-workload lane

1. **Demoted** empty-channel CFD → `simulation_results_cfd_channel_proxy` (`has_cfd=false` until bodyfit).
2. **Promoted** class-conditioned FEA alts → canonical `simulation_results_fea` (family loads: pressure / aero / mass×Ng).
3. **Bodyfit CFD** for aero Parts (fin/nose/fairing): `snappyHexMesh` + STL wall + `simpleFoam` → `artifacts/cfd_bodyfit/`.
4. **Fixes landed:** `.STL` case-sensitive resolve; binary STLs with `solid`-padded headers rewritten so OpenFOAM does not ASCII-crash (`write_stl_meters`).

```bash
PYTHONUNBUFFERED=1 python3.12 run_legacy_cfd_bodyfit.py --pilot 0 --workers 1
python3.12 run_legacy_cfd_bodyfit.py --ingest-only
cat data/physics_real_workload_summary.json
cat data/physics_gap_cfd_fix_summary.json
```

**TAO counts (legacy):** family FEA **2031**; bodyfit CFD **210/252** aero (83.3%); channel as primary **0**.
All aero Parts with resolvable STL are bodyfit-linked (incl. decimated MiniToolFins). Remaining **42** aero lack STL. Gap FEA with STL still unmeshable/pending: **~32**.

Do **not** touch rocket 8k roots. Keep workers low while rocket FEA/CFD runs.

### Physics-8k lane (other agent)

- Bodyfit: `artifacts/rocket_cfd_bodyfit/` + `run_rocket_cfd_bodyfit.py`
- FEA: `artifacts/rocket_fea_8k/`
- Leave alone.

---

## Hard rules

| Do | Don't |
|----|--------|
| FEA solid-only (`cadflow/msh_to_calculix.py`) | `gmsh.write()` mixed export |
| Bodyfit CFD (`snappyHexMesh_external`) | Empty channel / box-only CFD as training signal |
| Ask before Modal / 24B | Auto-train |
| Keep lanes separated (legacy vs 8k) | Rewrite each other’s artifact roots |

## Live numbers

- Legacy canonical FEA (family loads): **2031**
- Legacy CFD total (`has_cfd`): **371**
  - External aero bodyfit: **210/252**
  - Internal duct CFD: **161** (chamber 75 / valve 60 / nozzle 15 / injector 9 / igniter 2)
- Channel CFD as primary: **0**
- Summary: `data/physics_real_workload_summary.json`

## Legacy non-aero CFD — expanded (this agent)

Routing: `cadflow/legacy_cfd_routes.py` · Runner: `run_legacy_cfd_internal.py` · Artifacts: `artifacts/cfd_internal/`

**Expanded again 2026-07-24 AM:** `component_duct` (`simpleFoam`) for structure/generic/mechanism with geometry — batch **1076** Parts running (`legacy_cfd_component_duct.log`). Target: push legacy `has_cfd` well past 1k+. Bus/deployable still skipped (no freestream).

| Recipe | Solver | Intent |
|--------|--------|--------|
| `external_aero` | `simpleFoam` | fins/nose/fairing freestream |
| `chamber_internal` | `rhoSimpleFoam` | chambers + pressure-section twins |
| `nozzle_compressible` | `rhoCentralFoam` | nozzles / throats |
| `injector_orifice` | `simpleFoam` | injectors |
| `valve_feed` | `simpleFoam` | valves / manifolds / feed |
| `tank_pressure_flow` | `simpleFoam` | tanks / vessels |
| `turbopump_internal` | `simpleFoam` | impeller / inducer (when tagged) |
| `skip_exoatmospheric` | — | CubeSat/bus/deployable FEA only |

Do **not** freestream CubeSats. Internal flows are the point for propulsion hardware.

## Memory throttle (2026-07-23 ~23:10 PDT)
Rocket FEA restarted `--workers 2` (was 6); CFD `--workers 1` (was 2). ~20Gi MemAvailable at throttle; no CRITICAL in mem_watch. PIDs in `artifacts/rocket_fea.pid` / `artifacts/rocket_cfd_bodyfit.pid`.
