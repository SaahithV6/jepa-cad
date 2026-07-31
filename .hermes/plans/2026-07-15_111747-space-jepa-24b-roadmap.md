# Space JEPA 24B Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a space-systems JEPA model family that trains on real spacecraft CAD + solver traces using external compute, while keeping the local repo as the orchestration/control plane.

**Architecture:** The UI stays domain-generic. Domain behavior lives in a registry that routes each sample to a space-family adapter, solver stack, and verification policy. The backbone is a JEPA-style latent predictor trained on verified CAD/CAE trajectories; local hardware runs orchestration, preprocessing, evaluation, and small smoke runs, while Modal/other external compute handles heavy training and sweeps.

**Tech Stack:** PyTorch, current JEPA encoder/predictor stack, current distributed helpers in `utils/distributed.py`, current checkpointing/precision knobs in `train.py` + `configs/base.yaml`, NASA 3D Resources / NASA Open Data Portal / JPL space model sources, Modal for training jobs, optional Fireworks for inference or workflow helpers.

---

## Why this plan exists

We already have a working verified-data loop and a generic CAD/CAE UI, but the current Forge intent heuristics are only a prototype wedge. The long-term product should not hardcode domain classes into the UI. Instead, the system should load a domain family registry and train separate JEPA checkpoints per family.

The first family should be **space systems**:
- spacecraft bus / packaging
- propulsion / pressurization
- thermal control
- deployables / mechanisms
- structures / brackets / mounts

The target is roughly a **24B-class model family**, but the path should be staged:
1. prove the data pipeline and distributed training loop on a smaller backbone,
2. then scale width/depth and data volume,
3. then run the 24B training jobs on external compute.

---

## Current repo facts to preserve

- `train.py` already has:
  - distributed init / cleanup
  - gradient checkpointing support via model config
  - precision policy (`auto`, `fp32`, `fp16`, `bf16`)
  - checkpoint save/load
- `models/encoder.py` and `models/predictor.py` already support gradient checkpointing.
- `cadflow/pipeline.py` already builds verified CAD and can emit artifacts from structured geometry specs.
- The desktop app already has a generic Forge UI; it should remain generic and domain-agnostic.
- The current intent parser in `desktop/src/intentPlanner.ts` is a bridge-only prototype and should eventually be replaced by registry-driven planning.

---

## Scope

### In scope for the space family
- Public NASA/JPL/CubeSat-style CAD and mesh sources
- Spacecraft component graphs and assembly metadata
- Geometry canonicalization into training-ready samples
- Solver/verification traces where available
- A space-domain registry that selects the right preprocessing and objective heads
- External compute training jobs for a JEPA family targeted at space systems
- Evaluation and promotion gates for candidate checkpoints

### Out of scope for v1
- Full aero/hydro/dams specialization
- Hardcoded domain-specific Forge UI branches
- Large-scale multi-domain model zoo
- A full custom CAD authoring language
- LLM-driven geometry synthesis as the primary modeling method

---

## Data sources to prioritize

Use sources with public access and clear reuse terms first:

1. NASA 3D Resources
   - mission models, printable assets, textures
   - good for spacecraft / subsystem geometry

2. NASA 3D Resources GitHub mirror
   - easier to ingest automatically

3. NASA Open Data Portal datasets
   - Cassini Assembly
   - Satellite Kit
   - Shuttle Parts (Hi-Res)

4. JPL Solar System Simulator spacecraft models
   - useful for component assemblies and subsystem shapes

5. Additional open spacecraft / CubeSat CAD sources only if license review passes

Data types to extract:
- STEP / STP / IGES / OBJ / STL / FBX / mesh derivatives
- component and assembly graphs
- part labels / role tags
- metadata about mission role, subsystem, and interfaces
- solver-ready derived artifacts: point clouds, field sidecars, normalized meshes, B-rep exports where possible

---

## Target model family

### Family name
`space-jepa`

### Subfamilies / heads
- `spacecraft_bus`
- `space_propulsion`
- `space_thermal`
- `space_deployables`
- `space_structures`

### Shared backbone responsibilities
- encode geometry and assembly context
- predict masked latents
- condition on loads / constraints / materials / mission role
- support verifier-conditioned learning from solver traces

### Task heads to add over time
- geometry completion / masked part prediction
- solver residual prediction
- validity / watertightness / topology confidence
- stress / deformation surrogate heads
- assembly compatibility / interface risk heads
- optimization scoring heads

---

## External compute strategy

### Local machine
Use the laptop only for:
- data download and normalization
- dataset inspection
- smoke training on tiny subsets
- evaluation and checkpoint comparison
- orchestrating external runs

### Modal / external compute
Use Modal for:
- preprocessing big CAD sets
- long training runs
- grid sweeps over depth/width/embed_dim
- repeated verification-gated retraining
- export of checkpoints and evaluation artifacts

### Fireworks / optional external services
Use for:
- serving / inference helpers
- workflow support around reports or annotation
- not as the main training backend

---

## 24B scaling path

Do not jump straight to 24B in one shot.

### Stage A: prove the family on a smaller backbone
- start with a smaller JEPA config that matches the current trainer
- verify the space data loader and evaluation head end-to-end
- confirm external compute job submission works

### Stage B: scale memory-efficiently
- increase `model.embed_dim`, encoder layers, predictor layers
- keep `model.gradient_checkpointing = true`
- tune precision (`bf16` preferred on supported GPUs, otherwise `fp16`)
- keep batch size modest and use accumulation

### Stage C: scale to the 24B family
- move to distributed sharding / multi-GPU training on Modal or a similar external setup
- add domain adapters instead of separate full backbones where possible
- keep one shared base + multiple family heads
- compare against smaller checkpoints using the existing probe / verification flow

The 24B target should be a **checkpoint family**, not just a single monolith.

---

## Implementation phases

### Phase 1: Define the space registry and schema
**Objective:** Make space-model routing explicit and remove hidden UI hardcoding.

**Files likely to change:**
- `desktop/src/intentPlanner.ts` (replace ad-hoc heuristics with registry lookup)
- `desktop/src/types.ts` (add registry-friendly plan types if needed)
- `configs/space/*.yaml` (new family configs)
- `cadflow/registry.py` or similar new registry module

**Tasks:**
1. Define a `SpaceDomainSpec` structure: family, subsystem, required inputs, solver stack, verification rules.
2. Add registry entries for the five initial space subfamilies.
3. Make the UI accept a generic domain hint instead of hardcoded class-specific behavior.
4. Add tests that the registry resolves a domain spec without special-casing nozzle/bracket/etc.

**Verification:**
- registry resolution test passes
- Forge still works with generic intent text
- no UI logic depends on a baked-in domain class list

---

### Phase 2: Build the space data ingestion pipeline
**Objective:** Normalize real space CAD and metadata into model-ready shards.

**Files likely to change:**
- `data/ingest.py` or a new `data/space_ingest.py`
- `data/parsers.py`
- `cadflow/ingest.py` if the package wrapper needs to expose the new source type
- `tests/test_space_ingest.py`

**Tasks:**
1. Add source discovery for NASA / JPL public assets.
2. Filter sources by license/usage policy.
3. Canonicalize geometry into a shared sample format:
   - points
   - fields
   - component labels
   - provenance
   - subsystem tags
4. Build sharded training data from verified CAD/solver outputs.
5. Add a smoke test on a small public sample.

**Verification:**
- ingest a real public spacecraft sample
- create a shard with points/fields/metadata
- round-trip the shard through the existing parser pipeline

---

### Phase 3: Add space-family training configs
**Objective:** Make training configurable per space subfamily.

**Files likely to change:**
- `configs/base.yaml`
- new `configs/space/*.yaml`
- `utils/config.py` if needed for family overlays
- `tests/test_train_scaling.py`

**Tasks:**
1. Create a base space config.
2. Add overrides for each subfamily.
3. Add model size presets:
   - small smoke
   - medium scale
   - large external-compute run
4. Keep backward-compatible precision/checkpoint knobs.

**Verification:**
- config load test passes
- a small smoke config trains on a tiny curated sample
- model size presets resolve correctly

---

### Phase 4: Wire external compute jobs
**Objective:** Run preprocessing and training remotely while keeping the repo local-first.

**Files likely to change:**
- new `infra/modal/*.py` or `infra/compute/*.py`
- `README.md`
- maybe `cadflow/cli.py` if a new command is needed

**Tasks:**
1. Write a remote training job entrypoint for Modal.
2. Make the job accept a config + dataset shard location + checkpoint target.
3. Upload training artifacts and checkpoints back to a known path.
4. Make the local repo able to launch and inspect those runs.

**Verification:**
- a remote job can start from a config
- it produces a checkpoint artifact path
- local code can load the checkpoint for probe/eval

---

### Phase 5: Add space-model evaluation and promotion
**Objective:** Only promote checkpoints that beat the current best on the space eval suite.

**Files likely to change:**
- `eval/probe.py`
- new `eval/space_eval.py`
- `cadflow/flywheel_loop.py`
- `tests/test_space_eval.py`

**Tasks:**
1. Define eval metrics for each subfamily.
2. Compare candidate vs baseline checkpoint.
3. Promote only when the candidate wins on the relevant metrics.
4. Record provenance and family metadata in the flywheel.

**Verification:**
- candidate/baseline comparison test passes
- promotion only happens on a real win
- promoted checkpoint retains family metadata

---

### Phase 6: Scale toward the 24B model family
**Objective:** Move from the smoke backbone to a serious external-compute run.

**Files likely to change:**
- `models/encoder.py`
- `models/predictor.py`
- `train.py`
- `configs/space/*.yaml`
- `tests/test_jepa_core.py`

**Tasks:**
1. Increase width/depth in a controlled preset.
2. Keep checkpointing on.
3. Use bf16 where supported.
4. Add a large-model external run preset.
5. Add distributed sharding if the current distributed helpers are not enough.

**Verification:**
- one medium and one large preset launch successfully
- no regression in the current smoke tests
- a 24B-class run path exists, even if the first pass is a staged job rather than a full training completion

---

## Recommended initial data wedge

Start with the following public/space-heavy assets:
- Cassini Assembly / Cassini subsystem models
- Satellite Kit assets
- Shuttle Parts hi-res assets
- JPL spacecraft models
- any NASA 3D Resources spacecraft/mission geometry that is cleanly licensable

This gives you a realistic starting corpus for:
- spacecraft bus structures
- antenna/panel assemblies
- tanks and propulsion modules
- thermal and deployable components

---

## Immediate implementation order

1. Create the space registry and schema.
2. Add a space data ingestion path.
3. Add a space training config overlay.
4. Create a Modal training entrypoint.
5. Add space eval/probe comparison.
6. Scale model presets toward 24B.

---

## Risks / tradeoffs

- Public space CAD is heterogeneous; normalization will matter more than model size at first.
- Some sources may not have clean STEP/B-rep; mesh-to-latent and component-graph paths will be important.
- A 24B model is likely unrealistic on the local laptop, so external compute must own the heavy lifting.
- Hardcoded domain heuristics in the UI will eventually fight the registry; they should be removed early.
- A monolithic 24B checkpoint may be less useful than a shared base plus domain adapters.

---

## Open questions

1. Which source should be the first canonical public dataset: Cassini, Shuttle Parts, Satellite Kit, or a curated NASA 3D Resources subset?
2. Do we want a single shared backbone with adapters, or separate checkpoints per space subfamily from day one?
3. Should the first remote job target Modal only, or also support a second provider fallback?
4. Which verification metrics are required before a space checkpoint is considered promotable?
5. How much of the first corpus should be mesh-only versus CAD/B-rep?

---

## Success criteria for the first milestone

- The UI no longer depends on hardcoded space/nozzle/bracket logic.
- A public space CAD source can be ingested and normalized.
- A small JEPA smoke run can train on the space data path.
- External compute can launch a larger run from the same config.
- The eval loop can compare candidate and baseline checkpoints.
- The repo has a clear path from smoke model to 24B-scale training.

---

## Notes for later specialization

Once the space family is stable, split the registry into finer heads/adapters for:
- propulsion
- thermal
- structures
- attitude/control packaging
- entry/landing/reentry if needed later

Aero, hydro, and civil/dam families can reuse the same pattern after the space family proves the loop.
