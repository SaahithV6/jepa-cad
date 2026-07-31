# Spaceflight Data Ingestion and Verification Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a spaceflight-wide ingestion system that can discover, verify, normalize, and prove usability for as many public and reference data sources as possible across spacecraft, launch vehicles, propulsion, tanks/feed systems, thermal, TPS, mechanisms, and integration.

**Architecture:** Use a registry-first source map backed by source-specific adapters. Every source must pass three gates before it becomes training-relevant: (1) availability/license/metadata verification, (2) conversion into normalized shard-ready artifacts, and (3) end-to-end consumption by the existing dataset/train/probe loop. Treat documents, CAD, meshes, and mission metadata as different source classes, but normalize them into the same provenance-rich manifest format.

**Tech Stack:** Python, `cadflow`, `data.ingest`, `data.parsers`, `data.prepare_data`, `cadflow.flywheel`, `cadflow.promotion`, `cadflow.project`, `cadflow.cloud`, `pytest`, optional OCR/document extraction tooling, optional mesh/CAD converters, and the current training/probe pipeline.

---

## Scope

### In scope
- Everything spaceflight-related:
  - launch vehicles and stages
  - spacecraft buses and payloads
  - propulsion systems and engine internals
  - tanks, pressurization, feed systems
  - thermal control and TPS / reentry hardware
  - fairings, interstages, deployables, mechanisms, docking hardware
  - valves, seals, bearings, pumps, turbomachinery
  - mission metadata and hardware catalogs
  - public patents, technical reports, diagrams, manuals, and CAD repositories
- Verification that the data can actually be used by the current pipeline.
- Provenance and license tracking for every source.

### Out of scope for this plan
- Running Modal production jobs
- Building a new solver stack
- UI polish unrelated to ingestion/provenance/usability
- Proprietary/private data acquisition without explicit permission

---

## Current repo anchors

The plan should align with the existing pipeline seams already present in the repo:
- `cadflow/datasets.py` — source registry and taxonomy
- `docs/space-data-sources.md` — human-readable manifest of known sources
- `data/ingest.py` / `cadflow/ingest.py` — raw/flywheel ingestion to curated shards
- `cadflow/project.py` — project intake and domain routing
- `cadflow/cloud.py` — cloud training plan construction
- `data.parsers` / `data.prepare_data` — raw conversion and shard writing
- `tests/test_ingest_and_e2e.py` and related tests — smoke verification surface
- `eval/probe.py` — downstream usability check

---

## Plan

### Task 1: Expand the spaceflight source ontology

**Objective:** Make the registry cover the full spaceflight domain instead of only a few high-value examples.

**Files:**
- Modify: `cadflow/datasets.py`
- Modify: `docs/space-data-sources.md`
- Add/update tests: `tests/test_family_configs.py` or a new registry coverage test

**Source classes to cover:**
- Public institutional archives: NASA, ESA, JAXA, ISRO, CNSA, DLR, CNES, ROSCOSMOS public archives where accessible
- Technical report repositories: NTRS, ESA docs, JAXA repositories, university libraries
- Patents: propulsion, pressurization, tanks/feed, mechanisms, thermal, separation hardware
- Open CAD and parts libraries: STEP/STL/IGES repositories, model rocketry libraries, open engine repos
- Mission and spacecraft metadata databases
- Reference-only industrial catalogs with license review
- Simulation datasets: aero/thermal/structural/CFD/FEA if publicly accessible
- Historical scans, blueprints, exploded views, maintenance manuals, assembly docs

**Verification:**
- The registry should be able to categorize each source into a subsystem family and a source type.
- Run a coverage check that every source has: key, title, URL, domain, use cases, license note, and recommended family tags.
- Ensure the generated docs list every source and clearly separate public/institutional vs reference-only sources.

---

### Task 2: Add source validation metadata before download

**Objective:** Determine whether each source is usable before spending time downloading or converting it.

**Files:**
- Add or modify: `cadflow/datasets.py`
- Add or modify: `cadflow/cloud.py`
- Add or modify: `cadflow/project.py`
- Add tests: a new source-validation test file, or extend `tests/test_project_intake_and_cloud_plan.py`

**Validation fields to track per source:**
- URL reachability
- Content type / payload type
- License or usage note presence
- Source class: PDF, HTML, GitHub repo, dataset hub, STEP/STL/mesh archive, image archive, API, manual scan
- Expected extraction path: direct download, crawl, API pull, OCR, git clone, archive mirror
- Whether the source is reference-only or training-eligible
- Whether the source can be mirrored locally
- Whether it has enough metadata to be routed automatically

**Verification:**
- A validation pass should produce a machine-readable status per source: `reachable`, `blocked`, `ambiguous`, `needs_manual_review`.
- At least one sample from each major source class should be validated end-to-end.

---

### Task 3: Build source-specific adapters for the main spaceflight source classes

**Objective:** Make ingestion automatic by source type instead of hand-curated one-offs.

**Files:**
- Add: `cadflow/source_adapters.py` or equivalent new module
- Modify: `data/ingest.py`
- Modify: `cadflow/ingest.py`
- Modify: `cadflow/cli.py`
- Add tests: new adapter tests and ingestion tests

**Adapters to implement first:**
1. **PDF/report adapter**
   - download PDF
   - extract text
   - retain figures/tables metadata if available
   - index report title, section headings, and figure captions
2. **HTML/docs adapter**
   - fetch page text and linked assets
   - preserve title, headings, captions, and image links
3. **GitHub CAD repo adapter**
   - clone or archive download
   - find `.step`, `.stp`, `.stl`, `.igs`, `.iges`, `.obj`, `.fbx`, `.3mf`
   - capture repo metadata and license file
4. **Dataset hub adapter**
   - pull dataset cards and metadata
   - download sample artifacts when safe
5. **Patent adapter**
   - ingest claims, abstract, figures, and key drawing pages
6. **Image/scan adapter**
   - OCR the text
   - preserve image provenance and page numbers

**Verification:**
- Each adapter should have at least one fixture-backed test.
- Each adapter must emit a common manifest structure with source URI, artifact paths, checksum, and conversion status.
- Failures must be recorded rather than silently skipped.

---

### Task 4: Normalize all ingested artifacts into a single provenance manifest

**Objective:** Make every source traceable and reusable for training, retrieval, and audit.

**Files:**
- Modify: `data/ingest.py`
- Modify: `cadflow/promotion.py` if needed
- Add tests: `tests/test_ingest_and_e2e.py`, new manifest tests

**Manifest requirements:**
- source key and source URL
- fetched timestamp
- artifact type(s)
- checksum(s)
- license/usage note
- extraction method
- page ranges / figure indices / file names
- family tags
- eligibility flags: `training_eligible`, `reference_only`, `needs_review`

**Verification:**
- Every curated shard should point back to a source record.
- Re-running ingestion on the same source should produce stable deduplication or explicit versioning.
- The manifest should be sufficient to reconstruct where each training shard came from.

---

### Task 5: Prove the data can actually be used by the training pipeline

**Objective:** Move from “downloaded data” to “usable training input” with real smoke tests.

**Files:**
- Modify: `tests/test_ingest_and_e2e.py`
- Modify: `tests/test_cadflow_pipeline.py`
- Modify: `eval/probe.py` if required for stronger smoke checks
- Possibly modify: `data/dataset.py`, `train.py`, `cadflow/e2e.py`

**Verification ladder:**
1. **Ingestion smoke:** source artifacts become curated shards
2. **Dataset smoke:** `CADSimulationDataset` can load the shards
3. **Training smoke:** `train.py` can run a short step/epoch on the ingested data
4. **Probe smoke:** `eval/probe.py` can run on the resulting checkpoint
5. **End-to-end smoke:** `cadflow/e2e.py` can ingest -> train -> verify in one run

**Acceptance criterion:**
- At least one real source from each major source class can be ingested and consumed by the pipeline without manual intervention.

---

### Task 6: Create a spaceflight verification matrix

**Objective:** Define what “usable at all” means for each source and refuse weak data early.

**Files:**
- Add: `docs/spaceflight-verification-matrix.md`
- Modify: `docs/space-data-sources.md`
- Add tests for verification policy if needed

**Verification matrix dimensions:**
- fetchable
- parsable
- licensable
- deduplicated
- geometry-preserving
- text-preserving
- training-eligible
- evaluation-eligible
- source-repeatable
- audit-complete

**Verification policy examples:**
- Public NASA/ESA/JAXA reports and diagrams: eligible if text/figure extraction succeeds
- Open CAD repos: eligible if geometry import succeeds and license is known
- Patent PDFs: eligible for text/figure extraction, but mark as reference-heavy
- Marketplace/catalog assets: reference-only unless explicit permission is verified

---

### Task 7: Add source-quality scoring and prioritization

**Objective:** Focus ingestion effort where the corpus will most improve the model.

**Files:**
- Modify: `cadflow/datasets.py`
- Add or modify: `cadflow/cloud.py`
- Add tests around source ranking / prioritization

**Priority scoring signals:**
- direct relevance to spaceflight hardware
- geometric richness
- presence of explicit dimensions or assembly views
- engine/tank/feed/mechanism specificity
- availability of CAD or structured metadata
- quality of license situation
- ability to verify automatically

**Verification:**
- The cloud plan should prefer high-signal source classes first.
- The source ordering should be stable and reproducible.

---

### Task 8: Add a “can we use it at all?” gate for every new source

**Objective:** Prevent source bloat and keep only sources that clear the usability bar.

**Files:**
- Add: `tests/test_source_eligibility.py`
- Possibly add: `cadflow/source_quality.py`
- Modify: `docs/space-data-sources.md`

**Gate criteria:**
- URL or archive path resolves
- License or permission note exists
- Extraction path is known
- Artifact can be converted or indexed
- Source can be attached to a manifest record
- At least one downstream consumer can read the output

**Verification:**
- Sources that fail the gate are retained only as `blocked` or `manual_review`, not as training sources.

---

## Suggested execution order

1. Expand the ontology and source registry.
2. Add validation metadata and source classes.
3. Build adapters for PDFs, HTML, GitHub CAD repos, patents, and scans.
4. Normalize everything into one provenance manifest.
5. Add end-to-end smoke tests proving dataset/train/probe consumption.
6. Add the verification matrix and source-quality scoring.
7. Keep expanding sources only if they clear the gate.

---

## Practical source buckets to keep hunting

### Highest-value buckets
- NASA NTRS reports with diagrams/tables
- ESA spacecraft diagrams and 3D models
- JAXA engine and turbopump papers
- Patents for injectors, nozzles, pressurization, valves, seals, mechanisms
- Open CAD repos with STEP/STL and explicit licenses
- Mission hardware catalogs with structured metadata

### Secondary buckets
- university thesis/report archives
- conference papers with appendices and figures
- archival scans and museum blueprints
- launch vehicle and spacecraft manufacturer brochures
- simulation result archives and benchmark datasets

### Reference-only buckets
- marketplace CAD assets
- non-redistributable vendor catalogs
- community uploads with unclear rights

---

## Definition of done

The ingestion effort is complete enough to rely on when:
- the registry covers the major spaceflight subsystem families
- each major source type has a working adapter or a clearly documented manual path
- every source has a machine-readable usability status
- at least one source from each major class has been ingested successfully
- the curated output can be loaded by the training pipeline and exercised by a smoke train/probe run
- the docs clearly distinguish training-eligible from reference-only data

---

## Risks and tradeoffs

- The biggest risk is collecting many sources that are interesting but not actually usable.
- Patent and catalog data often need extra OCR/figure extraction and rights review.
- Open CAD repositories vary wildly in part quality and license clarity.
- A source registry without automated verification becomes a list, not a corpus.
- The right bias is to keep the gate strict and expand only where the pipeline can prove it can consume the data.

---

## Immediate next implementation slice

If executing this plan, start with:
1. a source validation report generator,
2. a single PDF/report adapter,
3. a single CAD repo adapter,
4. an end-to-end smoke test that proves those outputs train.

Once those four pieces work, the rest becomes systematic source expansion rather than guesswork.
