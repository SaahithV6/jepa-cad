# Ambitious CAD/CAE Agent System Implementation Plan

> **For Hermes:** This is a plan-only document. Do not implement yet. Use this plan task-by-task, with verification gates after each stage.

**Goal:** Turn `jepa-cad` into a professional-grade, agentic CAD/CAE system that can ingest a user specification, plan and execute geometry generation, solver-backed verification, and report generation, while preserving accountability at every layer.

**Architecture:** Keep the existing repository as the foundation, but shift it from a pure representation-learning scaffold into a layered orchestration system. The product should have: (1) an agent layer for planning, decomposition, tool selection, and iteration; (2) a geometry/CAD layer using FOSS tools such as FreeCAD / OpenCascade / CadQuery / pythonOCC; (3) a custom sculpting layer for freeform surfaces, direct modeling, and hybrid solid/surface edits; (4) a JEPA-driven intelligent modeling layer that builds latent context from geometry, physics, and materials, then feeds iterative design/test cycles; (5) a simulation layer that shells out to OpenFOAM and FEA/MBD tools; (6) a verification layer for constraints, provenance, and traceable pass/fail checks; and (7) a reporting/output layer that produces human-readable design packets and machine-readable artifacts. JEPA is therefore not just a retrieval sidecar — it becomes the core informed-modeling loop that helps the system propose, evaluate, and refine candidate designs. The long-term product should also support an RSI-style improvement loop: each verified design/simulation cycle feeds back into the data flywheel, producing better models, better proposals, and better assemblies over time.

**Tech Stack:** Python, PyTorch, YAML config, pytest, FreeCAD/OpenCascade/CadQuery, OpenFOAM, FEA/MBD FOSS tools, structured logging, checkpointing, job manifests, artifact folders, and later a thin UI/API if needed.

---

## Scope Decision

This project is **not** a ground-up replacement for CAD solvers or physics engines. Instead, it is an orchestration system that uses proven FOSS tools for heavy lifting and wraps them in an agentic workflow with accountability. The geometry side should explicitly support both parametric generation and custom sculpting so the system can handle freeform industrial design as well as engineered components. The LLM should be treated as a planner and assistant, not as an unchecked geometry author unless/until we have strong training data, supervised IO traces, and robust verification. JEPA is the learning and reasoning backbone for repeated modeling/test cycles: context + geometry + physics + materials in, candidate design updates and latent predictions out.

- Every run gets a manifest
- Every tool invocation is logged
- Every generated asset is traceable to inputs and configuration
- Every solver result is paired with a validation summary
- Every design proposal has a confidence/verification status
- Every sculpted geometry edit has a provenance trail and rollback path
- Every JEPA cycle records the context window, target prediction, downstream test outcome, and whether the result was promoted into the training set

That is the only way to make the product credible for professionals and investors.

---

## Day 1: Re-scope the repo into a real system

### Task 1: Freeze the product contract and define the system layers

**Objective:** Establish the exact product promise, success criteria, and the top-level architecture so future work stays coherent.

**Files:**
- Create: `.hermes/plans/2026-07-13_ambitious-cad-agent-plan.md`
- Modify later: `README.md`, `configs/base.yaml`

**Output:**
- A short product contract:
  - input: engineering spec / prompt / constraints
  - output: CAD model or sculpted geometry + solver-backed validation + report
  - accountability: logs, manifests, reproducibility, approvals
- A layer diagram in text:
  - agent planner
  - CAD generator / sculptor
  - solver runner
  - verifier
  - reporter

**Verification:**
- We can explain the product in 30 seconds without hand-waving.
- We can list what this system does and does not do in v1.

---

### Task 2: Audit the existing repo and map current code to the new architecture

**Objective:** Identify what can be reused, what needs refactoring, and what should stay as a subsystem.

**Files likely to inspect:**
- `README.md`
- `configs/base.yaml`
- `train.py`
- `data/dataset.py`
- `data/transforms.py`
- `data/prepare_data.py`
- `data/synthetic.py`
- `models/jepa.py`
- `models/encoder.py`
- `models/predictor.py`
- `eval/probe.py`
- `utils/*.py`
- `tests/*.py`

**Expected mapping:**
- JEPA model = representation / retrieval / ranking subsystem
- synthetic data = smoke-test / benchmark data generator
- data prep = future import pipeline for CAD/solver artifacts
- train loop = future orchestration backbone or research module

**Verification:**
- Produce a table: file, current role, future role, keep/refactor/deprecate.

---

### Task 3: Define the first real product wedge

**Objective:** Choose the first constrained domain so the system can become functional quickly instead of trying to solve “all engineering” at once.

**Candidate wedge:**
- professional part/assembly planning and generation for one high-value part family
- with analysis and validation hooks baked in

**Files:**
- Update later: `README.md`
- Update later: `configs/base.yaml`
- Create later: `docs/scope.md` or equivalent

**Questions to settle before code:**
- Which first part family?
- Which solver chain is mandatory for v1?
- What outputs are required for sign-off?

**Verification:**
- One sentence product wedge that is narrow enough to ship, but ambitious enough to matter.

---

### Task 4: Introduce an artifact-first job system

**Objective:** Make every run reproducible and auditable before building fancy agent behavior.

**Files likely to create:**
- `core/jobs.py`
- `core/artifacts.py`
- `core/manifest.py`
- `core/provenance.py`
- `tests/test_jobs.py`

**Design:**
- Each request becomes a job object
- Each job has:
  - input spec
  - config snapshot
  - toolchain used
  - generated artifacts
  - solver outputs
  - verification summary
  - status
- Store outputs under a structured folder, e.g. `runs/<run_id>/...`

**TDD steps:**
1. Write tests that a job manifest is created.
2. Write tests that artifact paths are deterministic.
3. Implement minimal manifest serialization.
4. Run tests and verify pass.

**Verification:**
- A single job can be reconstructed from files on disk.

---

## Day 2: Build the execution engine and accountability layers

### Task 5: Add the agent planning layer

**Objective:** Build the orchestration logic that converts a user spec into an explicit plan of actions.

**Files likely to create:**
- `agent/planner.py`
- `agent/constraints.py`
- `agent/tool_router.py`
- `tests/test_planner.py`

**Responsibilities:**
- parse user intent and constraints
- choose whether to generate, modify, sculpt, simulate, or verify
- emit a structured action plan
- refuse unsafe or underspecified requests with a clear gap report
- keep the LLM in a planning role unless deterministic tooling has produced geometry that passes validation

**Verification:**
- Given a sample spec, the planner produces an ordered list of steps and required tools.
- The plan is serializable and inspectable.

---

### Task 6: Wrap FOSS CAD tools behind stable interfaces

**Objective:** Make geometry generation/modification a pluggable backend so the agent can use FreeCAD/CadQuery/OpenCascade without caring about implementation details.

**Files likely to create:**
- `cad/backend.py`
- `cad/freecad_backend.py`
- `cad/cadquery_backend.py`
- `cad/sculpting.py`
- `cad/exporters.py`
- `tests/test_cad_backend.py`

**Design:**
- Common interface for:
  - create primitive
  - parameterize part
  - sculpt freeform surfaces
  - blend / fillet / loft / sweep / shell / boolean ops
  - modify geometry
  - export STEP/STL/IGES
  - emit metadata
- Start with the simplest backend that can be automated reliably.
- Add a hard rule that any LLM-authored geometry must be validated by deterministic geometry checks and, when possible, solver checks before being treated as a candidate design.

**Verification:**
- A trivial parametric part can be created and exported end-to-end.
- A simple sculpted surface or freeform edit can also be generated and exported.

---

### Task 7: Wrap solver execution as a job step with strict logs

**Objective:** Run OpenFOAM / FEA / MBD tools as external steps with captured stdout, stderr, exit codes, and artifact paths.

**Files likely to create:**
- `solvers/base.py`
- `solvers/openfoam.py`
- `solvers/fea.py`
- `solvers/mbd.py`
- `tests/test_solver_wrappers.py`

**Design:**
- Each solver run gets:
  - input deck path
  - working directory
  - command line
  - runtime output
  - status
  - generated result files
- The system must not pretend a solver succeeded when it failed.

**Verification:**
- A dummy solver or mock command can be executed and captured.
- Failures are reflected in the job record.

---

### Task 8: Add verification gates and accountability scoring

**Objective:** Prevent silent failures by requiring explicit validation after generation and simulation.

**Files likely to create:**
- `verify/checks.py`
- `verify/report.py`
- `tests/test_verification.py`

**Checks to include:**
- geometry validity
- config completeness
- solver completion status
- required artifacts present
- constraint satisfaction
- unit consistency where applicable
- provenance completeness

**Accountability output:**
- pass / warn / fail
- reasons
- missing inputs
- next recommended action

**Verification:**
- A run cannot be marked complete unless required checks pass or are explicitly waived.

---

### Task 9: Make JEPA the intelligent modeling loop

**Objective:** Preserve the current research value while upgrading JEPA into the system’s latent world-model for informed modeling, design iteration, and test-cycle prediction.

**Files likely to modify:**
- `models/jepa.py`
- `eval/probe.py`
- `README.md`
- `configs/base.yaml`
- later: `agent/feedback.py`, `verify/history.py`

**Role of JEPA in the new system:**
- geometry embeddings
- context encoding for design state
- physics/material-aware latent prediction
- similarity search / retrieval
- ranking candidate designs
- proposal generation for next design iteration
- test-cycle memory so the system learns from what failed or passed

**Verification:**
- The repo story explicitly says JEPA powers the informed modeling cycle.
- The README reflects the larger platform and the role of JEPA inside it.
- The design loop can log context → prediction → solver/test result → updated proposal.

---

### Task 10: Build reporting, review outputs, and the data flywheel

**Objective:** Generate a professional output package for users and stakeholders, and capture verified outcomes so the system can learn from its own successes and failures.

**Files likely to create:**
- `reports/generate.py`
- `reports/templates/summary.md`
- `reports/templates/checklist.md`
- `data/feedback.py`
- `data/registry.py`
- `tests/test_reports.py`

**Report contents:**
- user request summary
- design intent
- generated artifact list
- solver results
- validation status
- open risks
- recommended follow-up actions
- promotion decision: add to training set / reject / hold for review

**Verification:**
- A completed job emits a readable report without manual editing.
- A verified job can be promoted into a dataset registry with traceable provenance.

---

### Task 11: Add the RSI-style iterative improvement loop

**Objective:** Turn the product into a continuously improving system where verified design iterations produce better data, which trains better models, which then produce better assemblies.

**Files likely to create:**
- `train/curation.py`
- `train/registry.py`
- `train/evaluate.py`
- `tests/test_curated_feedback.py`

**Responsibilities:**
- accept verified solver/design outcomes only
- attach provenance and quality labels
- reject low-confidence or failed outputs from the training corpus unless explicitly useful as negative examples
- maintain train/val/test splits by job lineage
- compare new model versions against prior versions before promotion
- keep an auditable model registry so the system can say which model produced which design

**Verification:**
- A run can be promoted into a curated dataset only if it passes explicit checks.
- A new model version must beat the previous one on the agreed evaluation set before becoming default.
- The system can show a clear improvement loop from data → model → design → verification → better data.

---

### Task 12: Finalize the interactive assembly improvement loop

**Objective:** Ensure the end product supports continuous design iteration for assemblies, not just one-shot generation.

**Files likely to create:**
- `agent/iteration.py`
- `agent/assembly_loop.py`
- `tests/test_assembly_loop.py`

**Responsibilities:**
- accept an existing assembly or an initial concept
- generate improvement candidates
- run targeted simulation/verification cycles
- rank candidates by objective and constraint satisfaction
- carry forward only improvements that pass validation

**Verification:**
- The system can iterate on the same assembly multiple times and demonstrate measurable improvement over successive cycles.

---

**Objective:** Generate a professional output package for users and stakeholders.

**Files likely to create:**
- `reports/generate.py`
- `reports/templates/summary.md`
- `reports/templates/checklist.md`
- `tests/test_reports.py`

**Report contents:**
- user request summary
- design intent
- generated artifact list
- solver results
- validation status
- open risks
- recommended follow-up actions

**Verification:**
- A completed job emits a readable report without manual editing.

---

## Product quality gates

Before calling the system “real”, all of the following should be true:

1. A user spec can be turned into a structured job.
2. The planner emits a concrete execution plan.
3. The CAD backend can generate/export at least one part family.
4. Solver wrappers can run and record outputs.
5. Verification gates can fail loudly and explain why.
6. The artifact tree is reproducible from a job manifest.
7. Reports are generated automatically.
8. Existing JEPA code still has a meaningful role.

---

## Risks and tradeoffs

### Risk: scope explosion
The product can become too broad too quickly. Mitigation: enforce a narrow first wedge and reuse the job/manifest layer to keep work organized.

### Risk: solver fragility
External tools can fail for environmental reasons. Mitigation: wrap every solver in a strict interface with logs, exit codes, and mockable tests.

### Risk: overpromising autonomy
Full automatic design is not reliable on day one. Mitigation: make the system explicitly state confidence, assumptions, and required human approvals.

### Risk: JEPA distraction
The current repo may tempt us to focus on model training only. Mitigation: keep JEPA as one component, not the center of the product.

---

## Suggested execution order after confirmation

1. Lock the first part family / wedge.
2. Introduce the manifest and artifact system.
3. Add the planner.
4. Add CAD backend wrappers.
5. Add solver wrappers.
6. Add verification gates.
7. Add reporting.
8. Reframe JEPA inside the new architecture.
9. Update README and config.
10. Run tests and smoke the full pipeline.

---

## Confirmation needed

This plan is intentionally ambitious and assumes we are building a serious professional system, not just a research prototype.

**Please confirm one of these directions before I execute anything:**

- **A.** Proceed with this full ambitious system plan
- **B.** Narrow the first wedge further before execution
- **C.** Change the priority order of CAD / solver / planner / reporting
