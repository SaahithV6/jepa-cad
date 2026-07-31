# Native Solver Bootstrap and Wiring Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make the CAD/CAE path work end-to-end with deterministic geometry, explicit solver discovery, native solver execution when configured, and a clear fallback/diagnostic story.

**Architecture:** Add a small runtime/config layer that resolves solver binaries and shared-library paths from explicit repo-visible knobs instead of ad hoc shell state. Teach the CLI, pipeline, and adapters to consume that runtime object, then add a doctor-style diagnostic path and tests that prove both native and fallback execution behave correctly. Keep the existing deterministic fallback path, but make native execution the first-class, testable path whenever the solver environment is present.

**Tech Stack:** Python 3.12, pytest, argparse, pathlib, subprocess, dataclasses, the existing `cadflow/` orchestration modules.

---

## Current state to preserve

- `cadflow/cli.py` already supports `run` and `promote`.
- `cadflow/pipeline.py` already builds geometry, verifies it, runs a solver, and records flywheel entries.
- `cadflow/solver.py` already normalizes solver results and probes binaries on `PATH`.
- `cadflow/adapters.py` already has OpenFOAM / FEA / MBD adapters with fallback execution.
- The repo already passes through the deterministic mock-cad path, and that must stay green.

## Problem to solve

The native path still depends on brittle environment assumptions:

- solver binaries are discovered only by `PATH`
- OpenFOAM startup expects environment setup that is not explicit in the repo
- fallback behavior is too opaque when native tools are missing or partially configured
- there is no first-class “show me what solver runtime you detected” command

The fix should make the runtime contract explicit and testable.

---

## Task 1: Add a first-class solver runtime/config resolver

**Objective:** Centralize solver binary/library discovery so the rest of the codebase consumes one object instead of guessing from shell state.

**Files:**
- Create: `cadflow/runtime.py`
- Modify: `cadflow/solver.py`
- Modify: `cadflow/adapters.py`
- Modify: `cadflow/cli.py`
- Modify: `cadflow/pipeline.py`
- Modify: `cadflow/__init__.py`

**Step 1: Define the runtime contract**

Create a small dataclass in `cadflow/runtime.py` that captures:
- solver root directory, if any
- solver bin directories
- solver library directories
- resolved command paths for `simpleFoam`, `blockMesh`, `ccx`, `cgx`, and any future native solver hooks
- the environment mapping that must be passed to `subprocess.run`
- a human-readable diagnostic summary

**Step 2: Make binary probing runtime-aware**

Update `cadflow/solver.py` so `probe_solver_binary(...)` can accept:
- explicit search directories
- an optional runtime object
- fallback to `shutil.which(...)` only when no runtime is provided

The probe result should report:
- which binary name was sought
- the resolved absolute path, if found
- the candidate directories used
- a precise missing-binary reason when not found

**Step 3: Propagate runtime through the orchestration stack**

Thread the runtime object through:
- `cadflow/cli.py` argument parsing
- `cadflow/pipeline.py` `run_pipeline(...)`
- `cadflow/adapters.py` solver adapter execution

Use a small, explicit API surface; do not add a large config framework.

**Step 4: Expose the runtime in package exports**

Update `cadflow/__init__.py` so the new runtime object and resolver helpers are importable from the package root.

**Verification:**
- `pytest tests/test_cadflow_solver.py -q`
- `pytest tests/test_cli_and_distributed.py -q`

Expected: existing tests still pass, and new runtime-aware probe tests can be added next.

---

## Task 2: Make adapters run native solvers with explicit env, not shell guesswork

**Objective:** Ensure the OpenFOAM/FEA/MBD adapters actually invoke the configured native tools and capture useful artifacts and diagnostics.

**Files:**
- Modify: `cadflow/adapters.py`
- Modify: `cadflow/solver.py`
- Modify: `cadflow/pipeline.py`

**Step 1: Pass the runtime environment into subprocess calls**

Update the adapter `run(...)` path so `subprocess.run(...)` receives:
- a merged environment from the runtime object
- explicit `cwd`
- captured stdout/stderr
- a deterministic timeout

Avoid relying on users sourcing shell init files manually.

**Step 2: Keep OpenFOAM as a real native path, but make it explicit**

Refactor `OpenFOAMAdapter` so it uses:
- a resolved `simpleFoam` or `blockMesh` path from runtime
- a case deck written into the working directory
- a clear success/failure result based on the native command exit code and parsed artifacts

If the native OpenFOAM runtime is incomplete, the adapter should fail with a direct diagnostic instead of silently falling back unless fallback is explicitly allowed.

**Step 3: Make FEA/MBD adapters follow the same pattern**

Ensure the `FEAAdapter` and `MBDAdapter`:
- use the runtime resolver for `ccx`, `cgx`, `mbdyn`, or the relevant command names
- write deterministic input decks
- parse output consistently
- return `SolverResult` with a clear `mode` field in metadata (`native`, `fallback`, or `unavailable`)

**Step 4: Preserve fallback behavior, but label it clearly**

When native tools are unavailable and fallback is allowed:
- keep producing a valid `SolverResult`
- annotate metadata with the reason the native path was skipped
- retain the case files as artifacts

**Verification:**
- `pytest tests/test_cadflow_adapters.py -q`
- `pytest tests/test_cadflow_pipeline.py -q`

Expected: native-path tests exercise the real command invocation logic, and fallback tests remain deterministic.

---

## Task 3: Add a doctor/diagnostic command for solver wiring

**Objective:** Make it obvious what the repo thinks the solver setup is, so debugging does not require shell spelunking.

**Files:**
- Create: `cadflow/doctor.py`
- Modify: `cadflow/cli.py`
- Modify: `tests/test_cli_and_distributed.py`

**Step 1: Implement a solver diagnostic report**

Add a small command that prints, in JSON or readable text:
- solver root
- discovered binaries
- discovered library directories
- whether each backend is native-ready
- whether fallback would be used
- the exact missing item for any unavailable backend

**Step 2: Wire the command into the CLI**

Add `cadflow doctor` to `cadflow/cli.py` so users can run a single command to confirm the setup.

**Step 3: Make the diagnostics actionable**

The output should tell the user:
- what is missing
- which env var or config knob controls it
- what the code will do next (native execution or fallback)

**Verification:**
- `python -m cadflow.cli doctor`
- Add a unit test in `tests/test_cli_and_distributed.py` for the new subcommand

Expected: the command exits cleanly and clearly reports native/fallback readiness.

---

## Task 4: Add tests that prove the wiring works without depending on the host machine

**Objective:** Make the solver bootstrap reproducible in CI and on machines that do not have the native solvers installed.

**Files:**
- Create: `tests/test_cadflow_runtime.py`
- Modify: `tests/test_cadflow_solver.py`
- Modify: `tests/test_cadflow_adapters.py`
- Modify: `tests/test_cli_and_distributed.py`
- Modify: `tests/test_cadflow_pipeline.py`

**Step 1: Test runtime discovery with fake binaries**

Use `tmp_path` to create tiny fake executables named:
- `simpleFoam`
- `blockMesh`
- `ccx`
- `cgx`

Point the runtime resolver at that temp directory and assert:
- the binaries are resolved correctly
- the diagnostics are clear
- missing tools are reported individually

**Step 2: Test environment propagation**

Add a subprocess test that verifies the adapter passes the runtime env into the child process.

**Step 3: Test fallback remains deterministic**

Keep the existing mock-cad and fallback behavior covered so CI does not depend on the local machine’s solver install.

**Step 4: Test the end-to-end CLI path**

Update the CLI smoke tests to assert:
- `python -m cadflow.cli run --mock-cad` still works
- `python -m cadflow.cli run --require-native-solver` fails with a useful message when tools are absent
- when a fake runtime is injected, the command succeeds

**Verification:**
- `pytest tests/test_cadflow_runtime.py -q`
- `pytest tests/test_cli_and_distributed.py tests/test_cadflow_adapters.py tests/test_cadflow_pipeline.py -q`
- final check: `pytest -q`

Expected: tests are self-contained and do not require the host to have OpenFOAM/CalculiX installed.

---

## Task 5: Update the README and keep the integration story honest

**Objective:** Document the solver wiring so the next person can use it without reverse-engineering the code.

**Files:**
- Modify: `README.md`
- Optionally modify: `.gitignore` if new artifacts are created by the plan

**Step 1: Document the runtime knobs**

Add a small section that explains:
- how to set the solver root/bin/lib dirs
- how `cadflow doctor` reports readiness
- how native vs fallback execution is chosen

**Step 2: Update the smoke-test commands**

Show one command for:
- mock CAD smoke test
- native solver smoke test
- promotion smoke test

**Step 3: Clarify fallback semantics**

State clearly that fallback is deterministic and useful for tests, while native execution is used when the runtime is configured.

**Verification:**
- read the updated README and ensure it matches the CLI flags and diagnostics

---

## Acceptance criteria

The work is done when all of the following are true:

1. `cadflow run` still works with `--mock-cad`.
2. `cadflow run --require-native-solver` produces a clear diagnostic when the native runtime is missing.
3. Native solver execution uses an explicit runtime configuration instead of ad hoc shell setup.
4. `cadflow doctor` reports exactly what is and is not available.
5. `pytest -q` passes.
6. The README matches the actual CLI and runtime behavior.

---

## Suggested execution order

1. Runtime resolver and probe wiring
2. Adapter execution/env propagation
3. Doctor command
4. Tests
5. Documentation

This keeps the code changes testable at each step and avoids building the docs before the behavior is settled.
