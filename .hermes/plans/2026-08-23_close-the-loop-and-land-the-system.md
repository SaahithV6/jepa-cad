# Goal: close the loop and land the system

**Set:** 2026-08-23. Supersedes nothing; this is the finishing pass over the
2026-07-13 ambitious-CAD-agent plan.

## Why this goal

The seven layers are built. An audit on 2026-08-23 found six of them real and
exercised — planner, CAD, sculpting, simulation, verification, reporting — and
one of them absent in the only sense that matters:

    JEPA references in autodesign.py, design_loop.py, planner.py,
    flywheel_loop.py, promotion.py:  0

The original plan's sharpest claim is that "JEPA is not just a retrieval
sidecar — it becomes the core informed-modeling loop that helps the system
propose, evaluate, and refine candidate designs." Today it is precisely a
sidecar: it trains, it gets probed, its checkpoints get promoted by probe
score, and no design decision anywhere is informed by it.

The corpus work of 2026-08-22/23 is what makes closing that gap possible at
all. Before it, the representation was indistinguishable from random
projections under an honest measurement (+0.019, mixed sign across splits).
After it, +0.057, monotone across training, positive in all six group splits.
A flywheel needs something to learn from; it now has one.

## Definition of done

The system takes a mission spec and returns an assembly whose every claim is
either solver-backed or labelled as an estimate, and the learned model measurably
earns its place in producing it.

1. **The learned model is in the loop.** `run_design_loop` screens candidates
   through `cadflow/surrogate.py` before spending a solver, records every
   prediction against the outcome that judged it, and falls back to solving
   everything when the checkpoint does not clear the bar.
2. **Screening is shown to help, or shown not to.** A measured comparison:
   solver calls spent, and best design found, with screening on versus off.
   A null here gets written down, not buried.
3. **Disputed labels are resolved, not discarded.** 20% of stress records are
   dropped today because sibling records disagree. Where the disagreement is a
   load case rather than a contradiction, keep both and key them by load case.
4. **End to end, on the record.** `x kg to y km` produces an assembly, a mass
   closure, a stability margin, solver-verified stress on the driving part, and
   a packet — run fresh, with the numbers in the report matching the artefacts.
5. **Propulsion re-probed.** Nozzle geometry matches its labels for the first
   time; the thrust/Isp/expansion-ratio question is now answerable and gets an
   answer either way.
6. **Everything green.** Full suite passing, no skipped verification, no claim
   in any docstring that the code does not do.

## Rules carried forward from this week

These are not style preferences; each one is a bug that shipped.

- Verify the artefact, not the call. A generator that reported 1961/1961 wrote
  1,961 empty files.
- A label that could have been computed from the input is not a measurement.
- Split by mesh, never by record; 87% of meshes appear in more than one.
- An interval that omits a source of variance is not a confidence interval.
- A check that cannot observe the failure it looks for is not a check.
- Filenames belonging to another system are keys, not ours to renumber.
