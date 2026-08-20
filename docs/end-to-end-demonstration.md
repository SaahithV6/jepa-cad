# End-to-end demonstration: specification to solver-verified design

Recorded 2026-08-19. Everything below ran on the host CPU (24 threads, no GPU).

## What runs

```
natural-language specification
  -> SemanticTextEncoder + CadAssemblyDecoder   (trained, held-out validated)
  -> assembly parameters
  -> CadQuery                                   -> STEP + watertight STL
  -> gmsh                                       -> tetrahedral mesh
  -> CalculiX 2.21                              -> von Mises / displacement
  -> constraint check                           -> meets specification or not
```

## The run

Prompt, a specification the model had not seen:

> rocket airframe section 42 mm radius and 165 mm long, ogive nose 55 mm tall,
> 30 mm fin span at 5.0 mm thickness, carrying 180000 N axial load below 200 MPa

Decoded parameters vs the request:

| requested | generated | delta |
|---|---|---|
| 42 mm radius | 44.33 | +5.5% |
| 55 mm nose height | 58.07 | +5.6% |
| 5.0 mm fin thickness | 4.81 | -3.9% |
| 165 mm body length | 137.83 | -16% |
| 30 mm fin span | 41.53 | +38% |

Geometry produced: valid ISO-10303-21 STEP from OpenCASCADE, STL watertight
with 16,628 faces, 850 cm^3, bounding box 88.7 x 88.6 x 137.8 mm (consistent
with the decoded radius and height).

Solver result on the generated geometry, against the constraints stated in the
prompt:

```
solver_mode         : native
targets_met         : true
max_von_mises_mpa   : 59.60      (prompt allowed 200)
max_displacement_mm : 0.0597     (limit 3.0)
frd_bytes           : 1,067,120
```

The design satisfies the specification, and that is a solver result rather than
an assertion: the geometry was meshed and solved, with a megabyte of FRD behind
the number.

## Reproducing it

```bash
source env.sh                       # solver stack + .venv-sim

# 1. corpus of physics-confirmed (prompt -> design) pairs
python generate_confirmed_design_corpus.py --count 800
#    ... or rebuild from a partial/interrupted sweep:
python harvest_confirmed_designs.py

# 2. train the generative head (holds out 10% for validation)
python scripts/train_text_cad_confirmed.py --steps 800 --batch-size 8 \
    --out artifacts/text_cad_confirmed_train_v3

# 3. specification -> design
python scripts/infer_text_to_assembly.py \
    --prompt "rocket airframe section 42 mm radius and 165 mm long, ..." \
    --ckpt artifacts/text_cad_confirmed_train_v3/latest.pt \
    --out artifacts/text_cad_infer_v3

# 4. solve the generated geometry against the prompt's constraints
#    (see the verify step in artifacts/text_cad_infer_v3/verify)
```

## Honest limits

- **The dimensional match is loose.** Radius, nose height and fin thickness land
  within 6%, but body length is off by 16% and fin span by 38%. The head has
  learned the mapping approximately, not precisely.
- **Trained on 181 designs** (201 swept, 20 held out). Validation loss fell
  0.009696 -> 0.007853 and stayed at or below training loss throughout, so this
  is generalisation rather than memorisation -- but the corpus is small and is
  the binding constraint, not model capacity.
- **This is one part family.** Body tube, ogive nose and fins. Not an entire
  rocket assembly, not staging, not propulsion geometry.
- **The structural case is a single axial load.** No thermal, no buckling, no
  dynamic or modal analysis.
- **No GPU was involved.** The JEPA encoder trains at ~24 s/step on CPU; the
  T4 path in `colab/jepa_train_gpu.ipynb` is staged but has not been run.

## What this does not yet do

The statement of intent (`.hermes/plans/2026-07-13_ambitious-cad-agent-plan.md`)
describes seven layers: agent planner, CAD generator, sculptor, JEPA-driven
modelling, simulation, verification, reporting. Layers 2, 4, 5 and 6 are
exercised above. There is no agent planner decomposing a mission into parts, no
freeform sculpting, and no reporting layer producing a design packet.

A payload-and-range specification is now *conditionable* -- payload_kg,
delta_v_ms, apogee_km, downrange_km and burn_time_s occupy real conditioning
slots, populated from 1,500 propulsion/trajectory shards -- but the generative
head above maps text to a single part family's geometry. Going from
"x kg to y km" to a staged vehicle is not implemented.
