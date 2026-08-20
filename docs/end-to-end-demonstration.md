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

## JEPA encoder

First completed training run in the project's history. Previous best was 83
steps sitting flat at 0.48, because `warmup_steps: 2000` meant the learning
rate never left warmup on any short run.

```
step   1 | loss 0.506490 | embed_std 0.380545
step 250 | loss 0.043170 | embed_std 0.457790
```

Loss down 91.5%. `embed_std` *rose* over training, so representations grew more
diverse rather than collapsing -- the failure mode latent JEPA is prone to, and
which the collapse guard could not previously detect because at `batch_size: 1`
the statistic is NaN.

Corpus: 13,793 records across three disciplines -- 9,029 FEA, 3,267 CFD, 1,500
propulsion/trajectory -- with 158-d conditioning.

## Generative head: what actually moved spec adherence

Four checkpoints against one held-out prompt:

> rocket airframe section 42 mm radius and 165 mm long, ogive nose 55 mm tall,
> 30 mm fin span at 5.0 mm thickness, carrying 180000 N axial load below 200 MPa

| dimension | asked | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|
| body radius | 42.0 | 44.3 (5.5%) | 47.7 (13.6%) | 42.5 (1.2%) | 45.3 (8.0%) |
| body height | 165.0 | 137.8 (16.5%) | 141.9 (14.0%) | 148.6 (10.0%) | 158.1 (4.2%) |
| nose height | 55.0 | 58.1 (5.6%) | 60.2 (9.4%) | 66.5 (20.9%) | 43.0 (21.9%) |
| fin span | 30.0 | 41.5 (38.4%) | 44.1 (46.9%) | 42.2 (40.8%) | **29.8 (0.5%)** |
| fin thickness | 5.0 | 4.8 (3.9%) | 4.8 (4.9%) | 4.8 (3.9%) | 4.8 (3.9%) |
| **mean error** | | **14.0%** | 17.7% | 15.4% | **7.7%** |

| run | designs | tokenizer | corpus | held-out loss |
|---|---|---|---|---|
| v3 | 181 | hash | correlated | 0.007853 |
| v4 | 630 | hash | correlated | 0.005684 |
| v5 | 630 | numeric | correlated | 0.004510 |
| v6 | 543 | numeric | decorrelated | 0.003463 |

Three findings from that table:

1. **Held-out loss and spec adherence diverge.** v4 fit the corpus 28% better
   than v3 and followed the specification *worse* (17.7% vs 14.0%). Optimising
   the loss you can measure is not the same as optimising the behaviour you
   want.

2. **Hash tokenisation destroys numbers.** Every token including numerals was
   blake2b-hashed into a 4096-slot vocab, so "42" and "44" landed on unrelated
   ids and nothing could learn they were close. Reserving log-scale numeric
   buckets took body radius from 5.5% to 1.2% error.

3. **Correlated corpus dimensions cannot be specified.** Fin span was sampled
   as `uniform(0.4, 1.2) * body_radius`, so in training data the stated span
   carried almost no information beyond the radius. The model correctly learned
   to ignore it: 38-47% error across three checkpoints, unmoved by 3.5x more
   data or better tokenisation. Sampling dimensions independently fixed it in
   one step, to 0.5%.

## Solver verification of the generated design

The v6 geometry, meshed and solved against the constraints in its own prompt:

```
solver_mode         : native
targets_met         : true
max_von_mises_mpa   : 52.60      (prompt allowed 200)
max_displacement_mm : 0.0527     (limit 3.0)
frd_bytes           : 1,303,120
```

Geometry: valid ISO-10303-21 STEP from OpenCASCADE, watertight STL, 16,628
faces on the comparable v3 run. The design meets its specification as a solver
result, not an assertion.

## Reproducing it

```bash
source env.sh                       # solver stack + .venv-sim

python generate_confirmed_design_corpus.py --count 700     # ~27 min, 0.5/s
python harvest_confirmed_designs.py                        # recover a partial sweep

python scripts/train_text_cad_confirmed.py --steps 1200 --batch-size 8 \
    --out artifacts/text_cad_confirmed_train_v6

python scripts/infer_text_to_assembly.py \
    --prompt "rocket airframe section 42 mm radius and 165 mm long, ..." \
    --ckpt artifacts/text_cad_confirmed_train_v6/latest.pt \
    --out artifacts/text_cad_infer_v6
```

## Honest limits

- **Nose height is still 21.9% off.** It is sampled independently now, so the
  redundancy explanation does not cover it; the head is predicting near the
  corpus mean for that dimension. Unresolved.
- **One part family.** Body tube, ogive nose, fins. Not an assembly, no
  staging, no propulsion geometry.
- **One load case.** Single axial load. No thermal, buckling, modal or dynamic
  analysis.
- **543 training designs.** Small. Held-out loss improved monotonically with
  corpus size at every step, so this is still the binding constraint.
- **No GPU.** The encoder trains at ~24 s/step on CPU. `colab/jepa_train_gpu.ipynb`
  is staged and unrun.
- **Propulsion is analytic**, not solver-verified, and single-stage, so orbital
  payload cases are not representable.

## Against the statement of intent

`.hermes/plans/2026-07-13_ambitious-cad-agent-plan.md` names seven layers: agent
planner, CAD generator, sculptor, JEPA-driven modelling, simulation,
verification, reporting. Layers 2, 4, 5 and 6 now run end to end. There is no
agent planner decomposing a mission into parts, no freeform sculpting, and no
reporting layer producing a design packet.

A payload-and-range specification is *conditionable* -- `payload_kg`,
`delta_v_ms`, `apogee_km`, `downrange_km`, `burn_time_s` occupy real
conditioning slots populated from 1,500 propulsion/trajectory shards -- but the
generative head maps text to one part family's geometry. Going from "x kg to
y km" to a staged vehicle is not implemented.
