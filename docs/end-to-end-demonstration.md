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

---

## Mission-level: what works and where it stops

The airframe result above answers "build me a section with these dimensions".
The harder question is "deliver x kg to y km", which needs the propulsion
sizing and mass budget too. `generate_mission_corpus.py` builds 681 such pairs,
each verified twice: flown through a gravity turn, and structurally solved in
CalculiX under its own liftoff thrust. The decoder was widened from 8 airframe
dimensions to 14 (adding chamber pressure, expansion ratio, propellant,
structural and payload mass, and throat area).

Asked to *deliver 12 kg to 95 km apogee using LOX/RP-1 at 55 bar*:

| quantity | requested | designed | flown |
|---|---|---|---|
| chamber pressure | 55 bar | 55.6 bar (v8) | - |
| payload | 12 kg | 14.3 kg | - |
| apogee | 95 km | - | **703-929 km** |

**Directly stated design parameters are followed** -- chamber pressure to 1.1%,
payload to within a few kg, and the vehicle it produces is coherent: 167 kg
propellant, 42 kg structure, 224 kg gross, a credible sounding rocket.

**Stated mission outcomes are not achieved.** Apogee comes out 7-10x high.

Two things this is *not*: it is not underdetermination (adding throat area so
the vehicle can be flown from its own parameters changed nothing), and it is
not a shortage of training signal in the usual sense.

It is the corpus construction. Designs are sampled forward and labelled with
whatever mission results, so mission outcomes are extremely skewed: apogee
spans 12.6 to 5,219 km -- 415x, 2.6 decades -- with a median of 437 km. A 95 km
request sits at the 7.9th percentile, and only 15.6% of the corpus lies below
150 km. Asked for something in that sparse tail, the head returns something
near the bulk of the distribution.

Going from "design -> outcome" data to "outcome -> design" capability is an
inverse problem, and the corpus has to be built for it: rejection-sample
designs so that mission outcomes are roughly uniform in log space, rather than
accepting whatever apogee falls out of a uniform sweep over design parameters.
That is the next piece of work, and it is corpus design rather than model
capacity.

---

## Mission conditioning: three causes, two fixed

Asked to deliver a payload to a stated apogee, the head initially overshot by
7-10x. Three separate causes, diagnosed by measurement rather than guessed.

### 1. Corpus outcome coverage (fixed)

Designs were sampled uniformly and labelled with whatever mission resulted, so
outcomes were skewed across 2.6 decades: apogee 12.6-5,219 km, median 437, a
95 km request at the 7.9th percentile. Rejection sampling into 12 log-spaced
apogee bins fixed the coverage exactly -- 50 per bin from 15 to 3,000 km, at
17,643 attempts for 600 accepted, with rejects costing no solver time because
the trajectory runs before CalculiX.

Effect: a 95 km request went from 929 km flown to 559 km.

### 2. Model capacity (fixed)

`scripts/train_text_cad_confirmed.py` hardcoded a smoke-test model -- embed_dim
64, one text-encoder layer -- matching its own defaults of 40 steps at batch 2.
A real corpus was being pushed through a text->parameter path far too small.

Sweeping the requested apogee over 100x moved the design by 2.9%:

```
asked   20 km -> mass ratio 3.83, gross 255.0 kg
asked 2000 km -> mass ratio 3.94, gross 246.9 kg
```

At embed_dim 256 with 4 text layers, propellant mass responds monotonically and
by 72.5% over the same sweep, and flown apogee crosses the requested value
between 400 and 2000 km. Model size is now a flag.

### 3. The training objective (not fixed)

What remains is structural. The head is trained on mean-squared error over
normalised *design parameters*, but apogee depends on mass ratio
*exponentially*. Physics requires, for LOX/RP-1 at Isp 300 s:

| apogee | delta-v | required mass ratio |
|---|---|---|
| 20 km | 0.85 km/s | 1.33 |
| 95 km | 1.84 km/s | 1.87 |
| 400 km | 3.78 km/s | 3.62 |
| 2000 km | 8.46 km/s | 17.71 |

A 13x span. The model produces 3.03-4.29, a 1.4x span, and 100x of requested
apogee compresses into roughly 2x of flown outcome (431-855 km).

This is what minimising parameter MSE *should* do. The MSE-optimal predictor is
the conditional mean, and in a space where the outcome is exponential in the
parameter, the conditional mean of the parameters is nowhere near the parameter
that achieves the mean outcome. Better coverage and more capacity both help,
and neither addresses it, because the loss cannot see the physics.

Two ways out, neither attempted here:

- **Reparameterise so the target is linear in the outcome.** Predict delta-v or
  log mass ratio and derive the masses from it, instead of predicting masses
  directly. Cheap, and it makes parameter error proportional to mission error.
- **Put the physics in the loss.** Score the achieved apogee, not the parameter
  vector -- a differentiable trajectory, or a policy-gradient estimator over the
  existing integrator.

The first is the obvious next step and is a small change to the corpus and the
decode path.

### 3b. Reparameterising the target (partial)

Predicting log mass ratio instead of propellant mass, so that parameter error
is proportional to delta-v error rather than wildly uneven across the range:

```
MR      dv km/s   err if MR off by 0.5   err if ln(MR) off by 0.1
 1.33      0.84              939 m/s                    294 m/s
 1.87      1.84              697 m/s                    294 m/s
 3.62      3.78              381 m/s                    294 m/s
17.71      8.46               82 m/s                    294 m/s
```

Predicting MR, the same absolute error costs 11x more delta-v at the low end
than the high end, so the head is pushed toward high MR where mistakes are
cheap -- which is exactly the collapse observed.

Result, sweeping requested apogee:

| asked | before flown | error | after flown | error |
|---|---|---|---|---|
| 20 km | 750.1 | 1.57 dec | **19.9** | **0.00 dec** |
| 95 km | 431.4 | 0.66 dec | 44.7 | 0.33 dec |
| 400 km | 622.4 | 0.19 dec | 139.5 | 0.46 dec |
| 2000 km | 854.7 | 0.37 dec | 31.5 | 1.80 dec |

Mean error barely moved, 0.698 to 0.647 decades. The character changed
entirely: flown dynamic range went from 2.0x to 7.0x against a 100x request
span, the collapse to a narrow band is gone, and a 20 km request is now hit
exactly. The high end regressed, and that is not a corpus limit -- the corpus
contains mass ratios to 8.81.

So the loss-alignment fix removed the failure it targeted and exposed the next
one. What remains is that nothing in training ever scores the mission itself.
The head is still fitted to parameter vectors; the trajectory integrator that
decides whether the mission closes is never consulted during training. Options,
in increasing cost:

- Score achieved apogee with a policy-gradient estimator over the existing
  integrator. No differentiability required, and the integrator already runs at
  roughly 3 designs per second.
- Make the trajectory differentiable and train through it directly.

Both are physics-in-the-loop training. Neither is a corpus or capacity change,
which is why neither of those two fixes could substitute for it.

### 4. The target was a relation, not a function (fixed)

The deepest cause, and the one that explains why the previous three fixes each
helped and none sufficed.

Rejection sampling gave uniform coverage of mission outcomes, but each
specification was still paired with *one of many* vehicles that happen to reach
that altitude. At a given apogee, log mass ratio spanned 0.41 to 2.18, because
payload, propellant, chamber pressure and drag all varied independently.

Mean-squared error converges to the conditional mean of that set. Apogee is
nonlinear in the parameters, so the mean of designs that each achieve X does not
itself achieve X. The model was being asked to predict a set and was returning
its average, which is not a member of it. No loss function repairs that, and it
is why better coverage, more capacity and log-space reparameterisation each
moved the needle without fixing the behaviour.

`solve_mission_corpus.py` makes the target a function. Everything that varies is
stated in the prompt, and the one remaining degree of freedom -- mass ratio --
is solved by bisection until the vehicle actually flies the requested altitude.
Apogee is monotonic in mass ratio at fixed everything-else, so bisection
converges: 290 of 300 specifications solved in 1.7 minutes, 290 unique prompts,
no prompt with more than one design, targeting error 0.93% mean and 1.99% worst.

Result, asking for 8 kg to a swept altitude and flying what came back:

| asked | flown | ratio | mass ratio |
|---|---|---|---|
| 20 km | 20.9 | 1.05x | 1.76 |
| 35 km | 33.3 | 0.95x | 2.15 |
| 60 km | 51.8 | 0.86x | 2.54 |
| 95 km | 72.6 | 0.76x | 2.83 |
| 150 km | 108.1 | 0.72x | 3.19 |
| 250 km | 187.2 | 0.75x | 3.68 |
| 400 km | 330.9 | 0.83x | 4.25 |
| 650 km | 669.9 | 1.03x | 5.19 |
| 1000 km | 1022.4 | 1.02x | 5.88 |
| 1600 km | 398.5 | 0.25x | 6.29 |

Mean absolute error 0.120 decades, against 0.647 before -- a 5.4x improvement --
with 9 of 10 within 2x of target and three within 5%. Mass ratio rises
monotonically with the request, so the model has learned the physical
relationship rather than the corpus average.

Held-out generative loss fell to 1.0e-4, an order of magnitude below anything
the sampled corpora produced, with train and validation within 25% of each
other.

The remaining failure is the top of the range: 1600 km returns 398 km. That
specification sits at the edge of the solved grid, and the vehicle it implies
(1009 kg gross) is heavier than anything else in the corpus.

**The lesson generalises beyond this project.** When a specification admits many
valid answers, regression on the parameters learns the average answer, and for
any nonlinear system the average answer is not a valid one. Either the
specification must determine the design, or the loss must score the outcome.
Making the corpus a function was far cheaper here than putting physics in the
loss, and it is what actually worked.

### 5. Staging (fixed)

Two stages, solved the same way -- specification determines the design, total
mass ratio bisected until the stack flies the requested altitude, spent
structure jettisoned at separation. 414 of 486 specifications solved in 1.5
minutes.

Asking for 8 kg to a swept altitude on lox/rp1 at 55 bar, and flying what came
back:

| asked | flown | ratio | gross | separation |
|---|---|---|---|---|
| 50 km | 59.3 | 1.19x | 27.4 kg | 26 s |
| 100 km | 105.5 | 1.05x | 33.6 kg | 28 s |
| 200 km | 228.0 | 1.14x | 44.1 kg | 31 s |
| 400 km | 408.4 | 1.02x | 60.5 kg | 33 s |
| 800 km | 759.8 | 0.95x | 93.2 kg | 35 s |
| 1500 km | 1571.5 | 1.05x | 169.0 kg | 36 s |
| 3000 km | 2783.5 | 0.93x | 362.0 kg | 37 s |
| 6000 km | 4920.1 | 0.82x | 1109.8 kg | 38 s |

Mean absolute error 0.041 decades across a 120x span of requested altitude,
8 of 8 within 2x and 7 of 8 within 20%.

## Progression

| model | corpus | stages | mean error |
|---|---|---|---|
| v11 | sampled | 1 | 0.698 dec |
| v12 | sampled, log mass ratio | 1 | 0.647 dec |
| v13 | solved | 1 | 0.120 dec |
| v14 | solved | 2 | **0.041 dec** |

Reproduce the evaluation with `scripts/eval_mission_targeting.py <checkpoint>`.

---

## Correction pass: physics bugs found by researching the failures

Every one of these produced plausible numbers, which is why they survived.

### Trajectory

1. **Wrong root for apogee.** Apsides solve `2 eps r^2 + 2 mu r - L^2 = 0`. For a
   bound orbit `eps < 0`, so the denominator is negative and the `+sqrt` root is
   *perigee* while `-sqrt` is *apogee*. The code took `+sqrt` and reported the
   perigee. At r = 6,571 km and v = 10.2 km/s that is 200 km instead of
   33,190 km, so apogee looked pinned just above current altitude and then
   jumped to escape, leaving every target between unreachable.

2. **Flat-earth delta-v.** `sqrt(2 g h)` assumes constant gravity: it overstates
   burnout speed by 1.7x at 12,000 km and 2.7x at 40,000 km. The planner then
   escalated to architectures the mission did not need and declared reachable
   missions impossible. Replaced with the exact energy form.

3. **Integration truncated by t_max.** High-energy flights hit the 4,000 s limit
   and were extrapolated from whatever mid-flight state they were in, making
   apogee discontinuous in propellant. Integration now stops once the stack is
   ballistic and above 200 km, where the conic result is exact.

4. **Denormal ambient pressure.** The exponential atmosphere underflows to a
   denormal at extreme altitude: nonzero, so the vacuum guard passed it, but
   small enough that the pressure ratio underflowed to zero and `0 ** negative`
   raised.

### Search

5. **Bisection kept the last iterate, not the closest** (in two solvers). A probe
   landing on an escape trajectory has infinite apogee and discarded converged
   solves.

6. **Stage count hardcoded to "one, two, else three"**, so candidates of four or
   five silently built three-stage vehicles.

7. **Ascent profile fixed at 3 degrees.** Tuned for sounding rockets; on a
   high-energy flight it turns the vehicle horizontal early, so beyond a point
   extra propellant buys downrange rather than altitude and apogee stops
   responding at all -- non-monotonic, which no amount of bisection resolves.
   Now searched.

8. **Two solvers for one problem.** The corpus generator carried its own
   simplified solver -- fixed two stages, fixed split, fixed profile, narrow
   bracket -- and could not solve above ~6,000 km while the planner reached
   200,000 km. The trained model then failed exactly at that corpus edge:
   12,000 km asked, 318 km flown. The corpus is now generated *through* the
   planner, so there is one implementation.

### Data

9. **Throat area was a `0.0` placeholder** with a "filled below" comment that
   never filled, so all 414 records taught the decoder that throat area is zero.
   The evaluation missed it because the flight path recomputed the throat and
   ignored the prediction -- a vestigial output is exactly where a silent zero
   survives.

10. **Throat area spans four decades** (16.5 to 140,672 mm2, median 758), so a
    linear scale put the median at 0.038. Log-scaled.

### Structure

11. **Structural coefficient was a constant 0.14** and the CAD components were
    *solid* cylinders, which is why FEA margins came out at 200-300x. Structure
    is now sized from load: for a thin cylinder in axial compression the wall is
    set by buckling, not yield, with the NASA SP-8007 knockdown iterated for
    imperfection sensitivity. Yield-only sizing is out by 40x at 5 kN.
    `planner.plan_sized` runs the fixed point this creates -- structural mass
    depends on loads, loads on vehicle mass, vehicle mass on structural mass --
    converging in 4-8 damped iterations. For 20 kg to 400 km the coefficient
    converges 0.140 -> 0.086 and gross mass falls 172 -> 130 kg. The assumption
    was changing architecture decisions: 25 kg to 4,000 km closes on two stages
    where the assumed value forced three.

## Result

Planner chooses the architecture, model sizes the vehicle, vehicle flown with
its own predicted engine:

| asked | stages | flown | ratio |
|---|---|---|---|
| 50 km | 1 | 50.3 | 1.01x |
| 100 km | 1 | 78.4 | 0.78x |
| 200 km | 1 | 192.1 | 0.96x |
| 400 km | 1 | 373.9 | 0.93x |
| 800 km | 2 | 677.3 | 0.85x |
| 1500 km | 2 | 1552.9 | 1.04x |
| 3000 km | 2 | 2937.1 | 0.98x |
| 6000 km | 3 | 5950.7 | 0.99x |
| 12000 km | 3 | 8766.7 | 0.73x |
| 25000 km | 3 | 20151.3 | 0.81x |
| 50000 km | 3 | 41488.7 | 0.83x |

Mean 0.051 decades over a 1,000x span, 11 of 11 within 2x, five within 4%.
