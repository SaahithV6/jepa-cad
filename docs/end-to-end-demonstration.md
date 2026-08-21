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

## Nose shape, and the wave-drag model

The airframe was built entirely from cylinders and boxes, so the "nose cone" was
a cylinder with a flat forward face. `cadflow/profiles.py` and the `revolve_profile`
backend op replace it with a real surface of revolution in four standard rocketry
families -- tangent ogive, conical, elliptical, and von Karman. All four revolve
to within 0.000% of their analytic volume, at 2 faces each because the meridian
is fitted as a single spline.

Making the shape *matter* took two attempts, and the first one is worth recording
because the failure was instructive.

Karman's slender-body integral is

    D/q = -(1/2 pi) INT INT S''(x1) S''(x2) ln|x1 - x2| dx1 dx2

The first attempt discretised it on a uniform grid with the log singularity
absorbed into the diagonal via INT INT_cell ln|x-y| = h^2 (ln h - 3/2). Against
Sears-Haack it came out high by a factor that drifted 4.59 -> 5.20 across
n=200..6400 without settling, so it was deleted rather than shipped.

The fault was the numerical method, not the physics. The quadrature was already
correct -- integrating ln|x-y| over [0,L]^2 with a constant integrand converges
to the exact L^2(ln L - 3/2) to +0.03% at n=1600. What it could not handle was
that S'' is *singular* at the ends of exactly the shapes that matter: the Haack
family has S ~ x^(3/2) at the tip, hence S'' ~ x^(-1/2), and no uniform grid
represents that.

The fix is to integrate in theta under x = (L/2)(1 - cos theta), evaluated at
midpoints so no node ever lands on an end. The Jacobian (L/2) sin(theta) vanishes
at the ends exactly fast enough to cancel the singularity -- S'' * w stays finite
-- and the clustering resolves the region that carries the drag. Sears-Haack then
converges to six figures and Richardson-extrapolates to the same 1.395282 at
every n from 100 to 6400, with D/q L^2/R^4 constant to one part in 1e6 across
four bodies of different length and radius.

That value is (9 pi^3 / 2) R^4 / L^2, equivalently 128 V^2/(pi L^4), which is
16/3 times the 24 V^2/(pi L^4) that is often quoted. Since the quadrature is
validated independently, the discrepancy is a constant in the physics rather than
in the numerics, and the magnitudes alone do not settle it: at fineness 10 the
two candidates give wave-drag coefficients of 0.111 and 0.021, and real slender
bodies sit between. So the module is used **relatively**, normalised to a tangent
ogive, where any common prefactor cancels exactly. Cd keeps its absolute level
from the measured CFD corpus.

What licenses the relative use is a check that holds whatever the prefactor is.
The von Karman ogive is by construction the minimum-wave-drag nose for a given
length and base radius, and it comes out minimal at every fineness ratio tested.
The tangent ogive lands 4-20% above it and converges toward it as the nose slims,
and the cone penalty falls with fineness, 2.83 -> 1.69, which is the right trend:
shape matters most when blunt. tests/test_wave_drag.py holds 15 such checks.

Two limits of the theory are real, and the second one was found only after the
first version had already been committed pricing cones.

The nose must be *pointed*, S'(0) = 0 at the tip. An elliptical nose is not: it
meets the axis with infinite slope, and a blunt nose at supersonic speed has a
detached bow shock the theory does not model at all.

The nose must also meet the cylinder *tangentially*, and this is the condition
that decides whether there is an answer at all. A slope break puts a jump in S',
and linearised slender-body wave drag is logarithmically divergent at a slope
discontinuity -- it has no limit, so no quadrature converges to anything. At
fineness 3, refining n from 750 to 12,000:

    ogive        0.2442  0.2447  0.2449  0.2450  0.2451     converged
    vonkarman    0.2249  0.2257  0.2260  0.2262  0.2263     converged
    conical      0.4266  0.4596  0.5109  0.5281  0.5846     still climbing

The cause is the joint slope dr/dz: -1e-5 for the ogive and -1e-3 for von Karman,
both tangent to within rounding, against -0.1667 for the cone. The first version
priced cones anyway. Its number moved 7% between adjacent fineness values and
grew without bound under refinement -- noise with a plausible magnitude, which
is the worst kind, and it would have gone into the corpus as a conditioning
signal for the model to fit. `shape_factor` now refuses any non-tangent nose,
and only tangent shapes are sampled.

What survives is smaller but real: von Karman against the tangent ogive, smooth
in fineness, n-converged, 13% better at fineness 1.5 falling to 3.5% at 5.5 as
the advantage shrinks in the slender limit, which is what must happen.

The CFD stack cannot stand in here either: those routes run simpleFoam, which is
incompressible steady RANS and produces no wave drag at all.

Shape now reaches the trajectory. `cd_multiplier` scales the wave-drag share of
Cd and returns exactly 1.0 for an ogive, so the default vehicle flies precisely
as it did before. For 25 kg to 4,000 km:

  von Karman   CD 0.4037   gross 1106.6 kg   apogee 4175.9 km
  ogive        CD 0.4200   gross 1106.6 kg   apogee 4120.5 km

Same hardware, 55 km further on the better nose. The corpus samples tangent nose
shapes and fineness, and carries `nose_wave_factor` as a 42nd conditioning slot
-- shape as the physically meaningful scalar rather than a category index, so
the model sees a continuous quantity it can interpolate.

Supporting geometry is exact: `profile_volume` and `wetted_area` are exact for a
polyline meridian, with the cone reproducing pi R sqrt(R^2 + L^2) and pi R^2 L/3
to 0.0000%. At R=0.5, L=2.0 an elliptical nose wets 5.062 against a cone's 3.238
-- 56% more skin for the same length and base radius.


## Modal analysis

The FEA only ever ran `*STATIC`, so nothing in the loop could see a resonance,
and `first_mode_hz` was a conditioning slot with nothing populating it. A part
sized only for steady load can still be destroyed by one -- fin flutter is the
classic case.

`generate_modal_case_inp` writes a CalculiX `*FREQUENCY` deck, which also needs
`*DENSITY`: a modal analysis is a mass problem and the static deck has no reason
to carry a density. It reuses the mesh the static run already produced, so a
component's modal solve costs a fraction of a second rather than a re-mesh.

Validated against the one case with an exact answer, the first bending mode of a
uniform cantilever, f1 = (1.875104)^2/(2 pi) sqrt(EI/(rho A L^4)). For a
120 x 10 x 10 mm steel beam that is 580.2 Hz:

    element size   elements   f1 Hz   FE/theory
        5.0 mm          734   731.9      1.261
        4.0 mm         1179   701.1      1.208
        3.0 mm         2712   644.7      1.111
        2.2 mm         6514   619.1      1.067
        1.7 mm        13005   605.3      1.043

Monotone, converging, and converging from above -- the signature of linear
tetrahedra being too stiff. The test therefore asserts convergence rather than a
tolerance: refining must move the answer toward theory, the result must stay
above it, and the finer of two meshes must land within 15%. A fixed tolerance
would have been measuring the mesh rather than the solver.

The parser needed its own test because it was wrong. CalculiX prints mode,
eigenvalue, frequency in rad/time, frequency in cycles/time, and an imaginary
part; reading the last column took the imaginary part, identically zero for an
undamped eigenproblem, so every frequency was discarded as rigid-body and a
converged solve returned nothing at all.

Every component in the packet now reports its first mode, and the ordering is
the physical one -- 8,158 Hz for the short stubby thrust structure down to
1,342 Hz for the long thin stage 2 tank.

## The nozzle, verified against compressible CFD

The nozzle sizing has always been the ideal-rocket isentropic relations, and
nothing had ever checked them against a flow solve. The existing CFD routes
could not: they run simpleFoam, which is incompressible, and a nozzle is nothing
but compressibility. rhoCentralFoam is density-based, built for exactly this,
and was installed the whole time.

The result, and it is a clean one. A supersonic state is imposed at A/A* = 1.2
and the solver expands it to A/A* = 4.0 on its own:

    half-angle   CFD exit Mach   error vs isentropic
      36.2 deg      2.8863           -1.83%
      20.1 deg      2.9384           -0.06%
      10.4 deg      2.9402           -0.00%

Exact agreement once the nozzle is slender enough for quasi-1D theory to be the
right theory. The deficit at steep angles is not numerical error, it is nozzle
divergence loss, which quasi-1D theory does not contain.

Getting there took four failures, each of which said something specific.

*The mesh.* The obvious construction -- one block with the nozzle wall as a
polyLine edge -- does not work. blockMesh distributes points along the curved
top edge by arc length and along the straight bottom edge by x, so the two
disagree about where each column belongs and the cells shear. checkMesh found
140 negative-volume cells, max skewness 262, and 97-degree non-orthogonality;
rhoCentralFoam died in sqrt after nine steps with a max Courant number 1,950
times its mean, one tangled cell setting the timestep for the whole domain. The
fix is to mesh a plain orthogonal box and warp each point's y by h(x): every
column keeps its own x, and checkMesh reports skewness 2.1 and "Mesh OK".

*No gradient.* With the interior at chamber pressure, a fixed-value inlet at
that same pressure and an extrapolating outlet, there is no pressure gradient
anywhere. The solver ran to completion having moved nothing: exit Mach 1.8e-14
after 152 s of compute.

*The subsonic inlet.* Dropping the interior to 2% of chamber and ramping the
inlet like a valve opening got the flow started and then blew up anyway. A
diagnostic dump caught the interior at 26,347 K and 4.9 MPa against a 3,000 K,
2 MPa chamber -- hotter and at higher pressure than the reservoir feeding it,
which is thermodynamically impossible and unmistakably a boundary condition
rather than the mesh. Fixing p and T at a subsonic inlet while extrapolating U
leaves the momentum flux under-determined. Lowering the Courant number,
switching to Minmod, and adding a wave-transmissive outlet each only moved the
moment it blew up, which is how it became clear the timestep was not the
problem.

The well-posed version drops the subsonic region entirely. A supersonic inlet
has every characteristic entering, so fixing p, T and U there is the correct and
complete specification; a supersonic outlet has every characteristic leaving, so
extrapolating everything is correct. Both boundaries are then exactly determined
and nothing is under-specified. What is being tested is unchanged in substance:
only the inlet state is given, and the expansion is the solver's own.

*The wrong plane, and the wrong component.* Two post-processing bugs that would
each have produced a plausible-looking wrong number. blockMesh numbers cells
with i varying fastest, so averaging "the last few percent" of the cell list
takes the top row along the whole wall rather than the exit plane. And the
area-Mach relation predicts the speed, not its projection on the axis: in a
diverging nozzle the flow has a real radial component, and averaging Ux instead
of |U| understates the exit Mach.

## The conditioning contract, and three slots that were never reaching the model

Adding a conditioning slot is three edits in three files: the slot goes in
`CONDITIONING_QUANTITIES`, the value gets computed in the corpus, and the value
has to appear in the *shard metrics* under exactly the slot's name -- because
the graph ingest seeds conditioning by looking slot names up in a node's own
properties. Miss the third and nothing breaks: the run succeeds, the corpus
holds the value, and the model conditions on a zero.

`nose_wave_factor` was doing exactly that. Computed from a validated integral,
written into every corpus record, and invisible.

Fixing it was one line. Writing the test that would have caught it found two
more immediately. The corpus emits `delta_v_ideal_m_s` and `max_q_pa`; the slots
are named `delta_v_ms` and `max_dynamic_pressure_kpa`. Neither had ever been
populated -- including mission delta-v, which is the single most descriptive
number about a launch vehicle. The second is a unit change as well as a rename,
so a name-only fix would have been wrong by a factor of 1000.

An audit of all 42 slots against the graph found four with zero occurrences
anywhere in 327,700 nodes: `nose_wave_factor` and the three inertia components.
A first pass with grep claimed ten, which was wrong -- several are populated
through `FAMILY_PHYSICS` target names rather than as literal keys, so the graph
had to be checked rather than the source.

`Ixx/Iyy/Izz` are exactly computable from geometry that is already being built,
and they are what attitude control and stability need, so `backends` gained
`mass_properties`. Against an analytic 10 x 20 x 30 mm aluminium box, mass and
all three principal moments agree to 0.0000%.

Then the guard earned itself twice over. A 1500-record generation launched
before the `delta_v_ms` fix finished *after* a corrected 600-record run, wrote
the same files, and silently reverted the corpus. The contract test failed by
name on exactly the two slots. The generator now takes an exclusive lock and
refuses to run alongside another, and writes through a temporary file and a
rename so a reader never sees a half-written corpus.

## Coupons are not the vehicle

Worth stating plainly, because a packet section briefly claimed otherwise.

The CAD body radius is clamped to 50 mm so the parts stay meshable. The
trajectory's reference diameter for the 25 kg / 4,000 km mission is 569 mm. That
is a factor of 8 in radius, and it means the six analysed parts carry 672 g of
structure against the 151 kg the planner sized for the same vehicle. Every
stress, mode and margin in the component table is a statement about a
representative section, not about the flight article.

This is a deliberate trade, not an oversight: meshing a 569 mm shell at a 3 mm
wall with three elements through the thickness is a very different computation
from meshing a 71 mm one, and the loop would not close in reasonable time. But
the packet has to say which object each number is about, so it now reports the
coupon stack and the flight vehicle as two separate sections with the scale
factor written out between them.

The flight vehicle is built only from numbers the planner produced -- stage
lengths from propellant volume at LOX/RP-1 bulk density, each stage a uniform
cylinder of its wet mass:

    length             4.16 m
    diameter           569 mm
    wet mass           1106.6 kg   (planner gross 1106.6 kg)
    centre of gravity  1.841 m from the aft end, 44% of length
    pitch inertia      1288.7 kg m^2
    roll inertia       44.8 kg m^2

The wet mass reproducing the planner's gross exactly is what confirms this
describes the same vehicle that flew the trajectory, and it is asserted rather
than eyeballed. Pitch inertia is 29x roll, as it must be for something 4.16 m
long and 0.57 m across.

The stage model is coarse: a real stage has domes, a dry engine at one end, and
a propellant level that moves through the burn. Centre of pressure is not
reported at all, because it needs the Barrowman set and half-remembered
coefficients do not belong next to numbers that are exact.

## Centre of pressure, derived rather than looked up

I first declined to report one at all, on the grounds that the Barrowman
coefficients are exactly the kind of half-remembered constant that had already
produced one wrong answer in this document. That was the wrong call. Slender-body
theory gives the nose contribution from its volume,

    X_cp = L - V_nose / A_base

measured from the tip, and the nose volume is computed exactly from the meridian.
Nothing has to be recalled.

It reproduces the two families whose values are exact constants -- a cone at
2L/3 and a von Karman ogive at L/2, both to the last digit -- and it is better
than the table for the third. Books quote a single 0.466 L for a tangent ogive.
The real value depends on fineness:

    fineness  1.0    0.4300
    fineness  2.5    0.4606
    fineness  5.0    0.4651
    fineness  8.0    0.4661

0.466 is the slender limit. Below fineness 4, which is where sounding-rocket
noses actually live, the tabulated number is simply wrong and this one is not.

The validity boundary then reappeared unprompted: an elliptical nose gives
0.333 L against a tabulated 0.5, because it meets the axis with infinite slope
and slender-body theory does not describe a blunt nose. That is the same shape,
excluded for the same reason, as in the wave-drag work -- two independent
derivations disagreeing with the tables on the same nose, for the same stated
cause.

A cylinder generates no normal force in this theory, so a finless vehicle's
centre of pressure is its nose's. For the sized vehicle that puts the CP at
z = 3.468 m against a CG at 1.841 m -- forward of it, a static margin of -2.86
calibers, unstable. That is the correct answer and the reason fins exist.

Fin centre of pressure is *not* implemented. The Barrowman fin set is
semi-empirical and I have no independent way to check it here, and the pattern
in this document is unambiguous: every formula shipped without a validation
route turned out to be wrong. So no full-vehicle static margin is claimed. What
would settle it is a CFD normal-force run on the fins, which is a real piece of
work rather than a missing constant.
