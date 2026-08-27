# Design packet

**Specification:** deliver 200 kg payload to 400 km apogee using lox/rp1 at 55 bar chamber pressure

## Architecture

- mission needs about 3.67 km/s; at Isp 325 s a single stage would need mass ratio 3.16
- that exceeds the 2.52 a stage can close, so at least 2 stages are required
- 2 stage(s) closes the mission at 1248.0 kg gross; simpler architectures are preferred so this is selected

## Structural mass closure

Designed at a structural coefficient of **0.140**. Solving it as a fixed point -- size the vehicle, size its walls from the resulting loads, recompute the coefficient, repeat -- converges to **0.230**.

> The mass model wants 0.230 where the design asserts 0.140, so this vehicle is optimistic: built to its own structural model it would be heavier than planned and would fall short of the target. Re-run with `--solve-structure` to design at the solved value. It is reported rather than silently absorbed because it changes the architecture -- a heavier structure closes fewer stages, so the same mission needs more of them.


## Vehicle: 2 stage(s)

| stage | propellant | structure | expansion |
|---|---|---|---|
| 1 | 584.92 kg | 190.60 kg | 12 |
| 2 | 205.51 kg | 66.97 kg | 30 |

payload 200.0 kg, gross 1248.0 kg


**What sets the wall thicknesses.** 3 of 6 components are at minimum gauge rather than sized by strength, carrying 35% of the analysed structural mass. Those walls cannot be thinned -- they are already as thin as the process allows -- so a structural coefficient above flown practice is partly a consequence of building a small vehicle rather than a design fault. The levers that remain are fewer stages, a larger vehicle, or a material with a lower minimum gauge; thinning walls is not one of them.


At minimum gauge: nose cone, fin set, stage 2 tank.


**Structure against flown hardware.** Stage 1 structural coefficient is 0.2458. Ten flown stages from Saturn V's S-IC to Electron's first stage span 0.036 to 0.118, median 0.080. Above the flown range 0.036-0.118, by 108%. The reference set contains no stage below 10200 kg wet and this one is 776 kg, so a heavier fraction is expected: tank mass follows area while propellant follows volume. The comparison is an extrapolation and does not by itself indicate an error. Reference figures are secondary-source and unverified against primary mass statements; treat as a regime check, not a validation.

## Mission verification

Flown apogee **401.6 km** against 400 km requested (**0.4%** error); downrange 16.2 km, max-Q 83.5 kPa, separations 45.0 s.

Peak axial acceleration **5.2 g** (stage 1 5.2 g, stage 2 4.2 g). This is what sizes the structure, and it is a property of the architecture rather than a choice: thrust is held while mass falls, so acceleration climbs through each burn.


## Nozzle

| quantity | value |
|---|---|
| throat / exit radius | 36.4 / 125.9 mm |
| area ratio | 12.0 |
| contour | 80% bell, 267 mm long, exiting at 9 deg |
| divergence efficiency | 0.9938 |
| wall / mass | 1.50 mm, 1.92 kg of Inconel |

The contour is a quadratic pinned by four constraints that are all given or forced -- throat radius, the exit radius the area ratio demands, and the flow angle at each end -- so nothing about it is read off a chart. Its shape now has a consequence: divergence loss multiplies thrust, and a 25 degree exit would cost 4.7% of specific impulse against this one's 0.6%. The wall is offset along the surface normal, so it is constant-thickness sheet rather than thinning where the contour is steep.

## Thermal

| quantity | value |
|---|---|
| throat diameter | 72.7 mm |
| throat heat flux | 33.6 MW/m^2 at a 800 K wall |
| adiabatic wall temperature | 3563 K |
| total heat into the walls | 2428 kW, 2.39% of exhaust power |
| regenerative cooling | rp1 rises 306 K to 606 K (limit 700 K) |
| cooling closes | **yes**, margin +94 K |

Gas properties are the real equilibrium mixture's -- Prandtl 0.591, Reynolds 2.24e+06 -- not a textbook value for air. The convective correlation is the standard turbulent form; what makes it checkable is its scalings, and heat flux is asserted to go as throat diameter to the -0.200 and chamber pressure to the 0.8. The cooling check is an energy balance and involves no correlation at all.
| peak skin temperature | 734 K at 21.9 km |

> The skin reaches 734 K, past the 450 K at which aluminium keeps useful strength. Every allowable in the component table below is a room-temperature value, so those margins do not hold at this condition: the vehicle needs a thermal protection system, a different skin material, or a trajectory that spends less time fast in thick air. This is a radiation-equilibrium steady state and the vehicle passes through quickly, so it is an upper bound rather than what the structure actually reaches -- but it is far enough past the limit to matter.
## Component verification

| component | load case | load | wall | driver | buckling margin | shell p95 | p99 | peak | 1st mode | mass | Izz | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nose cone | drag plus normal force at max-Q, 1.5 ultimate | 246 N | 0.80 mm | minimum gauge | 467.74x | 1.5 MPa | 2 | 2 MPa | 2101 Hz | 51 g | 6.85e-05 | PASS |
| thrust structure | engine thrust into the aft ring | 45856 N | 1.70 mm | strength | 12.67x | 139.2 MPa | 212 | 558 MPa | 4800 Hz | 64 g | 8.41e-05 | PASS |
| fin set | one fin's share of fin normal force at 5 deg, 1.5 ultimate | 200 N | 0.80 mm | minimum gauge | 576.32x | 11.4 MPa | 24* | 71 MPa | 538 Hz | 88 g | 1.54e-04 | PASS |
| stage 1 tank | carries 584.9 kg propellant at 5.2 g | 63704 N | 2.36 mm | strength | 18.30x | 154.4 MPa | 196 | 436 MPa | 2234 Hz | 227 g | 2.93e-04 | PASS |
| interstage 1/2 | transmits lower-stage thrust to the stage above | 63704 N | 2.36 mm | strength | 18.30x | 151.4 MPa | 203 | 906 MPa | 4162 Hz | 109 g | 1.41e-04 | PASS |
| stage 2 tank | carries 205.5 kg propellant at 4.2 g | 19494 N | 0.80 mm | minimum gauge | 6.00x | 127.6 MPa | 136 | 183 MPa | 1261 Hz | 72 g | 8.23e-05 | PASS |

## Coupon stack (what was analysed)

The six analysed parts stacked nose-forward: 0.664 m in 6 sections, 611 g of structure, centre of gravity 0.297 m from the aft end, Ixx 1.757e-02 and Izz 8.771e-04 kg m^2.

These are **coupons, not the vehicle**. Body radius is clamped to 50 mm so the parts stay meshable, while this mission's reference diameter is 592 mm -- a factor of 8.0 in radius, so the coupons carry 611 g against the 258 kg of structure the planner sized. The stresses and modes above are about representative sections; the mass properties that describe the flight vehicle are below.

## Flight vehicle

| quantity | value |
|---|---|
| length | 3.41 m |
| diameter | 592 mm |
| wet mass | 1248.0 kg |
| centre of gravity from aft end | 1.680 m (49% of length) |
| pitch/yaw inertia Ixx | 1211.8 kg m^2 |
| roll inertia Izz | 54.7 kg m^2 |

## Flight loads on the assembly

| quantity | value |
|---|---|
| load case | max-Q 83.5 kPa at 5 deg incidence |
| aerodynamic side load | nose 4.0 kN + fins 9.3 kN |
| peak bending moment | 3.2 kN m at 1.47 m from aft (43% of length) |
| skin stress at that station | axial 24.0 + bending 14.6 = **38.6 MPa** on a 0.80 mm wall |
| that wall | sized here for the flight loads at 296 mm radius, driven by **minimum gauge** -- not carried over from the coupons, which are built at a clamped radius |
| margin against 700 MPa allowable | 18.17 |
| load set closes | shear 2.01e-12 N, moment 1.52e-01 N m -> **True** |

The load set is checked for closure rather than assumed to balance: a free vehicle must return shear and moment to zero at its aft end, and a distribution that does not still draws a smooth and entirely plausible moment curve.


## Skin buckling

| quantity | value |
|---|---|
| radius over thickness | 370 |
| classical critical stress | 330.5 MPa (a perfect cylinder; no real shell reaches it) |
| knockdown, compression / bending | 0.370 / 0.489 (NASA SP-8007) |
| allowable, compression / bending | 122.2 / 161.5 MPa |
| interaction R_c + R_b | 0.29 (must not exceed 1) |
| buckling margin | **3.49**, governed by compression |
| yield margin for comparison | 18.2 |

Buckling governs. The yield margin of 18.2 overstates the real one by a factor of 5.2, and every component margin in the table above is a yield margin computed the same way.

- Internal pressure is not credited. A pressurised tank buckles at a substantially higher stress than this, so the result is conservative for a tank and correct as written for an unpressurised interstage.


## Bending modes of the assembly

| quantity | value |
|---|---|
| first elastic bending mode | **58.3 Hz** free-free |
| next modes | 159.7 Hz, 312.8 Hz, 517.6 Hz |
| rigid-body modes found | 2 (a free planar beam has exactly 2) -> **True** |
| section | 0.80 mm wall at 296 mm radius, E 200 GPa |

Control bandwidth has to sit well below 58.3 Hz -- the usual allowance is a factor of five to ten -- or the autopilot drives the structure instead of steering it. This packet does not size a control system, so that comparison is left open rather than claimed as satisfied.


## Propellant slosh

| tank | fill | slosh mode | participating mass | vs first bending |
|---|---|---|---|---|
| stage 1 | 90% | 2.64 Hz | 38 kg (7% of liquid) | ratio 0.045 |
| stage 1 | 50% | 2.64 Hz | 38 kg (13% of liquid) | ratio 0.045 |
| stage 1 | 20% | 2.62 Hz | 37 kg (32% of liquid) | ratio 0.045 |
| stage 2 | 90% | 2.64 Hz | 38 kg (20% of liquid) | ratio 0.045 |
| stage 2 | 50% | 2.61 Hz | 37 kg (36% of liquid) | ratio 0.045 |
| stage 2 | 20% | 2.24 Hz | 27 kg (66% of liquid) | ratio 0.038 |

Closest approach to the first bending mode is a ratio of 0.045 -- separated by more than an octave.

Structural coupling is not a concern at these frequencies. Control coupling may still be: the slosh modes here sit at a few hertz, which is where launch vehicle control bandwidth normally lives, and this packet does not size a control system. Baffles are not modelled.


## Control authority

| quantity | value |
|---|---|
| condition | max-Q 83.5 kPa at 5 deg incidence |
| aerodynamic moment about the CG | 7.9 kN m |
| thrust x gimbal arm | 35.3 kN x 1.68 m |
| gimbal deflection required | **7.6 deg** |
| assumed available | 8.0 deg (typical production engine, not a hardware specification) |
| authority | **True** (utilisation 0.95) |

- The vehicle is statically stable, so this deflection is spent overcoming its own fins rather than correcting an instability. Larger fins buy static margin and cost control authority.

| frequency | value |
|---|---|
| rigid-body pitch (weathercock) at max-Q | 1.37 Hz |
| stage 1 slosh at max-Q | 2.64 Hz |
| ratio | **1.92** |
| first bending | 58.3 Hz |
| usable control band | No usable band: flying the vehicle needs at least 4.12 Hz while the slosh caps bandwidth at 0.53 Hz. The slosh lies inside the required band rather than above it, so added damping cannot open it: this needs a notch filter at that frequency, hardware that moves the mode, or an autopilot designed to fly through it. |
| lowest flexible mode / rigid-body mode | **1.92** |
| verdict robust to the separation rules | **True** |

The verdict holds across every rigid-body margin from 1.5 to 3 and every flexible separation from 1.5 to 5, so it is a property of the vehicle rather than of the factors chosen. The lowest flexible mode sits at 1.92 times the rigid-body mode; below about 3 there is no bandwidth that dominates one without exciting the other.


**Baffles cannot close this.** the control bandwidth this vehicle needs (4.12 Hz) is above its slosh frequency (2.64 Hz), so the mode lies inside the control band rather than above it. Baffles add damping but do not move the frequency, so no baffle closes this. It needs a notch filter at the slosh frequency, a tank that sloshes elsewhere, or an autopilot designed to fly through the mode


**Static margin was traded for control authority.** Fins sized purely for stability left the engine short of gimbal; the loop searched downward until it could steer.

| target margin | fin span | CNa | gimbal needed | ok |
|---|---|---|---|---|
| 1.50 cal | 642 mm | 10.77 /rad | 18.90 deg | no |
| 1.40 cal | 580 mm | 9.57 /rad | 15.59 deg | no |
| 1.30 cal | 529 mm | 8.62 /rad | 12.97 deg | no |
| 1.20 cal | 487 mm | 7.83 /rad | 10.86 deg | no |
| 1.10 cal | 450 mm | 7.18 /rad | 9.11 deg | no |
| 1.00 cal | 419 mm | 6.62 /rad | 7.63 deg | yes |

Fins resized for 1.00 calibers of static margin, down from 1.50, so the engine can trim the vehicle at max-Q within 8.0 degrees.



## Stage separation

| separation | spent | upper | closing rate | coast to clear |
|---|---|---|---|---|
| 1/2 | 190.6 kg | 472.5 kg | 1.68 m/s | **0.53 s** |

Plume clearance is taken as 1.5 body diameters (0.89 m), because a vacuum plume spreads well beyond the nozzle that produced it. Tip-off and plume impingement on the spent stage are not modelled; this answers only whether the gap opens fast enough.


## Stability

| quantity | value |
|---|---|
| fin span (each of 4) | 419 mm |
| fin root / tip chord | 592 / 296 mm, sweep 355 mm |
| fin drag, absent from the flown trajectory | Cd 0.0170 on body frontal area, **4%** of the body's 0.378 |
| fin planform vs body frontal area | 2.7x |

The apogee under mission verification is a finless vehicle's. Adding this drag costs roughly 1% of apogee, about 23 kg of gross mass to recover -- small against the other uncertainties here, and always in the optimistic direction. Interference drag where the fins meet the body is not included. It is real and positive, so this is a floor on fin drag rather than an estimate of it. Skin friction uses a single representative coefficient of 0.003; the true value varies by about a factor of two across an ascent.

| centre of pressure | 1.088 m from aft |
| centre of gravity | 1.680 m from aft |
| static margin | 1.00 calibers (target 1.0, sized for liftoff) |
| normal force slope | nose 2.00 + fins 4.62 = 6.62 /rad |

| burn state | vehicle mass | centre of gravity | static margin |
|---|---|---|---|
| liftoff | 1248.0 kg | 1.680 m | 1.00 cal |
| stage 1 burnout | 663.1 kg | 2.243 m | 1.95 cal |

Fins are sized by solving for the span that meets the margin, not assumed and then checked. The nose centre of pressure comes from slender-body theory as L - V/A_base, which needs only the nose volume and reproduces the exact families (cone 2L/3, von Karman L/2) to the last digit. The fin set is Barrowman, whose CN_alpha converges onto Jones' slender-wing result pi AR/2 as aspect ratio goes to zero and whose unswept-rectangular centre of pressure is exactly the quarter chord -- two limits with known answers, which is what makes its constants checkable rather than merely quoted.

## Assembly

The vehicle built as geometry: 4.22 m over 9 parts, 9 of them exported to STEP under `cad/`.

| part | kind | station | length | mass |
|---|---|---|---|---|
| nozzle | revolve | 0.000 m | 267 mm | 0.65 kg |
| stage 1 tank | shell | 0.000 m | 2082 mm | 31.22 kg |
| fin 1 | extrude | 0.148 m | 592 mm | 4.21 kg |
| fin 2 | extrude | 0.148 m | 592 mm | 4.21 kg |
| fin 3 | extrude | 0.148 m | 592 mm | 4.21 kg |
| fin 4 | extrude | 0.148 m | 592 mm | 4.21 kg |
| interstage 1/2 | loft | 2.082 m | 184 mm | 2.53 kg |
| stage 2 tank | shell | 2.266 m | 864 mm | 11.92 kg |
| nose cone | revolve | 3.130 m | 1090 mm | 10.90 kg |

### Does the mass budget hold the vehicle?

| term | mass |
|---|---|
| skin, as drawn | 74.1 kg |
| engine, from liftoff thrust at T/W 60 | 59.9 kg |
| accounted for | 134.0 kg |
| structural budget | 257.6 kg |
| slack | +123.5 kg |

> The budget holds, with 123.5 kg left for plumbing, avionics, tank domes and separation hardware -- none of which the geometry draws.

Stage lengths come from propellant volume at LOX/RP-1 bulk density and each stage is a uniform cylinder of its wet mass -- coarse, since a real stage has domes, a dry engine at one end and a moving liquid level, but built only from numbers the planner produced, and the wet mass reproduces the planner's gross of 1248.0 kg. Pitch inertia is 22x roll, as it must be for a long thin vehicle.

Sizing is on p95. Across a 16x mesh refinement on one part the median moved 2.8%, p95 13.8%, p99 36.6% and the peak 268% -- p99 is not stable enough to design against at the mesh densities this loop can afford, because with ~1,200 nodes its top percent is a dozen nodes all on the same corner. A starred p99 marks a part whose p99 exceeds twice its p95: that is a stress concentration wanting a fillet or a doubler, not a wall that wants thickening. The peak column never converges and is shown only to locate it.

Wall thickness and buckling margin size the thin shell; the shell FEA column is CalculiX on that same hollow geometry, so the meshed part is the part being designed rather than a solid billet with the same outer dimensions. The 1st mode column is a CalculiX *FREQUENCY solve on that same mesh, clamped at the aft face: it is what a static check cannot see, and it is the quantity a flutter or coupled-loads assessment starts from. Mass and Izz are computed on that same solid, exact for the geometry, in kg and kg m^2 about the centroid.

## Assembly verification

| check | result | detail |
|---|---|---|
| skin buckling | **PASS** | interaction 0.29, margin 3.49 governed by compression |
| slosh / pitch-mode separation | **PASS** | slosh 2.64 Hz against pitch 1.37 Hz, ratio 1.92 |
| control bandwidth window | **FAIL** | needs >= 4.12 Hz, capped at 0.53 Hz by slosh |
| thrust vector control authority | **PASS** | 7.6 deg required against 8.0 available |
| stage separation clearance | **PASS** | 1 separation(s), longest coast to plume clearance 0.53 s |

1 assembly-level check(s) failed while every component passed its own coupon test. The components are not wrong -- each one survives the load it was given. What fails is the vehicle they add up to, and no per-part analysis can see it.


Allowable 700 MPa, derived from inconel-718 at 1030 MPa (typical (non-statistical)) with a yield factor of safety of 1.25 and a 0.85 knockdown. All 6 components passed: **True**. Assembly checks passed: **False**. Overall: **False**


This allowable is not certifiable and the packet should not be read as though it were. The catalogue carries typical strengths rather than A- or B-basis values, so the knockdown above stands in for a statistical basis that does not exist here:

- Catalogue strength is a typical value, not an A- or B-basis allowable. A knockdown stands in for the statistical basis and is a judgement, not a tolerance bound.

Discretisation error is measured separately and is not included in any margin quoted here. Against an exact Lame solution (`artifacts/verification/fea_mesh_convergence.json`) the C3D4 linear tetrahedra this pipeline writes converge first-order in stress (p = 1.31) and read 9.8% low on surface stress at 34,493 elements, against a 40,000 element budget; quadratic C3D10 elements reach a lower field error with 661.

On the real corpus parts that number is larger and two-sided. Solving twelve components at identical meshes and loads under both element types (`artifacts/verification/element_order_ab.json`) moved the p95 this loop sizes against by a median of 1.1% but a range of -13.9% to +14.5%. The small median belongs to smooth parts like body tubes; the double-digit ends belong to fins and nose cones, where the field is dominated by stress concentrations. Read the component margins below as carrying at least that much numerical uncertainty.
