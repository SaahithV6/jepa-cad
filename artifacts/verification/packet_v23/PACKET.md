# Design packet

**Specification:** deliver 25 kg payload to 4000 km apogee using lox/rp1 at 55 bar chamber pressure

## Architecture

- mission needs about 9.38 km/s; at Isp 325 s a single stage would need mass ratio 18.90
- that exceeds the 2.37 a stage can close, so at least 4 stages are required
- 4 stage(s) closes the mission at 1806.6 kg gross; simpler architectures are preferred so this is selected

## Structural mass closure

Designed at a structural coefficient of **0.140**. Solving it as a fixed point -- size the vehicle, size its walls from the resulting loads, recompute the coefficient, repeat -- converges to **0.249**.

> The mass model wants 0.249 where the design asserts 0.140, so this vehicle is optimistic: built to its own structural model it would be heavier than planned and would fall short of the target. Re-run with `--solve-structure` to design at the solved value. It is reported rather than silently absorbed because it changes the architecture -- a heavier structure closes fewer stages, so the same mission needs more of them.


## Vehicle: 4 stage(s)

| stage | propellant | structure | expansion |
|---|---|---|---|
| 1 | 973.89 kg | 344.53 kg | 12 |
| 2 | 253.21 kg | 89.58 kg | 30 |
| 3 | 65.83 kg | 23.29 kg | 60 |
| 4 | 23.13 kg | 8.18 kg | 80 |

payload 25.0 kg, gross 1806.6 kg


**What sets the wall thicknesses.** 5 of 10 components are at minimum gauge rather than sized by strength, carrying 29% of the analysed structural mass. Those walls cannot be thinned -- they are already as thin as the process allows -- so a structural coefficient above flown practice is partly a consequence of building a small vehicle rather than a design fault. The levers that remain are fewer stages, a larger vehicle, or a material with a lower minimum gauge; thinning walls is not one of them.


At minimum gauge: nose cone, fin set, stage 3 tank, interstage 3/4, stage 4 tank.


**Structure against flown hardware.** Stage 1 structural coefficient is 0.2613. Ten flown stages from Saturn V's S-IC to Electron's first stage span 0.036 to 0.118, median 0.080. Above the flown range 0.036-0.118, by 121%. The reference set contains no stage below 10200 kg wet and this one is 1318 kg, so a heavier fraction is expected: tank mass follows area while propellant follows volume. The comparison is an extrapolation and does not by itself indicate an error. Reference figures are secondary-source and unverified against primary mass statements; treat as a regime check, not a validation.

## Mission verification

Flown apogee **4118.3 km** against 4000 km requested (**3.0%** error); downrange 82.3 km, max-Q 87.6 kPa, separations 51.0 s, 125.0 s, 212.6 s.

Peak axial acceleration **6.6 g** (stage 1 6.6 g, stage 2 5.1 g, stage 3 3.5 g, stage 4 3.2 g). This is what sizes the structure, and it is a property of the architecture rather than a choice: thrust is held while mass falls, so acceleration climbs through each burn.


## Nozzle

| quantity | value |
|---|---|
| throat / exit radius | 44.1 / 152.7 mm |
| area ratio | 12.0 |
| contour | 80% bell, 324 mm long, exiting at 9 deg |
| divergence efficiency | 0.9938 |
| wall / mass | 1.50 mm, 2.82 kg of Inconel |

The contour is a quadratic pinned by four constraints that are all given or forced -- throat radius, the exit radius the area ratio demands, and the flow angle at each end -- so nothing about it is read off a chart. Its shape now has a consequence: divergence loss multiplies thrust, and a 25 degree exit would cost 4.7% of specific impulse against this one's 0.6%. The wall is offset along the surface normal, so it is constant-thickness sheet rather than thinning where the contour is steep.

## Thermal

| quantity | value |
|---|---|
| throat diameter | 88.2 mm |
| throat heat flux | 32.4 MW/m^2 at a 800 K wall |
| adiabatic wall temperature | 3563 K |
| total heat into the walls | 3435 kW, 2.30% of exhaust power |
| regenerative cooling | rp1 rises 294 K to 594 K (limit 700 K) |
| cooling closes | **yes**, margin +106 K |

Gas properties are the real equilibrium mixture's -- Prandtl 0.591, Reynolds 2.72e+06 -- not a textbook value for air. The convective correlation is the standard turbulent form; what makes it checkable is its scalings, and heat flux is asserted to go as throat diameter to the -0.200 and chamber pressure to the 0.8. The cooling check is an energy balance and involves no correlation at all.
| peak skin temperature | 863 K at 30.9 km |

> The skin reaches 863 K. That is inside the 920 K service limit of inconel-718 -- the repair loop selected that alloy for this reason, having started from aluminium, which stops being useful at 450 K. The margins below still use a room-temperature allowable, so they are optimistic by an amount the catalogue has no data to quantify. This is a radiation-equilibrium steady state and the vehicle passes through quickly, so it is an upper bound rather than what the structure actually reaches.
## Component verification

| component | load case | load | wall | driver | buckling margin | shell p95 | p99 | peak | 1st mode | mass | Izz | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nose cone | drag plus normal force at max-Q, 1.5 ultimate | 331 N | 0.80 mm | minimum gauge | 340.15x | 1.7 MPa | 2 | 2 MPa | 1768 Hz | 198 g | 3.42e-04 | PASS |
| thrust structure | engine thrust into the aft ring | 67415 N | 2.20 mm | strength | 14.81x | 166.9 MPa | 250 | 535 MPa | 4249 Hz | 321 g | 5.37e-04 | PASS |
| fin set | one fin's share of fin normal force at 5 deg, 1.5 ultimate | 200 N | 0.80 mm | minimum gauge | 563.14x | 12.4 MPa | 25* | 74 MPa | 336 Hz | 362 g | 8.53e-04 | PASS |
| stage 1 tank | carries 973.9 kg propellant at 6.6 g | 117774 N | 3.85 mm | strength | 27.42x | 148.0 MPa | 212 | 556 MPa | 1911 Hz | 1416 g | 2.28e-03 | PASS |
| interstage 1/2 | transmits lower-stage thrust to the stage above | 117774 N | 3.85 mm | strength | 27.42x | 152.4 MPa | 201 | 274 MPa | 4380 Hz | 683 g | 1.10e-03 | PASS |
| stage 2 tank | carries 253.2 kg propellant at 5.1 g | 24214 N | 0.86 mm | strength | 5.54x | 134.7 MPa | 139 | 173 MPa | 1053 Hz | 302 g | 4.41e-04 | PASS |
| interstage 2/3 | transmits lower-stage thrust to the stage above | 24214 N | 0.86 mm | strength | 5.54x | 136.6 MPa | 154 | 333 MPa | 2629 Hz | 146 g | 2.13e-04 | PASS |
| stage 3 tank | carries 65.8 kg propellant at 3.5 g | 4981 N | 0.80 mm | minimum gauge | 23.36x | 33.4 MPa | 35 | 40 MPa | 1081 Hz | 257 g | 3.12e-04 | PASS |
| interstage 3/4 | transmits lower-stage thrust to the stage above | 4981 N | 0.80 mm | minimum gauge | 23.36x | 32.8 MPa | 39 | 72 MPa | 2536 Hz | 124 g | 1.51e-04 | PASS |
| stage 4 tank | carries 23.1 kg propellant at 3.2 g | 1788 N | 0.80 mm | minimum gauge | 66.21x | 13.7 MPa | 25 | 359 MPa | 1205 Hz | 232 g | 2.30e-04 | PASS |

## Coupon stack (what was analysed)

The six analysed parts stacked nose-forward: 1.289 m in 10 sections, 4040 g of structure, centre of gravity 0.464 m from the aft end, Ixx 4.392e-01 and Izz 6.833e-03 kg m^2.

These are **coupons, not the vehicle**. Body radius is clamped to 50 mm so the parts stay meshable, while this mission's reference diameter is 670 mm -- a factor of 8.0 in radius, so the coupons carry 4040 g against the 466 kg of structure the planner sized. The stresses and modes above are about representative sections; the mass properties that describe the flight vehicle are below.

## Flight vehicle

| quantity | value |
|---|---|
| length | 4.37 m |
| diameter | 670 mm |
| wet mass | 1806.6 kg |
| centre of gravity from aft end | 1.861 m (43% of length) |
| pitch/yaw inertia Ixx | 2163.1 kg m^2 |
| roll inertia Izz | 101.3 kg m^2 |

## Flight loads on the assembly

| quantity | value |
|---|---|
| load case | max-Q 87.6 kPa at 5 deg incidence |
| aerodynamic side load | nose 5.4 kN + fins 15.0 kN |
| peak bending moment | 6.9 kN m at 1.83 m from aft (42% of length) |
| skin stress at that station | axial 35.4 + bending 24.5 = **59.9 MPa** on a 0.80 mm wall |
| that wall | sized here for the flight loads at 335 mm radius, driven by **minimum gauge** -- not carried over from the coupons, which are built at a clamped radius |
| margin against 700 MPa allowable | 11.70 |
| load set closes | shear -8.84e-12 N, moment 3.24e-01 N m -> **True** |

The load set is checked for closure rather than assumed to balance: a free vehicle must return shear and moment to zero at its aft end, and a distribution that does not still draws a smooth and entirely plausible moment curve.


## Skin buckling

| quantity | value |
|---|---|
| radius over thickness | 419 |
| classical critical stress | 292.2 MPa (a perfect cylinder; no real shell reaches it) |
| knockdown, compression / bending | 0.350 / 0.472 (NASA SP-8007) |
| allowable, compression / bending | 102.2 / 138.0 MPa |
| interaction R_c + R_b | 0.52 (must not exceed 1) |
| buckling margin | **1.91**, governed by compression |
| yield margin for comparison | 11.7 |

Buckling governs. The yield margin of 11.7 overstates the real one by a factor of 6.1, and every component margin in the table above is a yield margin computed the same way.

- Internal pressure is not credited. A pressurised tank buckles at a substantially higher stress than this, so the result is conservative for a tank and correct as written for an unpressurised interstage.


## Bending modes of the assembly

| quantity | value |
|---|---|
| first elastic bending mode | **49.5 Hz** free-free |
| next modes | 132.3 Hz, 249.7 Hz, 396.4 Hz |
| rigid-body modes found | 2 (a free planar beam has exactly 2) -> **True** |
| section | 0.80 mm wall at 335 mm radius, E 200 GPa |

Control bandwidth has to sit well below 49.5 Hz -- the usual allowance is a factor of five to ten -- or the autopilot drives the structure instead of steering it. This packet does not size a control system, so that comparison is left open rather than claimed as satisfied.


## Propellant slosh

| tank | fill | slosh mode | participating mass | vs first bending |
|---|---|---|---|---|
| stage 1 | 90% | 2.48 Hz | 55 kg (6% of liquid) | ratio 0.050 |
| stage 1 | 50% | 2.48 Hz | 55 kg (11% of liquid) | ratio 0.050 |
| stage 1 | 20% | 2.47 Hz | 54 kg (28% of liquid) | ratio 0.050 |
| stage 2 | 90% | 2.48 Hz | 55 kg (24% of liquid) | ratio 0.050 |
| stage 2 | 50% | 2.43 Hz | 52 kg (41% of liquid) | ratio 0.049 |
| stage 2 | 20% | 2.00 Hz | 36 kg (70% of liquid) | ratio 0.040 |
| stage 3 | 90% | 2.10 Hz | 39 kg (66% of liquid) | ratio 0.043 |
| stage 3 | 50% | 1.69 Hz | 25 kg (77% of liquid) | ratio 0.034 |
| stage 3 | 20% | 1.10 Hz | 11 kg (83% of liquid) | ratio 0.022 |
| stage 4 | 90% | 1.38 Hz | 17 kg (81% of liquid) | ratio 0.028 |
| stage 4 | 50% | 1.04 Hz | 10 kg (83% of liquid) | ratio 0.021 |
| stage 4 | 20% | 0.66 Hz | 4 kg (84% of liquid) | ratio 0.013 |

Closest approach to the first bending mode is a ratio of 0.050 -- separated by more than an octave.

Structural coupling is not a concern at these frequencies. Control coupling may still be: the slosh modes here sit at a few hertz, which is where launch vehicle control bandwidth normally lives, and this packet does not size a control system. Baffles are not modelled.


## Control authority

| quantity | value |
|---|---|
| condition | max-Q 87.6 kPa at 5 deg incidence |
| aerodynamic moment about the CG | 12.3 kN m |
| thrust x gimbal arm | 51.9 kN x 1.86 m |
| gimbal deflection required | **7.3 deg** |
| assumed available | 8.0 deg (typical production engine, not a hardware specification) |
| authority | **True** (utilisation 0.91) |

- The vehicle is statically stable, so this deflection is spent overcoming its own fins rather than correcting an instability. Larger fins buy static margin and cost control authority.

| frequency | value |
|---|---|
| rigid-body pitch (weathercock) at max-Q | 1.28 Hz |
| stage 1 slosh at max-Q | 2.48 Hz |
| ratio | **1.93** |
| first bending | 49.5 Hz |
| usable control band | No usable band: flying the vehicle needs at least 3.85 Hz while the slosh caps bandwidth at 0.50 Hz. The slosh lies inside the required band rather than above it, so added damping cannot open it: this needs a notch filter at that frequency, hardware that moves the mode, or an autopilot designed to fly through it. |
| lowest flexible mode / rigid-body mode | **1.93** |
| verdict robust to the separation rules | **True** |

The verdict holds across every rigid-body margin from 1.5 to 3 and every flexible separation from 1.5 to 5, so it is a property of the vehicle rather than of the factors chosen. The lowest flexible mode sits at 1.93 times the rigid-body mode; below about 3 there is no bandwidth that dominates one without exciting the other.


**Baffles cannot close this.** the control bandwidth this vehicle needs (3.85 Hz) is above its slosh frequency (2.48 Hz), so the mode lies inside the control band rather than above it. Baffles add damping but do not move the frequency, so no baffle closes this. It needs a notch filter at the slosh frequency, a tank that sloshes elsewhere, or an autopilot designed to fly through the mode


| mode | frequency | vs crossover | stabilisation |
|---|---|---|---|
| slosh | 2.48 Hz | 0.64x | **phase** |
| first bending | 49.45 Hz | 12.84x | **gain** |

slosh sits below crossover and must be phase stabilised: the control design has to model it rather than filter it out. Standard practice, and not something this packet can verify, since it does not design a control system.


**Static margin was traded for control authority.** Fins sized purely for stability left the engine short of gimbal; the loop searched downward until it could steer.

| target margin | fin span | CNa | gimbal needed | ok |
|---|---|---|---|---|
| 1.50 cal | 892 mm | 13.69 /rad | 22.60 deg | no |
| 1.40 cal | 800 mm | 12.06 /rad | 18.42 deg | no |
| 1.30 cal | 726 mm | 10.78 /rad | 15.20 deg | no |
| 1.20 cal | 666 mm | 9.74 /rad | 12.63 deg | no |
| 1.10 cal | 615 mm | 8.89 /rad | 10.54 deg | no |
| 1.00 cal | 572 mm | 8.17 /rad | 8.79 deg | no |
| 0.90 cal | 534 mm | 7.56 /rad | 7.31 deg | yes |

Fins resized for 0.90 calibers of static margin, down from 1.50, so the engine can trim the vehicle at max-Q within 8.0 degrees.



## Stage separation

| separation | spent | upper | closing rate | coast to clear |
|---|---|---|---|---|
| 1/2 | 344.5 kg | 488.2 kg | 2.05 m/s | **0.49 s** |
| 2/3 | 89.6 kg | 145.4 kg | 1.94 m/s | **0.52 s** |
| 3/4 | 23.3 kg | 56.3 kg | 1.70 m/s | **0.59 s** |

Plume clearance is taken as 1.5 body diameters (1.00 m), because a vacuum plume spreads well beyond the nozzle that produced it. Tip-off and plume impingement on the spent stage are not modelled; this answers only whether the gap opens fast enough.


## Stability

| quantity | value |
|---|---|
| fin span (each of 4) | 534 mm |
| fin root / tip chord | 670 / 335 mm, sweep 402 mm |
| fin drag, absent from the flown trajectory | Cd 0.0190 on body frontal area, **5%** of the body's 0.378 |
| fin planform vs body frontal area | 3.0x |

The apogee under mission verification is a finless vehicle's. Adding this drag costs roughly 1% of apogee, about 23 kg of gross mass to recover -- small against the other uncertainties here, and always in the optimistic direction. Interference drag where the fins meet the body is not included. It is real and positive, so this is a floor on fin drag rather than an estimate of it. Skin friction uses a single representative coefficient of 0.003; the true value varies by about a factor of two across an ascent.

| centre of pressure | 1.258 m from aft |
| centre of gravity | 1.861 m from aft |
| static margin | 0.90 calibers (target 0.9, sized for liftoff) |
| normal force slope | nose 2.00 + fins 5.56 = 7.56 /rad |

| burn state | vehicle mass | centre of gravity | static margin |
|---|---|---|---|
| liftoff | 1806.6 kg | 1.861 m | 0.90 cal |
| stage 1 burnout | 832.8 kg | 2.454 m | 1.78 cal |

Fins are sized by solving for the span that meets the margin, not assumed and then checked. The nose centre of pressure comes from slender-body theory as L - V/A_base, which needs only the nose volume and reproduces the exact families (cone 2L/3, von Karman L/2) to the last digit. The fin set is Barrowman, whose CN_alpha converges onto Jones' slender-wing result pi AR/2 as aspect ratio goes to zero and whose unswept-rectangular centre of pressure is exactly the quarter chord -- two limits with known answers, which is what makes its constants checkable rather than merely quoted.

## Assembly

The vehicle built as geometry: 5.61 m over 13 parts, 13 of them exported to STEP under `cad/`.

| part | kind | station | length | mass |
|---|---|---|---|---|
| nozzle | revolve | 0.000 m | 324 mm | 0.95 kg |
| stage 1 tank | shell | 0.000 m | 2709 mm | 45.97 kg |
| fin 1 | extrude | 0.167 m | 670 mm | 7.73 kg |
| fin 2 | extrude | 0.167 m | 670 mm | 7.73 kg |
| fin 3 | extrude | 0.167 m | 670 mm | 7.73 kg |
| fin 4 | extrude | 0.167 m | 670 mm | 7.73 kg |
| interstage 1/2 | loft | 2.709 m | 208 mm | 3.25 kg |
| stage 2 tank | shell | 2.917 m | 832 mm | 12.99 kg |
| interstage 2/3 | loft | 3.749 m | 191 mm | 2.75 kg |
| stage 3 tank | shell | 3.940 m | 256 mm | 3.67 kg |
| interstage 3/4 | loft | 4.196 m | 176 mm | 2.32 kg |
| stage 4 tank | shell | 4.372 m | 200 mm | 2.64 kg |
| nose cone | revolve | 4.572 m | 1043 mm | 9.98 kg |

### Does the mass budget hold the vehicle?

| term | mass |
|---|---|
| skin, as drawn | 115.5 kg |
| engine, from liftoff thrust at T/W 60 | 88.1 kg |
| accounted for | 203.6 kg |
| structural budget | 465.6 kg |
| slack | +262.0 kg |

> The budget holds, with 262.0 kg left for plumbing, avionics, tank domes and separation hardware -- none of which the geometry draws.

Stage lengths come from propellant volume at LOX/RP-1 bulk density and each stage is a uniform cylinder of its wet mass -- coarse, since a real stage has domes, a dry engine at one end and a moving liquid level, but built only from numbers the planner produced, and the wet mass reproduces the planner's gross of 1806.6 kg. Pitch inertia is 21x roll, as it must be for a long thin vehicle.

Sizing is on p95. Across a 16x mesh refinement on one part the median moved 2.8%, p95 13.8%, p99 36.6% and the peak 268% -- p99 is not stable enough to design against at the mesh densities this loop can afford, because with ~1,200 nodes its top percent is a dozen nodes all on the same corner. A starred p99 marks a part whose p99 exceeds twice its p95: that is a stress concentration wanting a fillet or a doubler, not a wall that wants thickening. The peak column never converges and is shown only to locate it.

Wall thickness and buckling margin size the thin shell; the shell FEA column is CalculiX on that same hollow geometry, so the meshed part is the part being designed rather than a solid billet with the same outer dimensions. The 1st mode column is a CalculiX *FREQUENCY solve on that same mesh, clamped at the aft face: it is what a static check cannot see, and it is the quantity a flutter or coupled-loads assessment starts from. Mass and Izz are computed on that same solid, exact for the geometry, in kg and kg m^2 about the centroid.

## Flight barrel, at flight scale

| quantity | value |
|---|---|
| mesh | 768 shell elements, 800 nodes |
| membrane stress | 35.61 MPa |
| closed form | 35.40 MPa (+0.61%) |
| peak, at the clamped end | 37.28 MPa |

This is the flight barrel, not a coupon. The component table above analyses parts at 42 mm radius because a 0.80 mm wall cannot be resolved with solid elements at 335 mm -- it would take on the order of a billion tetrahedra. Shell elements carry the thickness as a property, so the same barrel is a few hundred elements.

- Compared on the middle 90% of nodes: the clamped base introduces a bending boundary layer the membrane solution does not model, and it perturbs the stress in opposite directions under axial load and under pressure.

| stage | length | wall | driver | axial load | stress | vs closed form |
|---|---|---|---|---|---|---|
| 1 | 2.71 m | 0.80 mm | minimum gauge | 32 kN | 19.0 MPa | +0.62% |
| 2 | 0.70 m | 0.80 mm | minimum gauge | 9 kN | 5.7 MPa | +0.52% |
| 3 | 0.30 m | 0.80 mm | minimum gauge | 4 kN | 2.2 MPa | +0.49% |
| 4 | 0.30 m | 0.80 mm | minimum gauge | 2 kN | 1.0 MPa | +0.45% |

Each barrel is sized for the mass it supports at peak axial acceleration, at the flight radius -- not carried over from the coupons, whose walls were sized at 42 mm and mean nothing here. Stage 1 is worst at 19.0 MPa against a 700 MPa allowable.



## Assembly verification

| check | result | detail |
|---|---|---|
| skin buckling | **PASS** | interaction 0.52, margin 1.91 governed by compression |
| slosh / pitch-mode separation | **PASS** | slosh 2.48 Hz against pitch 1.28 Hz, ratio 1.93 |
| flexible mode stabilisation | **REQUIRED** | slosh below crossover; phase stabilisation required and not verified here |
| thrust vector control authority | **PASS** | 7.3 deg required against 8.0 available |
| stage separation clearance | **PASS** | 3 separation(s), longest coast to plume clearance 0.59 s |
| flight barrel stress | **PASS** | 768 shell elements at 335 mm radius: 35.6 MPa against a 700 MPa allowable, +0.6% from closed form; worst of 4 per-stage barrels 19.0 MPa |

1 check(s) marked REQUIRED are neither passed nor failed: they name real work this packet cannot do. Claiming they pass would report checks that never ran; claiming they fail would suggest defects where there are none.

- **flexible mode stabilisation** -- slosh below crossover; phase stabilisation required and not verified here.


Allowable 700 MPa, derived from inconel-718 at 1030 MPa (typical (non-statistical)) with a yield factor of safety of 1.25 and a 0.85 knockdown. All 10 coupons passed: **True**. Assembly checks failed: **0**. Requirements unverified: **1**.

Overall: **INCOMPLETE** -- nothing failed, but the packet cannot call a design verified while a requirement it did not check remains outstanding.


This allowable is not certifiable and the packet should not be read as though it were. The catalogue carries typical strengths rather than A- or B-basis values, so the knockdown above stands in for a statistical basis that does not exist here:

- Catalogue strength is a typical value, not an A- or B-basis allowable. A knockdown stands in for the statistical basis and is a judgement, not a tolerance bound.
- Allowable is a room-temperature value applied at 863 K. No yield-versus-temperature data is available for this material, so the real hot allowable is lower by an unquantified amount.

Discretisation error is measured separately and is not included in any margin quoted here. Against an exact Lame solution (`artifacts/verification/fea_mesh_convergence.json`) the C3D4 linear tetrahedra this pipeline writes converge first-order in stress (p = 1.31) and read 9.8% low on surface stress at 34,493 elements, against a 40,000 element budget; quadratic C3D10 elements reach a lower field error with 661.

On the real corpus parts that number is larger and two-sided. Solving twelve components at identical meshes and loads under both element types (`artifacts/verification/element_order_ab.json`) moved the p95 this loop sizes against by a median of 1.1% but a range of -13.9% to +14.5%. The small median belongs to smooth parts like body tubes; the double-digit ends belong to fins and nose cones, where the field is dominated by stress concentrations. Read the component margins below as carrying at least that much numerical uncertainty.
