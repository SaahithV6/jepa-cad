# Design packet

**Specification:** deliver 1000 kg payload to 250 km apogee using lox/rp1 at 55 bar chamber pressure

## Architecture

- mission needs about 2.93 km/s; at Isp 325 s a single stage would need mass ratio 2.51
- that is within what a stage can close at structural coefficient 0.2070730938865051 (limit 2.99), so try a single stage first
- 1 stage(s) closes the mission at 4279.2 kg gross; simpler architectures are preferred so this is selected

## Structural mass closure

Designed at a structural coefficient of **0.140**. Solving it as a fixed point -- size the vehicle, size its walls from the resulting loads, recompute the coefficient, repeat -- converges to **0.251**.

> The mass model wants 0.251 where the design asserts 0.140, so this vehicle is optimistic: built to its own structural model it would be heavier than planned and would fall short of the target. Re-run with `--solve-structure` to design at the solved value. It is reported rather than silently absorbed because it changes the architecture -- a heavier structure closes fewer stages, so the same mission needs more of them.


## Vehicle: 1 stage(s)

| stage | propellant | structure | expansion |
|---|---|---|---|
| 1 | 2600.17 kg | 679.04 kg | 12 |

payload 1000.0 kg, gross 4279.2 kg


**What sets the wall thicknesses.** 2 of 4 components are at minimum gauge rather than sized by strength, carrying 11% of the analysed structural mass. Those walls cannot be thinned -- they are already as thin as the process allows -- so a structural coefficient above flown practice is partly a consequence of building a small vehicle rather than a design fault. The levers that remain are fewer stages, a larger vehicle, or a material with a lower minimum gauge; thinning walls is not one of them.


At minimum gauge: nose cone, fin set.


**Structure against flown hardware.** Stage 1 structural coefficient is 0.2071. Ten flown stages from Saturn V's S-IC to Electron's first stage span 0.036 to 0.118, median 0.080. Above the flown range 0.036-0.118, by 75%. The reference set contains no stage below 10200 kg wet and this one is 3279 kg, so a heavier fraction is expected: tank mass follows area while propellant follows volume. The comparison is an extrapolation and does not by itself indicate an error. Reference figures are secondary-source and unverified against primary mass statements; treat as a regime check, not a validation.

## Mission verification

Flown apogee **245.6 km** against 250 km requested (**1.8%** error); downrange 12.0 km, max-Q 116.0 kPa, separations none.

Peak axial acceleration **9.6 g** (stage 1 9.6 g). This is what sizes the structure, and it is a property of the architecture rather than a choice: thrust is held while mass falls, so acceleration climbs through each burn.


## Nozzle

| quantity | value |
|---|---|
| throat / exit radius | 73.6 / 255.1 mm |
| area ratio | 12.0 |
| contour | 80% bell, 542 mm long, exiting at 9 deg |
| divergence efficiency | 0.9938 |
| wall / mass | 1.50 mm, 7.84 kg of Inconel |

The contour is a quadratic pinned by four constraints that are all given or forced -- throat radius, the exit radius the area ratio demands, and the flow angle at each end -- so nothing about it is read off a chart. Its shape now has a consequence: divergence loss multiplies thrust, and a 25 degree exit would cost 4.7% of specific impulse against this one's 0.6%. The wall is offset along the surface normal, so it is constant-thickness sheet rather than thinning where the contour is steep.

## Thermal

| quantity | value |
|---|---|
| throat diameter | 147.3 mm |
| throat heat flux | 29.2 MW/m^2 at a 800 K wall |
| adiabatic wall temperature | 3563 K |
| total heat into the walls | 8652 kW, 2.07% of exhaust power |
| regenerative cooling | rp1 rises 266 K to 566 K (limit 700 K) |
| cooling closes | **yes**, margin +134 K |

Gas properties are the real equilibrium mixture's -- Prandtl 0.591, Reynolds 4.54e+06 -- not a textbook value for air. The convective correlation is the standard turbulent form; what makes it checkable is its scalings, and heat flux is asserted to go as throat diameter to the -0.200 and chamber pressure to the 0.8. The cooling check is an energy balance and involves no correlation at all.
| peak skin temperature | 979 K at 36.2 km |

> The skin reaches 979 K, past the 450 K at which aluminium keeps useful strength. Every allowable in the component table below is a room-temperature value, so those margins do not hold at this condition: the vehicle needs a thermal protection system, a different skin material, or a trajectory that spends less time fast in thick air. This is a radiation-equilibrium steady state and the vehicle passes through quickly, so it is an upper bound rather than what the structure actually reaches -- but it is far enough past the limit to matter.
## Component verification

| component | load case | load | wall | driver | buckling margin | shell p95 | p99 | peak | 1st mode | mass | Izz | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nose cone | drag plus normal force at max-Q, 1.5 ultimate | 621 N | 0.80 mm | minimum gauge | 174.97x | 2.7 MPa | 3 | 4 MPa | 1465 Hz | 93 g | 2.28e-04 | PASS |
| thrust structure | engine thrust into the aft ring | 188152 N | 5.16 mm | strength | 31.22x | 175.2 MPa | 234 | 722 MPa | 4933 Hz | 343 g | 7.73e-04 | PASS |
| fin set | one fin's share of fin normal force at 5 deg, 1.5 ultimate | 200 N | 0.80 mm | minimum gauge | 543.70x | - | - | hull | - | 163 g | 5.26e-04 | FAIL |
| stage 1 tank | carries 2600.2 kg propellant at 9.6 g | 401551 N | 11.02 mm | strength | 70.34x | 155.6 MPa | 177 | 247 MPa | 1629 Hz | 1768 g | 3.55e-03 | PASS |

## Coupon stack (what was analysed)

The six analysed parts stacked nose-forward: 0.575 m in 4 sections, 2366 g of structure, centre of gravity 0.300 m from the aft end, Ixx 4.200e-02 and Izz 5.292e-03 kg m^2.

These are **coupons, not the vehicle**. Body radius is clamped to 50 mm so the parts stay meshable, while this mission's reference diameter is 893 mm -- a factor of 8.9 in radius, so the coupons carry 2366 g against the 679 kg of structure the planner sized. The stresses and modes above are about representative sections; the mass properties that describe the flight vehicle are below.

## Flight vehicle

| quantity | value |
|---|---|
| length | 4.96 m |
| diameter | 893 mm |
| wet mass | 4279.2 kg |
| centre of gravity from aft end | 2.615 m (53% of length) |
| pitch/yaw inertia Ixx | 9528.3 kg m^2 |
| roll inertia Izz | 426.5 kg m^2 |

## Flight loads on the assembly

| quantity | value |
|---|---|
| load case | max-Q 116.0 kPa at 5 deg incidence |
| aerodynamic side load | nose 12.7 kN + fins 31.1 kN |
| peak bending moment | 15.0 kN m at 2.15 m from aft (43% of length) |
| skin stress at that station | axial 56.8 + bending 16.0 = **72.8 MPa** on a 1.50 mm wall |
| that wall | sized here for the flight loads at 446 mm radius, driven by **buckling** -- not carried over from the coupons, which are built at a clamped radius |
| margin against 188 MPa allowable | 2.58 |
| load set closes | shear -5.54e-12 N, moment 7.93e-01 N m -> **True** |

The load set is checked for closure rather than assumed to balance: a free vehicle must return shear and moment to zero at its aft end, and a distribution that does not still draws a smooth and entirely plausible moment curve.


## Skin buckling

| quantity | value |
|---|---|
| radius over thickness | 297 |
| classical critical stress | 141.7 MPa (a perfect cylinder; no real shell reaches it) |
| knockdown, compression / bending | 0.406 / 0.518 (NASA SP-8007) |
| allowable, compression / bending | 57.5 / 73.3 MPa |
| interaction R_c + R_b | 1.21 (must not exceed 1) |
| buckling margin | **0.83**, governed by compression |
| yield margin for comparison | 2.6 |

**The skin buckles.** Interaction is 1.21 against a limit of 1. The yield margin of 2.6 above is real and irrelevant: this wall folds before it yields.

- Internal pressure is not credited. A pressurised tank buckles at a substantially higher stress than this, so the result is conservative for a tank and correct as written for an unpressurised interstage.


## Bending modes of the assembly

| quantity | value |
|---|---|
| first elastic bending mode | **26.2 Hz** free-free |
| next modes | 73.3 Hz, 143.5 Hz, 236.1 Hz |
| rigid-body modes found | 2 (a free planar beam has exactly 2) -> **True** |
| section | 1.50 mm wall at 446 mm radius, E 69 GPa |

Control bandwidth has to sit well below 26.2 Hz -- the usual allowance is a factor of five to ten -- or the autopilot drives the structure instead of steering it. This packet does not size a control system, so that comparison is left open rather than claimed as satisfied.


## Propellant slosh

| tank | fill | slosh mode | participating mass | vs first bending |
|---|---|---|---|---|
| stage 1 | 90% | 2.15 Hz | 130 kg (6% of liquid) | ratio 0.082 |
| stage 1 | 50% | 2.15 Hz | 130 kg (10% of liquid) | ratio 0.082 |
| stage 1 | 20% | 2.14 Hz | 129 kg (25% of liquid) | ratio 0.082 |

Closest approach to the first bending mode is a ratio of 0.082 -- separated by more than an octave.

Structural coupling is not a concern at these frequencies. Control coupling may still be: the slosh modes here sit at a few hertz, which is where launch vehicle control bandwidth normally lives, and this packet does not size a control system. Baffles are not modelled.


## Control authority

| quantity | value |
|---|---|
| condition | max-Q 116.0 kPa at 5 deg incidence |
| aerodynamic moment about the CG | 46.9 kN m |
| thrust x gimbal arm | 144.7 kN x 2.62 m |
| gimbal deflection required | **7.1 deg** |
| assumed available | 8.0 deg (typical production engine, not a hardware specification) |
| authority | **True** (utilisation 0.89) |

- The vehicle is statically stable, so this deflection is spent overcoming its own fins rather than correcting an instability. Larger fins buy static margin and cost control authority.

| frequency | value |
|---|---|
| rigid-body pitch (weathercock) at max-Q | 1.19 Hz |
| stage 1 slosh at max-Q | 2.15 Hz |
| ratio | **1.80** |
| first bending | 26.2 Hz |
| usable control band | No usable band: flying the vehicle needs at least 3.58 Hz while the slosh caps bandwidth at 0.43 Hz. The slosh lies inside the required band rather than above it, so added damping cannot open it: this needs a notch filter at that frequency, hardware that moves the mode, or an autopilot designed to fly through it. |
| lowest flexible mode / rigid-body mode | **1.80** |
| verdict robust to the separation rules | **True** |

The verdict holds across every rigid-body margin from 1.5 to 3 and every flexible separation from 1.5 to 5, so it is a property of the vehicle rather than of the factors chosen. The lowest flexible mode sits at 1.80 times the rigid-body mode; below about 3 there is no bandwidth that dominates one without exciting the other.


**Baffles cannot close this.** the control bandwidth this vehicle needs (3.58 Hz) is above its slosh frequency (2.15 Hz), so the mode lies inside the control band rather than above it. Baffles add damping but do not move the frequency, so no baffle closes this. It needs a notch filter at the slosh frequency, a tank that sloshes elsewhere, or an autopilot designed to fly through the mode


**Static margin was traded for control authority.** Fins sized purely for stability left the engine short of gimbal; the loop searched downward until it could steer.

| target margin | fin span | CNa | gimbal needed | ok |
|---|---|---|---|---|
| 1.50 cal | 847 mm | 9.22 /rad | 11.94 deg | no |
| 1.40 cal | 772 mm | 8.29 /rad | 10.00 deg | no |
| 1.30 cal | 709 mm | 7.53 /rad | 8.42 deg | no |
| 1.20 cal | 656 mm | 6.90 /rad | 7.11 deg | yes |

Fins resized for 1.20 calibers of static margin, down from 1.50, so the engine can trim the vehicle at max-Q within 8.0 degrees.



## Stability

| quantity | value |
|---|---|
| fin span (each of 4) | 656 mm |
| fin root / tip chord | 893 / 446 mm, sweep 536 mm |
| fin drag, absent from the flown trajectory | Cd 0.0172 on body frontal area, **5%** of the body's 0.378 |
| fin planform vs body frontal area | 2.8x |

The apogee under mission verification is a finless vehicle's. Adding this drag costs roughly 1% of apogee, about 23 kg of gross mass to recover -- small against the other uncertainties here, and always in the optimistic direction. Interference drag where the fins meet the body is not included. It is real and positive, so this is a floor on fin drag rather than an estimate of it. Skin friction uses a single representative coefficient of 0.003; the true value varies by about a factor of two across an ascent.

| centre of pressure | 1.544 m from aft |
| centre of gravity | 2.615 m from aft |
| static margin | 1.20 calibers (target 1.2, sized for liftoff) |
| normal force slope | nose 2.00 + fins 4.90 = 6.90 /rad |

| burn state | vehicle mass | centre of gravity | static margin |
|---|---|---|---|
| liftoff | 4279.2 kg | 2.615 m | 1.20 cal |
| burnout | 1679.0 kg | 3.514 m | 2.21 cal |

Fins are sized by solving for the span that meets the margin, not assumed and then checked. The nose centre of pressure comes from slender-body theory as L - V/A_base, which needs only the nose volume and reproduces the exact families (cone 2L/3, von Karman L/2) to the last digit. The fin set is Barrowman, whose CN_alpha converges onto Jones' slender-wing result pi AR/2 as aspect ratio goes to zero and whose unswept-rectangular centre of pressure is exactly the quarter chord -- two limits with known answers, which is what makes its constants checkable rather than merely quoted.

## Assembly

The vehicle built as geometry: 5.86 m over 7 parts, 7 of them exported to STEP under `cad/`.

| part | kind | station | length | mass |
|---|---|---|---|---|
| nozzle | revolve | 0.000 m | 542 mm | 2.65 kg |
| stage 1 tank | shell | 0.000 m | 4071 mm | 92.19 kg |
| fin 1 | extrude | 0.223 m | 893 mm | 15.56 kg |
| fin 2 | extrude | 0.223 m | 893 mm | 15.56 kg |
| fin 3 | extrude | 0.223 m | 893 mm | 15.56 kg |
| fin 4 | extrude | 0.223 m | 893 mm | 15.56 kg |
| nose cone | revolve | 4.071 m | 1786 mm | 29.40 kg |

### Does the mass budget hold the vehicle?

| term | mass |
|---|---|
| skin, as drawn | 186.5 kg |
| engine, from liftoff thrust at T/W 60 | 246.0 kg |
| accounted for | 432.4 kg |
| structural budget | 679.0 kg |
| slack | +246.6 kg |

> The budget holds, with 246.6 kg left for plumbing, avionics, tank domes and separation hardware -- none of which the geometry draws.

Stage lengths come from propellant volume at LOX/RP-1 bulk density and each stage is a uniform cylinder of its wet mass -- coarse, since a real stage has domes, a dry engine at one end and a moving liquid level, but built only from numbers the planner produced, and the wet mass reproduces the planner's gross of 4279.2 kg. Pitch inertia is 22x roll, as it must be for a long thin vehicle.

Sizing is on p95. Across a 16x mesh refinement on one part the median moved 2.8%, p95 13.8%, p99 36.6% and the peak 268% -- p99 is not stable enough to design against at the mesh densities this loop can afford, because with ~1,200 nodes its top percent is a dozen nodes all on the same corner. A starred p99 marks a part whose p99 exceeds twice its p95: that is a stress concentration wanting a fillet or a doubler, not a wall that wants thickening. The peak column never converges and is shown only to locate it.

Wall thickness and buckling margin size the thin shell; the shell FEA column is CalculiX on that same hollow geometry, so the meshed part is the part being designed rather than a solid billet with the same outer dimensions. The 1st mode column is a CalculiX *FREQUENCY solve on that same mesh, clamped at the aft face: it is what a static check cannot see, and it is the quantity a flutter or coupled-loads assessment starts from. Mass and Izz are computed on that same solid, exact for the geometry, in kg and kg m^2 about the centroid.

## Assembly verification

| check | result | detail |
|---|---|---|
| skin buckling | **FAIL** | interaction 1.21, margin 0.83 governed by compression |
| slosh / pitch-mode separation | **PASS** | slosh 2.15 Hz against pitch 1.19 Hz, ratio 1.80 |
| control bandwidth window | **FAIL** | needs >= 3.58 Hz, capped at 0.43 Hz by slosh |
| thrust vector control authority | **PASS** | 7.1 deg required against 8.0 available |

2 assembly-level check(s) failed while every component passed its own coupon test. The components are not wrong -- each one survives the load it was given. What fails is the vehicle they add up to, and no per-part analysis can see it.


Allowable 188 MPa, derived from al-6061-t6 at 276 MPa (typical (non-statistical)) with a yield factor of safety of 1.25 and a 0.85 knockdown. All 4 components passed: **False**. Assembly checks passed: **False**. Overall: **False**


This allowable is not certifiable and the packet should not be read as though it were. The catalogue carries typical strengths rather than A- or B-basis values, so the knockdown above stands in for a statistical basis that does not exist here:

- Catalogue strength is a typical value, not an A- or B-basis allowable. A knockdown stands in for the statistical basis and is a judgement, not a tolerance bound.

Discretisation error is measured separately and is not included in any margin quoted here. Against an exact Lame solution (`artifacts/verification/fea_mesh_convergence.json`) the C3D4 linear tetrahedra this pipeline writes converge first-order in stress (p = 1.31) and read 9.8% low on surface stress at 34,493 elements, against a 40,000 element budget; quadratic C3D10 elements reach a lower field error with 661.

On the real corpus parts that number is larger and two-sided. Solving twelve components at identical meshes and loads under both element types (`artifacts/verification/element_order_ab.json`) moved the p95 this loop sizes against by a median of 1.1% but a range of -13.9% to +14.5%. The small median belongs to smooth parts like body tubes; the double-digit ends belong to fins and nose cones, where the field is dominated by stress concentrations. Read the component margins below as carrying at least that much numerical uncertainty.
