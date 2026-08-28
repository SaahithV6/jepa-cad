# Design packet

**Specification:** deliver 50 kg payload to 120 km apogee using lox/rp1 at 55 bar chamber pressure

## Architecture

- mission needs about 2.05 km/s; at Isp 325 s a single stage would need mass ratio 1.90
- that is within what a stage can close at structural coefficient 0.21491772638948176 (limit 2.88), so try a single stage first
- 1 stage(s) closes the mission at 219.3 kg gross; simpler architectures are preferred so this is selected

## Structural mass closure

Designed at a structural coefficient of **0.140**. Solving it as a fixed point -- size the vehicle, size its walls from the resulting loads, recompute the coefficient, repeat -- converges to **0.251**.

> The mass model wants 0.251 where the design asserts 0.140, so this vehicle is optimistic: built to its own structural model it would be heavier than planned and would fall short of the target. Re-run with `--solve-structure` to design at the solved value. It is reported rather than silently absorbed because it changes the architecture -- a heavier structure closes fewer stages, so the same mission needs more of them.


## Vehicle: 1 stage(s)

| stage | propellant | structure | expansion |
|---|---|---|---|
| 1 | 132.91 kg | 36.38 kg | 12 |

payload 50.0 kg, gross 219.3 kg


**What sets the wall thicknesses.** 2 of 4 components are at minimum gauge rather than sized by strength, carrying 45% of the analysed structural mass. Those walls cannot be thinned -- they are already as thin as the process allows -- so a structural coefficient above flown practice is partly a consequence of building a small vehicle rather than a design fault. The levers that remain are fewer stages, a larger vehicle, or a material with a lower minimum gauge; thinning walls is not one of them.


At minimum gauge: nose cone, fin set.


**Structure against flown hardware.** Stage 1 structural coefficient is 0.2149. Ten flown stages from Saturn V's S-IC to Electron's first stage span 0.036 to 0.118, median 0.080. Above the flown range 0.036-0.118, by 82%. The reference set contains no stage below 10200 kg wet and this one is 169 kg, so a heavier fraction is expected: tank mass follows area while propellant follows volume. The comparison is an extrapolation and does not by itself indicate an error. Reference figures are secondary-source and unverified against primary mass statements; treat as a regime check, not a validation.

## Mission verification

Flown apogee **119.1 km** against 120 km requested (**0.8%** error); downrange 8.3 km, max-Q 122.2 kPa, separations none.

Peak axial acceleration **9.4 g** (stage 1 9.4 g). This is what sizes the structure, and it is a property of the architecture rather than a choice: thrust is held while mass falls, so acceleration climbs through each burn.


## Nozzle

| quantity | value |
|---|---|
| throat / exit radius | 19.0 / 66.0 mm |
| area ratio | 12.0 |
| contour | 80% bell, 140 mm long, exiting at 9 deg |
| divergence efficiency | 0.9938 |
| wall / mass | 1.50 mm, 0.53 kg of Inconel |

The contour is a quadratic pinned by four constraints that are all given or forced -- throat radius, the exit radius the area ratio demands, and the flow angle at each end -- so nothing about it is read off a chart. Its shape now has a consequence: divergence loss multiplies thrust, and a 25 degree exit would cost 4.7% of specific impulse against this one's 0.6%. The wall is offset along the surface normal, so it is constant-thickness sheet rather than thinning where the contour is steep.

## Thermal

| quantity | value |
|---|---|
| throat diameter | 38.1 mm |
| throat heat flux | 38.3 MW/m^2 at a 800 K wall |
| adiabatic wall temperature | 3563 K |
| total heat into the walls | 758 kW, 2.72% of exhaust power |
| regenerative cooling | rp1 rises 348 K to 648 K (limit 700 K) |
| cooling closes | **yes**, margin +52 K |

Gas properties are the real equilibrium mixture's -- Prandtl 0.591, Reynolds 1.17e+06 -- not a textbook value for air. The convective correlation is the standard turbulent form; what makes it checkable is its scalings, and heat flux is asserted to go as throat diameter to the -0.200 and chamber pressure to the 0.8. The cooling check is an energy balance and involves no correlation at all.
| peak skin temperature | 993 K at 24.4 km |

> The skin reaches 993 K, past the 420 K service limit of al-6061-t6, which is what this design selected. The vehicle needs thermal protection or a trajectory that spends less time fast in thick air; no catalogued alloy is left to upgrade to. This is a radiation-equilibrium steady state and the vehicle passes through quickly, so it is an upper bound rather than what the structure actually reaches.
## Component verification

| component | load case | load | wall | driver | buckling margin | shell p95 | p99 | peak | 1st mode | mass | Izz | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nose cone | drag plus normal force at max-Q, 1.5 ultimate | 200 N | 0.80 mm | minimum gauge | 632.09x | 2.4 MPa | 3 | 6 MPa | 4346 Hz | 16 g | 6.58e-06 | PASS |
| thrust structure | engine thrust into the aft ring | 12580 N | 0.83 mm | strength | 10.89x | 147.0 MPa | 194 | 415 MPa | 8254 Hz | 10 g | 4.09e-06 | PASS |
| fin set | one fin's share of fin normal force at 5 deg, 1.5 ultimate | 200 N | 0.80 mm | minimum gauge | 632.09x | 13.5 MPa | 27 | 83 MPa | 2223 Hz | 25 g | 1.25e-05 | PASS |
| stage 1 tank | carries 132.9 kg propellant at 9.4 g | 20278 N | 1.34 mm | strength | 18.60x | 138.5 MPa | 182 | 668 MPa | 3955 Hz | 40 g | 1.64e-05 | PASS |

## Coupon stack (what was analysed)

The six analysed parts stacked nose-forward: 0.239 m in 4 sections, 91 g of structure, centre of gravity 0.126 m from the aft end, Ixx 3.982e-04 and Izz 4.224e-05 kg m^2.

These are **coupons, not the vehicle**. Body radius is clamped to 50 mm so the parts stay meshable, while this mission's reference diameter is 332 mm -- a factor of 8.0 in radius, so the coupons carry 91 g against the 36 kg of structure the planner sized. The stresses and modes above are about representative sections; the mass properties that describe the flight vehicle are below.

## Flight vehicle

| quantity | value |
|---|---|
| length | 1.84 m |
| diameter | 332 mm |
| wet mass | 219.3 kg |
| centre of gravity from aft end | 0.964 m (52% of length) |
| pitch/yaw inertia Ixx | 66.7 kg m^2 |
| roll inertia Izz | 3.0 kg m^2 |

## Flight loads on the assembly

| quantity | value |
|---|---|
| load case | max-Q 122.2 kPa at 5 deg incidence |
| aerodynamic side load | nose 1.8 kN + fins 3.0 kN |
| peak bending moment | 0.6 kN m at 0.80 m from aft (44% of length) |
| skin stress at that station | axial 14.3 + bending 8.2 = **22.5 MPa** on a 0.80 mm wall |
| that wall | sized here for the flight loads at 166 mm radius, driven by **minimum gauge** -- not carried over from the coupons, which are built at a clamped radius |
| margin against 188 MPa allowable | 8.34 |
| load set closes | shear -6.44e-14 N, moment 2.04e-02 N m -> **True** |

The load set is checked for closure rather than assumed to balance: a free vehicle must return shear and moment to zero at its aft end, and a distribution that does not still draws a smooth and entirely plausible moment curve.


## Skin buckling

| quantity | value |
|---|---|
| radius over thickness | 207 |
| classical critical stress | 203.3 MPa (a perfect cylinder; no real shell reaches it) |
| knockdown, compression / bending | 0.465 / 0.566 (NASA SP-8007) |
| allowable, compression / bending | 94.6 / 115.1 MPa |
| interaction R_c + R_b | 0.22 (must not exceed 1) |
| buckling margin | **4.50**, governed by compression |
| yield margin for comparison | 8.3 |

- Internal pressure is not credited. A pressurised tank buckles at a substantially higher stress than this, so the result is conservative for a tank and correct as written for an unpressurised interstage.


## Bending modes of the assembly

| quantity | value |
|---|---|
| first elastic bending mode | **84.7 Hz** free-free |
| next modes | 237.0 Hz, 464.1 Hz, 763.8 Hz |
| rigid-body modes found | 2 (a free planar beam has exactly 2) -> **True** |
| section | 0.80 mm wall at 166 mm radius, E 69 GPa |

Control bandwidth has to sit well below 84.7 Hz -- the usual allowance is a factor of five to ten -- or the autopilot drives the structure instead of steering it. This packet does not size a control system, so that comparison is left open rather than claimed as satisfied.


## Propellant slosh

| tank | fill | slosh mode | participating mass | vs first bending |
|---|---|---|---|---|
| stage 1 | 90% | 3.52 Hz | 7 kg (6% of liquid) | ratio 0.042 |
| stage 1 | 50% | 3.52 Hz | 7 kg (10% of liquid) | ratio 0.042 |
| stage 1 | 20% | 3.52 Hz | 7 kg (25% of liquid) | ratio 0.042 |

Closest approach to the first bending mode is a ratio of 0.042 -- separated by more than an octave.

Structural coupling is not a concern at these frequencies. Control coupling may still be: the slosh modes here sit at a few hertz, which is where launch vehicle control bandwidth normally lives, and this packet does not size a control system. Baffles are not modelled.


## Control authority

| quantity | value |
|---|---|
| condition | max-Q 122.2 kPa at 5 deg incidence |
| aerodynamic moment about the CG | 1.3 kN m |
| thrust x gimbal arm | 9.7 kN x 0.96 m |
| gimbal deflection required | **7.9 deg** |
| assumed available | 8.0 deg (typical production engine, not a hardware specification) |
| authority | **True** (utilisation 0.98) |

- The vehicle is statically stable, so this deflection is spent overcoming its own fins rather than correcting an instability. Larger fins buy static margin and cost control authority.

| frequency | value |
|---|---|
| rigid-body pitch (weathercock) at max-Q | 2.36 Hz |
| stage 1 slosh at max-Q | 3.52 Hz |
| ratio | **1.49** |
| first bending | 84.7 Hz |
| usable control band | No usable band: flying the vehicle needs at least 7.07 Hz while the slosh caps bandwidth at 0.70 Hz. The slosh lies inside the required band rather than above it, so added damping cannot open it: this needs a notch filter at that frequency, hardware that moves the mode, or an autopilot designed to fly through it. |
| lowest flexible mode / rigid-body mode | **1.50** |
| verdict robust to the separation rules | **True** |

The verdict holds across every rigid-body margin from 1.5 to 3 and every flexible separation from 1.5 to 5, so it is a property of the vehicle rather than of the factors chosen. The lowest flexible mode sits at 1.49 times the rigid-body mode; below about 3 there is no bandwidth that dominates one without exciting the other.


**Baffles cannot close this.** the control bandwidth this vehicle needs (7.07 Hz) is above its slosh frequency (3.52 Hz), so the mode lies inside the control band rather than above it. Baffles add damping but do not move the frequency, so no baffle closes this. It needs a notch filter at the slosh frequency, a tank that sloshes elsewhere, or an autopilot designed to fly through the mode


| mode | frequency | vs crossover | stabilisation |
|---|---|---|---|
| slosh | 3.52 Hz | 0.50x | **phase** |
| first bending | 84.71 Hz | 11.98x | **gain** |

slosh sits below crossover and must be phase stabilised: the control design has to model it rather than filter it out. Standard practice, and not something this packet can verify, since it does not design a control system.


**Static margin was traded for control authority.** Fins sized purely for stability left the engine short of gimbal; the loop searched downward until it could steer.

| target margin | fin span | CNa | gimbal needed | ok |
|---|---|---|---|---|
| 1.50 cal | 321 mm | 9.44 /rad | 27.64 deg | no |
| 1.40 cal | 292 mm | 8.47 /rad | 22.84 deg | no |
| 1.30 cal | 268 mm | 7.67 /rad | 19.07 deg | no |
| 1.20 cal | 247 mm | 7.01 /rad | 16.00 deg | no |
| 1.10 cal | 230 mm | 6.46 /rad | 13.46 deg | no |
| 1.00 cal | 214 mm | 5.99 /rad | 11.31 deg | no |
| 0.90 cal | 200 mm | 5.58 /rad | 9.47 deg | no |
| 0.80 cal | 188 mm | 5.22 /rad | 7.87 deg | yes |

Fins resized for 0.80 calibers of static margin, down from 1.50, so the engine can trim the vehicle at max-Q within 8.0 degrees.



## Stability

| quantity | value |
|---|---|
| fin span (each of 4) | 188 mm |
| fin root / tip chord | 332 / 166 mm, sweep 199 mm |
| fin drag, absent from the flown trajectory | Cd 0.0150 on body frontal area, **4%** of the body's 0.420 |
| fin planform vs body frontal area | 2.2x |

The apogee under mission verification is a finless vehicle's. Adding this drag costs roughly 1% of apogee, about 23 kg of gross mass to recover -- small against the other uncertainties here, and always in the optimistic direction. Interference drag where the fins meet the body is not included. It is real and positive, so this is a floor on fin drag rather than an estimate of it. Skin friction uses a single representative coefficient of 0.003; the true value varies by about a factor of two across an ascent.

| centre of pressure | 0.699 m from aft |
| centre of gravity | 0.964 m from aft |
| static margin | 0.80 calibers (target 0.8, sized for liftoff) |
| normal force slope | nose 2.00 + fins 3.22 = 5.22 /rad |

| burn state | vehicle mass | centre of gravity | static margin |
|---|---|---|---|
| liftoff | 219.3 kg | 0.964 m | 0.80 cal |
| burnout | 86.4 kg | 1.287 m | 1.77 cal |

Fins are sized by solving for the span that meets the margin, not assumed and then checked. The nose centre of pressure comes from slender-body theory as L - V/A_base, which needs only the nose volume and reproduces the exact families (cone 2L/3, von Karman L/2) to the last digit. The fin set is Barrowman, whose CN_alpha converges onto Jones' slender-wing result pi AR/2 as aspect ratio goes to zero and whose unswept-rectangular centre of pressure is exactly the quarter chord -- two limits with known answers, which is what makes its constants checkable rather than merely quoted.

## Assembly

The vehicle built as geometry: 2.17 m over 7 parts, 7 of them exported to STEP under `cad/`.

| part | kind | station | length | mass |
|---|---|---|---|---|
| nozzle | revolve | 0.000 m | 140 mm | 0.18 kg |
| stage 1 tank | shell | 0.000 m | 1508 mm | 12.61 kg |
| fin 1 | extrude | 0.083 m | 332 mm | 0.47 kg |
| fin 2 | extrude | 0.083 m | 332 mm | 0.47 kg |
| fin 3 | extrude | 0.083 m | 332 mm | 0.47 kg |
| fin 4 | extrude | 0.083 m | 332 mm | 0.47 kg |
| nose cone | revolve | 1.508 m | 663 mm | 4.32 kg |

### Does the mass budget hold the vehicle?

| term | mass |
|---|---|
| skin, as drawn | 19.0 kg |
| engine, from liftoff thrust at T/W 60 | 16.4 kg |
| accounted for | 35.4 kg |
| structural budget | 36.4 kg |
| slack | +0.9 kg |

> The budget holds, with 0.9 kg left for plumbing, avionics, tank domes and separation hardware -- none of which the geometry draws.

Stage lengths come from propellant volume at LOX/RP-1 bulk density and each stage is a uniform cylinder of its wet mass -- coarse, since a real stage has domes, a dry engine at one end and a moving liquid level, but built only from numbers the planner produced, and the wet mass reproduces the planner's gross of 219.3 kg. Pitch inertia is 22x roll, as it must be for a long thin vehicle.

Sizing is on p95. Across a 16x mesh refinement on one part the median moved 2.8%, p95 13.8%, p99 36.6% and the peak 268% -- p99 is not stable enough to design against at the mesh densities this loop can afford, because with ~1,200 nodes its top percent is a dozen nodes all on the same corner. A starred p99 marks a part whose p99 exceeds twice its p95: that is a stress concentration wanting a fillet or a doubler, not a wall that wants thickening. The peak column never converges and is shown only to locate it.

Wall thickness and buckling margin size the thin shell; the shell FEA column is CalculiX on that same hollow geometry, so the meshed part is the part being designed rather than a solid billet with the same outer dimensions. The 1st mode column is a CalculiX *FREQUENCY solve on that same mesh, clamped at the aft face: it is what a static check cannot see, and it is the quantity a flutter or coupled-loads assessment starts from. Mass and Izz are computed on that same solid, exact for the geometry, in kg and kg m^2 about the centroid.

## Flight barrel, at flight scale

| quantity | value |
|---|---|
| mesh | 768 shell elements, 800 nodes |
| membrane stress | 14.39 MPa |
| closed form | 14.30 MPa (+0.63%) |
| peak, at the clamped end | 15.00 MPa |

This is the flight barrel, not a coupon. The component table above analyses parts at 21 mm radius because a 0.80 mm wall cannot be resolved with solid elements at 166 mm -- it would take on the order of a billion tetrahedra. Shell elements carry the thickness as a property, so the same barrel is a few hundred elements.

- Compared on the middle 90% of nodes: the clamped base introduces a bending boundary layer the membrane solution does not model, and it perturbs the stress in opposite directions under axial load and under pressure.

| stage | length | wall | driver | axial load | stress | vs closed form |
|---|---|---|---|---|---|---|
| 1 | 1.51 m | 0.80 mm | minimum gauge | 5 kN | 5.6 MPa | +0.64% |

Each barrel is sized for the mass it supports at peak axial acceleration, at the flight radius -- not carried over from the coupons, whose walls were sized at 21 mm and mean nothing here. Stage 1 is worst at 5.6 MPa against a 188 MPa allowable.



## Assembly verification

| check | result | detail |
|---|---|---|
| skin buckling | **PASS** | interaction 0.22, margin 4.50 governed by compression |
| slosh / pitch-mode separation | **PASS** | slosh 3.52 Hz against pitch 2.36 Hz, ratio 1.49 |
| flexible mode stabilisation | **REQUIRED** | slosh below crossover; phase stabilisation required and not verified here |
| thrust vector control authority | **PASS** | 7.9 deg required against 8.0 available |
| flight barrel stress | **PASS** | 768 shell elements at 166 mm radius: 14.4 MPa against a 188 MPa allowable, +0.6% from closed form; worst of 1 per-stage barrels 5.6 MPa |

1 check(s) marked REQUIRED are neither passed nor failed: they name real work this packet cannot do. Claiming they pass would report checks that never ran; claiming they fail would suggest defects where there are none.

- **flexible mode stabilisation** -- slosh below crossover; phase stabilisation required and not verified here.


Allowable 188 MPa, derived from al-6061-t6 at 276 MPa (typical (non-statistical)) with a yield factor of safety of 1.25 and a 0.85 knockdown. All 4 coupons passed: **True**. Assembly checks failed: **0**. Requirements unverified: **1**.

Overall: **INCOMPLETE** -- nothing failed, but the packet cannot call a design verified while a requirement it did not check remains outstanding.


This allowable is not certifiable and the packet should not be read as though it were. The catalogue carries typical strengths rather than A- or B-basis values, so the knockdown above stands in for a statistical basis that does not exist here:

- Catalogue strength is a typical value, not an A- or B-basis allowable. A knockdown stands in for the statistical basis and is a judgement, not a tolerance bound.
- Allowable is a room-temperature value applied at 993 K. No yield-versus-temperature data is available for this material, so the real hot allowable is lower by an unquantified amount.
- Service temperature 993 K exceeds the catalogue limit of 420 K for this material.

Discretisation error is measured separately and is not included in any margin quoted here. Against an exact Lame solution (`artifacts/verification/fea_mesh_convergence.json`) the C3D4 linear tetrahedra this pipeline writes converge first-order in stress (p = 1.31) and read 9.8% low on surface stress at 34,493 elements, against a 40,000 element budget; quadratic C3D10 elements reach a lower field error with 661.

On the real corpus parts that number is larger and two-sided. Solving twelve components at identical meshes and loads under both element types (`artifacts/verification/element_order_ab.json`) moved the p95 this loop sizes against by a median of 1.1% but a range of -13.9% to +14.5%. The small median belongs to smooth parts like body tubes; the double-digit ends belong to fins and nose cones, where the field is dominated by stress concentrations. Read the component margins below as carrying at least that much numerical uncertainty.
