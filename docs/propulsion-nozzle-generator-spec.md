# Propulsion / Nozzle Generator Spec

## Goal
Create a deterministic, solver-backed nozzle generator that emits training cases the JEPA stack can learn from as **physics relations**, not just text or shapes.

The generator should produce:
- a nozzle geometry family
- scalar operating inputs
- solver-backed or analytic outputs
- graph-traceable artifacts for dataset / shard / sample ingestion

## Scope extension

This generator is not limited to nozzles. The same family-aware synthetic pipeline should cover missing or underrepresented spaceflight subdomains, including:
- liquid engines and turbopumps
- rotating detonation engines
- aerospikes and alternate nozzle families
- ion thrusters and electric propulsion
- tanks / feed systems
- structures and load-bearing assemblies
- thermal-management layouts

The nozzle case is just one family in a broader synthetic corpus.

## Tool stack boundary
Use open-source or analytic backends only:

1. **Geometry synthesis**: analytic Rao bell or conical contour builder
2. **Thermochemistry**: Cantera equilibrium / frozen-flow solve
3. **Flow field**: OpenFOAM subset for verification and local field labels
4. **Thermal / stress**: analytic thin-wall + Bartz-style heat-transfer estimates by default; optional CalculiX subset for verification
5. **Packaging**: existing `Dataset` / `Shard` / `Sample` / `GraphDocument` pipeline

## Exact graph schema

### Node types
Use the existing open-world schema; the nozzle generator only needs the following node contracts.

#### 1) `Dataset`
One node per sweep family.

Required properties:
- `name`
- `manifest_path`
- `processed_dir`
- `source_count`
- `shard_count`

Nozzle-specific properties:
- `family` = `propulsion_nozzle`
- `sweep_id`
- `solver_stack` = `analytic|cantera|openfoam|calculix`
- `fidelity` = `analytic|solver-backed|hybrid`
- `parameter_schema_version`

#### 2) `Shard`
One node per written shard or solver batch.

Required properties:
- `name`
- `shard_path`
- `source_path`
- `format`
- `index`

Nozzle-specific properties:
- `sweep_id`
- `case_count`
- `fidelity`
- `solver_stack`

#### 3) `Sample`
One node per generated nozzle case.

Required properties:
- `name`
- `num_points`
- `num_fields`

Nozzle-specific properties:
- `case_id`
- `sweep_id`
- `propellant_pair`
- `nozzle_family`
- `solver_stack`
- `fidelity`
- `graph_metadata`
- `parametric_summary`
- `physical_summary`

#### 4) `RocketNozzle`
The design entity being synthesized.

Required properties:
- `name`
- `part_class = nozzle`

Nozzle fields:
- `chamber_pressure`
- `expansion_ratio`
- `contraction_ratio`
- `throat_diameter`
- `exit_diameter`
- `area_ratio`
- `construction_ratio`
- `wall_thickness`
- `cooling_method`
- `regen_channel_count`
- `film_cooling`
- `mixture_ratio`
- `mass_flow`
- `heat_flux`

Add these case-level fields as well:
- `propellant_pair`
- `thrust`
- `isp`
- `cstar`
- `wall_stress`
- `coolant_delta_p`
- `safety_factor`

#### 5) `Dimension`
One node per swept input variable.

Required properties:
- `name`
- `value`
- `unit`

For nozzles, use dimensions for:
- `Pc`
- `eps`
- `At`
- `MR`
- `t_wall`
- `contraction_ratio`
- `regen_channel_count`
- `coolant_delta_p`
- `safety_factor`

#### 6) `Material`
Categorical material node.

Required properties:
- `name`
- `family`
- `properties`

#### 7) `Process`
Manufacturing or thermal-management process node.

Required properties:
- `process_kind`
- `parameters`

For the nozzle generator, use process kinds such as:
- `regenerative_cooling`
- `film_cooling`
- `ablative`
- `additive_manufacture`

#### 8) `SimulationCase`
One node per analytic or solver-backed evaluation.

Required properties:
- `solver`
- `inputs`
- `outputs`
- `status`

Recommended solver values:
- `analytic-isentropic`
- `analytic-thermal`
- `cantera-equilibrium`
- `openfoam-rans`
- `calculix-thermoelastic`

#### 9) `Analogue`
Canonical summary record for a case or shard.

Properties should hold:
- `summary`
- `feature_summary`
- `parametric_summary`
- `physical_summary`

#### 10) `Statistic`
Sweep-level aggregates.

Use for counts and summary metrics like:
- case count
- solver success rate
- mean/max heat flux
- mean/max wall stress
- mean Isp

---

### Edge types
Use these relations exactly:

- `CONTAINS`
  - `Dataset -> Shard`
  - `Shard -> Sample`
- `HAS_SAMPLE`
  - `Dataset -> Sample`
  - `Shard -> Sample`
- `PART_OF`
  - `Sample -> RocketNozzle`
  - `RocketNozzle -> Subsystem` when grouping nozzle family under propulsion
- `HAS_DIMENSION`
  - `RocketNozzle -> Dimension`
  - `SimulationCase -> Dimension` when the case stores derived dimensions
- `MADE_OF`
  - `RocketNozzle -> Material`
- `MANUFACTURED_BY`
  - `RocketNozzle -> Process`
- `SIMULATED_IN`
  - `RocketNozzle -> SimulationCase`
  - `Sample -> SimulationCase`
- `VERIFIED_BY`
  - `SimulationCase -> TestCase`
  - `TestCase -> SimulationCase`
- `SOURCE_OF_TRUTH_FOR`
  - `SimulationCase -> RocketNozzle`
  - `SimulationCase -> Dimension`
- `AFFECTS`
  - `Dimension -> SimulationCase`
  - `Material -> SimulationCase`
  - `Process -> SimulationCase`
- `VARIANT_OF`
  - `RocketNozzle -> RocketNozzle`
  - `SimulationCase -> SimulationCase`
- `HAS_ANALOGUE`
  - `Dataset -> Analogue`
  - `Shard -> Analogue`
  - `Sample -> Analogue`
- `ANALOGUE_OF`
  - `Analogue -> Sample`
  - `Analogue -> SimulationCase`
  - `Analogue -> RocketNozzle`
- `HAS_STATISTIC`
  - `Dataset -> Statistic`

### Required edge properties
Every nozzle edge should carry at least one of:
- `role`
- `fidelity`
- `solver`
- `unit`
- `level_index`
- `fixed`

---

## Minimal artifact layout

Use one directory per sweep family and keep the layout small and deterministic:

```text
artifacts/nozzle_synth/
  manifest.json
  graph/
    spaceflight-graph.json
  sweeps/
    sweep_pc_eps_mr.yaml
    sweep_wall_material_cooling.yaml
    sweep_propellant_geometry.yaml
  cases/
    <case_id>/
      inputs.json
      geometry.json
      contour.csv
      solver/
        analytic.json
        cantera.json
        openfoam/
          system/
          constant/
          0/
        calculix/
          model.inp
          result.json
      outputs.json
      graph.jsonl
  shards/
    nozzle_000000.npz
    nozzle_000001.npz
  logs/
    generator.log
    solver.log
```

### File contract

#### `manifest.json`
Top-level run manifest.
Must include:
- `generator_version`
- `parameter_schema_version`
- `sweeps`
- `case_count`
- `solver_stack`
- `created_at`

#### `inputs.json`
Per-case scalar input record.
Must include:
- `Pc`
- `eps`
- `At`
- `MR`
- `t_wall`
- `material`
- `cooling_method`
- `propellant_pair`
- `nozzle_family`
- `contraction_ratio`

#### `geometry.json`
Deterministic contour summary.
Must include:
- `throat_radius`
- `exit_radius`
- `contour_kind`
- `contour_points_ref`
- `length`
- `area_ratio`

#### `contour.csv`
Wall/centerline contour samples for reproducible geometry rebuilds.

#### `solver/analytic.json`
Default analytic outputs for every case.
Must include:
- `thrust`
- `isp`
- `cstar`
- `exit_mach`
- `chamber_temperature`
- `mass_flow`
- `wall_heat_flux_max`
- `wall_stress_max`
- `coolant_delta_p`
- `safety_factor_min`

#### `solver/cantera.json`
Thermochemistry outputs.
Must include:
- `gamma`
- `mw`
- `tc`
- `cstar`
- `isp_vac`
- `isp_sea_level` when applicable

#### `solver/openfoam/`
Only for a subset of cases.
Keep the mesh and probe outputs needed for validation:
- wall pressure / temperature
- wall heat flux
- exit-plane Mach
- residual history

#### `outputs.json`
Merged canonical label file used by the graph and shard writer.
Must include the final supervised fields:
- `thrust`
- `isp`
- `cstar`
- `wall_heat_flux`
- `wall_stress`
- `coolant_delta_p`
- `safety_factor`
- `solver_confidence`
- `fidelity`

#### `graph.jsonl`
One JSON record per graph node/edge created for the case.

#### `*.npz`
Training shard payload.
Suggested tensor contract:
- `points`: `(N, 3)` nozzle surface / wall sample coordinates
- `fields`: `(N, F)` local physics channels
- `max_stress`: scalar
- `case_vector`: scalar input vector
- `target_vector`: scalar output vector
- `fidelity`: scalar code

### Recommended field channels
Use 6–8 per-point channels, in this order:
1. `pressure`
2. `temperature`
3. `mach`
4. `wall_heat_flux`
5. `wall_shear_stress`
6. `wall_temperature`
7. `wall_stress`
8. `safety_factor`

---

## First sweeps to materialize
Materialize these before expanding the family.

### Sweep 1 — thermo/geometry coupling
Purpose: teach the model how chamber pressure, expansion ratio, and mixture ratio move thrust/Isp/c*.

Fixed:
- `propellant_pair = LOX/LH2`
- `t_wall = 1.0 mm`
- `material = Inconel 718`
- `cooling_method = regenerative`
- `nozzle_family = Rao bell`

Sweep axes:
- `Pc ∈ {3, 10, 20} MPa`
- `eps ∈ {20, 50, 100}`
- `MR ∈ {4.5, 5.5, 6.5}`

Output focus:
- `thrust`
- `isp`
- `cstar`
- `exit_mach`
- `chamber_temperature`

Run type:
- analytic + Cantera for every point
- OpenFOAM on a small verification subset

### Sweep 2 — thermal / structural coupling
Purpose: teach wall thickness, material choice, and cooling method to move heat flux, stress, coolant pressure drop, and safety factor.

Fixed:
- `propellant_pair = LOX/RP-1`
- `Pc = 10 MPa`
- `eps = 20`
- `MR = 2.6`
- `At = fixed to baseline thrust`
- `nozzle_family = Rao bell`

Sweep axes:
- `t_wall ∈ {0.5, 1.0, 2.0} mm`
- `material ∈ {Inconel 718, CuCrZr, Haynes 230}`
- `cooling_method ∈ {regenerative, film, ablative}`

Output focus:
- `wall_heat_flux_max`
- `wall_stress_max`
- `coolant_delta_p`
- `safety_factor_min`

Run type:
- analytic thin-wall + heat-transfer relation for all points
- CalculiX and/or OpenFOAM on a subset for verification

### Sweep 3 — propellant / family transfer
Purpose: force the representation to learn that propellant chemistry changes the full nozzle response, not just the shape.

Fixed:
- `Pc = 7 MPa`
- `eps = 30`
- `t_wall = 1.0 mm`
- `material = Inconel 718`
- `cooling_method = regenerative`

Sweep axes:
- `propellant_pair ∈ {LOX/RP-1, LOX/CH4, LOX/LH2}`
- `contraction_ratio ∈ {2.0, 3.0, 4.0}`
- `nozzle_family ∈ {conical, Rao bell, plug}`

Output focus:
- `cstar`
- `isp`
- `thrust`
- `wall_heat_flux_max`
- `exit_mach`

Run type:
- analytic + Cantera for every point
- OpenFOAM only on representative corners

---

## Generation rules
1. **Deterministic seeding**: hash the sweep row into the case ID and random seed.
2. **Parameter sweep first**: generate the whole Cartesian grid before any adaptive sampling.
3. **Analytic first, solver second**: every case gets an analytic baseline; only selected cases get OpenFOAM / CalculiX.
4. **Keep provenance explicit**: every output must trace back to a `SimulationCase` and a `RocketNozzle` node.
5. **Write graph alongside data**: every shard write should emit graph nodes/edges in the same transaction or batch.
6. **Promote only verified cases**: solver-backed cases should be marked `verified`; analytic-only cases should stay `reference` or `analogue`.

## Practical first implementation
If you only build one thing next, build the sweep runner that:
- reads a sweep YAML
- expands it into case rows
- generates analytic geometry + thermochemistry
- optionally launches OpenFOAM/CalculiX for sampled rows
- writes `inputs.json`, `outputs.json`, `graph.jsonl`, and `*.npz`
- appends a `Dataset` + `Shard` + `Sample` + `SimulationCase` graph fragment

That closes the propulsion/nozzle gap in a way the JEPA stack can actually learn from.