# Spaceflight Graph Schema

This repository now treats spaceflight data as a **property graph** with an open-world schema.

## Design goals

- Represent **parts**, **assemblies**, **features**, **dimensions**, **materials**, **processes**, **simulations**, **tests**, **failures**, **subsystems**, **vehicles**, **missions**, and **sources**.
- Allow different part families to carry **different attribute sets**.
- Keep a shared base schema so everything is still searchable and traceable.
- Preserve provenance so every graph node can be tied back to source material.
- Support both **geometry learning** and **association learning**.

## Base node contract

Every node has a shared core:

- `id`
- `type`
- `source_ref`
- `provenance`
- `domain`
- `confidence`
- `tags`
- `version`

## Specialized node types

The schema includes dedicated node types for:

- `Source`
- `Document`
- `Part`
- `Assembly`
- `Feature`
- `Dimension`
- `Material`
- `Process`
- `SimulationCase`
- `TestCase`
- `FailureMode`
- `Subsystem`
- `Vehicle`
- `Mission`
- `RocketNozzle`
- `Tank`
- `Valve`
- `StructurePart`

## Example: rocket nozzle specialization

Rocket nozzles can carry operating-point and construction data that would not make sense for every part type:

- chamber pressure
- expansion ratio
- contraction ratio
- throat diameter
- exit diameter
- area ratio
- construction ratio
- wall thickness
- cooling method
- regenerative channel count
- film cooling flag
- mixture ratio
- mass flow
- heat flux

This is intentionally **not** a fixed universal part schema. It is a type-specific extension schema.

## Edge types

The graph uses these core relationships:

- `PART_OF`
- `HAS_FEATURE`
- `HAS_DIMENSION`
- `MADE_OF`
- `MANUFACTURED_BY`
- `SIMULATED_IN`
- `TESTED_IN`
- `VERIFIED_BY`
- `AFFECTS`
- `VARIANT_OF`
- `SOURCE_OF_TRUTH_FOR`
- `MENTIONS`
- `RECOMMENDED_FOR`

## Immediate graph export

The current source registry can be materialized as a graph document with:

- one `Source` node per registry item
- domain nodes
- use-case feature nodes
- recommendation nodes
- traceability edges linking the source to semantic categories

That gives the model a relational layer over the corpus immediately, even before full part extraction is complete.

## CLI

Render the schema:

```bash
python -m cadflow.cli graph-schema --json
```

Export the current source registry graph:

```bash
python -m cadflow.cli graph-export --json
```

## Why this is adaptable

This schema is open-world:

- base fields are shared everywhere
- type-specific extensions are allowed
- unknown fields can be preserved in the graph payload
- future part families can add their own schema nodes without breaking old data

That means a rocket nozzle can have chamber-pressure and cooling-channel fields, while a tank can have pressure and pressurant fields, and a mechanism can have entirely different attributes.
