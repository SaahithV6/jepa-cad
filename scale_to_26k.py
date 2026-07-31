#!/usr/bin/env python3.12
"""Spaceflight data ingestion with 26k source target and phase prioritization."""
import json
from pathlib import Path

plan = {
    "phase_1_immediate": {
        "target": 7768,
        "description": "Current + immediate downloads + parametric generation",
        "sources": {
            "existing_corpus": 1768,
            "openrocket": 500,
            "birds4_cad": 300,
            "grabcad_filtered": 500,
            "thingiverse": 400,
            "nasa_artemis": 100,
            "parametric_variants": 4200,
        },
        "timeline": "2-3 days",
        "status": "IN_PROGRESS",
    },
    "phase_2_expansion": {
        "target": 15000,
        "description": "Additional CAD repositories + procedural generation",
        "additions": {
            "github_aerospace_repos": 2000,  # Search github.com for rocket/satellite projects
            "thingiverse_full_sweep": 1500,  # Complete aerospace + 3D printable
            "openscad_library_variations": 2500,  # Procedural variants
            "freecad_parametric_sweep": 1500,  # Extended parameter space
            "cad_exchange_aerospace": 1000,  # CAD Exchange library
            "commercial_models": 500,  # PTC Creo, Solidworks community
        },
        "timeline": "3-5 days",
        "blockers": "None",
    },
    "phase_3_scaling": {
        "target": 26000,
        "description": "Reach 26k unique parts through aggressive procedural generation + synthetic variants",
        "strategy": {
            "procedural_generation": {
                "openscad_full_sweep": 5000,  # Multi-parameter procedural
                "geometry_transformation": 3000,  # Rotate/scale/deform existing
                "topological_variants": 2000,  # Different topology, same function
                "material_variants": 1500,  # Same geometry, different material specs
            },
            "synthetic_assembly": {
                "part_combinations": 2000,  # Assemble existing parts into new subassemblies
                "parametric_interpolation": 1000,  # Interpolate between designs
            },
            "data_augmentation": {
                "mesh_refinement_variants": 2000,  # Different mesh densities
                "noise_injection": 1000,  # Slightly deformed geometries
            },
        },
        "timeline": "5-7 days additional",
        "total_unique": 26000,
    },
    "execution_plan": {
        "step_1": "Phase 1 execution (IN_PROGRESS): Remesh 391 failed parts, download OpenRocket+BIRDS4, generate 4,200 parametric variants",
        "step_2": "Phase 1 completion: Gmsh mesh all 7,768 parts (-clscale 0.2), FEA/CFD simulations, TAO graph ingestion",
        "step_3": "Phase 2 start: Search GitHub + CAD repositories, download + convert formats",
        "step_4": "Phase 2 generation: Run full procedural generation suite",
        "step_5": "Phase 3 synthetic: Generate procedural variants + assembly combinations",
        "step_6": "Final: Validate 26k unique geometries, physics coverage, graph structure",
    },
    "dataset_breakdown_26k": {
        "real_hardware": 2000,  # Actual spacecraft/rocket CAD
        "parametric_generation": 8000,  # Algorithmic variations
        "procedural_geometry": 10000,  # OpenSCAD, FreeCAD, etc.
        "synthetic_variants": 6000,  # Transformations, combinations
    },
    "physics_coverage_26k": {
        "fea_simulations": 26000,
        "cfd_simulations": 26000,
        "total_physics_points": 52000,
    },
    "graph_structure_target": {
        "part_nodes": 26000,
        "physics_evaluation_nodes": 52000,
        "assembly_nodes": 5000,
        "parametric_space_nodes": 10000,
        "total_graph_nodes": 93000,
        "expected_edges": 500000,
    },
    "resource_requirements": {
        "disk_storage": "500GB+ (meshes + simulations)",
        "cpu_time": "14-21 days (with 8-worker parallelization)",
        "memory": "64GB RAM recommended",
        "gpu_time": "Training only (separate)",
    },
    "milestones": {
        "day_0_1": "Phase 1 meshing complete (7,768 parts)",
        "day_2_3": "Phase 1 FEA/CFD complete, TAO ingestion",
        "day_4_6": "Phase 2 sources downloaded + parametric generation",
        "day_7_10": "Phase 2 meshing + physics complete",
        "day_11_18": "Phase 3 procedural + synthetic generation",
        "day_19_21": "Full 26k dataset ready for JEPA training",
    },
}

# Save
output_path = Path('.hermes/26k_ingestion_strategy.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(plan, f, indent=2)

print("26K INGESTION STRATEGY")
print("=" * 80)
print(f"\nPhase 1 (IN_PROGRESS): {plan['phase_1_immediate']['target']} parts")
for source, count in plan['phase_1_immediate']['sources'].items():
    print(f"  {source}: {count}")

print(f"\nPhase 2 (QUEUED): +{plan['phase_2_expansion']['target'] - plan['phase_1_immediate']['target']} parts")
for source, count in plan['phase_2_expansion']['additions'].items():
    print(f"  {source}: {count}")

print(f"\nPhase 3 (STRETCH): +{plan['phase_3_scaling']['target'] - plan['phase_2_expansion']['target']} parts")
print("  Procedural + synthetic variants")

print(f"\nFINAL TARGET: {plan['phase_3_scaling']['target']:,} unique parts")
print(f"Physics coverage: {plan['physics_coverage_26k']['total_physics_points']:,} simulations")
print(f"Graph nodes: {plan['graph_structure_target']['total_graph_nodes']:,}")
print(f"Graph edges: {plan['graph_structure_target']['expected_edges']:,}")

print(f"\nTimeline: 21 days (parallel execution)")
print(f"Disk: {plan['resource_requirements']['disk_storage']}")
print(f"Reference: {output_path}")
