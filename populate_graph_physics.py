#!/usr/bin/env python3.12
"""
Populate graph with actual physics parameters, test cases, and solver setups.
Attaches CalculiX/OpenFOAM cases with proper boundary conditions and parameters.
"""
import json
import sys
from pathlib import Path
from uuid import uuid4
from typing import Any

# Physics parameter ranges from spec and experience
PHYSICS_REGIMES = {
    'nose_cone': {
        'flow_type': 'external',
        'mach_range': [0.1, 8.0],
        'temperature_range': [250, 3000],  # K
        'altitude_range': [0, 100_000],  # m
        'reynolds_range': [1e4, 1e8],
        'heat_flux_range': [10, 500],  # kW/m²
        'materials': ['Al-7075', 'Ti-6Al-4V', 'Carbon Phenolic'],
    },
    'nozzle': {
        'flow_type': 'internal',
        'chamber_pressure_range': [10, 350],  # bar
        'expansion_ratio_range': [5, 150],
        'throat_area_range': [0.01, 10],  # cm²
        'wall_temp_range': [300, 2000],  # K
        'heat_flux_range': [100, 2000],  # kW/m²
        'reynolds_range': [1e5, 1e7],
        'materials': ['Inconel X', 'Copper-Beryllium', 'Rhenium', 'Niobium'],
        'cooling_methods': ['ablative', 'regenerative', 'film_cooling', 'transpiration'],
    },
    'tank': {
        'flow_type': 'internal',
        'pressure_range': [5, 500],  # bar
        'temperature_range': [77, 300],  # K (cryogenic to room)
        'fluid': ['LOX', 'LH2', 'RP-1', 'methane'],
        'wall_stress_range': [10, 500],  # MPa
        'buckling_margin': [1.5, 5.0],
        'materials': ['Al-2014', 'Stainless-304', 'Titanium'],
    },
    'fin': {
        'flow_type': 'external',
        'mach_range': [0.5, 5.0],
        'temperature_range': [220, 2500],
        'reynolds_range': [1e5, 1e9],
        'shock_angle_range': [20, 60],  # degrees
        'bending_stress_range': [50, 300],  # MPa
        'materials': ['Al-7075-T6', 'Carbon-Phenolic', 'CFRP'],
    },
    'combustion_chamber': {
        'flow_type': 'internal',
        'chamber_pressure_range': [50, 300],  # bar
        'temperature_range': [2500, 3500],  # K
        'mass_flux_range': [5000, 25000],  # kg/m²/s
        'heat_flux_range': [500, 3000],  # kW/m²
        'l_star_range': [1.0, 2.5],  # m
        'materials': ['Inconel X', 'Tungsten', 'Molybdenum'],
    },
}

SOLVER_CASES = {
    'fea': {
        'solver': 'CalculiX',
        'sim_kinds': ['static_structural', 'thermal_stress', 'modal', 'buckling'],
        'load_cases': [
            {'name': 'axial_6g', 'accel': 6.0, 'direction': 'z'},
            {'name': 'lateral_4g', 'accel': 4.0, 'direction': 'xy'},
            {'name': 'combined_load', 'accel': 7.0, 'direction': 'xyz'},
            {'name': 'pressure', 'value': 100, 'unit': 'bar'},
            {'name': 'thermal_gradient', 'delta_t': 500},
        ],
        'mesh_sizes': ['coarse', 'medium', 'fine'],
    },
    'cfd': {
        'solver': 'OpenFOAM',
        'sim_kinds': ['subsonic', 'transonic', 'supersonic', 'hypersonic'],
        'turbulence_models': ['laminar', 'spalart_allmaras', 'k_omega_sst', 'k_epsilon'],
        'boundary_conditions': [
            {'type': 'inlet', 'mach': 2.0, 'temp': 300, 'pressure': 101325},
            {'type': 'wall', 'temp': 500, 'roughness': 'smooth'},
            {'type': 'outlet', 'pressure': 101325},
            {'type': 'symmetry'},
        ],
        'schemes': ['euler', 'rk2', 'rk3'],
    },
}

def populate_physics_targets(graph: dict, physics_regimes: dict) -> int:
    """Populate PhysicsTarget nodes with actual physics parameters."""
    count = 0
    
    # Group PhysicsTarget nodes by part type
    physics_nodes = [n for n in graph['nodes'] if n['type'] == 'PhysicsTarget']
    part_nodes = {n['id']: n for n in graph['nodes'] if n['type'] == 'Part'}
    
    for pnode in physics_nodes:
        # Find associated part
        part_id = None
        for edge in graph['edges']:
            if edge.get('target') == pnode['id'] and edge['source'] in part_nodes:
                part_id = edge['source']
                break
        
        if not part_id:
            continue
        
        part = part_nodes[part_id]
        part_name = part.get('label', part['id']).lower()
        
        # Match to physics regime
        regime = None
        for regime_name, regime_data in physics_regimes.items():
            if regime_name in part_name:
                regime = regime_data
                break
        
        if regime:
            pnode['attributes'] = {
                'flow_type': regime.get('flow_type', 'unknown'),
                'mach_range': regime.get('mach_range'),
                'temperature_range': regime.get('temperature_range'),
                'reynolds_range': regime.get('reynolds_range'),
                'heat_flux_range': regime.get('heat_flux_range'),
                'pressure_range': regime.get('pressure_range'),
                'materials': regime.get('materials', []),
                'cooling_methods': regime.get('cooling_methods'),
            }
            # Filter out None values
            pnode['attributes'] = {k: v for k, v in pnode['attributes'].items() if v is not None}
            count += 1
    
    return count

def populate_solver_setups(graph: dict, solver_cases: dict) -> int:
    """Populate SolverSetup nodes with CalculiX and OpenFOAM parameters."""
    count = 0
    
    solver_nodes = [n for n in graph['nodes'] if n['type'] == 'SolverSetup']
    
    for snode in solver_nodes:
        solver_type = snode.get('properties', {}).get('solver', 'fea')
        
        if solver_type not in solver_cases:
            continue
        
        cases = solver_cases[solver_type]
        
        if solver_type == 'fea':
            # Add CalculiX-specific parameters
            snode['properties'].update({
                'solver_name': cases['solver'],
                'sim_kinds': cases['sim_kinds'],
                'load_cases': cases['load_cases'],
                'mesh_sizes': cases['mesh_sizes'],
                'material_models': ['linear_elastic', 'plastic', 'hyperelastic'],
                'contact_type': ['none', 'frictionless', 'frictional'],
                'damping': 0.05,
            })
        
        elif solver_type == 'cfd':
            # Add OpenFOAM-specific parameters
            snode['properties'].update({
                'solver_name': cases['solver'],
                'sim_kinds': cases['sim_kinds'],
                'turbulence_models': cases['turbulence_models'],
                'boundary_conditions': cases['boundary_conditions'],
                'schemes': cases['schemes'],
                'solver_libs': ['simpleFoam', 'rhoCentralFoam', 'sonicFoam'],
                'convergence_tol': 1e-4,
            })
        
        count += 1
    
    return count

def attach_test_cases(graph: dict) -> int:
    """Attach actual test cases to SimulationCase nodes."""
    count = 0
    
    test_case_nodes = [n for n in graph['nodes'] if n['type'] == 'TestCase']
    sim_case_nodes = {n['id']: n for n in graph['nodes'] if n['type'] == 'SimulationCase'}
    
    for tcase in test_case_nodes:
        # Find associated SimulationCase
        sim_id = None
        for edge in graph['edges']:
            if edge.get('target') == tcase['id'] and edge['source'] in sim_case_nodes:
                sim_id = edge['source']
                break
        
        if not sim_id or not tcase.get('properties'):
            continue
        
        sim_case = sim_case_nodes[sim_id]
        solver_type = sim_case.get('properties', {}).get('solver', 'fea')
        
        # Generate test parameters based on solver type
        if solver_type == 'fea':
            test_params = {
                'load_cases': [
                    {'type': 'axial_6g', 'value': 6.0},
                    {'type': 'lateral_4g', 'value': 4.0},
                    {'type': 'combined', 'value': 7.0},
                ],
                'mesh_refinement': 'fine',
                'material_id': 'al_7075_t6',
                'time_steps': 100,
            }
        else:
            test_params = {
                'turbulence_model': 'k_omega_sst',
                'boundary_conditions': {
                    'inlet_mach': 2.5,
                    'inlet_temperature': 300,
                    'inlet_pressure': 101325,
                    'wall_temperature': 500,
                },
                'time_steps': 1000,
                'cfl': 0.8,
            }
        
        tcase['properties']['test_parameters'] = test_params
        count += 1
    
    return count

def fix_edge_labels(graph: dict) -> int:
    """Fix missing edge labels with appropriate relationship types."""
    count = 0
    
    node_map = {n['id']: n for n in graph['nodes']}
    
    for edge in graph['edges']:
        if edge.get('label') == 'unknown' or not edge.get('label'):
            source_node = node_map.get(edge['source'])
            target_node = node_map.get(edge['target'])
            
            if not source_node or not target_node:
                continue
            
            source_type = source_node['type']
            target_type = target_node['type']
            
            # Infer label from node types
            if source_type == 'Part' and target_type == 'SolverSetup':
                edge['label'] = 'requires_solver'
            elif source_type == 'SolverSetup' and target_type == 'PhysicsTarget':
                edge['label'] = 'solves_for'
            elif source_type == 'PhysicsTarget' and target_type == 'SimulationCase':
                edge['label'] = 'requires_case'
            elif source_type == 'SimulationCase' and target_type == 'TestCase':
                edge['label'] = 'contains_test'
            elif source_type == 'Part' and target_type == 'PhysicsTarget':
                edge['label'] = 'has_physics_target'
            elif source_type == 'Assembly' and target_type == 'Part':
                edge['label'] = 'contains_part'
            elif source_type == 'Material' and target_type == 'Part':
                edge['label'] = 'is_made_of'
            else:
                edge['label'] = f'{source_type.lower()}_to_{target_type.lower()}'
            
            count += 1
    
    return count

def main():
    print("=" * 70)
    print("POPULATING GRAPH WITH PHYSICS PARAMETERS & TEST CASES")
    print("=" * 70)
    
    graph_path = Path('artifacts/jepa-train-bundle/graph.json')
    backup_path = Path('artifacts/jepa-train-bundle/graph.backup.json')
    
    if not graph_path.exists():
        print("✗ Graph not found")
        return 1
    
    # Backup original
    with open(graph_path) as f:
        graph = json.load(f)
    
    with open(backup_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"✓ Backed up to {backup_path}")
    
    # Populate PhysicsTarget attributes
    print("\n[PHYSICS TARGETS]")
    count = populate_physics_targets(graph, PHYSICS_REGIMES)
    print(f"✓ Populated {count} PhysicsTarget nodes with physics regimes")
    
    # Populate SolverSetup with real solver parameters
    print("\n[SOLVER SETUPS]")
    count = populate_solver_setups(graph, SOLVER_CASES)
    print(f"✓ Populated {count} SolverSetup nodes with CalculiX/OpenFOAM parameters")
    
    # Attach test cases with parameters
    print("\n[TEST CASES]")
    count = attach_test_cases(graph)
    print(f"✓ Attached test parameters to {count} TestCase nodes")
    
    # Fix edge labels
    print("\n[EDGE LABELS]")
    count = fix_edge_labels(graph)
    print(f"✓ Fixed/added labels to {count} edges")
    
    # Save updated graph
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"\n✓ Saved updated graph to {graph_path}")
    
    # Verify
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    physics_populated = len([n for n in graph['nodes'] if n['type'] == 'PhysicsTarget' and n.get('attributes')])
    solver_populated = len([n for n in graph['nodes'] if n['type'] == 'SolverSetup' and 'solver_name' in n.get('properties', {})])
    test_populated = len([n for n in graph['nodes'] if n['type'] == 'TestCase' and 'test_parameters' in n.get('properties', {})])
    edges_labeled = len([e for e in graph['edges'] if e.get('label') and e['label'] != 'unknown'])
    
    print(f"\nPhysicsTarget nodes with attributes: {physics_populated}/2159")
    print(f"SolverSetup nodes with solver params: {solver_populated}/2159")
    print(f"TestCase nodes with test params: {test_populated}/2108")
    print(f"Edges with correct labels: {edges_labeled}/{len(graph['edges'])}")
    
    print("\n✅ GRAPH POPULATION COMPLETE")
    return 0

if __name__ == '__main__':
    sys.exit(main())
