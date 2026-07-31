#!/usr/bin/env python3.12
"""Extract actual spaceflight CAD components from real sources."""
import json
import shutil
from pathlib import Path
import subprocess

print("=" * 80)
print("EXTRACTING ACTUAL SPACEFLIGHT CAD COMPONENTS")
print("=" * 80)

output_dir = Path('data/spaceflight_components')
output_dir.mkdir(parents=True, exist_ok=True)

# Phase 1: OpenRocket designs (real rockets)
print("\n[PHASE 1] OpenRocket - Real Rocket Designs")
openrocket_dir = Path('data/raw/downloads/openrocket')
ork_files = list(openrocket_dir.rglob('*.ork'))

ork_output = output_dir / 'openrocket_designs'
ork_output.mkdir(exist_ok=True)

for ork in ork_files[:100]:  # Sample first 100 designs
    try:
        shutil.copy(ork, ork_output / ork.name)
    except:
        pass

print(f"  ✓ {len(list(ork_output.glob('*.ork')))} OpenRocket designs extracted")

# Phase 2: BIRDS4-CAD satellite hardware
print("\n[PHASE 2] BIRDS4-CAD - University Satellite Hardware (Verified Flight)")
birds_dir = Path('data/raw/downloads/birds4')
birds_cad = list(birds_dir.rglob('*.step')) + list(birds_dir.rglob('*.stp')) + list(birds_dir.rglob('*.iges'))

birds_output = output_dir / 'birds4_hardware'
birds_output.mkdir(exist_ok=True)

for cad in birds_cad:
    try:
        shutil.copy(cad, birds_output / cad.name)
    except:
        pass

print(f"  ✓ {len(list(birds_output.glob('*')))} BIRDS4 CAD files extracted")

# Phase 3: Generate actual parametric rocket components
print("\n[PHASE 3] Parametric Spaceflight Components")

components = {
    'nozzle': {
        'parameters': ['chamber_pressure_bar', 'expansion_ratio', 'throat_diameter_mm', 'material'],
        'ranges': {
            'chamber_pressure_bar': [1, 5, 10, 20, 50],
            'expansion_ratio': [5, 10, 20, 40, 100],
            'throat_diameter_mm': [10, 20, 30, 50, 75],
            'material': ['copper', 'tungsten', 'ceramic', 'carbon_steel'],
        },
    },
    'tank': {
        'parameters': ['internal_diameter_mm', 'length_mm', 'internal_pressure_bar', 'material'],
        'ranges': {
            'internal_diameter_mm': [100, 200, 500, 1000, 2000],
            'length_mm': [500, 1000, 2000, 3000, 5000],
            'internal_pressure_bar': [1, 5, 10, 20, 50],
            'material': ['aluminum', 'steel', 'titanium', 'composite'],
        },
    },
    'turbopump': {
        'parameters': ['flow_rate_kg_s', 'inlet_pressure_bar', 'outlet_pressure_bar', 'stages'],
        'ranges': {
            'flow_rate_kg_s': [1, 5, 10, 50, 100],
            'inlet_pressure_bar': [1, 5, 10],
            'outlet_pressure_bar': [50, 100, 200, 300],
            'stages': [1, 2, 3, 4],
        },
    },
    'fairing': {
        'parameters': ['length_mm', 'diameter_mm', 'nose_cone_type', 'material'],
        'ranges': {
            'length_mm': [500, 1000, 2000, 3000],
            'diameter_mm': [1000, 2000, 3000, 4000],
            'nose_cone_type': ['ogive', 'conical', 'parabolic'],
            'material': ['aluminum', 'composite', 'titanium'],
        },
    },
    'strut': {
        'parameters': ['length_mm', 'cross_section_mm', 'material', 'type'],
        'ranges': {
            'length_mm': [500, 1000, 2000, 3000, 5000],
            'cross_section_mm': [50, 100, 200, 300],
            'material': ['aluminum', 'titanium', 'steel', 'composite'],
            'type': ['tubular', 'solid', 'truss'],
        },
    },
    'injector': {
        'parameters': ['flow_rate_kg_s', 'inlet_diameter_mm', 'hole_diameter_mm', 'num_holes'],
        'ranges': {
            'flow_rate_kg_s': [1, 5, 10, 50, 100],
            'inlet_diameter_mm': [20, 30, 50, 75, 100],
            'hole_diameter_mm': [2, 5, 10, 20],
            'num_holes': [4, 9, 16, 25, 49],
        },
    },
}

configs = []
for comp_type, comp_info in components.items():
    ranges = comp_info['ranges']
    
    # Generate combinations
    from itertools import product
    param_lists = [ranges[p] for p in comp_info['parameters']]
    
    for combo in product(*param_lists):
        config = {
            'id': f'{comp_type}_{len(configs):05d}',
            'type': comp_type,
        }
        for param, value in zip(comp_info['parameters'], combo):
            config[param] = value
        configs.append(config)

comp_output = output_dir / 'parametric_components'
comp_output.mkdir(exist_ok=True)
(comp_output / 'component_configs.json').write_text(json.dumps(configs, indent=2))

print(f"  ✓ {len(configs)} parametric spaceflight component configurations generated")

# Summary
print("\n" + "=" * 80)
print("SPACEFLIGHT COMPONENT INVENTORY")
print("=" * 80)

inventory = {
    'openrocket_designs': len(list(ork_output.glob('*.ork'))),
    'birds4_hardware': len(list(birds_output.glob('*'))),
    'parametric_components': len(configs),
    'total_sources': len(list(ork_output.glob('*.ork'))) + len(list(birds_output.glob('*'))) + len(configs),
}

for source, count in inventory.items():
    print(f"  {source}: {count}")

print(f"\nTotal components available: {inventory['total_sources']}")
print(f"Output directory: {output_dir}")
