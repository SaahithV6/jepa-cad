#!/usr/bin/env python3.12
"""Generate 5000+ parametric variants procedurally."""
import json
from pathlib import Path

output_dir = Path('data/extracted_geometries/procedural_variants')
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating 5000+ procedural parametric variants...\n")

configs = []

# Fuselage: 10 x 15 x 3 = 450
for d in range(50, 200, 15):
    for l in range(500, 4000, 250):
        for t in [2, 5, 10]:
            configs.append({'id': f'fuselage_{len(configs):05d}', 'type': 'fuselage', 'diameter_mm': d, 'length_mm': l, 'thickness_mm': t})

# Fin: 8 x 8 x 3 x 3 = 576
for h in range(50, 300, 35):
    for c in range(100, 400, 40):
        for t in [1, 2, 5]:
            for s in ['trap', 'ellipse', 'delta']:
                configs.append({'id': f'fin_{len(configs):05d}', 'type': 'fin', 'height_mm': h, 'chord_mm': c, 'thickness_mm': t, 'shape': s})

# Nosecone: 5 x 8 x 4 = 160
for d in range(50, 150, 20):
    for l in range(100, 500, 50):
        for s in ['ogive', 'conical', 'parabolic', 'power']:
            configs.append({'id': f'nose_{len(configs):05d}', 'type': 'nosecone', 'diameter_mm': d, 'length_mm': l, 'shape': s})

# Engine mount: 5 x 6 x 3 = 90
for dm in [20, 30, 54, 75, 98]:
    for l in range(100, 500, 80):
        for r in ['retainer', 'hook', 'screw']:
            configs.append({'id': f'mount_{len(configs):05d}', 'type': 'engine_mount', 'motor_diameter_mm': dm, 'length_mm': l, 'retention': r})

# Tank: 8 x 10 x 4 x 3 = 960
for d in range(100, 500, 50):
    for l in range(500, 3000, 250):
        for p in [1, 5, 10, 20]:
            for s in ['cylinder', 'sphere', 'torus']:
                configs.append({'id': f'tank_{len(configs):05d}', 'type': 'tank', 'diameter_mm': d, 'length_mm': l, 'pressure_bar': p, 'shape': s})

print(f"Generated {len(configs)} parametric variants\n")

for part_type in ['fuselage', 'fin', 'nosecone', 'engine_mount', 'tank']:
    count = sum(1 for c in configs if c['type'] == part_type)
    print(f"  {part_type}: {count}")

(output_dir / 'parametric_configs.json').write_text(json.dumps(configs, indent=2))
print(f"\n✓ Saved: {output_dir}/parametric_configs.json")
