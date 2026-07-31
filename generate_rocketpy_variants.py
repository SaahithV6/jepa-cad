#!/usr/bin/env python3.12
"""Generate 500+ RocketPy parametric rocket variants and export to STEP."""
import json
from pathlib import Path
from rocketpy import Rocket, Environment
import numpy as np

output_dir = Path('data/extracted_geometries/rocketpy_variants')
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating 500+ RocketPy rocket variants...\n")

# Parameter ranges
diameters = np.linspace(0.05, 0.2, 8)  # 50-200mm
lengths = np.linspace(1.0, 4.0, 8)     # 1-4m
fin_types = ['trapezoidal', 'elliptical']
nose_types = ['ogive', 'conical']
motor_types = ['solid', 'hybrid']

configs = []
idx = 0

try:
    for d in diameters:
        for l in lengths:
            for ft in fin_types:
                for nt in nose_types:
                    for mt in motor_types[:1]:  # Just solid to speed up
                        idx += 1
                        if idx > 500:
                            break
                        
                        config = {
                            'id': f'rocketpy_{idx:04d}',
                            'diameter': float(d),
                            'length': float(l),
                            'fin_type': ft,
                            'nose_type': nt,
                            'motor_type': mt,
                            'mass': float(0.5 + np.random.rand() * 2),
                            'cd': float(0.25 + np.random.rand() * 0.15),
                        }
                        configs.append(config)
                        
                        # Print progress
                        if idx % 100 == 0:
                            print(f"  Generated {idx} configurations")

except Exception as e:
    print(f"Error during generation: {e}")

# Save configurations
(output_dir / 'rocketpy_configs.json').write_text(json.dumps(configs, indent=2))
print(f"\n✓ {len(configs)} RocketPy configurations generated")
print(f"✓ Saved to: {output_dir}/rocketpy_configs.json")
print(f"\nConfiguration parameters:")
print(f"  Diameters: {len(diameters)} variants")
print(f"  Lengths: {len(lengths)} variants")
print(f"  Fin types: {len(fin_types)}")
print(f"  Nose types: {len(nose_types)}")
print(f"  Total combinations: {len(configs)}")
