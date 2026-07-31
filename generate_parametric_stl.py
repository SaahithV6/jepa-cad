#!/usr/bin/env python3.12
"""Generate 2191 parametric rocket parts as STL via OpenSCAD."""
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

configs_path = Path('data/extracted_geometries/procedural_variants/parametric_configs.json')
with open(configs_path) as f:
    configs = json.load(f)

output_dir = Path('data/generated_geometry')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating {len(configs)} parametric parts via OpenSCAD...\n")

def generate_scad_stl(config, idx, total):
    """Generate SCAD script and convert to STL."""
    try:
        part_id = config['id']
        part_type = config['type']
        
        # Create SCAD based on part type
        if part_type == 'fuselage':
            scad = f"""
cylinder(d={config['diameter_mm']}, h={config['length_mm']}, $fn=32);
"""
        elif part_type == 'fin':
            scad = f"""
linear_extrude({config['thickness_mm']})
  polygon([[0,0], [{config['chord_mm']},0], [{config['chord_mm']/2},{config['height_mm']}, [0,{config['height_mm']}]]);
"""
        elif part_type == 'nosecone':
            scad = f"""
cylinder(d1={config['diameter_mm']}, d2=0, h={config['length_mm']}, $fn=32);
"""
        elif part_type == 'engine_mount':
            scad = f"""
cylinder(d={config['motor_diameter_mm']}, h={config['length_mm']}, $fn=16);
"""
        elif part_type == 'tank':
            scad = f"""
cylinder(d={config['diameter_mm']}, h={config['length_mm']}, $fn=32);
"""
        else:
            scad = "cube([10,10,10]);"
        
        # Write SCAD
        scad_file = output_dir / f"{part_id}.scad"
        scad_file.write_text(scad)
        
        # Convert to STL
        stl_file = output_dir / f"{part_id}.stl"
        result = subprocess.run(
            ['openscad', '-o', str(stl_file), str(scad_file)],
            capture_output=True,
            timeout=30,
        )
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{total}] Generated STL files")
        
        return stl_file.exists() and stl_file.stat().st_size > 100
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(generate_scad_stl, cfg, i, len(configs)): i
        for i, cfg in enumerate(configs)
    }
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(configs)} parametric parts generated as STL")
print(f"Output: {output_dir}")
