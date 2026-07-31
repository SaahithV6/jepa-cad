#!/usr/bin/env python3.12
"""Mesh all 8000 synthetic variants with Gmsh (-clscale 0.2)."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

os.environ['GMSH_NOPOPUP'] = '1'

variants = sorted(list(Path('data/synthetic_variants').glob('*.step')))
output_dir = Path('artifacts/meshes_8k')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Meshing {len(variants)} synthetic variants...\n")

def mesh_variant(step_file, idx, total):
    """Mesh a single STEP file."""
    try:
        var_id = step_file.stem
        msh_file = output_dir / f"{var_id}.msh"
        
        result = subprocess.run(
            ['gmsh', str(step_file), '-3', '-format', 'msh2', '-o', str(msh_file), '-clscale', '0.2', '-nopopup'],
            capture_output=True,
            timeout=180,
        )
        
        if (idx + 1) % 1000 == 0:
            print(f"  [{idx+1}/{total}] Meshed")
        
        return msh_file.exists() and msh_file.stat().st_size > 100000
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(mesh_variant, var, i, len(variants)): i for i, var in enumerate(variants)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(variants)} meshes generated")
print(f"Output: {output_dir}")
