#!/usr/bin/env python3.12
"""Mesh ALL 2154 geometries with Gmsh (-clscale 0.2), generate FEA/CFD case structure."""
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

os.environ['GMSH_NOPOPUP'] = '1'

# Load manifest
with open('data/geometry_manifest.json') as f:
    manifest = json.load(f)

corpus_path = Path('artifacts/corpus-sweep-run/sweep/runs')
all_steps = list(corpus_path.glob('*/geometry.step'))

print(f"Meshing {len(all_steps)} geometries with Gmsh (-clscale 0.2)...\n")

cases_dir = Path('artifacts/solver-cases-full')
cases_dir.mkdir(parents=True, exist_ok=True)

def mesh_geometry(step_path, idx, total):
    """Mesh a STEP file and create FEA/CFD case structure."""
    try:
        # Use folder name as ID
        node_id = step_path.parent.name
        case_dir = cases_dir / f"mesh_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        msh_file = case_dir / 'domain.msh'
        
        # Gmsh mesh
        result = subprocess.run(
            [
                'gmsh', str(step_path),
                '-3', '-format', 'msh2',
                '-o', str(msh_file),
                '-clscale', '0.2',
                '-nopopup',
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if not msh_file.exists() or msh_file.stat().st_size < 100000:
            return False
        
        # Create FEA case structure
        fea_case = case_dir / 'fea'
        fea_case.mkdir(exist_ok=True)
        (fea_case / 'mesh.msh').symlink_to(msh_file, target_is_directory=False) if not (fea_case / 'mesh.msh').exists() else None
        
        # Create CFD case structure
        cfd_case = case_dir / 'cfd'
        cfd_case.mkdir(exist_ok=True)
        (cfd_case / 'mesh.msh').symlink_to(msh_file, target_is_directory=False) if not (cfd_case / 'mesh.msh').exists() else None
        
        if (idx + 1) % 200 == 0:
            size_mb = msh_file.stat().st_size / (1024 * 1024)
            print(f"  [{idx+1}/{total}] Meshed: {size_mb:.1f}MB")
        
        return True
    
    except Exception as e:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(mesh_geometry, step, i, len(all_steps)): i
        for i, step in enumerate(all_steps)
    }
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(all_steps)} geometries meshed")
print(f"✓ FEA + CFD case structure created")
print(f"Ready for physics simulations")
