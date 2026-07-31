#!/usr/bin/env python3.12
"""Remesh the 391 failed parts with finer scale (-clscale 0.1)."""
import json
import subprocess
from pathlib import Path
import os

os.environ['GMSH_NOPOPUP'] = '1'

with open('artifacts/jepa-train-bundle/graph.json') as f:
    graph = json.load(f)

parts = [n for n in graph['nodes'] if n['type'] == 'Part']
meshes_dir = Path('artifacts/solver-cases-full')

# Collect successful IDs
successful_ids = set()
for mesh in meshes_dir.glob('fea_*/mesh.msh'):
    if mesh.stat().st_size > 1000000:
        node_id = mesh.parent.name.replace('fea_', '')
        successful_ids.add(node_id)

failed_parts = [p for p in parts if p['id'] not in successful_ids]

print(f"Remeshing {len(failed_parts)} failed parts with finer scale (-clscale 0.1)...\n")

successful = 0
for i, part in enumerate(failed_parts):
    node_id = part['id']
    geom_ref = part['properties'].get('geometry_ref', '')
    step_path = Path(geom_ref)
    
    if not step_path.exists():
        continue
    
    case_dir = meshes_dir / f"fea_{node_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    msh_file = case_dir / 'mesh.msh'
    
    try:
        # Use finer scale for these difficult parts
        result = subprocess.run(
            [
                'gmsh',
                str(step_path),
                '-3',
                '-format', 'msh2',
                '-o', str(msh_file),
                '-clscale', '0.1',  # Very fine (10% default)
                '-nopopup',
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        
        if msh_file.exists() and msh_file.stat().st_size > 500000:  # >500KB
            successful += 1
            if (i + 1) % 50 == 0:
                size_mb = msh_file.stat().st_size / (1024 * 1024)
                print(f"  [{i+1}/{len(failed_parts)}] {node_id}: {size_mb:.1f}MB")
    
    except:
        pass
    
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(failed_parts)} ({successful} successful)")

print(f"\n✓ {successful}/{len(failed_parts)} previously-failed parts remeshed")
