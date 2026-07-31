#!/usr/bin/env python3.12
"""Generate high-detail Gmsh meshes from STEP files (sequential, millions of elements)."""
import json
import subprocess
import os
from pathlib import Path

os.environ['GMSH_NOPOPUP'] = '1'

class DetailedMeshGenerator:
    def __init__(self, cases_base: Path):
        self.cases_base = cases_base
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def generate_detailed_mesh(self, node_id: str, idx: int, total: int, geometry_ref: str) -> bool:
        """Generate high-detail mesh from STEP file sequentially."""
        case_dir = self.cases_base / f"cfd_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return False
            
            msh_file = case_dir / 'domain.msh'
            
            # High-detail mesh with balanced refinement
            # Target: 500K-2M elements per part = 20-100MB mesh files
            result = subprocess.run(
                [
                    'gmsh',
                    str(step_path),
                    '-3',                          # 3D mesh
                    '-format', 'msh2',              # MSH2 format
                    '-o', str(msh_file),
                    '-clmax', '0.2',                # Max element size: 0.2mm
                    '-clmin', '0.05',               # Min element size: 0.05mm
                    '-nopopup',
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes per part
            )
            
            # Check for success
            if not msh_file.exists():
                return False
            
            size_mb = msh_file.stat().st_size / (1024 * 1024)
            if size_mb < 20:  # Require at least 20MB for detail
                return False
            
            # Parse element count from Gmsh output
            if 'elements' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'elements' in line:
                        print(f"  [{idx+1}/{total}] {node_id}: {size_mb:.1f}MB mesh")
                        break
            
            return True
        
        except Exception as e:
            return False
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating high-detail Gmsh meshes for {len(parts)} parts (sequential)...\n")
        
        successful = 0
        for i, node in enumerate(parts):
            result = self.generate_detailed_mesh(
                node['id'],
                i,
                len(parts),
                node['properties'].get('geometry_ref', '')
            )
            
            if result:
                successful += 1
            
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(parts)} ({successful} successful)")
        
        print(f"\n✓ {successful}/{len(parts)} high-detail meshes generated")
        print(f"Expected: 50-200MB per mesh file")

if __name__ == '__main__':
    gen = DetailedMeshGenerator(Path('artifacts/solver-cases-full'))
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
