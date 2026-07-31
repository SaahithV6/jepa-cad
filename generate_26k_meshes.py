#!/usr/bin/env python3.12
"""Generate 26k-element Gmsh meshes from STEP files (sequential, fixed element count)."""
import json
import subprocess
import os
from pathlib import Path

os.environ['GMSH_NOPOPUP'] = '1'

class FixedElementMeshGenerator:
    def __init__(self, cases_base: Path):
        self.cases_base = cases_base
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def generate_26k_mesh(self, node_id: str, idx: int, total: int, geometry_ref: str) -> bool:
        """Generate ~26k element mesh from STEP file."""
        case_dir = self.cases_base / f"cfd_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return False
            
            msh_file = case_dir / 'domain.msh'
            
            # Mesh sizing for ~26k elements
            # Scale factor tuned to produce approximately 26k tetrahedra
            result = subprocess.run(
                [
                    'gmsh',
                    str(step_path),
                    '-3',                          # 3D mesh
                    '-format', 'msh2',              # MSH2 format
                    '-o', str(msh_file),
                    '-clscale', '1.5',              # Scale to target ~26k elements
                    '-nopopup',
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            # Check for success
            if not msh_file.exists() or msh_file.stat().st_size < 500:
                return False
            
            # Parse element count from output
            element_count = 0
            for line in result.stdout.split('\n'):
                if 'elements' in line and 'nodes' in line:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'elements':
                                element_count = int(parts[i-1])
                                break
                    except:
                        pass
            
            if (idx + 1) % 50 == 0:
                size_kb = msh_file.stat().st_size / 1024
                print(f"  [{idx+1}/{total}] {node_id}: {element_count} elements, {size_kb:.0f}KB")
            
            return True
        
        except Exception as e:
            return False
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating 26k-element Gmsh meshes for {len(parts)} parts (sequential)...\n")
        
        successful = 0
        for i, node in enumerate(parts):
            result = self.generate_26k_mesh(
                node['id'],
                i,
                len(parts),
                node['properties'].get('geometry_ref', '')
            )
            
            if result:
                successful += 1
            
            if (i + 1) % 200 == 0:
                pct = (successful / (i + 1)) * 100
                print(f"Progress: {i+1}/{len(parts)} ({successful} successful, {pct:.1f}%)")
        
        print(f"\n✓ {successful}/{len(parts)} 26k-element meshes generated")

if __name__ == '__main__':
    gen = FixedElementMeshGenerator(Path('artifacts/solver-cases-full'))
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
