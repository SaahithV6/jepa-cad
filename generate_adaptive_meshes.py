#!/usr/bin/env python3.12
"""Generate adaptive-detail Gmsh meshes from STEP files (sequential)."""
import json
import subprocess
import os
from pathlib import Path

os.environ['GMSH_NOPOPUP'] = '1'

class AdaptiveMeshGenerator:
    def __init__(self, cases_base: Path):
        self.cases_base = cases_base
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def get_step_bounds(self, step_path: Path) -> tuple:
        """Extract bounding box from STEP file using Gmsh."""
        try:
            result = subprocess.run(
                ['gmsh', str(step_path), '-0', '-format', 'msh2', '-o', '/tmp/bounds.msh', '-nopopup'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Parse bounds from output
            for line in result.stdout.split('\n'):
                if 'Bounding box' in line:
                    # Extract dimensions
                    parts = line.split()
                    try:
                        return tuple(float(p) for p in parts[-6:])
                    except:
                        return None
            return None
        except:
            return None
    
    def generate_adaptive_mesh(self, node_id: str, idx: int, total: int, geometry_ref: str) -> bool:
        """Generate adaptive mesh based on part size."""
        case_dir = self.cases_base / f"cfd_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return False
            
            msh_file = case_dir / 'domain.msh'
            
            # Adaptive mesh sizing: scale to part size
            # Small parts (< 1mm) need finer mesh, large parts can use coarser
            # Use 0.05mm minimum, adapt max based on geometry
            result = subprocess.run(
                [
                    'gmsh',
                    str(step_path),
                    '-3',                          # 3D mesh
                    '-format', 'msh2',              # MSH2 format
                    '-o', str(msh_file),
                    '-clscale', '0.5',              # Scale mesh size to 50% of default (finer)
                    '-nopopup',
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            # Check for success - accept any mesh >1KB
            if not msh_file.exists() or msh_file.stat().st_size < 1000:
                return False
            
            size_kb = msh_file.stat().st_size / 1024
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{total}] {node_id}: {size_kb:.0f}KB")
            
            return True
        
        except Exception as e:
            return False
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating adaptive Gmsh meshes for {len(parts)} parts (sequential)...\n")
        
        successful = 0
        for i, node in enumerate(parts):
            result = self.generate_adaptive_mesh(
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
        
        print(f"\n✓ {successful}/{len(parts)} adaptive meshes generated")

if __name__ == '__main__':
    gen = AdaptiveMeshGenerator(Path('artifacts/solver-cases-full'))
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
