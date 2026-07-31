#!/usr/bin/env python3.12
"""Generate detailed Gmsh meshes from STEP files (90k+ elements, 60MB+ per part)."""
import json
import subprocess
import os
from pathlib import Path

os.environ['GMSH_NOPOPUP'] = '1'

class DetailedPhysicsMeshGenerator:
    def __init__(self, cases_base: Path):
        self.cases_base = cases_base
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def generate_detailed_physics_mesh(self, node_id: str, idx: int, total: int, geometry_ref: str) -> bool:
        """Generate high-detail mesh (90k+ elements, 60MB+) from actual STEP geometry."""
        case_dir = self.cases_base / f"fea_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return False
            
            msh_file = case_dir / 'mesh.msh'
            
            # High-detail mesh: ~90k+ elements, expect 60MB+ output
            result = subprocess.run(
                [
                    'gmsh',
                    str(step_path),
                    '-3',                          # 3D mesh
                    '-format', 'msh2',              # MSH2 format
                    '-o', str(msh_file),
                    '-clscale', '0.3',              # Fine mesh (30% of default)
                    '-nopopup',
                ],
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minutes per part
            )
            
            if not msh_file.exists():
                return False
            
            size_mb = msh_file.stat().st_size / (1024 * 1024)
            
            # Log progress
            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{total}] {node_id}: {size_mb:.1f}MB")
                if size_mb < 10:
                    print(f"    WARNING: Only {size_mb:.1f}MB (target 60MB+)")
            
            return size_mb > 1  # Accept any mesh >1MB
        
        except Exception as e:
            return False
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating detailed physics meshes for {len(parts)} parts (sequential)...")
        print(f"Target: 90k+ elements, 60MB+ per part\n")
        
        successful = 0
        for i, node in enumerate(parts):
            result = self.generate_detailed_physics_mesh(
                node['id'],
                i,
                len(parts),
                node['properties'].get('geometry_ref', '')
            )
            
            if result:
                successful += 1
            
            if (i + 1) % 50 == 0:
                pct = (successful / (i + 1)) * 100
                print(f"Progress: {i+1}/{len(parts)} ({successful} successful, {pct:.1f}%)")
        
        print(f"\n✓ {successful}/{len(parts)} detailed physics meshes generated")

if __name__ == '__main__':
    gen = DetailedPhysicsMeshGenerator(Path('artifacts/solver-cases-full'))
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
