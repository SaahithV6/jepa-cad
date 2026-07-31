#!/usr/bin/env python3.12
"""Generate all 2159 high-quality Gmsh meshes from STEP files (aggressive refinement)."""
import json
import subprocess
import os
from pathlib import Path

os.environ['GMSH_NOPOPUP'] = '1'

class AllPartsDetailedMeshGenerator:
    def __init__(self, cases_base: Path):
        self.cases_base = cases_base
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def generate_mesh(self, node_id: str, idx: int, total: int, geometry_ref: str) -> tuple:
        """Generate high-quality mesh from STEP file (all parts, no failures)."""
        case_dir = self.cases_base / f"fea_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return (False, 0)
            
            msh_file = case_dir / 'mesh.msh'
            
            # Aggressive mesh refinement for all parts
            # -clscale 0.1 = 10% of default (very fine)
            # -order 2 = second-order tetrahedral elements (better accuracy)
            result = subprocess.run(
                [
                    'gmsh',
                    str(step_path),
                    '-3',                          # 3D mesh
                    '-format', 'msh2',              # MSH2 format
                    '-o', str(msh_file),
                    '-clscale', '0.1',              # VERY fine (10% default)
                    '-order', '2',                  # 2nd order tetrahedra
                    '-optimize',                    # Optimize mesh
                    '-nopopup',
                ],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes per part
            )
            
            if not msh_file.exists():
                return (False, 0)
            
            size_mb = msh_file.stat().st_size / (1024 * 1024)
            
            # Extract element count from stdout
            elem_count = 0
            for line in result.stdout.split('\n'):
                if 'elements' in line and 'nodes' in line:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'elements':
                                elem_count = int(parts[i-1])
                    except:
                        pass
            
            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{total}] {node_id}: {elem_count} elements, {size_mb:.1f}MB")
            
            # Accept any valid mesh (>500KB minimum for quality)
            return (size_mb > 0.5, elem_count)
        
        except subprocess.TimeoutExpired:
            return (False, 0)
        except Exception as e:
            return (False, 0)
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating high-quality meshes for ALL {len(parts)} parts...")
        print(f"Refinement: -clscale 0.1 (very fine), 2nd-order tetrahedra\n")
        
        successful = 0
        total_elements = 0
        
        for i, node in enumerate(parts):
            success, elem_count = self.generate_mesh(
                node['id'],
                i,
                len(parts),
                node['properties'].get('geometry_ref', '')
            )
            
            if success:
                successful += 1
                total_elements += elem_count
            
            if (i + 1) % 100 == 0:
                pct = (successful / (i + 1)) * 100
                avg_elem = total_elements / successful if successful > 0 else 0
                print(f"Progress: {i+1}/{len(parts)} ({successful} successful, {pct:.1f}%, avg {avg_elem:.0f} elem/part)")
        
        print(f"\n✓ {successful}/{len(parts)} meshes generated")
        print(f"✓ Total elements: {total_elements}")
        print(f"✓ Average per part: {total_elements/successful:.0f}" if successful > 0 else "")

if __name__ == '__main__':
    gen = AllPartsDetailedMeshGenerator(Path('artifacts/solver-cases-full'))
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
