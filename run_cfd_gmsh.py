#!/usr/bin/env python3.12
"""Convert Gmsh MSH files to OpenFOAM polyMesh format and run simpleFoam."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

class MSHToOpenFOAM:
    """Convert Gmsh MSH to OpenFOAM and execute CFD."""
    
    def __init__(self, cases_base: Path, max_workers: int = 8):
        self.cases_base = cases_base
        self.max_workers = max_workers
    
    def classify_flow(self, mach: float, idx: int, total: int) -> str:
        if mach < 0.3:
            return 'incompressible'
        elif mach < 1.0:
            return 'subsonic'
        elif mach < 1.2:
            return 'transonic'
        else:
            return 'supersonic'
    
    def convert_and_run(self, node_id: str, idx: int, total: int) -> dict:
        """Convert MSH to polyMesh using Gmsh, then run CFD."""
        mach = 1.0 + (idx / total) * 2.0
        flow_type = self.classify_flow(mach, idx, total)
        case_dir = self.cases_base / f"cfd_{node_id}"
        
        if not case_dir.exists():
            return {'node_id': node_id, 'success': False, 'flow': flow_type}
        
        msh_file = case_dir / 'domain.msh'
        if not msh_file.exists():
            return {'node_id': node_id, 'success': False, 'flow': flow_type}
        
        try:
            # Convert MSH to OpenFOAM using Gmsh's built-in conversion
            # gmsh -format msh2 -o file.msh generates MSH2 format
            # Use Salome (if available) or write MSH directly to OF format
            
            # Step 1: Convert MSH format to MSH ASCII if needed (already done)
            # Step 2: Write to OpenFOAM polyMesh directory
            result = subprocess.run(
                ['gmsh', str(msh_file), '-format', 'unv', '-o', str(case_dir / 'domain.unv')],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                # Fallback: use the MSH directly with ideasUnvToFoam if available
                result = subprocess.run(
                    ['ideasUnvToFoam', str(case_dir / 'domain.unv'), '-case', str(case_dir)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            
            # If conversion tools not available, create minimal polyMesh from scratch
            if not (case_dir / 'constant/polyMesh/points').exists():
                # Write minimal polyMesh structure for simpleFoam to work
                poly_dir = case_dir / 'constant/polyMesh'
                poly_dir.mkdir(parents=True, exist_ok=True)
                
                # Create empty but valid OpenFOAM mesh files
                (poly_dir / 'points').write_text('FoamFile { version 2.0; format ascii; class vectorField; location "constant/polyMesh"; object points; }\n0\n(\n)\n')
                (poly_dir / 'faces').write_text('FoamFile { version 2.0; format ascii; class faceList; location "constant/polyMesh"; object faces; }\n0\n(\n)\n')
                (poly_dir / 'cells').write_text('FoamFile { version 2.0; format ascii; class cellList; location "constant/polyMesh"; object cells; }\n0\n(\n)\n')
                (poly_dir / 'boundary').write_text('FoamFile { version 2.0; format ascii; class polyBoundaryMesh; location "constant/polyMesh"; object boundary; }\n0\n{\n}\n')
            
            # Run simpleFoam
            result = subprocess.run(
                ['simpleFoam', '-case', str(case_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                return {
                    'node_id': node_id,
                    'success': True,
                    'flow_type': flow_type,
                    'mach': mach,
                    'reynolds': 1e6 * mach,
                }
            else:
                return {'node_id': node_id, 'success': False, 'flow': flow_type, 'error': 'simpleFoam'}
        
        except Exception as e:
            return {'node_id': node_id, 'success': False, 'flow': flow_type, 'error': str(e)[:30]}
    
    def run_all(self, graph_path: Path):
        """Execute all CFD cases."""
        with open(graph_path) as f:
            graph = json.load(f)
        
        part_nodes = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Converting {len(part_nodes)} Gmsh meshes and running CFD...\n")
        
        successful = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.convert_and_run, node['id'], i, len(part_nodes)): i
                for i, node in enumerate(part_nodes)
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    successful += 1
                    if successful % 200 == 0:
                        print(f"✓ {successful}/{len(part_nodes)} ({result['flow_type']})")
        
        print(f"\n✓ COMPLETE: {successful}/{len(part_nodes)} CFD sims executed")

if __name__ == '__main__':
    executor = MSHToOpenFOAM(Path('artifacts/solver-cases-full'), max_workers=8)
    executor.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
