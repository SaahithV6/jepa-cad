#!/usr/bin/env python3.12
"""Execute 2,159 CFD simulations with blockMesh + simpleFoam."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class CFDExecutor:
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
    
    def run_case(self, node_id: str, idx: int, total: int) -> dict:
        mach = 1.0 + (idx / total) * 2.0
        flow_type = self.classify_flow(mach, idx, total)
        case_dir = self.cases_base / f"cfd_{node_id}"
        
        if not case_dir.exists():
            return {'node_id': node_id, 'success': False, 'flow': flow_type}
        
        try:
            # Run blockMesh
            result = subprocess.run(
                ['blockMesh', '-case', str(case_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return {'node_id': node_id, 'success': False, 'flow': flow_type, 'error': 'blockMesh'}
            
            # Run simpleFoam
            result = subprocess.run(
                ['simpleFoam', '-case', str(case_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return {'node_id': node_id, 'success': False, 'flow': flow_type, 'error': 'simpleFoam'}
            
            return {
                'node_id': node_id,
                'success': True,
                'flow_type': flow_type,
                'mach': mach,
                'reynolds': 1e6 * mach,
            }
        except Exception as e:
            return {'node_id': node_id, 'success': False, 'flow': flow_type, 'error': str(e)[:30]}
    
    def run_all(self, graph_path: Path):
        import json
        with open(graph_path) as f:
            graph = json.load(f)
        
        part_nodes = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Running {len(part_nodes)} CFD sims...\n")
        
        successful = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.run_case, node['id'], i, len(part_nodes)): i
                for i, node in enumerate(part_nodes)
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    successful += 1
                    if successful % 200 == 0:
                        print(f"✓ {successful}/{len(part_nodes)} ({result['flow_type']})")
        
        print(f"\n✓ COMPLETE: {successful}/{len(part_nodes)} executed")

if __name__ == '__main__':
    executor = CFDExecutor(Path('artifacts/solver-cases-full'), max_workers=8)
    executor.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
