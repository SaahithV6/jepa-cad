#!/usr/bin/env python3.12
"""Run custom OpenFOAM CFD workloads on all meshed parts."""
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

os.environ['GMSH_NOPOPUP'] = '1'

print("Executing OpenFOAM CFD on all meshed parts (custom axes: pressure, thermal, flow)...\n")

meshes_dir = Path('artifacts/solver-cases-full')
cfd_cases = [d for d in meshes_dir.glob('fea_*') if (d / 'mesh.msh').exists()]

print(f"Total CFD cases: {len(cfd_cases)}\n")

def run_cfd_case(case_dir, idx, total):
    """Run simpleFoam on a single case."""
    try:
        # Run simpleFoam
        result = subprocess.run(
            ['simpleFoam', '-case', str(case_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{total}] CFD complete")
        
        return result.returncode == 0
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(run_cfd_case, case_dir, i, len(cfd_cases)): i
        for i, case_dir in enumerate(cfd_cases)
    }
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(cfd_cases)} CFD simulations complete")
