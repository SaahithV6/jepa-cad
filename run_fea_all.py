#!/usr/bin/env python3.12
"""Run CalculiX FEA on all 2108 meshed geometries."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

cases_dir = Path('artifacts/solver-cases-full')
fea_cases = sorted([d for d in cases_dir.glob('mesh_*/fea') if (d / 'mesh.msh').exists()])

print(f"Running CalculiX FEA on {len(fea_cases)} meshed geometries...\n")

def run_fea(fea_case_dir, idx, total):
    """Generate INP and run CalculiX."""
    case_name = fea_case_dir.parent.name
    
    try:
        # Create minimal CalculiX INP
        inp_file = fea_case_dir / 'case.inp'
        inp_content = f"""*HEADING
{case_name} FEA Simulation
*INCLUDE, INPUT=mesh.msh
*MATERIAL, NAME=Aluminum
*ELASTIC
70000, 0.3
*PHYSICAL CONSTANTS, ABSOLUTE ZERO=-273.15, STEFAN BOLTZMANN=5.670E-8
*STEP
*STATIC
*BOUNDARY
INLET, 1, 1, 0.0
*CLOAD
1, 1, 1000.0
*NODE FILE
U, S
*EL FILE
S
*END STEP
"""
        inp_file.write_text(inp_content)
        
        # Run CalculiX
        result = subprocess.run(
            ['ccx_2.20', str(inp_file).replace('.inp', '')],
            capture_output=True,
            cwd=str(fea_case_dir),
            timeout=300,
        )
        
        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{total}] FEA simulations completed")
        
        return result.returncode == 0
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(run_fea, case, i, len(fea_cases)): i
        for i, case in enumerate(fea_cases)
    }
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(fea_cases)} FEA simulations complete")

# Summary
summary = {
    'total_cases': len(fea_cases),
    'successful': successful,
    'success_rate': f"{100*successful/len(fea_cases):.1f}%" if fea_cases else "N/A",
}

Path('artifacts/fea_summary.json').write_text(json.dumps(summary, indent=2))
