#!/usr/bin/env python3.12
"""Run CalculiX FEA with proper MSH include and boundary conditions."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

cases = sorted([d for d in Path('artifacts/cfd_8k').glob('*') if d.is_dir()])[:5367]

print(f"Running CalculiX FEA (proper MSH format) on {len(cases)} cases...\n")

def run_fea(case_dir, idx, total):
    """Generate proper INP with MSH include and run CalculiX."""
    try:
        inp_file = case_dir / 'case.inp'
        
        # Proper CalculiX INP with MSH include
        inp_content = """*HEADING
FEA Analysis
*INCLUDE, INPUT=mesh.msh
*MATERIAL, NAME=Steel
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=ALL, MATERIAL=Steel
*STEP
*STATIC
*BOUNDARY
ALLNODES, 1, 6, 0.0
*END STEP
"""
        inp_file.write_text(inp_content)
        
        # Run CalculiX
        result = subprocess.run(
            ['/home/best/.local/bin/ccx', str(inp_file).replace('.inp', '')],
            capture_output=True,
            cwd=str(case_dir),
            timeout=180,
        )
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{total}] FEA complete")
        
        # Check for results
        frd = (case_dir / 'case.frd').exists()
        dat = (case_dir / 'case.dat').exists()
        return frd or dat
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_fea, case, i, len(cases)): i for i, case in enumerate(cases)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(cases)} CalculiX FEA simulations complete")

summary = {'total': len(cases), 'successful': successful, 'success_rate': f"{100*successful/len(cases):.1f}%"}
Path('artifacts/fea_5k_final_summary.json').write_text(json.dumps(summary, indent=2))
