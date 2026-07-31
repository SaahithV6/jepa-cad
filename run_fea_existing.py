#!/usr/bin/env python3.12
"""Run CalculiX FEA on all existing fea_part:* cases."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

cases_dir = Path('artifacts/solver-cases-full')
fea_cases = sorted([d for d in cases_dir.glob('fea_part:*') if (d / 'mesh.msh').exists()])

print(f"Running CalculiX FEA on {len(fea_cases)} cases...\n")

def run_fea(case_dir, idx, total):
    """Generate INP and run CalculiX."""
    case_name = case_dir.name
    
    try:
        # Create CalculiX INP
        inp_file = case_dir / 'case.inp'
        inp_content = f"""*HEADING
{case_name} FEA
*INCLUDE, INPUT=mesh.msh
*MATERIAL, NAME=Steel
*ELASTIC
210000, 0.3
*STEP
*STATIC
*BOUNDARY
1, 1, 1, 0.0
*CLOAD
1, 1, 1000.0
*NODE FILE
U, S
*END STEP
"""
        inp_file.write_text(inp_content)
        
        # Run CalculiX
        result = subprocess.run(
            ['ccx_2.20', str(inp_file).replace('.inp', '')],
            capture_output=True,
            cwd=str(case_dir),
            timeout=180,
        )
        
        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{total}] FEA complete")
        
        return result.returncode == 0
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_fea, case, i, len(fea_cases)): i for i, case in enumerate(fea_cases)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(fea_cases)} FEA simulations complete")

summary = {
    'total_cases': len(fea_cases),
    'successful': successful,
    'success_rate': f"{100*successful/len(fea_cases):.1f}%" if fea_cases else "N/A",
}

Path('artifacts/fea_summary.json').write_text(json.dumps(summary, indent=2))
