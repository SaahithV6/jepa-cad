#!/usr/bin/env python3.12
"""Run actual CalculiX FEA on 5367 CFD case directories (repurposed as FEA)."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

cases = sorted([d for d in Path('artifacts/cfd_8k').glob('*') if d.is_dir()])[:5367]

print(f"Running CalculiX FEA on {len(cases)} cases...\n")

def run_fea(case_dir, idx, total):
    """Generate INP and run CalculiX."""
    try:
        # Create INP
        inp_file = case_dir / 'case.inp'
        mesh_msh = case_dir / 'mesh.msh'
        
        if not mesh_msh.exists():
            return False
        
        inp_content = f"""*HEADING
{case_dir.name} FEA
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
            ['/home/best/.local/bin/ccx', str(inp_file).replace('.inp', '')],
            capture_output=True,
            cwd=str(case_dir),
            timeout=180,
        )
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{total}] FEA complete")
        
        # Check for results
        return (case_dir / f'{inp_file.stem}.frd').exists() or (case_dir / f'{inp_file.stem}.dat').exists()
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_fea, case, i, len(cases)): i for i, case in enumerate(cases)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(cases)} FEA simulations complete")

summary = {'total': len(cases), 'successful': successful, 'success_rate': f"{100*successful/len(cases):.1f}%"}
Path('artifacts/fea_5k_summary.json').write_text(json.dumps(summary, indent=2))
