#!/usr/bin/env python3.12
"""Run actual OpenFOAM simpleFoam on 5367 CFD cases."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

cases = sorted([d for d in Path('artifacts/cfd_8k').glob('*') if d.is_dir()])[:5367]

print(f"Running OpenFOAM simpleFoam on {len(cases)} CFD cases...\n")

def run_openfoam(case_dir, idx, total):
    """Run simpleFoam on a case."""
    try:
        # Ensure case structure exists
        for d in ['0', 'system', 'constant']:
            (case_dir / d).mkdir(exist_ok=True)
        
        # Create minimal control files if missing
        if not (case_dir / 'system/controlDict').exists():
            (case_dir / 'system/controlDict').write_text(
                "application simpleFoam;\nstartFrom startTime;\nstartTime 0;\n"
                "stopAt endTime;\nendTime 100;\ndeltaT 1;\n"
                "writeControl timeStep;\nwriteInterval 100;\n"
            )
        if not (case_dir / 'system/fvSchemes').exists():
            (case_dir / 'system/fvSchemes').write_text(
                "ddtSchemes { default steadyState; }\n"
                "gradSchemes { default Gauss linear; }\n"
                "divSchemes { default none; div(phi,U) Gauss upwind; }\n"
                "laplacianSchemes { default Gauss linear corrected; }\n"
            )
        if not (case_dir / 'system/fvSolution').exists():
            (case_dir / 'system/fvSolution').write_text(
                "solvers { p { solver GAMG; } U { solver smoothSolver; } }\n"
                "SIMPLE { nNonOrthogonalCorrectors 2; }\n"
            )
        
        # Run simpleFoam
        result = subprocess.run(
            ['/home/best/.local/bin/simpleFoam', '-case', str(case_dir)],
            capture_output=True,
            cwd=str(case_dir),
            timeout=300,
        )
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{total}] CFD complete")
        
        # Check for results
        return (case_dir / '100').exists() or (case_dir / 'postProcessing').exists()
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_openfoam, case, i, len(cases)): i for i, case in enumerate(cases)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(cases)} OpenFOAM CFD simulations complete")

summary = {'total': len(cases), 'successful': successful, 'success_rate': f"{100*successful/len(cases):.1f}%"}
Path('artifacts/cfd_5k_summary.json').write_text(json.dumps(summary, indent=2))
