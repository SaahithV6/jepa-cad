#!/usr/bin/env python3.12
"""Generate CFD case structure from FEA meshes and run OpenFOAM."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

os.environ['GMSH_NOPOPUP'] = '1'

cases_dir = Path('artifacts/solver-cases-full')
fea_cases = [d for d in cases_dir.glob('fea_*') if (d / 'mesh.msh').exists()]

print(f"Creating CFD cases from {len(fea_cases)} FEA meshes...\n")

def setup_and_run_cfd(fea_case_dir, idx, total):
    """Create CFD structure and run simpleFoam."""
    node_id = fea_case_dir.name.replace('fea_', '')
    cfd_case_dir = cases_dir / f"cfd_{node_id}"
    cfd_case_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy mesh
        import shutil
        mesh_src = fea_case_dir / 'mesh.msh'
        mesh_dst = cfd_case_dir / 'mesh.msh'
        if not mesh_dst.exists():
            shutil.copy(mesh_src, mesh_dst)
        
        # Create OpenFOAM case structure
        for d in ['0', 'system', 'constant']:
            (cfd_case_dir / d).mkdir(exist_ok=True)
        
        # Minimal controlDict
        (cfd_case_dir / 'system/controlDict').write_text(
            "application simpleFoam;\n"
            "startFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 100;\n"
            "deltaT 1;\nwriteControl timeStep;\nwriteInterval 100;\n"
        )
        (cfd_case_dir / 'system/fvSchemes').write_text(
            "ddtSchemes { default steadyState; }\n"
            "gradSchemes { default Gauss linear; }\n"
            "divSchemes { default none; div(phi,U) Gauss upwind; }\n"
            "laplacianSchemes { default Gauss linear corrected; }\n"
        )
        (cfd_case_dir / 'system/fvSolution').write_text(
            "solvers { p { solver GAMG; } U { solver smoothSolver; } }\n"
            "SIMPLE { nNonOrthogonalCorrectors 2; }\n"
        )
        
        # Run simpleFoam
        result = subprocess.run(
            ['simpleFoam', '-case', str(cfd_case_dir)],
            capture_output=True,
            timeout=300,
        )
        
        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{total}] CFD complete")
        
        return result.returncode == 0
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(setup_and_run_cfd, case, i, len(fea_cases)): i
        for i, case in enumerate(fea_cases)
    }
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"\n✓ {successful}/{len(fea_cases)} CFD simulations complete")
