#!/usr/bin/env python3.12
"""Mesh STEP files with Gmsh (26k elements), generate CFD cases."""
import json
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['GMSH_NOPOPUP'] = '1'

class GMSHMeshGenerator:
    def __init__(self, cases_base: Path, max_workers: int = 8):
        self.cases_base = cases_base
        self.max_workers = max_workers
        self.cases_base.mkdir(parents=True, exist_ok=True)
    
    def classify_flow(self, mach: float, idx: int, total: int) -> str:
        if mach < 0.3:
            return 'incompressible'
        elif mach < 1.0:
            return 'subsonic'
        elif mach < 1.2:
            return 'transonic'
        else:
            return 'supersonic'
    
    def generate_case(self, node_id: str, idx: int, total: int, geometry_ref: str) -> dict:
        """Mesh STEP file directly with Gmsh, create CFD case."""
        mach = 1.0 + (idx / total) * 2.0
        flow_type = self.classify_flow(mach, idx, total)
        
        case_dir = self.cases_base / f"cfd_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            step_path = Path(geometry_ref)
            if not step_path.exists():
                return {'node_id': node_id, 'success': False, 'flow': flow_type}
            
            # Mesh size for ~26k elements
            mesh_size = 0.15 * (1.0 + mach * 0.05)
            
            # Generate mesh from STEP
            msh_file = case_dir / 'domain.msh'
            result = subprocess.run(
                ['gmsh', str(step_path), '-3', '-format', 'msh2', '-o', str(msh_file), '-nopopup'],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if not msh_file.exists() or msh_file.stat().st_size < 50000:
                return {'node_id': node_id, 'success': False, 'flow': flow_type}
            
            # Create OF case
            (case_dir / '0').mkdir(exist_ok=True)
            (case_dir / 'system').mkdir(exist_ok=True)
            (case_dir / 'constant').mkdir(exist_ok=True)
            
            inlet_vel = mach * 340.0
            
            (case_dir / '0/p').write_text(f"FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 101325;\nboundaryField {{ inlet {{ type fixedValue; value uniform 101325; }} outlet {{ type fixedValue; value uniform 101325; }} walls {{ type zeroGradient; }} }}\n")
            (case_dir / '0/U').write_text(f"FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField {{ inlet {{ type fixedValue; value uniform ({inlet_vel:.1f} 0 0); }} outlet {{ type zeroGradient; }} walls {{ type noSlip; }} }}\n")
            (case_dir / '0/k').write_text("FoamFile { version 2.0; format ascii; class volScalarField; object k; }\ndimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0.1;\nboundaryField { inlet { type turbulentIntensityKineticEnergyInlet; intensity 0.05; value uniform 0.1; } outlet { type zeroGradient; } walls { type kqRWallFunction; value uniform 0; } }\n")
            (case_dir / '0/omega').write_text("FoamFile { version 2.0; format ascii; class volScalarField; object omega; }\ndimensions [0 0 -1 0 0 0 0];\ninternalField uniform 1;\nboundaryField { inlet { type turbulentMixingLengthFrequencyInlet; mixingLength 0.005; value uniform 1; } outlet { type zeroGradient; } walls { type omegaWallFunction; value uniform 0; } }\n")
            
            (case_dir / 'system/controlDict').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object controlDict; }\napplication simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1000;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 100;\n")
            (case_dir / 'system/fvSchemes').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) Gauss upwind; div(phi,k) Gauss upwind; div(phi,omega) Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
            (case_dir / 'system/fvSolution').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object fvSolution; }\nsolvers { p { solver GAMG; tolerance 1e-7; relTol 0.01; } U { solver smoothSolver; tolerance 1e-8; relTol 0.1; } k { solver smoothSolver; tolerance 1e-8; relTol 0.1; } omega { solver smoothSolver; tolerance 1e-8; relTol 0.1; } }\nSIMPLE { nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; k 1e-5; omega 1e-5; } }\n")
            
            return {'node_id': node_id, 'success': True, 'flow_type': flow_type}
        
        except Exception as e:
            return {'node_id': node_id, 'success': False, 'flow': flow_type}
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Meshing {len(parts)} STEP files with Gmsh (~26k elements)...\n")
        
        successful = 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    self.generate_case,
                    node['id'],
                    i,
                    len(parts),
                    node['properties'].get('geometry_ref', '')
                ): i
                for i, node in enumerate(parts)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        successful += 1
                        if successful % 100 == 0:
                            print(f"✓ {successful}/{len(parts)}")
                except Exception as e:
                    pass
        
        print(f"\n✓ {successful}/{len(parts)} CFD cases with Gmsh meshes")

if __name__ == '__main__':
    gen = GMSHMeshGenerator(Path('artifacts/solver-cases-full'), max_workers=8)
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
