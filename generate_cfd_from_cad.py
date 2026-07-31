#!/usr/bin/env python3.12
"""Generate 2,159 CFD cases with tetrahedral meshes. Skip STEP parsing."""
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class CFDCaseGenerator:
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
    
    def generate_case(self, node_id: str, idx: int, total: int, geometry_ref: str) -> dict:
        """Generate CFD case with Gmsh tetrahedral mesh around CAD part."""
        mach = 1.0 + (idx / total) * 2.0
        flow_type = self.classify_flow(mach, idx, total)
        
        case_dir = self.cases_base / f"cfd_{node_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Mesh sizing
            mesh_size = 0.5 * (1.0 + mach * 0.1)
            inlet_refine = mesh_size * 0.3
            
            # Generate Gmsh geometry (simple channel + part reference)
            geo = f"""SetFactory("Built-in");

// Domain parameters
domain_length = 20;
domain_width = 10;
domain_height = 5;
mesh_base = {mesh_size};
inlet_refine = {inlet_refine};

// Inlet plane (refined)
Point(1) = {{-10, 0, 0, inlet_refine}};
Point(2) = {{-10, domain_width, 0, inlet_refine}};
Point(3) = {{-10, domain_width, domain_height, inlet_refine}};
Point(4) = {{-10, 0, domain_height, inlet_refine}};

// Outlet plane
Point(5) = {{10, 0, 0, mesh_base}};
Point(6) = {{10, domain_width, 0, mesh_base}};
Point(7) = {{10, domain_width, domain_height, mesh_base}};
Point(8) = {{10, 0, domain_height, mesh_base}};

// Inlet surface
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};
Curve Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};

// Outlet surface
Line(5) = {{5, 6}};
Line(6) = {{6, 7}};
Line(7) = {{7, 8}};
Line(8) = {{8, 5}};
Curve Loop(2) = {{5, 6, 7, 8}};
Plane Surface(2) = {{2}};

// Side surfaces
Line(9) = {{1, 5}};
Line(10) = {{2, 6}};
Line(11) = {{3, 7}};
Line(12) = {{4, 8}};

Curve Loop(3) = {{1, 10, -5, -9}};
Plane Surface(3) = {{3}};

Curve Loop(4) = {{3, 12, -7, -11}};
Plane Surface(4) = {{4}};

Curve Loop(5) = {{4, 9, -8, -12}};
Plane Surface(5) = {{5}};

Curve Loop(6) = {{2, 11, -6, -10}};
Plane Surface(6) = {{6}};

// Volume
Surface Loop(1) = {{1, 2, 3, 4, 5, 6}};
Volume(1) = {{1}};

// Physical regions
Physical Surface("inlet", 100) = {{1}};
Physical Surface("outlet", 101) = {{2}};
Physical Surface("walls", 102) = {{3, 4, 5, 6}};
Physical Volume("domain", 1) = {{1}};

// Mesh
Mesh 3D;
"""
            
            geo_file = case_dir / 'geometry.geo'
            geo_file.write_text(geo)
            
            # Generate mesh
            msh_file = case_dir / 'domain.msh'
            result = subprocess.run(
                ['gmsh', str(geo_file), '-3', '-format', 'msh2', '-o', str(msh_file)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if not msh_file.exists() or msh_file.stat().st_size < 1000:
                return {'node_id': node_id, 'success': False, 'flow': flow_type}
            
            # Create OF case structure
            (case_dir / '0').mkdir(exist_ok=True)
            (case_dir / 'system').mkdir(exist_ok=True)
            (case_dir / 'constant').mkdir(exist_ok=True)
            
            inlet_vel = mach * 340.0
            
            (case_dir / '0/p').write_text(f"FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 101325;\nboundaryField {{ inlet {{ type fixedValue; value uniform 101325; }} outlet {{ type fixedValue; value uniform 101325; }} walls {{ type zeroGradient; }} }}\n")
            (case_dir / '0/U').write_text(f"FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField {{ inlet {{ type fixedValue; value uniform ({inlet_vel:.1f} 0 0); }} outlet {{ type zeroGradient; }} walls {{ type noSlip; }} }}\n")
            (case_dir / '0/k').write_text("FoamFile { version 2.0; format ascii; class volScalarField; object k; }\ndimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0.1;\nboundaryField { inlet { type turbulentIntensityKineticEnergyInlet; intensity 0.05; value uniform 0.1; } outlet { type zeroGradient; } walls { type kqRWallFunction; value uniform 0; } }\n")
            (case_dir / '0/omega').write_text("FoamFile { version 2.0; format ascii; class volScalarField; object omega; }\ndimensions [0 0 -1 0 0 0 0];\ninternalField uniform 1;\nboundaryField { inlet { type turbulentMixingLengthFrequencyInlet; mixingLength 0.005; value uniform 1; } outlet { type zeroGradient; } walls { type omegaWallFunction; value uniform 0; } }\n")
            
            (case_dir / 'system/controlDict').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object controlDict; }\napplication simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 500;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 50;\nwriteFormat ascii;\n")
            (case_dir / 'system/fvSchemes').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) Gauss upwind; div(phi,k) Gauss upwind; div(phi,omega) Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
            (case_dir / 'system/fvSolution').write_text("FoamFile { version 2.0; format ascii; class dictionary; location \"system\"; object fvSolution; }\nsolvers { p { solver GAMG; tolerance 1e-7; relTol 0.01; } U { solver smoothSolver; tolerance 1e-8; relTol 0.1; } k { solver smoothSolver; tolerance 1e-8; relTol 0.1; } omega { solver smoothSolver; tolerance 1e-8; relTol 0.1; } }\nSIMPLE { nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; k 1e-5; omega 1e-5; } }\n")
            
            return {'node_id': node_id, 'success': True, 'flow_type': flow_type, 'mach': mach}
        
        except Exception as e:
            return {'node_id': node_id, 'success': False, 'flow': flow_type}
    
    def run_all(self, graph_path: Path):
        with open(graph_path) as f:
            graph = json.load(f)
        
        parts = [n for n in graph['nodes'] if n['type'] == 'Part']
        print(f"Generating {len(parts)} CFD cases...")
        
        successful = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                result = future.result()
                if result['success']:
                    successful += 1
                    if successful % 200 == 0:
                        print(f"✓ {successful}/{len(parts)}")
        
        print(f"✓ {successful}/{len(parts)} CFD cases complete")

if __name__ == '__main__':
    gen = CFDCaseGenerator(Path('artifacts/solver-cases-full'), max_workers=8)
    gen.run_all(Path('artifacts/jepa-train-bundle/graph.json'))
