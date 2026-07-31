#!/usr/bin/env python3.12
"""Discover and catalog all spaceflight geometry sources for JEPA training data."""

import json
from pathlib import Path

sources = {
    "NASA_Artemis": {
        "url": "https://nasa-public-data.s3.amazonaws.com",
        "description": "SLS, Orion spacecraft STEP files",
        "format": ["STEP", "IGES"],
        "priority": 1,
        "coverage": ["rocket_structure", "spacecraft_bus", "thermal_protection", "docking_mechanism"],
    },
    "ESA_Scifleet": {
        "url": "https://www.esa.int/ESA_Multimedia/Missions",
        "description": "European spacecraft CAD models",
        "format": ["STEP"],
        "priority": 1,
        "coverage": ["satellite", "launcher", "service_module"],
    },
    "JAXA_Datasets": {
        "url": "https://www.jaxa.jp/",
        "description": "Japan Aerospace Exploration Agency models",
        "format": ["STEP", "CAD"],
        "priority": 2,
        "coverage": ["launcher", "satellite", "thermal"],
    },
    "OpenRocket": {
        "url": "https://github.com/OpenRocket/OpenRocket",
        "description": "Parametric rocket design engine + STEP export",
        "format": ["STEP", "parametric_API"],
        "priority": 1,
        "coverage": ["nose_cone", "fin", "tube", "engine_mount", "parachute_bay"],
        "note": "~500+ community designs available",
    },
    "BIRDS4_CAD": {
        "url": "https://github.com/BIRDSOpenSource/BIRDS4-CAD",
        "description": "University satellite real hardware CAD",
        "format": ["STEP", "IGES"],
        "priority": 1,
        "coverage": ["cubesat", "structure", "solar_panel", "antenna", "bus"],
        "note": "Verified flight-ready designs",
    },
    "BIRDSX_CAD": {
        "url": "https://github.com/BIRDSOpenSource/BIRDSX-CAD",
        "description": "Extended university satellite library",
        "format": ["STEP"],
        "priority": 2,
        "coverage": ["satellite_variants", "structure", "payload_bay"],
    },
    "RocketPy": {
        "url": "https://github.com/RocketPy/RocketPy",
        "description": "Python rocket flight simulator with parametric geometry export",
        "format": ["STEP", "parametric_API"],
        "priority": 1,
        "coverage": ["rocket_assembly", "trajectory_optimized_shapes"],
        "note": "Can generate variations by parameters",
    },
    "SpaceX_OpenSource": {
        "url": "https://github.com/SpaceX",
        "description": "SpaceX public technical documentation (limited CAD)",
        "format": ["PDF_specs", "technical_drawings"],
        "priority": 3,
        "coverage": ["starship_specs", "falcon_specs"],
        "note": "Mostly specs, limited CAD",
    },
    "BlueOrigin_Public": {
        "url": "https://www.blueorigin.com/",
        "description": "New Shepard, New Glenn technical specs",
        "format": ["technical_drawings"],
        "priority": 3,
        "coverage": ["suborbital_launcher", "orbital_launcher"],
    },
    "Relativity_Space": {
        "url": "https://www.relativityspace.com/",
        "description": "3D printed rocket parts + parametric design",
        "format": ["CAD", "parametric"],
        "priority": 2,
        "coverage": ["printed_structures", "tank", "engine_nozzle"],
    },
    "Rocket_Lab": {
        "url": "https://www.rocketlabusa.com/",
        "description": "Electron launcher specs + ISS resupply",
        "format": ["technical_specs"],
        "priority": 2,
        "coverage": ["small_launcher", "cubesat_dispenser"],
    },
    "Axiom_Space": {
        "url": "https://www.axiomspace.com/",
        "description": "Commercial space station modules",
        "format": ["CAD"],
        "priority": 2,
        "coverage": ["pressurized_module", "docking_adaptor", "solar_array"],
    },
    "TU_Delft_Aerospace": {
        "url": "https://www.tudelft.nl/",
        "description": "University drone + rocket projects",
        "format": ["STEP", "CAD"],
        "priority": 2,
        "coverage": ["student_rocket", "subscale_models"],
    },
    "AIAA_Archive": {
        "url": "https://www.aiaa.org/",
        "description": "American Institute CAD archive",
        "format": ["CAD", "models"],
        "priority": 2,
        "coverage": ["competition_designs", "published_research"],
    },
    "GrabCAD": {
        "url": "https://grabcad.com/",
        "description": "Public CAD library (filter: aerospace, rocket)",
        "format": ["STEP", "IGES", "multiple"],
        "priority": 2,
        "coverage": ["community_designs", "engine_models", "structural_parts"],
        "note": "500K+ models, filter for rocket/aerospace",
    },
    "Thingiverse_Aerospace": {
        "url": "https://www.thingiverse.com/",
        "description": "3D printable rocket + spacecraft designs",
        "format": ["STL", "STEP"],
        "priority": 3,
        "coverage": ["3d_printable_parts", "model_rockets"],
    },
    "FreeCAD_Workbenches": {
        "url": "https://wiki.freecadweb.org/",
        "description": "FreeCAD Rocket workbench + macros",
        "format": ["parametric_API", "FCStd"],
        "priority": 1,
        "coverage": ["parametric_rocket", "custom_geometries"],
        "note": "Procedural generation",
    },
    "OpenSCAD_Designs": {
        "url": "https://www.thingiverse.com/",
        "description": "OpenSCAD rocket design scripts (parametric)",
        "format": ["SCAD", "parametric"],
        "priority": 2,
        "coverage": ["procedural_parts", "parametric_variants"],
    },
    "ITAR_Export": {
        "url": "https://www.itar.gov/",
        "description": "Open-source, non-ITAR aerospace projects",
        "format": ["STEP", "CAD"],
        "priority": 1,
        "coverage": ["amateur_rocket", "nanosatellite"],
    },
}

parametric_generators = {
    "RocketPy_synthesis": {
        "tool": "RocketPy",
        "parameters": ["fin_type", "nose_cone_type", "diameter", "length", "mass", "cp", "cg"],
        "output_format": "STEP",
        "variants_per_design": 20,
    },
    "FreeCAD_Rocket_Workbench": {
        "tool": "FreeCAD + macro",
        "parameters": ["motor_diameter", "length", "fin_count", "fin_shape", "material"],
        "output_format": "STEP",
        "variants_per_design": 50,
    },
    "OpenSCAD_parametric": {
        "tool": "OpenSCAD",
        "parameters": ["r", "h", "thickness", "angle", "profile"],
        "output_format": "STL->STEP",
        "variants_per_design": 100,
    },
    "CoolProp_thermal": {
        "tool": "CoolProp Python",
        "parameters": ["pressure", "temperature", "fluid", "pipe_diameter"],
        "output_format": "mesh + properties",
        "variants_per_design": 30,
    },
    "OpenFOAM_nozzle_gen": {
        "tool": "snappyHexMesh + scripts",
        "parameters": ["expansion_ratio", "throat_diameter", "chamber_pressure", "cooling"],
        "output_format": "blockMeshDict -> mesh",
        "variants_per_design": 40,
    },
}

print("=" * 80)
print("SPACEFLIGHT GEOMETRY SOURCE INVENTORY")
print("=" * 80)
print(f"\nTotal sources discovered: {len(sources)}")
print(f"Parametric generators: {len(parametric_generators)}\n")

# Priority 1 (highest value)
p1 = {k: v for k, v in sources.items() if v.get("priority") == 1}
print(f"PRIORITY 1 (Immediate):")
for name, info in p1.items():
    print(f"  ✓ {name}: {info['description']}")
    print(f"    Coverage: {', '.join(info['coverage'])}")
    if 'note' in info:
        print(f"    Note: {info['note']}")

print(f"\nPARAMETRIC GENERATORS:")
for name, gen in parametric_generators.items():
    print(f"  ✓ {name} ({gen['tool']})")
    print(f"    Params: {', '.join(gen['parameters'][:3])}...")
    print(f"    Variants per design: {gen['variants_per_design']}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. Priority 1 sources → Download all available CAD (expect 1-5GB)
2. OpenRocket + RocketPy → Generate 1000+ parametric variants
3. FreeCAD workbench → Create procedural parts library
4. GrabCAD filter → Download aerospace subset (~500 models)
5. Integrate to graph with metadata + parametrization
6. Run FEA/CFD on all → Physics training data
7. Train JEPA on 5000+ components + variants
""")

# Save to reference
ref_path = Path('data/spaceflight_sources.json')
ref_path.parent.mkdir(parents=True, exist_ok=True)
with open(ref_path, 'w') as f:
    json.dump({'sources': sources, 'parametric_generators': parametric_generators}, f, indent=2)
print(f"\nReference saved to: {ref_path}")
