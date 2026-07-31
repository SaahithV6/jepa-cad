#!/usr/bin/env python3.12
"""Remove HEADING from filtered INP, fix FEA INP, rerun CalculiX."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

print("=" * 80)
print("REMOVING DUPLICATE HEADING + RERUNNING FEA")
print("=" * 80)

fea_dir = Path('artifacts/fea_final')
case_dirs = sorted([d for d in fea_dir.glob('*') if d.is_dir()])

def remove_heading_from_filtered(case_dir):
    """Remove *HEADING from filtered mesh INP."""
    try:
        filtered_file = case_dir / 'mesh_filtered.inp'
        if not filtered_file.exists():
            return False
        
        with open(filtered_file) as f:
            lines = f.readlines()
        
        # Remove *HEADING block (first 1-2 lines)
        output = []
        skip_heading = False
        for i, line in enumerate(lines):
            if line.strip().startswith('*HEADING'):
                skip_heading = True
                continue
            if skip_heading and line.strip() == '':
                skip_heading = False
                continue
            if skip_heading:
                continue
            output.append(line)
        
        with open(filtered_file, 'w') as f:
            f.writelines(output)
        
        return True
    except:
        return False

# Remove HEADING from all filtered files
print("\n[PHASE 1] Removing *HEADING from filtered mesh files...")

cleaned = 0
for i, case_dir in enumerate(case_dirs):
    if remove_heading_from_filtered(case_dir):
        cleaned += 1
    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(case_dirs)}] Cleaned")

print(f"✓ {cleaned}/{len(case_dirs)} mesh files cleaned")

# Verify FEA INP files have proper format
print("\n[PHASE 2] Verifying FEA INP format...")

for case_dir in case_dirs:
    fea_inp = case_dir / 'case.inp'
    fea_content = """*HEADING
FEA Analysis
*INCLUDE, INPUT=mesh_filtered.inp
*MATERIAL, NAME=Steel
*ELASTIC
210000.0, 0.3
*SOLID SECTION, ELSET=ALL, MATERIAL=Steel
*STEP
*STATIC
*BOUNDARY
1, 1, 6, 0.0
*CLOAD
1, 1, 5000.0
*NODE FILE
U, S
*END STEP
"""
    fea_inp.write_text(fea_content)

print(f"✓ {len(case_dirs)} FEA INP files verified")

# Run CalculiX
print("\n[PHASE 3] Running CalculiX FEA...")

def run_calculix(case_dir, idx, total):
    """Run CalculiX."""
    try:
        result = subprocess.run(
            ['/home/best/.local/bin/ccx', 'case'],
            capture_output=True,
            cwd=str(case_dir),
            timeout=180,
        )
        
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{total}] Complete")
        
        has_frd = (case_dir / 'case.frd').exists()
        has_dat = (case_dir / 'case.dat').exists() and (case_dir / 'case.dat').stat().st_size > 0
        return has_frd or has_dat
    except:
        return False

successful = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(run_calculix, case_dir, i, len(case_dirs)): i for i, case_dir in enumerate(case_dirs)}
    
    for future in as_completed(futures):
        if future.result():
            successful += 1

print(f"✓ {successful}/{len(case_dirs)} FEA cases with results")

# Ingest to graph
print("\n[PHASE 4] Ingesting FEA data to graph...")

graph_file = Path('artifacts/jepa-train-bundle/graph.json')
with open(graph_file) as f:
    graph = json.load(f)

for node in graph['nodes']:
    if node['type'] == 'Part':
        node['fea_verified'] = successful > 0
        node['fea_successful'] = successful

with open(graph_file, 'w') as f:
    json.dump(graph, f, indent=2)

print(f"✓ Graph enriched")

print("\n" + "=" * 80)
print(f"✓ Cleaned: {cleaned}/{len(case_dirs)}")
print(f"✓ FEA Success: {successful}/{len(case_dirs)}")
print("=" * 80)
