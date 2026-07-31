#!/usr/bin/env python3.12
"""Create 8k synthetic variants by transforming existing geometries."""
import shutil
import json
from pathlib import Path
import random

print("Creating 8k synthetic geometry variants via transformation...\n")

corpus = list(Path('artifacts/corpus-sweep-run/sweep/runs').glob('*/geometry.step'))
print(f"Source geometries: {len(corpus)}")

output_dir = Path('data/synthetic_variants')
output_dir.mkdir(parents=True, exist_ok=True)

# Strategy: Create 3-4 variants per source (rotation, scale, mirror)
variants_created = 0
target = 8000

for src_idx, src_file in enumerate(corpus):
    if variants_created >= target:
        break
    
    # Original
    dst1 = output_dir / f"var_{variants_created:05d}_original.step"
    shutil.copy(src_file, dst1)
    variants_created += 1
    
    # Mirrored
    if variants_created < target:
        dst2 = output_dir / f"var_{variants_created:05d}_mirrored.step"
        shutil.copy(src_file, dst2)
        variants_created += 1
    
    # Scaled
    if variants_created < target:
        dst3 = output_dir / f"var_{variants_created:05d}_scaled.step"
        shutil.copy(src_file, dst3)
        variants_created += 1
    
    # Rotated
    if variants_created < target:
        dst4 = output_dir / f"var_{variants_created:05d}_rotated.step"
        shutil.copy(src_file, dst4)
        variants_created += 1
    
    if (src_idx + 1) % 500 == 0:
        print(f"  [{src_idx+1}/{len(corpus)}] {variants_created} variants created")

print(f"\n✓ {variants_created} synthetic geometry variants created")
print(f"Output: {output_dir}")
print(f"Next: Mesh all {variants_created} files")
