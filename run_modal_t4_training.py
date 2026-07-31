#!/usr/bin/env python3.12
"""
Direct Modal T4 training runner.
500 steps, saves checkpoint locally.
No complexity, no sync issues.
"""
import sys
import os
from pathlib import Path

os.environ['JEPA_MODAL_GPU'] = 'T4'
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

sys.path.insert(0, str(Path(__file__).parent))

from cadflow.modal_training import launch_modal_training

print("=" * 70)
print("MODAL T4 TRAINING: 500 STEPS")
print("=" * 70)

result = launch_modal_training(
    project_root='.',
    goal='500-step production test on T4',
    raw_dirs=[],
    out_dir=Path('artifacts/modal-t4-500step'),
    family='space',
    config='configs/families/space_24b.yaml',
    data_source='graph',
    probe_data_source='graph',
    graph_path='artifacts/jepa-train-bundle/graph.json',
    num_points=2048,
    max_steps=500,
    grad_accum_steps=2,
    extra_overrides=['train.batch_size=8'],
)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE")
print("=" * 70)
print(f"\nRun ID: {result.run_id}")
print(f"Output: artifacts/modal-t4-500step/")
print("\nCheckpoint will be at: artifacts/modal-t4-500step/latest.pt")
