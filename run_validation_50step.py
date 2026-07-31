#!/usr/bin/env python3.12
"""
Quick 50-step validation on Modal T4.
Tests if model learns from graph data correctly.
Fast feedback: ~2-3 minutes.
"""
import sys
import os
from pathlib import Path

os.environ['JEPA_MODAL_GPU'] = 'T4'
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

sys.path.insert(0, str(Path(__file__).parent))

from cadflow.modal_training import launch_modal_training

print("=" * 70)
print("QUICK VALIDATION: 50 STEPS ON T4")
print("=" * 70)

result = launch_modal_training(
    project_root='.',
    goal='quick validation - graph data quality check',
    raw_dirs=[],
    out_dir=Path('artifacts/validation-50step'),
    family='space',
    config='configs/families/space_24b.yaml',
    data_source='graph',
    probe_data_source='graph',
    graph_path='artifacts/jepa-train-bundle/graph.json',
    num_points=2048,
    max_steps=50,
    grad_accum_steps=2,
    extra_overrides=['train.batch_size=8'],
)

print("\n" + "=" * 70)
print("✅ VALIDATION COMPLETE")
print("=" * 70)
print(f"\nRun ID: {result.run_id}")
print(f"Steps: 50")
print(f"Output: artifacts/validation-50step/")
print("\nCheck the logs to verify:")
print("  • Loss is decreasing")
print("  • Model converges on graph data")
print("  • No device/dtype errors")
