#!/usr/bin/env python3.12
"""Direct Modal training runner - bypasses CLI import chain."""
import sys
import os
from pathlib import Path

# Use T4 GPU instead of A100 (free tier)
os.environ['JEPA_MODAL_GPU'] = 'T4'
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

# Now import and run
from cadflow.modal_training import launch_modal_training

result = launch_modal_training(
    project_root='.',
    goal='space part world model 500-step validation run',
    raw_dirs=[],
    out_dir=Path('artifacts/modal-500step-run'),
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

print(f"✓ Training completed: {result}")
