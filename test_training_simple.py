#!/usr/bin/env python3.12
"""
Direct 500-step JEPA training test.
Saves checkpoint locally for app integration.
No Modal sync complexity.
"""
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("DIRECT 500-STEP JEPA TRAINING TEST")
print("=" * 70)

# Verify graph exists
graph_path = Path('artifacts/jepa-train-bundle/graph.json')
if not graph_path.exists():
    print(f"✗ Graph not found at {graph_path}")
    sys.exit(1)
print(f"✓ Graph: {graph_path} ({graph_path.stat().st_size / 1024 / 1024:.1f} MB)")

# Verify config exists
config_path = Path('configs/families/space_24b.yaml')
if not config_path.exists():
    print(f"✗ Config not found at {config_path}")
    sys.exit(1)
print(f"✓ Config: {config_path}")

# Create output directory
out_dir = Path('artifacts/test-500step-output')
out_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {out_dir}")

print("\n" + "=" * 70)
print("RUNNING 500-STEP TRAINING")
print("=" * 70 + "\n")

# Run training
cmd = [
    sys.executable,
    'train.py',
    '--config', str(config_path),
    '--family', 'space',
    '--data-source', 'graph',
    '--set', f'data.graph_path={graph_path.absolute()}',
    '--set', f'data.graph_data_root={graph_path.parent.absolute()}',
    '--max-steps', '500',
    '--set', 'train.batch_size=8',
    '--set', 'train.grad_accum_steps=2',
    '--set', f'checkpoint.checkpoint_dir={out_dir}',
    '--set', 'logging.log_dir=/tmp/logs',
]

print(f"Command:\n  {' '.join(cmd)}\n")

result = subprocess.run(cmd, cwd=Path(__file__).parent)

print("\n" + "=" * 70)
if result.returncode == 0:
    print("✅ TRAINING SUCCESSFUL")
    print("=" * 70)
    
    # List outputs
    checkpoints = list(out_dir.glob('*.pt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"\n✓ Latest checkpoint: {latest.name}")
        print(f"  Size: {latest.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"\nReady for app integration at: {latest}")
    else:
        print("\n! No checkpoints saved")
else:
    print(f"✗ TRAINING FAILED (exit code: {result.returncode})")
    print("=" * 70)
    sys.exit(1)
