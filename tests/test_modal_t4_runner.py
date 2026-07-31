"""Test: run_modal_t4_training.py is correctly configured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_modal_t4_runner_syntax():
    """Test that run_modal_t4_training.py has valid syntax."""
    import ast
    with open('run_modal_t4_training.py') as f:
        ast.parse(f.read())

def test_modal_t4_runner_config():
    """Test that run_modal_t4_training.py has correct configuration."""
    with open('run_modal_t4_training.py') as f:
        src = f.read()
    
    assert "JEPA_MODAL_GPU'] = 'T4'" in src, "T4 GPU not configured"
    assert "max_steps=500" in src, "Steps not set to 500"
    assert "train.batch_size=8" in src, "Batch size not set to 8"
    assert "modal-t4-500step" in src, "Output directory not set correctly"
    assert "artifacts/jepa-train-bundle/graph.json" in src, "Graph path incorrect"
    assert "launch_modal_training(" in src, "launch_modal_training not called"

if __name__ == '__main__':
    test_modal_t4_runner_syntax()
    print("✓ test_modal_t4_runner_syntax passed")
    
    test_modal_t4_runner_config()
    print("✓ test_modal_t4_runner_config passed")
    
    print("\n✅ All tests passed")
