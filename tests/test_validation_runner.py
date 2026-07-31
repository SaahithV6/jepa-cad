"""Test: run_validation_50step.py is correctly configured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_validation_runner_syntax():
    """Test that run_validation_50step.py has valid syntax."""
    import ast
    with open('run_validation_50step.py') as f:
        ast.parse(f.read())

def test_validation_runner_config():
    """Test that run_validation_50step.py has correct configuration."""
    with open('run_validation_50step.py') as f:
        src = f.read()
    
    assert "JEPA_MODAL_GPU'] = 'T4'" in src, "T4 GPU not configured"
    assert "max_steps=50" in src, "Steps not set to 50"
    assert "train.batch_size=8" in src, "Batch size not set to 8"
    assert "validation-50step" in src, "Output directory not set correctly"
    assert "artifacts/jepa-train-bundle/graph.json" in src, "Graph path incorrect"
    assert "quick validation" in src, "Validation goal not documented"

if __name__ == '__main__':
    test_validation_runner_syntax()
    print("✓ test_validation_runner_syntax passed")
    
    test_validation_runner_config()
    print("✓ test_validation_runner_config passed")
    
    print("\n✅ All tests passed")
