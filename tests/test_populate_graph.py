"""Test: populate_graph_physics.py correctly populates the graph."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_populate_graph_physics_executable():
    """Test that populate_graph_physics.py is executable."""
    script_path = Path('populate_graph_physics.py')
    assert script_path.exists(), "populate_graph_physics.py not found"
    assert script_path.is_file(), "populate_graph_physics.py is not a file"

def test_populate_graph_physics_results():
    """Test that populate_graph_physics.py produced correct results."""
    graph_path = Path('artifacts/jepa-train-bundle/graph.json')
    assert graph_path.exists(), "Graph file not found"
    
    with open(graph_path) as f:
        graph = json.load(f)
    
    # Check nodes populated
    physics_nodes = [n for n in graph['nodes'] if n['type'] == 'PhysicsTarget' and n.get('attributes')]
    assert len(physics_nodes) >= 600, f"Expected 600+ PhysicsTarget nodes, got {len(physics_nodes)}"
    
    solver_nodes = [n for n in graph['nodes'] if n['type'] == 'SolverSetup' and 'solver_name' in n.get('properties', {})]
    assert len(solver_nodes) >= 1700, f"Expected 1700+ SolverSetup nodes, got {len(solver_nodes)}"
    
    test_nodes = [n for n in graph['nodes'] if n['type'] == 'TestCase' and 'test_parameters' in n.get('properties', {})]
    assert len(test_nodes) >= 2000, f"Expected 2000+ TestCase nodes, got {len(test_nodes)}"
    
    # Check edges labeled
    edges_with_labels = [e for e in graph['edges'] if e.get('label') and e['label'] != 'unknown']
    assert len(edges_with_labels) >= 160000, f"Expected 160000+ labeled edges, got {len(edges_with_labels)}"

def test_graph_backup_exists():
    """Test that graph backup was created."""
    backup_path = Path('artifacts/jepa-train-bundle/graph.backup.json')
    assert backup_path.exists(), "Graph backup not found"
    assert backup_path.stat().st_size > 0, "Graph backup is empty"

if __name__ == '__main__':
    tests = [
        ('test_populate_graph_physics_executable', test_populate_graph_physics_executable),
        ('test_populate_graph_physics_results', test_populate_graph_physics_results),
        ('test_graph_backup_exists', test_graph_backup_exists),
    ]
    
    print("=" * 70)
    print("TESTS: populate_graph_physics.py")
    print("=" * 70 + "\n")
    
    for test_name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {test_name}")
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
