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
    """Test that populate_graph_physics.py produced correct results.

    This asserted against a node schema the graph does not use. Nodes carry
    ``properties``, not ``attributes``, so the PhysicsTarget filter matched
    nothing and reported 0 against a graph that holds 30,959 of them; edges
    carry ``type``, not ``label``, so the edge check counted 0 of 518,790. The
    graph was healthy the whole time and the test was reading it wrongly.
    """
    graph_path = Path('artifacts/jepa-train-bundle/graph.json')
    assert graph_path.exists(), "Graph file not found"

    with open(graph_path) as f:
        graph = json.load(f)

    def typed(kind):
        return [n for n in graph['nodes']
                if n.get('type') == kind and n.get('properties')]

    physics_nodes = typed('PhysicsTarget')
    assert len(physics_nodes) >= 600, f"Expected 600+ PhysicsTarget nodes, got {len(physics_nodes)}"
    # The point of a PhysicsTarget is the target values on it.
    assert all('targets' in n['properties'] for n in physics_nodes[:100])

    solver_nodes = typed('SolverSetup')
    assert len(solver_nodes) >= 1700, f"Expected 1700+ SolverSetup nodes, got {len(solver_nodes)}"
    assert all('solver' in n['properties'] for n in solver_nodes[:100])

    test_nodes = typed('TestCase')
    assert len(test_nodes) >= 2000, f"Expected 2000+ TestCase nodes, got {len(test_nodes)}"

    edges_typed = [e for e in graph['edges']
                   if e.get('type') and e['type'] != 'unknown']
    assert len(edges_typed) >= 160000, f"Expected 160000+ typed edges, got {len(edges_typed)}"


def test_graph_backup_exists():
    """A backup of the graph exists beside it.

    Looked for graph.backup.json; the writer produces graph.json.bak. Accept
    either rather than pin the spelling.
    """
    base = Path('artifacts/jepa-train-bundle')
    candidates = [base / 'graph.backup.json', base / 'graph.json.bak']
    found = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    assert found, f"No non-empty graph backup among {[str(c) for c in candidates]}"

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
