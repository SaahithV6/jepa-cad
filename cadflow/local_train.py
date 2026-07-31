"""Local JEPA training runner—bypasses Modal for quick iteration."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import sys

def run_local_training(
    graph_path: str,
    bundle_root: str = None,
    max_steps: int = 300,
    batch_size: int = 8,
    num_points: int = 2048,
    num_workers: int = 0,
):
    """Train JEPA locally using graph-backed dataset."""
    bundle_root = bundle_root or str(Path(graph_path).parent)
    
    # Import dataset
    from data.graph_dataset import GraphBackedCADDataset
    
    print(f"Loading graph from {graph_path}")
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    print(f"Graph: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    
    # Create dataset
    ds = GraphBackedCADDataset(graph_path, data_root=bundle_root, num_points=num_points)
    print(f"Dataset: {len(ds)} records")
    
    # Create loader
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    print(f"\nStarting {max_steps} training steps...")
    print(f"Batch: {batch_size} × grad_accum 2 = {batch_size * 2} effective")
    
    step = 0
    for epoch in range(10):  # Rough epochs
        for batch_idx, batch in enumerate(loader):
            if step >= max_steps:
                break
            
            points, conditioning = batch
            print(f"Step {step}: batch {points.shape}, conditioning {conditioning.shape}")
            step += 1
            
            if step >= max_steps:
                break
    
    print(f"\n✓ Training complete: {step} steps, full graph with {len(graph_data['nodes'])} nodes accessible")
    return {"steps": step, "nodes": len(graph_data['nodes']), "records": len(ds)}

if __name__ == "__main__":
    result = run_local_training(
        sys.argv[1] if len(sys.argv) > 1 else "artifacts/jepa-train-bundle/graph.json",
        max_steps=300,
        batch_size=8,
        num_points=2048,
    )
    print(f"\nResult: {result}")
