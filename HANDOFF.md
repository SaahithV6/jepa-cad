# JEPA-CAD Handoff to Cursor

**Date:** 2026-07-23  
**Status:** FEA solid-only pipeline fixed; TAO graph ingest live; full CalculiX re-run in progress for physics-verified training data.

---

## Current State

### ✅ Complete
- **2,154 real spaceflight geometries** meshed (99.9% success rate)
- **42,286 TAO graph nodes** with 164,857 edges
- **Graph structure ingested** (Parts, Assemblies, Parameters, Relationships)
- **24B JEPA model architecture** defined and tested locally
- **Modal training infrastructure** verified (prior runs: 300-step + 500-step successful)
- **Local mesh validation** (~2,151 valid Gmsh MSH files, ~1-5MB each)
- **MSH→CalculiX blocker fixed** (`cadflow/msh_to_calculix.py`): direct MSH2 tet parse, no `gmsh.write()`
- **FRD → TAO ingest** maps sweep run ids in `geometry_ref` → Part nodes with real stress/disp

### ⚠️ In Progress
- **Full FEA re-run** with SI units (E=210 GPa, load=5 MN) across ~2,109 cases
- **CFD:** OpenFOAM still incomplete (deprioritized; FEA-first for Modal launch)

### 📊 Data Inventory
```
artifacts/
├── jepa-train-bundle/
│   ├── graph.json                 (42,286 nodes; Part.simulation_results_fea from FRD)
│   └── physics_verified_index.json
├── fea_final/{case_id}/
│   ├── mesh.msh
│   ├── mesh_solid.inp             (C3D4 only)
│   ├── case.inp
│   └── case.frd                   (valid when >100KB)
data/physics_verified_summary.json   (tracked summary for GitHub)
```

---

## How to Run Locally

### Prerequisites
```bash
cd /home/best/jepa-cad
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gmsh torch modal
```

### Path: FEA + TAO Graph Population
```bash
# Full re-run (SI units, solid-only):
PYTHONUNBUFFERED=1 python3.12 proper_msh_to_inp_fea.py --force --workers 8

# Resume failures only:
PYTHONUNBUFFERED=1 python3.12 proper_msh_to_inp_fea.py --resume --workers 8

# Re-ingest existing FRDs into graph:
python3.12 proper_msh_to_inp_fea.py --ingest-only

# Single case debug:
python3.12 proper_msh_to_inp_fea.py --single artifacts/fea_final/0024da0fdb17d12c
```

### Path: 24B JEPA Training (LOCAL pilot)
```bash
python3.12 launch_modal_training.py
```

---

## Cursor TODO (Prioritized)

### 🔴 CRITICAL: Finish FEA corpus then Modal train

#### Task 1–2: CalculiX INP + single-case verify ✅ DONE
Solid-only MSH2 path produces valid `.frd` (tested: ~0.28 mm max disp, ~250 MPa von Mises @ 5 MN).

#### Task 3: Scale FEA to all 2,109 cases — IN PROGRESS
```bash
PYTHONUNBUFFERED=1 python3.12 proper_msh_to_inp_fea.py --force --workers 8
```
Success criteria: ≥1500 parts with `physics_verified=true` and real `simulation_results_fea`.

#### Task 4: CFD — skip for Modal launch
Use FEA-only physics annotations for first 24B training run.

### 🟡 MEDIUM: Graph + Training

#### Task 5: Ingest Real FEA Results ✅ (via `--ingest-only` / end of batch)
#### Task 6: Launch 500-step JEPA on Modal after Task 3 completes

---

## Code Locations (Local)

| File | Purpose | Status |
|------|---------|--------|
| `cadflow/msh_to_calculix.py` | MSH2→solid INP, FRD parse, graph ingest | ✅ |
| `proper_msh_to_inp_fea.py` | Batch FEA + TAO update | ✅ |
| `data/physics_verified_summary.json` | Tracked corpus stats | ✅ |
| `filter_3d_run_fea.py` | Deprecated gmsh.write filter | ❌ do not use |
| `launch_modal_training.py` | 50-step local pilot | ✅ |

---

## Units (SI)
- Mesh coordinates: meters
- Young's modulus: 2.1e11 Pa
- Load: 5e6 N distributed on max-X face
- Reported stress: MPa (von Mises)
- Reported displacement: mm

## What NOT to Do

❌ Use `gmsh.write()` for CalculiX — exports 1D/2D elements that break the solver  
❌ Run deprecated scripts that rewrite `case.inp` to `mesh_filtered.inp`  
❌ Train Modal 24B until `data/physics_verified_summary.json` shows ≥1500 linked parts

---

**Handoff updated 2026-07-23.** FEA root cause fixed; full SI re-run + graph population in flight.
