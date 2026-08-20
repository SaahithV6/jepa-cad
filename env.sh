#!/usr/bin/env bash
# Activate the local sim environment.
#   source /home/lain_iwakura/Documents/jepa-cad-vm/env.sh
#
# Puts the recovered CalculiX/OpenFOAM wrappers on PATH and activates .venv-sim.
# Verified working on CachyOS: doctor_native_fea.py -> solver_mode=native.

_here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# solver wrappers (ccx, cgx, simpleFoam, blockMesh) live here
export PATH="$HOME/.local/bin:$PATH"

# the wrappers source this to set LD_LIBRARY_PATH for the extracted .deb trees
export CADFLOW_SOLVER_ROOT="$HOME/.local/cadflow-solvers"

# OpenFOAM 1912 locates its global etc/ via WM_PROJECT_DIR. The .deb wrappers
# don't set it, so simpleFoam fails with "Could not find mandatory etc entry
# 'controlDict'" without this.
export WM_PROJECT_DIR="$CADFLOW_SOLVER_ROOT/openfoam_1912.200626-2build3_amd64/usr/share/openfoam"

# Only 4 wrapper scripts exist in ~/.local/bin (ccx, cgx, simpleFoam, blockMesh),
# but the extracted .deb carries the full OpenFOAM toolset -- snappyHexMesh,
# surfaceFeatureExtract, checkMesh, decomposePar and ~200 more. The body-fitted
# CFD path needs snappyHexMesh, so put the real bin dirs on PATH and source the
# library environment directly rather than wrapping each binary.
if [[ -f "$HOME/.local/bin/cadflow-solver-env.sh" ]]; then
  source "$HOME/.local/bin/cadflow-solver-env.sh"
fi
export PATH="$CADFLOW_SOLVER_ROOT/openfoam_1912.200626-2build3_amd64/usr/bin:$PATH"
export PATH="$CADFLOW_SOLVER_ROOT/calculix-ccx_2.21-1_amd64/usr/bin:$PATH"

# CalculiX parallelism: prefer many concurrent cases over threads-per-case.
# Raise this only if you are running a single large solve.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -f "$_here/.venv-sim/bin/activate" ]]; then
  source "$_here/.venv-sim/bin/activate"
fi

echo "sim env active"
echo "  ccx:    $(command -v ccx || echo MISSING)"
echo "  python: $(command -v python)"
echo "  cores:  $(nproc --all) (run cases in parallel, not threads per case)"
unset _here
