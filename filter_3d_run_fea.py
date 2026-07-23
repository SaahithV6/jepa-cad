#!/usr/bin/env python3.12
"""DEPRECATED: use proper_msh_to_inp_fea.py (direct MSH2 parsing).

Filter converted INP to keep only 3D solid elements, rerun CalculiX FEA.
This approach is unreliable; kept for reference only.
"""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def filter_inp_to_3d_solids(case_dir: Path) -> bool:
    """Filter INP to keep only 3D solid elements (C3D4, C3D10, C3D20, etc.)."""
    try:
        inp_file = case_dir / "mesh_converted.inp"
        if not inp_file.exists():
            return False

        with inp_file.open(encoding="utf-8") as handle:
            lines = handle.readlines()

        output: list[str] = []
        in_nodeset = False
        in_elset = False
        in_surface = False

        skip_element_types = {"S3", "S4", "CPS3", "CPS4", "CPS6", "CPS8", "B31", "B32", "T3D2"}
        keep_element_types = {"C3D4", "C3D10", "C3D20", "C3D20R"}

        for line in lines:
            if any(f"type={stype}" in line for stype in skip_element_types):
                in_nodeset = False
                in_elset = False
                in_surface = True
                continue

            if any(f"type={ktype}" in line for ktype in keep_element_types):
                in_nodeset = False
                in_elset = False
                in_surface = False
                output.append(line)
                continue

            if in_surface:
                if line.startswith("*"):
                    in_surface = False
                else:
                    continue

            if not (in_nodeset or in_elset or in_surface):
                output.append(line)

        filtered_file = case_dir / "mesh_filtered.inp"
        filtered_file.write_text("".join(output), encoding="utf-8")
        return filtered_file.stat().st_size > 0
    except OSError:
        return False


def main() -> None:
    print("=" * 80)
    print("DEPRECATED: use proper_msh_to_inp_fea.py instead")
    print("=" * 80)

    fea_dir = Path("artifacts/fea_final")
    case_dirs = sorted(d for d in fea_dir.glob("*") if d.is_dir())

    filtered = sum(filter_inp_to_3d_solids(case_dir) for case_dir in case_dirs)
    print(f"Filtered: {filtered}/{len(case_dirs)}")


if __name__ == "__main__":
    main()
