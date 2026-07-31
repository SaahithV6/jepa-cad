"""Mass properties from STL × density."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadflow.mass_properties import mass_properties_from_stl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/openrocket_hardware_8k/parts/nose_cone_00000.stl"
if not SAMPLE.is_file():
    SAMPLE = next(ROOT.glob("data/openrocket_hardware_8k/**/nose_cone_*.stl"), None)


@pytest.mark.skipif(SAMPLE is None or not Path(SAMPLE).is_file(), reason="no sample STL")
def test_mass_properties_scales_with_density():
    a = mass_properties_from_stl(SAMPLE, 2700.0)
    b = mass_properties_from_stl(SAMPLE, 5400.0)
    assert a is not None and b is not None
    assert a.mass_kg > 0
    assert abs(b.mass_kg / a.mass_kg - 2.0) < 1e-3
    assert abs(b.inertia_kg_m2[0] / a.inertia_kg_m2[0] - 2.0) < 1e-3
    assert len(a.center_of_mass_m) == 3
