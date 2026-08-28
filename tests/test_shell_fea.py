"""Flight-scale stress, using the right element for a thin wall.

Every component this project analyses is a coupon, because body radius is
clamped so parts stay meshable. The packet is honest about that and it still
means the flight parts have never been analysed.

The clamp is arithmetic, not laziness. Resolving a 0.8 mm wall with solid
tetrahedra wants elements near a quarter of a millimetre, and a 4.37 m tank at
335 mm radius then needs on the order of two billion of them against a budget
of forty thousand. Solid elements are the wrong tool: a shell carries its
thickness as a property instead of meshing through it, and the same tank
becomes a few hundred elements that solve in under a second.

Both load cases have exact answers -- N/(2 pi r t) axially, p r / t in hoop for
an open barrel -- so this is checked against truth rather than against a finer
version of itself.
"""

import math
import shutil
import tempfile
from pathlib import Path

import pytest

from cadflow.shell_fea import (
    DEFAULT_CCX, analyse_barrel, cylinder_shell_mesh)

pytestmark = pytest.mark.skipif(
    not DEFAULT_CCX.exists(), reason="CalculiX not available")

# The actual flight tank this project designs, not a coupon.
BARREL = dict(radius_m=0.335, length_m=4.37, thickness_m=0.0008,
              youngs_pa=200e9)


@pytest.fixture(scope="module")
def work():
    d = Path(tempfile.mkdtemp(prefix="shellfea_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_the_seam_closes():
    """A barrel not joined round its circumference carries no hoop load.

    Duplicating the seam column produces a flat plate rolled up, which looks
    like a cylinder, meshes like a cylinder, and reports a pressure vessel with
    no hoop stress at all.
    """
    nodes, elements = cylinder_shell_mesh(0.335, 4.37, n_theta=16, n_axial=8)
    assert len(nodes) == 16 * 9          # no duplicated seam column
    assert len(elements) == 16 * 8       # wraps all the way round
    # every node belongs to at least two elements, so nothing is a free edge
    from collections import Counter
    seen = Counter(n for _e, ns in elements for n in ns)
    assert min(seen.values()) >= 2


def test_axial_stress_matches_the_closed_form_at_flight_scale(work):
    """N/(2 pi r t), on the real 335 mm tank rather than a 42 mm coupon."""
    r = analyse_barrel(work / "axial", axial_n=80_000.0, **BARREL)
    assert r.converged
    assert r.analytic_mpa == pytest.approx(
        80_000.0 / (2 * math.pi * 0.335 * 0.0008) / 1e6, rel=1e-9)
    assert abs(r.error_pct) < 3.0, r.as_dict()


def test_hoop_stress_uses_the_open_ended_reference(work):
    """p r / t, because this barrel has no end caps.

    The closed-end formula adds an axial pr/2t and lands at sqrt(3)/2 of the
    hoop stress. Using it here produced an 8.4% error that read as solver
    inaccuracy and was a wrong reference -- the same trap the Lame case was
    written to avoid, walked into anyway. A mismatched boundary condition shows
    up as a fixed offset that refinement never removes.
    """
    r = analyse_barrel(work / "press", pressure_pa=200_000.0, **BARREL)
    assert r.converged
    assert r.analytic_mpa == pytest.approx(
        200_000.0 * 0.335 / 0.0008 / 1e6, rel=1e-9)
    assert abs(r.error_pct) < 4.0, r.as_dict()


def test_it_is_tractable_where_solid_elements_are_not(work):
    """Hundreds of elements, not billions.

    This is the whole reason the module exists: the coupon clamp was forced by
    the element type, not by the physics.
    """
    r = analyse_barrel(work / "size", axial_n=80_000.0, **BARREL)
    assert r.elements < 5_000
    tets_needed = (2 * math.pi * 0.335 * 0.0008 * 4.37) / ((0.0008 / 3) ** 3 / 6)
    assert tets_needed > 1e9, "the comparison this module rests on"


def test_combined_loading_offers_no_analytic_column(work):
    """No closed form exists, so none is claimed.

    Presenting an approximation as the exact answer would be worse than
    presenting nothing, since the error column is what tells a reader whether
    to believe the run.
    """
    r = analyse_barrel(work / "both", axial_n=80_000.0, pressure_pa=200_000.0,
                       **BARREL)
    assert r.converged
    assert r.analytic_mpa is None and r.error_pct is None
    assert any("no single closed form" in n for n in r.notes)


def test_a_degenerate_mesh_is_refused():
    with pytest.raises(ValueError):
        cylinder_shell_mesh(0.0, 1.0)
    with pytest.raises(ValueError, match="too coarse"):
        cylinder_shell_mesh(0.3, 1.0, n_theta=4)


def test_a_result_past_linear_validity_is_refused(work):
    """Stress over modulus is a strain, whether or not anyone asked for it.

    Under full-suite load this solve once returned 2,697 MPa where it returns
    82 in isolation -- a factor of thirty-three -- and the FRD parsed cleanly,
    so nothing downstream could tell it was garbage. A linear elastic result
    implying more than a percent of strain describes nothing: the material has
    yielded and the small-displacement assumption is gone.

    The guard is independent of the closed-form comparison, so it still works
    for the combined load case where there is no analytic answer to check
    against.
    """
    from cadflow.shell_fea import MAX_LINEAR_STRAIN

    r = analyse_barrel(work / "overload", pressure_pa=20_000_000.0, **BARREL)
    assert not r.converged
    assert any("linear elastic" in n for n in r.notes)
    assert r.max_von_mises_mpa * 1e6 / BARREL["youngs_pa"] > MAX_LINEAR_STRAIN


def test_a_normal_run_is_far_inside_linear_validity(work):
    """The guard must not fire on the cases this module exists to run."""
    from cadflow.shell_fea import MAX_LINEAR_STRAIN

    r = analyse_barrel(work / "normal", axial_n=80_000.0, **BARREL)
    assert r.converged
    strain = r.max_von_mises_mpa * 1e6 / BARREL["youngs_pa"]
    assert strain < 0.1 * MAX_LINEAR_STRAIN, strain
