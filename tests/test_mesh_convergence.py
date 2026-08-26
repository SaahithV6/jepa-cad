"""Discretisation error in the structural solver, measured against Lame.

This project reported FEA stress as though it were stress. It is a mesh's
estimate of stress, and until now nothing measured the difference. The prior
evidence was a single sensitivity range -- "p95 moved 13.8% across a 16x
refinement" -- with no exact answer to converge toward, no observed order and no
error band, which is not a convergence study.

The verification case is a thick-walled cylinder under internal pressure, which
is the tank problem this project actually solves and which Lame gives in closed
form. Results are in artifacts/verification/fea_mesh_convergence.json; this
module pins the conclusions that should not silently change.

The finding that matters: the production path writes C3D4 linear tetrahedra,
whose stress field converges first-order, and CalculiX's own manual advises
against them. At 34,493 elements -- essentially the production ceiling of
40,000 -- C3D4 is still less accurate than C3D10 at 661 elements.
"""

import json
import math
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts/verification/fea_mesh_convergence.json"

pytestmark = pytest.mark.skipif(
    not ART.exists(),
    reason="run scripts/fea_mesh_convergence.py to produce the study")


@pytest.fixture(scope="module")
def study():
    return json.loads(ART.read_text())


def test_the_exact_solution_is_the_plane_strain_one(study):
    """Lame, with the axial stress the applied restraints actually produce.

    A closed-end or free-end formula against a plane-strain model differs by a
    fixed offset that no amount of refinement removes, and reads exactly like
    solver error. Pinning the number keeps the reference honest.
    """
    a, b, p, nu = 0.050, 0.075, 10.0e6, 0.33
    k = p * a ** 2 / (b ** 2 - a ** 2)
    sr, st = k * (1 - b ** 2 / a ** 2), k * (1 + b ** 2 / a ** 2)
    sz = nu * (sr + st)
    vm = math.sqrt(0.5 * ((sr - st) ** 2 + (st - sz) ** 2 + (sz - sr) ** 2))
    assert study["exact_peak_vm_mpa"] == pytest.approx(vm / 1e6, rel=1e-9)
    assert study["exact_peak_vm_mpa"] == pytest.approx(31.2953, abs=1e-3)


def test_stress_converges_at_the_theoretical_order(study):
    """First order for linear tets, second for quadratic.

    This is the check that the study measured discretisation error rather than
    something else. A model with a loose boundary condition or a mis-ordered
    element still produces a smooth-looking error sequence, but not one whose
    slope lands on the theoretical order for its element type.
    """
    orders = {et: {o["metric"]: o for o in study["analysis"][et]["orders"]}
              for et in ("C3D4", "C3D10")}
    lin = orders["C3D4"]["l2_error_mpa"]
    quad = orders["C3D10"]["l2_error_mpa"]
    assert 1.0 <= lin["order_p"] <= 1.6, lin
    assert 1.7 <= quad["order_p"] <= 2.4, quad
    assert lin["r_squared"] > 0.98 and quad["r_squared"] > 0.98
    assert quad["order_p"] > lin["order_p"] + 0.4


def test_surface_stress_is_where_linear_tets_hurt(study):
    """Pressure vessel sizing reads the bore, and C3D4 reads it worst.

    Radial stress at a pressurised surface must equal minus the applied
    pressure by traction equilibrium, at any mesh density -- it tests the load
    and the restraints, not the discretisation. C3D4 misses it by 30% on the
    coarse mesh and is still 9.8% low at the production element ceiling, while
    C3D10 is inside 1.4% from the coarsest mesh onward.
    """
    for et, coarse_tol, fine_tol in (("C3D4", 40.0, 11.0), ("C3D10", 2.0, 0.5)):
        rows = study["results"][et]
        assert abs(rows[0]["bore_radial_error_pct"]) < coarse_tol, (et, rows[0])
        assert abs(rows[-1]["bore_radial_error_pct"]) < fine_tol, (et, rows[-1])
    assert (abs(study["results"]["C3D4"][-1]["bore_radial_error_pct"])
            > 10 * abs(study["results"]["C3D10"][-1]["bore_radial_error_pct"]))


def test_quadratic_elements_beat_linear_at_a_fraction_of_the_cost(study):
    """The reason this is a finding and not a footnote.

    Refinement cannot buy C3D4 out of the gap: the coarsest C3D10 mesh here has
    a lower field error than the finest C3D4 mesh, which carries 52 times as
    many elements and sits at the production budget of 40,000.
    """
    lin = study["results"]["C3D4"][-1]
    quad = study["results"]["C3D10"][0]
    assert quad["l2_error_mpa"] < lin["l2_error_mpa"], (quad, lin)
    assert lin["elements"] > 40 * quad["elements"]
    assert lin["elements"] <= 40_000, "study should straddle the production cap"


def test_peak_stress_oscillates_and_is_reported_as_such(study):
    """An observed order of 5.5 for a linear tetrahedron is not a result.

    Richardson extrapolation returns a number for any three values, including
    an oscillating sequence it does not apply to. C3D4 peak stress went 31.94,
    31.74, 31.78 and the formula duly reported order 5.5 with a 0.05% error
    band -- roughly a hundredfold understatement of the real uncertainty. The
    classifier now names the behaviour instead of extrapolating through it.
    """
    g = study["analysis"]["C3D4"]["gci_peak"]
    assert g["convergence"] == "oscillatory"
    assert g["convergence_ratio_R"] < 0
    assert "order_p" not in g, "no order may be claimed outside the asymptotic range"
    assert g["oscillation_band_pct"] > 0.5


def test_the_field_norm_is_monotone_even_where_the_peak_is_not(study):
    """Why the L2 norm carries the conclusion.

    Peak stress is one node's value and can wander with mesh topology. The
    field norm integrates everything and falls at every single refinement for
    both element types, which is what makes a least-squares order meaningful.
    """
    for et in ("C3D4", "C3D10"):
        l2 = [r["l2_error_mpa"] for r in study["results"][et]]
        assert all(b < a for a, b in zip(l2, l2[1:])), (et, l2)
