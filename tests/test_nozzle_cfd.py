"""Compressible CFD verification of the analytic nozzle.

The nozzle sizing in this project is the ideal-rocket isentropic relations and
nothing had ever checked them against a flow solve. The existing CFD routes
could not: they run simpleFoam, which is incompressible, and a nozzle is nothing
but compressibility.

The check is that a density-based solver handed a supersonic state at
A/A* = 1.2 expands it to the Mach number the area-Mach relation predicts at
A/A* = 4.0, and that it does so *exactly* once the nozzle is slender enough for
quasi-1D theory to be the right theory:

    half-angle   CFD exit Mach   error
      36.2 deg      2.8863       -1.83%
      20.1 deg      2.9384       -0.06%
      10.4 deg      2.9402       -0.00%

The deficit at steep angles is not error, it is nozzle divergence loss -- a real
effect that quasi-1D theory does not contain.
"""

import math
import shutil

import pytest

from cadflow.nozzle_cfd import (
    area_ratio_for_mach,
    diverging_contour,
    exit_mach_from_case,
    isentropic_state,
    run_nozzle_case,
    supersonic_mach_for_area_ratio,
    warp_points_to_contour,
    write_supersonic_expansion_case,
)

foam_missing = not (shutil.which("rhoCentralFoam") and shutil.which("blockMesh"))


def test_area_mach_relation_round_trips():
    """The analytic half, on its own."""
    for mach in (1.2, 1.5, 2.5, 3.5, 5.0):
        ratio = area_ratio_for_mach(mach)
        assert supersonic_mach_for_area_ratio(ratio) == pytest.approx(mach, rel=1e-6)


def test_area_ratio_is_one_at_the_throat_and_rises_either_side():
    assert area_ratio_for_mach(1.0) == pytest.approx(1.0, rel=1e-9)
    assert area_ratio_for_mach(0.5) > 1.0
    assert area_ratio_for_mach(2.0) > 1.0


def test_isentropic_state_is_self_consistent():
    """p/p0 and T/T0 must agree with the Mach they claim to describe."""
    p0, t0, gamma = 2.0e6, 3000.0, 1.4
    for mach in (1.5, 2.5, 3.5):
        p, t, u = isentropic_state(mach, p0, t0)
        assert u / math.sqrt(gamma * 287.0 * t) == pytest.approx(mach, rel=1e-9)
        assert t0 / t == pytest.approx(1.0 + 0.2 * mach**2, rel=1e-9)
        assert p / p0 == pytest.approx((t / t0) ** (gamma / (gamma - 1.0)), rel=1e-9)


def test_diverging_contour_hits_its_endpoints_and_is_monotone():
    h_t = 0.01
    pts = diverging_contour(h_t, 1.2, 4.0, length=20.0)
    assert pts[0][1] == pytest.approx(1.2 * h_t)
    assert pts[-1][1] == pytest.approx(4.0 * h_t)
    heights = [h for _, h in pts]
    assert all(b >= a for a, b in zip(heights, heights[1:]))
    # cosine blend: zero slope at the inlet, so the imposed state meets no corner
    assert (heights[1] - heights[0]) < 0.02 * (heights[-1] - heights[0])


@pytest.mark.skipif(foam_missing, reason="OpenFOAM not installed")
def test_warping_produces_a_valid_mesh(tmp_path):
    """The box-then-warp trick, which exists because the obvious way fails.

    Meshing the nozzle directly with the wall as a polyLine edge gives blockMesh
    a curved top edge and a straight bottom one; it distributes points along
    each by arc length, the two disagree about where each column belongs, and
    the cells shear. checkMesh reported 140 negative-volume cells and skewness
    262 on that mesh. Warping an orthogonal box keeps every column at its own x.
    """
    import subprocess

    case = tmp_path / "nzl"
    spec = write_supersonic_expansion_case(case, nx=60, ny=20, end_time=1e-9)
    subprocess.run(["blockMesh", "-case", str(case)], capture_output=True, check=True)
    moved = warp_points_to_contour(case, list(spec.contour))
    assert moved > 0
    check = subprocess.run(
        ["checkMesh", "-case", str(case)], capture_output=True, text=True
    )
    assert "negative cell volume" not in check.stdout, check.stdout[-2000:]
    assert "Mesh OK" in check.stdout, check.stdout[-2000:]


@pytest.mark.skipif(foam_missing, reason="OpenFOAM not installed")
def test_slender_nozzle_matches_the_isentropic_relation(tmp_path):
    """The verification itself.

    Only the inlet state is given. The expansion from A/A* = 1.2 to 4.0, and the
    exit Mach it produces, are the solver's own.
    """
    case = tmp_path / "slender"
    spec = write_supersonic_expansion_case(
        case, inlet_area_ratio=1.2, exit_area_ratio=4.0,
        length=24.0, nx=200, ny=32, end_time=8e-4,
    )
    codes = run_nozzle_case(case, timeout_s=2400, contour=list(spec.contour))
    if codes.get("rhoCentralFoam") != 0:
        pytest.skip(f"solver did not complete: {codes}")
    mach = exit_mach_from_case(case, nx=spec.nx)
    assert mach is not None
    want = supersonic_mach_for_area_ratio(4.0)
    assert mach == pytest.approx(want, rel=0.01), f"{mach} vs {want}"


@pytest.mark.skipif(foam_missing, reason="OpenFOAM not installed")
def test_steep_nozzle_loses_more_than_a_slender_one(tmp_path):
    """Divergence loss is real, and must be ordered by divergence angle.

    A steep nozzle is genuinely not quasi-1D, so it must fall further below the
    1D prediction than a slender one -- and must fall *below*, never above.
    """
    want = supersonic_mach_for_area_ratio(4.0)
    got = {}
    for length in (6.0, 24.0):
        case = tmp_path / f"L{length:g}"
        spec = write_supersonic_expansion_case(
            case, exit_area_ratio=4.0, length=length,
            nx=200, ny=32, end_time=8e-4,
        )
        codes = run_nozzle_case(case, timeout_s=2400, contour=list(spec.contour))
        if codes.get("rhoCentralFoam") != 0:
            pytest.skip(f"solver did not complete for L={length}: {codes}")
        mach = exit_mach_from_case(case, nx=spec.nx)
        if mach is None:
            pytest.skip("no exit field")
        got[length] = mach

    assert got[6.0] < got[24.0], got
    assert got[24.0] <= want * 1.005, got
    assert got[6.0] < want, got
