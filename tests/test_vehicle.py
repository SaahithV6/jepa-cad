"""Whole-vehicle mass properties, checked against geometry solved as one piece.

Combining parts with the parallel-axis theorem is easy to get subtly wrong --
a missed shift, or shifting about the wrong axis -- and the result stays
plausible either way. So the test builds a shape twice: once as separate parts
combined arithmetically, and once as a single solid handed to the CAD kernel.
They must agree.
"""

import math

import pytest

from cadflow.vehicle import Placed, combine, static_margin

RHO = 2700.0


def _box(mass, ix, iy, iz, z=0.0, cz=0.0, name="p"):
    return Placed(name=name, mass_kg=mass, cx_m=0.0, cy_m=0.0, cz_m=cz,
                  Ixx_kg_m2=ix, Iyy_kg_m2=iy, Izz_kg_m2=iz, station_z_m=z)


def test_single_part_is_itself():
    p = _box(2.0, 0.1, 0.2, 0.3, z=5.0)
    got = combine([p])
    assert got["mass_kg"] == pytest.approx(2.0)
    assert got["cg_z_m"] == pytest.approx(5.0)
    assert got["Ixx_kg_m2"] == pytest.approx(0.1)
    assert got["Izz_kg_m2"] == pytest.approx(0.3)


def test_two_equal_masses_put_the_cg_between_them():
    got = combine([_box(1.0, 0, 0, 0, z=-1.0), _box(1.0, 0, 0, 0, z=1.0)])
    assert got["cg_z_m"] == pytest.approx(0.0)
    assert got["mass_kg"] == pytest.approx(2.0)
    # two point masses at +-1 m about the CG: I = sum m d^2 = 2 kg m^2
    assert got["Ixx_kg_m2"] == pytest.approx(2.0)
    assert got["Iyy_kg_m2"] == pytest.approx(2.0)
    # ...and nothing off the axis, so no roll inertia
    assert got["Izz_kg_m2"] == pytest.approx(0.0)


def test_cg_is_mass_weighted_not_averaged():
    got = combine([_box(3.0, 0, 0, 0, z=0.0), _box(1.0, 0, 0, 0, z=4.0)])
    assert got["cg_z_m"] == pytest.approx(1.0)


def test_empty_and_massless_vehicles_are_refused():
    with pytest.raises(ValueError):
        combine([])
    with pytest.raises(ValueError):
        combine([_box(0.0, 0, 0, 0)])


def test_parallel_axis_matches_the_solid_it_describes():
    """The real check: two stacked boxes, combined vs solved as one bar.

    A 20 x 20 x 100 mm bar is exactly two 20 x 20 x 50 mm boxes stacked, so the
    arithmetic must reproduce the closed form for the whole bar.
    """
    a, b, half = 0.020, 0.020, 0.050
    m_half = RHO * a * b * half
    # each half about its own centroid
    ixx_half = m_half * (b * b + half * half) / 12.0
    izz_half = m_half * (a * a + b * b) / 12.0

    got = combine([
        _box(m_half, ixx_half, ixx_half, izz_half, z=-half / 2.0),
        _box(m_half, ixx_half, ixx_half, izz_half, z=+half / 2.0),
    ])

    length = 2.0 * half
    m_full = RHO * a * b * length
    ixx_full = m_full * (b * b + length * length) / 12.0
    izz_full = m_full * (a * a + b * b) / 12.0

    assert got["mass_kg"] == pytest.approx(m_full, rel=1e-12)
    assert got["cg_z_m"] == pytest.approx(0.0, abs=1e-12)
    assert got["Ixx_kg_m2"] == pytest.approx(ixx_full, rel=1e-12)
    assert got["Izz_kg_m2"] == pytest.approx(izz_full, rel=1e-12)


def test_against_the_cad_kernel_on_the_same_geometry():
    """Same bar, but with the halves' own inertia coming from the CAD kernel."""
    cq = pytest.importorskip("cadquery")
    from cadflow.backends import get_backend

    backend = get_backend(prefer_real=True)
    if backend.name != "cadquery":
        pytest.skip("real CAD backend unavailable")

    # millimetres in CAD, metres out of mass_properties
    half = cq.Workplane("XY").box(20.0, 20.0, 50.0)
    full = cq.Workplane("XY").box(20.0, 20.0, 100.0)
    mp_half = backend.mass_properties(half, RHO)
    mp_full = backend.mass_properties(full, RHO)

    got = combine([
        Placed("lower", mp_half["mass_kg"], mp_half["cx_m"], mp_half["cy_m"],
               mp_half["cz_m"], mp_half["Ixx_kg_m2"], mp_half["Iyy_kg_m2"],
               mp_half["Izz_kg_m2"], station_z_m=-0.025),
        Placed("upper", mp_half["mass_kg"], mp_half["cx_m"], mp_half["cy_m"],
               mp_half["cz_m"], mp_half["Ixx_kg_m2"], mp_half["Iyy_kg_m2"],
               mp_half["Izz_kg_m2"], station_z_m=+0.025),
    ])

    assert got["mass_kg"] == pytest.approx(mp_full["mass_kg"], rel=1e-9)
    assert got["Ixx_kg_m2"] == pytest.approx(mp_full["Ixx_kg_m2"], rel=1e-9)
    assert got["Iyy_kg_m2"] == pytest.approx(mp_full["Iyy_kg_m2"], rel=1e-9)
    assert got["Izz_kg_m2"] == pytest.approx(mp_full["Izz_kg_m2"], rel=1e-9)


def test_static_margin_sign_and_scale():
    """Positive calibers means the centre of pressure is aft of the CG."""
    # +z forward: cp behind cg is the stable case
    assert static_margin(cg_z_m=1.0, cp_z_m=0.8, body_diameter_m=0.1) == pytest.approx(2.0)
    assert static_margin(cg_z_m=0.8, cp_z_m=1.0, body_diameter_m=0.1) < 0.0
    with pytest.raises(ValueError):
        static_margin(1.0, 0.5, 0.0)


def test_moving_mass_forward_moves_the_cg_forward():
    """Sanity on the direction, since a sign error here inverts stability."""
    aft = combine([_box(1.0, 0, 0, 0, z=0.0), _box(1.0, 0, 0, 0, z=1.0)])
    fwd = combine([_box(1.0, 0, 0, 0, z=0.0), _box(1.0, 0, 0, 0, z=3.0)])
    assert fwd["cg_z_m"] > aft["cg_z_m"]
    assert math.isfinite(fwd["Ixx_kg_m2"])
