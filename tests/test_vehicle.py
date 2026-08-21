"""Whole-vehicle mass properties, checked against geometry solved as one piece.

Combining parts with the parallel-axis theorem is easy to get subtly wrong --
a missed shift, or shifting about the wrong axis -- and the result stays
plausible either way. So the test builds a shape twice: once as separate parts
combined arithmetically, and once as a single solid handed to the CAD kernel.
They must agree.
"""

import math

import pytest

from cadflow.vehicle import (
    Placed, combine, cylinder_inertia, flight_vehicle_properties, static_margin)

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


def test_cylinder_inertia_matches_the_closed_forms():
    m, r, ell = 3.0, 0.2, 1.5
    ixx, izz = cylinder_inertia(m, r, ell)
    assert izz == pytest.approx(m * r * r / 2.0)
    assert ixx == pytest.approx(m * (3 * r * r + ell * ell) / 12.0)


def test_cylinder_inertia_against_the_cad_kernel():
    cq = pytest.importorskip("cadquery")
    from cadflow.backends import get_backend

    backend = get_backend(prefer_real=True)
    if backend.name != "cadquery":
        pytest.skip("real CAD backend unavailable")
    r_mm, l_mm = 100.0, 400.0
    solid = cq.Workplane("XY").cylinder(l_mm, r_mm)
    mp = backend.mass_properties(solid, RHO)
    ixx, izz = cylinder_inertia(mp["mass_kg"], r_mm / 1000.0, l_mm / 1000.0)
    assert ixx == pytest.approx(mp["Ixx_kg_m2"], rel=1e-4)
    assert izz == pytest.approx(mp["Izz_kg_m2"], rel=1e-4)


class _Stage:
    def __init__(self, prop, struct):
        self.prop_mass_kg = prop
        self.struct_mass_kg = struct


def test_flight_vehicle_conserves_mass():
    """Wet mass must reproduce the planner's gross exactly, or the model is
    describing a different vehicle than the one that flew the trajectory."""
    stages = [_Stage(762.71, 124.16), _Stage(167.42, 27.26)]
    payload = 25.0
    veh = flight_vehicle_properties(stages, payload, 0.2845)
    expected = sum(s.prop_mass_kg + s.struct_mass_kg for s in stages) + payload
    assert veh["mass_kg"] == pytest.approx(expected, rel=1e-12)


def test_flight_vehicle_cg_lies_inside_the_vehicle():
    stages = [_Stage(762.71, 124.16), _Stage(167.42, 27.26)]
    veh = flight_vehicle_properties(stages, 25.0, 0.2845)
    assert 0.0 < veh["cg_z_m"] < veh["length_m"]
    # heavy first stage is aft, so the CG must sit below mid-length
    assert veh["cg_z_m"] < 0.6 * veh["length_m"]


def test_flight_vehicle_is_long_and_thin_in_its_inertia():
    stages = [_Stage(762.71, 124.16), _Stage(167.42, 27.26)]
    veh = flight_vehicle_properties(stages, 25.0, 0.2845)
    assert veh["Ixx_kg_m2"] > 10.0 * veh["Izz_kg_m2"]
    assert veh["length_m"] > 4.0 * (2.0 * veh["radius_m"])


def test_a_bigger_stage_makes_a_longer_vehicle():
    small = flight_vehicle_properties([_Stage(100.0, 20.0)], 10.0, 0.2845)
    big = flight_vehicle_properties([_Stage(800.0, 120.0)], 10.0, 0.2845)
    assert big["length_m"] > small["length_m"]
    assert big["Ixx_kg_m2"] > small["Ixx_kg_m2"]


def test_flight_vehicle_rejects_a_nonsense_radius():
    with pytest.raises(ValueError):
        flight_vehicle_properties([_Stage(100.0, 20.0)], 10.0, 0.0)


def test_nose_cp_reproduces_the_exact_families():
    """Cone at 2L/3 and von Karman at L/2 are exact, so they pin the formula."""
    from cadflow.profiles import nose_profile
    from cadflow.vehicle import nose_center_of_pressure

    r, ell = 1.0, 5.0
    cone = nose_center_of_pressure(nose_profile(r, ell, "conical", 20000), r)
    vk = nose_center_of_pressure(nose_profile(r, ell, "vonkarman", 20000), r)
    assert cone / ell == pytest.approx(2.0 / 3.0, rel=1e-4)
    assert vk / ell == pytest.approx(0.5, rel=1e-4)


def test_ogive_cp_approaches_the_tabulated_value_as_it_slims():
    """Tables give a single 0.466 L; that is the slender limit, not a constant."""
    from cadflow.profiles import nose_profile
    from cadflow.vehicle import nose_center_of_pressure

    ratios = []
    for fineness in (1.0, 2.5, 5.0, 8.0):
        ell = 2.0 * fineness
        ratios.append(
            nose_center_of_pressure(nose_profile(1.0, ell, "ogive", 20000), 1.0) / ell
        )
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] == pytest.approx(0.466, abs=0.002)
    assert ratios[0] < 0.44


def test_nose_cp_is_forward_of_the_base():
    from cadflow.profiles import nose_profile
    from cadflow.vehicle import nose_center_of_pressure

    for shape in ("ogive", "conical", "vonkarman"):
        x = nose_center_of_pressure(nose_profile(1.0, 5.0, shape, 4000), 1.0)
        assert 0.0 < x < 5.0


def test_finless_vehicle_is_unstable():
    """The expected answer, not a bug: this is why fins exist."""
    from cadflow.profiles import nose_profile
    from cadflow.vehicle import body_alone_static_margin

    prof = nose_profile(0.2845, 1.5, "ogive", 4000)
    margin = body_alone_static_margin(prof, 0.2845,
                                      nose_tip_station_m=4.16, cg_z_m=1.841)
    assert margin < 0.0


def test_nose_cp_rejects_a_nonsense_radius():
    from cadflow.profiles import nose_profile
    from cadflow.vehicle import nose_center_of_pressure

    with pytest.raises(ValueError):
        nose_center_of_pressure(nose_profile(1.0, 5.0, "ogive", 100), 0.0)
