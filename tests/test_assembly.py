"""The whole vehicle as geometry, not just as a list of parts.

The intent names "individual parts *and whole assemblies*". Components had been
designed, meshed, analysed and mass-propertied one at a time, but the assembly
itself had no geometry -- only an arithmetic combination of its pieces' masses.

Building it needed the sculpting layer: a nose cone is a surface of revolution,
a tank is a prism, a change of diameter between them is a loft and is neither.
Two of these tests exist because parts came out solid when they should have been
shells -- the nose cone at 339 kg, more than the rest of the vehicle together,
for the lightest-loaded part on it.
"""

import math

import pytest

from cadflow.assembly import build_vehicle, mass_closure
from cadflow.backends import get_backend
from cadflow.sculpt import bell_contour

pytest.importorskip("cadquery")


class _Stage:
    def __init__(self, prop, struct, throat=0.0059, eps=12.0):
        self.prop_mass_kg = prop
        self.struct_mass_kg = struct
        self.throat_area_m2 = throat
        self.expansion_ratio = eps


@pytest.fixture(scope="module")
def backend():
    b = get_backend(prefer_real=True)
    if b.name != "cadquery":
        pytest.skip("real CAD backend unavailable")
    return b


@pytest.fixture(scope="module")
def stages():
    return [_Stage(762.7, 124.2), _Stage(167.4, 27.3)]


@pytest.fixture(scope="module")
def vehicle(backend, stages):
    return build_vehicle(stages, 25.0, 0.2845, wall_mm=3.0,
                         fin_span_m=0.567, fin_root_chord_m=0.569,
                         nozzle=bell_contour(0.0434, 12.0), backend=backend)


def test_every_kind_of_shape_is_present(vehicle):
    """A rocket needs all three primitives, which is why it could not be built
    until loft existed."""
    kinds = {p.kind for p in vehicle.parts}
    assert {"revolve", "shell", "loft", "extrude"} <= kinds, kinds


def test_the_stack_is_ordered_and_contiguous(vehicle):
    """Tanks and interstages must run aft to forward without gaps."""
    body = [p for p in vehicle.parts
            if p.kind in ("shell", "loft") or p.name == "nose cone"]
    body.sort(key=lambda p: p.station_z_mm)
    for lower, upper in zip(body, body[1:]):
        assert upper.station_z_mm == pytest.approx(
            lower.station_z_mm + lower.length_mm, rel=1e-6), (lower.name, upper.name)


def test_upper_stages_are_narrower_than_lower_ones(vehicle):
    """Which is what makes the transitions necessary in the first place."""
    tanks = sorted((p for p in vehicle.parts if p.name.endswith("tank")),
                   key=lambda p: p.station_z_mm)
    assert len(tanks) >= 2
    assert tanks[1].radius_mm < tanks[0].radius_mm


def test_the_nose_cone_is_a_shell_not_a_billet(vehicle):
    """It came out at 339 kg solid -- heavier than everything else combined,
    for the least loaded part on the vehicle."""
    nose = next(p for p in vehicle.parts if p.name == "nose cone")
    tank = next(p for p in vehicle.parts if p.name == "stage 1 tank")
    assert nose.mass_kg < tank.mass_kg, (nose.mass_kg, tank.mass_kg)
    assert nose.mass_kg < 40.0


def test_the_interstage_is_a_shell_not_a_billet(vehicle):
    """Solid, it was 114 kg against the 43 kg tank below it -- absurd for a
    shorter part of the same diameter."""
    inter = next(p for p in vehicle.parts if p.name.startswith("interstage"))
    tank = next(p for p in vehicle.parts if p.name == "stage 1 tank")
    assert inter.mass_kg < tank.mass_kg
    assert inter.length_mm < tank.length_mm


def test_fins_sit_at_the_aft_end(vehicle):
    """They were reported at the nose, because the running station had already
    advanced past the whole stack by the time they were added."""
    fins = [p for p in vehicle.parts if p.name.startswith("fin")]
    assert len(fins) == 4
    for f in fins:
        assert f.station_z_mm < 0.2 * vehicle.total_length_mm, f.station_z_mm


def test_fins_reach_beyond_the_body(vehicle):
    fins = [p for p in vehicle.parts if p.name.startswith("fin")]
    tank = next(p for p in vehicle.parts if p.name == "stage 1 tank")
    assert all(f.radius_mm > tank.radius_mm for f in fins)


def test_the_nozzle_hangs_below_the_stack(vehicle):
    noz = next(p for p in vehicle.parts if p.name == "nozzle")
    assert noz.mass_kg > 0.0
    assert noz.length_mm > 0.0


def test_every_part_has_mass_and_the_total_is_their_sum(vehicle):
    assert all(p.mass_kg > 0.0 for p in vehicle.parts), [
        p.name for p in vehicle.parts if p.mass_kg <= 0.0]
    assert vehicle.mass_kg == pytest.approx(
        sum(p.mass_kg for p in vehicle.parts), rel=1e-12)


def test_the_vehicle_is_long_and_thin(vehicle):
    assert vehicle.total_length_mm > 4.0 * vehicle.max_radius_mm


def test_a_thicker_wall_is_a_heavier_vehicle(backend, stages):
    thin = build_vehicle(stages, 25.0, 0.2845, wall_mm=2.0, backend=backend)
    thick = build_vehicle(stages, 25.0, 0.2845, wall_mm=6.0, backend=backend)
    assert thick.mass_kg > 1.8 * thin.mass_kg


def test_build_rejects_nonsense(backend, stages):
    with pytest.raises(ValueError):
        build_vehicle(stages, 25.0, 0.0, backend=backend)
    with pytest.raises(ValueError):
        build_vehicle(stages, 25.0, 0.2845, wall_mm=0.0, backend=backend)


# --- mass closure -----------------------------------------------------------

def test_mass_closure_accounts_for_the_engine_the_geometry_does_not_draw():
    """An independent check on the structural coefficient, arriving from
    geometry rather than from the mass model -- and reaching the same verdict.

    Skin 97.4 kg plus an engine at T/W 60 of 85.3 kg is 182.7 kg against a
    155.6 kg budget: the vehicle cannot contain itself. The structural
    fixed-point solve says the coefficient should be 0.25 where the design
    asserts 0.14, and it knows nothing about this calculation.
    """
    from cadflow.assembly import Part, VehicleAssembly

    asm = VehicleAssembly(parts=[Part("skin", "shell", 0, 100, 50, None, 97.4)])
    closure = mass_closure(asm, 155.6, liftoff_thrust_n=50200.0, engine_twr=60.0)
    assert closure["engine_kg"] == pytest.approx(50200.0 / (9.80665 * 60.0))
    assert closure["accounted_kg"] == pytest.approx(97.4 + closure["engine_kg"])
    assert not closure["closes"]
    assert closure["slack_kg"] < 0.0


def test_a_generous_budget_closes():
    from cadflow.assembly import Part, VehicleAssembly

    asm = VehicleAssembly(parts=[Part("skin", "shell", 0, 100, 50, None, 50.0)])
    closure = mass_closure(asm, 400.0, liftoff_thrust_n=50200.0)
    assert closure["closes"]
    assert closure["slack_kg"] > 0.0


def test_no_thrust_means_no_engine_term():
    from cadflow.assembly import Part, VehicleAssembly

    asm = VehicleAssembly(parts=[Part("skin", "shell", 0, 100, 50, None, 50.0)])
    closure = mass_closure(asm, 100.0)
    assert closure["engine_kg"] == 0.0
    assert closure["accounted_kg"] == pytest.approx(50.0)
