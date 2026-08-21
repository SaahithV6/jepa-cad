"""The sculpting layer: shapes that are not primitives.

The geometry layer could make cylinders, boxes, extrusions and surfaces of
revolution, which covers an airframe section and a nose cone. The nozzle -- the
single most shaped part of a rocket -- existed only as an area ratio, with no
contour, no wall and no mass.

Two of these tests exist because of bugs found while writing the layer. The
shell operation silently returned an expanded bounding box on every input,
including returning 865% of a hollow tube's volume as a solid block. And the
nozzle wall was offset radially rather than normally, which thins it by
cos(theta) exactly where the contour is steepest.
"""

import math

import pytest

from cadflow.backends import get_backend
from cadflow.sculpt import (
    CONICAL_REFERENCE_DEG,
    bell_contour,
    nozzle_solid,
    transition_sections,
    transition_solid,
)

cq = pytest.importorskip("cadquery")


@pytest.fixture(scope="module")
def backend():
    b = get_backend(prefer_real=True)
    if b.name != "cadquery":
        pytest.skip("real CAD backend unavailable")
    return b


def _circle(r, n=64):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


# --- bell nozzle contour ----------------------------------------------------

@pytest.mark.parametrize("eps", [4.0, 12.0, 30.0, 80.0])
def test_exit_radius_is_forced_by_the_area_ratio(eps):
    """Not fitted, not chosen: the area ratio the engine was sized for fixes it."""
    n = bell_contour(0.0335, eps)
    assert n.exit_radius_m == pytest.approx(0.0335 * math.sqrt(eps), rel=1e-12)
    assert n.contour[-1][0] == pytest.approx(n.exit_radius_m, rel=1e-12)
    assert n.contour[0][0] == pytest.approx(0.0335, rel=1e-12)


@pytest.mark.parametrize("frac", [0.6, 0.8, 1.0])
def test_length_is_the_stated_fraction_of_a_fifteen_degree_cone(frac):
    """Which is what "an 80% bell" means."""
    n = bell_contour(0.0335, 12.0, percent_bell=frac)
    cone = (n.exit_radius_m - n.throat_radius_m) / math.tan(
        math.radians(CONICAL_REFERENCE_DEG))
    assert n.length_m == pytest.approx(frac * cone, rel=1e-12)


def test_the_contour_leaves_and_arrives_at_the_angles_asked_for():
    """Two endpoints and two tangents determine a quadratic completely, so
    these are constraints rather than outcomes."""
    n = bell_contour(0.0335, 12.0, initial_angle_deg=32.0, exit_angle_deg=8.0)
    c = n.contour
    start = math.degrees(math.atan2(c[1][0] - c[0][0], c[1][1] - c[0][1]))
    end = math.degrees(math.atan2(c[-1][0] - c[-2][0], c[-1][1] - c[-2][1]))
    assert start == pytest.approx(32.0, abs=0.5)
    assert end == pytest.approx(8.0, abs=0.5)


def test_the_contour_only_ever_widens():
    """A diverging nozzle that narrows anywhere would choke twice."""
    n = bell_contour(0.0335, 30.0)
    radii = [r for r, _z in n.contour]
    zs = [z for _r, z in n.contour]
    assert all(b >= a - 1e-12 for a, b in zip(radii, radii[1:]))
    assert all(b > a for a, b in zip(zs, zs[1:]))


def test_divergence_efficiency_rewards_a_shallower_exit():
    """The loss the CFD work measured directly: a steep exit throws thrust
    sideways. 1.83% below quasi-1D at 36 degrees, 0.00% at 10.4."""
    steep = bell_contour(0.0335, 12.0, exit_angle_deg=20.0)
    shallow = bell_contour(0.0335, 12.0, exit_angle_deg=5.0)
    assert shallow.divergence_efficiency > steep.divergence_efficiency
    assert steep.divergence_efficiency == pytest.approx(
        0.5 * (1 + math.cos(math.radians(20.0))), rel=1e-12)
    assert shallow.divergence_efficiency > 0.99


def test_bell_contour_rejects_nonsense():
    with pytest.raises(ValueError):
        bell_contour(0.0, 12.0)
    with pytest.raises(ValueError):
        bell_contour(0.03, 0.9)                       # not diverging
    with pytest.raises(ValueError):
        bell_contour(0.03, 12.0, exit_angle_deg=40.0, initial_angle_deg=30.0)
    with pytest.raises(ValueError):
        bell_contour(0.03, 12.0, percent_bell=3.0)


# --- the nozzle as a solid --------------------------------------------------

def test_nozzle_solid_is_one_watertight_body(backend):
    n = bell_contour(0.0335, 12.0)
    solid = nozzle_solid(n, 0.0025, backend=backend)
    shape = solid.val() if hasattr(solid, "val") else solid
    assert len(shape.Solids()) == 1
    assert shape.isValid()


def test_nozzle_wall_volume_matches_pappus(backend):
    """Surface of revolution: mid-surface area times thickness, exactly."""
    n = bell_contour(0.0335, 12.0)
    t = 0.0025
    solid = nozzle_solid(n, t, backend=backend)
    want = 0.0
    for (r0, z0), (r1, z1) in zip(n.contour, n.contour[1:]):
        slant = math.hypot(r1 - r0, z1 - z0)
        mid = (r0 + r1) / 2 + t / 2 * math.cos(math.atan2(r1 - r0, z1 - z0))
        want += 2 * math.pi * mid * t * slant
    assert backend.volume(solid) / 1e9 == pytest.approx(want, rel=0.01)


def test_the_wall_is_constant_thickness_not_radially_offset(backend):
    """The bug: a radial offset gives t*cos(theta) across the sheet, so the
    wall thins exactly where the contour is steepest. At the throat, where the
    slope is 30 degrees, that is 87% of the intended thickness, and the solid
    came out 5% light."""
    t = 0.0025
    thick = bell_contour(0.0335, 12.0)
    thin = bell_contour(0.0335, 12.0)
    v_thick = backend.volume(nozzle_solid(thick, t, backend=backend))
    v_thin = backend.volume(nozzle_solid(thin, t / 2, backend=backend))
    # halving the thickness must halve the wall volume, to first order
    assert v_thin / v_thick == pytest.approx(0.5, abs=0.02)


def test_a_bigger_area_ratio_is_a_heavier_nozzle(backend):
    small = nozzle_solid(bell_contour(0.0335, 6.0), 0.0025, backend=backend)
    big = nozzle_solid(bell_contour(0.0335, 40.0), 0.0025, backend=backend)
    assert backend.volume(big) > 3.0 * backend.volume(small)


def test_nozzle_wall_thickness_must_be_positive(backend):
    with pytest.raises(ValueError):
        nozzle_solid(bell_contour(0.0335, 12.0), 0.0, backend=backend)


# --- shell ------------------------------------------------------------------

def test_shell_is_exact(backend):
    """It used to return an expanded bounding box, silently, on every input."""
    base = cq.Workplane("XY").cylinder(60, 20)
    want = math.pi * (20**2 - 18**2) * 60 + 2 * math.pi * 18**2 * 2
    assert backend.volume(backend.sculpt_offset(base, -2.0)) == pytest.approx(
        want, rel=1e-6)


def test_shelling_removes_material(backend):
    """The property the old implementation violated most spectacularly: a
    hollow tube came back at 865% of its own volume."""
    for solid in (cq.Workplane("XY").cylinder(60, 20),
                  cq.Workplane("XY").box(40, 40, 40)):
        before = backend.volume(solid)
        after = backend.volume(backend.sculpt_offset(solid, -2.0))
        assert after < before, (before, after)


def test_an_open_shell_holds_less_than_a_closed_one(backend):
    base = cq.Workplane("XY").cylinder(60, 20)
    closed = backend.volume(backend.sculpt_offset(base, -2.0))
    opened = backend.volume(backend.sculpt_offset(base, -2.0, open_face=">Z"))
    assert opened < closed


def test_an_impossible_shell_raises_rather_than_inventing_geometry(backend):
    """A wall thicker than the part cannot be shelled. Saying so is the whole
    difference between a bug and a constraint."""
    base = cq.Workplane("XY").cylinder(60, 20)
    with pytest.raises(ValueError, match="shell"):
        backend.sculpt_offset(base, -50.0)


def test_zero_offset_is_a_no_op(backend):
    base = cq.Workplane("XY").cylinder(60, 20)
    assert backend.volume(backend.sculpt_offset(base, 0.0)) == pytest.approx(
        backend.volume(base))


# --- loft -------------------------------------------------------------------

def test_loft_reproduces_a_conical_frustum(backend):
    """The exact volume is pi h (r0^2 + r0 r1 + r1^2)/3; the small shortfall is
    the polygon inscribed in the circle, which for 64 sides is 0.161%."""
    r0, r1, h = 30.0, 10.0, 50.0
    solid = backend.loft_sections([(0.0, _circle(r0)), (h, _circle(r1))],
                                  ruled=True)
    exact = math.pi * h / 3 * (r0 * r0 + r0 * r1 + r1 * r1)
    polygon_factor = (64 / (2 * math.pi)) * math.sin(2 * math.pi / 64)
    assert backend.volume(solid) == pytest.approx(exact * polygon_factor, rel=2e-3)


def test_loft_needs_at_least_two_sections(backend):
    with pytest.raises(ValueError):
        backend.loft_sections([(0.0, _circle(10.0))])


def test_loft_rejects_a_degenerate_section(backend):
    with pytest.raises(ValueError):
        backend.loft_sections([(0.0, _circle(10.0)), (5.0, [(0, 0), (1, 1)])])


# --- stage transition -------------------------------------------------------

def test_transition_meets_both_diameters(backend):
    sections = transition_sections(0.20, 0.15, 0.30)
    first = max(abs(x) for x, _y in sections[0][1])
    last = max(abs(x) for x, _y in sections[-1][1])
    assert first == pytest.approx(200.0, rel=1e-6)
    assert last == pytest.approx(150.0, rel=1e-6)


def test_transition_is_tangent_at_both_ends():
    """A cosine blend, so it joins each cylinder without a slope break -- the
    same reason a tangent ogive beats a cone, and the same consequence: a slope
    discontinuity is where drag and stress both concentrate."""
    sections = transition_sections(0.20, 0.15, 0.30, stations=40)
    radii = [max(abs(x) for x, _y in ring) for _z, ring in sections]
    start_slope = abs(radii[1] - radii[0])
    mid = len(radii) // 2
    mid_slope = abs(radii[mid + 1] - radii[mid])
    end_slope = abs(radii[-1] - radii[-2])
    assert start_slope < 0.1 * mid_slope
    assert end_slope < 0.1 * mid_slope


def test_transition_is_one_watertight_solid(backend):
    solid = transition_solid(0.20, 0.15, 0.30, backend=backend)
    shape = solid.val() if hasattr(solid, "val") else solid
    assert len(shape.Solids()) == 1
    assert shape.isValid()
    assert backend.volume(solid) > 0.0


def test_transition_rejects_nonsense():
    with pytest.raises(ValueError):
        transition_sections(0.0, 0.15, 0.30)
    with pytest.raises(ValueError):
        transition_sections(0.20, 0.15, 0.0)
