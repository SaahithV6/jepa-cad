"""Geometry of the sculpted parts: nose meridians and fin planforms.

Every assertion here is against a closed form, not against a previously
recorded value. A regression test that only says "the number did not change"
cannot tell a correct shape from one that has been wrong since the day it was
written -- which is exactly what happened when the nose cone was a cylinder.
"""

import math

import pytest

from cadflow.profiles import (
    NOSE_SHAPES,
    centred,
    fin_planform,
    nose_profile,
    planform_area,
    profile_volume,
    wetted_area,
)


@pytest.mark.parametrize("shape", NOSE_SHAPES)
def test_nose_meets_its_endpoints_exactly(shape):
    """r(0) = R at the base and r(L) = 0 at the tip, to the micron.

    The base radius has to match the body the nose joins or the revolve leaves a
    step, and the tip has to close or the solid is not watertight.
    """
    r, ell = 35.654, 57.0
    prof = nose_profile(r, ell, shape, 64)
    assert prof[0] == (r, 0.0)
    assert prof[-1] == (0.0, ell)
    assert all(0.0 <= q <= r + 1e-9 for q, _ in prof)
    zs = [z for _, z in prof]
    assert zs == sorted(zs)


def test_cone_volume_and_wetted_area_are_the_closed_forms():
    r, ell = 0.5, 2.0
    prof = nose_profile(r, ell, "conical", 2000)
    assert profile_volume(prof) == pytest.approx(math.pi * r * r * ell / 3.0, rel=1e-6)
    assert wetted_area(prof) == pytest.approx(math.pi * r * math.hypot(r, ell), rel=1e-6)


def test_ellipsoid_volume_is_two_thirds_of_the_cylinder():
    r, ell = 0.5, 2.0
    prof = nose_profile(r, ell, "elliptical", 4000)
    assert profile_volume(prof) == pytest.approx(
        2.0 / 3.0 * math.pi * r * r * ell, rel=1e-4
    )


def test_ogive_is_tangent_to_the_body_at_its_base():
    """The defining property of a *tangent* ogive: dr/dz = 0 where it meets the
    cylinder, so there is no slope break at the joint. This is why it costs less
    wave drag than a cone and why it does not raise a stress riser there."""
    r, ell = 1.0, 3.0
    prof = nose_profile(r, ell, "ogive", 4000)
    (r0, z0), (r1, z1) = prof[0], prof[1]
    assert abs((r1 - r0) / (z1 - z0)) < 1e-3


def test_cone_is_the_slenderest_and_ellipse_the_bluntest():
    """Volume ordering follows bluntness, which is a shape fact, not a fit."""
    r, ell = 0.5, 2.0
    vols = {s: profile_volume(nose_profile(r, ell, s, 2000)) for s in NOSE_SHAPES}
    assert vols["conical"] < vols["vonkarman"] < vols["ogive"] < vols["elliptical"]


def test_centred_shifts_without_changing_the_shape():
    prof = nose_profile(1.0, 4.0, "ogive", 32)
    moved = centred(prof, 4.0)
    assert moved[0][1] == pytest.approx(-2.0)
    assert moved[-1][1] == pytest.approx(2.0)
    assert profile_volume(moved) == pytest.approx(profile_volume(prof))


def test_nose_profile_rejects_nonsense():
    with pytest.raises(ValueError):
        nose_profile(0.0, 4.0, "ogive")
    with pytest.raises(ValueError):
        nose_profile(1.0, -1.0, "ogive")
    with pytest.raises(ValueError):
        nose_profile(1.0, 4.0, "not-a-shape")


def test_fin_planform_area_is_the_trapezoid_area():
    span, root, taper = 14.0, 18.0, 0.5
    pts = fin_planform(span, root, taper_ratio=taper)
    tip = root * taper
    assert planform_area(pts) == pytest.approx((root + tip) / 2.0 * span, rel=1e-9)


def test_fin_is_tapered_and_swept_rather_than_a_box():
    """A box fin has its whole area at the tip, where it does least for
    stability and most for root bending."""
    span, root = 14.0, 18.0
    pts = fin_planform(span, root, taper_ratio=0.5, sweep_frac=0.6)
    root_chord = pts[1][1] - pts[0][1]
    tip_chord = pts[2][1] - pts[3][1]
    assert root_chord == pytest.approx(root)
    assert tip_chord < root_chord                       # tapered
    assert pts[3][1] > pts[0][1]                        # leading edge swept aft
    assert planform_area(pts) < span * root             # smaller than the box


def test_fin_taper_ratio_one_is_a_parallelogram():
    pts = fin_planform(10.0, 20.0, taper_ratio=1.0, sweep_frac=0.0)
    assert planform_area(pts) == pytest.approx(200.0)


def test_fin_planform_rejects_nonsense():
    with pytest.raises(ValueError):
        fin_planform(0.0, 10.0)
    with pytest.raises(ValueError):
        fin_planform(10.0, 0.0)


# --- fin attachment ---------------------------------------------------------
# A component once came back with no usable result at all -- gmsh fell back to a
# convex hull, which for a hollow part is a solid billet. The cause was that the
# fin root embedded deeper than the wall was thick, so the fin punched through
# and left a sliver dangling inside the cavity. These check the geometry that
# caused it, without needing to mesh anything.

def _fin_root_x(constraints):
    """Radial station of the fin root in the built spec."""
    from scripts.smoke_params_to_assembly import constraints_to_geometry

    spec = constraints_to_geometry(constraints)
    fins = [p for p in spec["parts"] if p["kind"] == "extrude"]
    assert fins, "no fin in the spec"
    return fins[0]["params"]["at"][0]


@pytest.mark.parametrize("wall", [0.4, 0.8, 1.5, 3.0, 6.0, 12.0])
def test_fin_root_never_passes_through_the_wall(wall):
    """The bug, stated directly: the root must stay within the shell.

    The old embed depth was min(2.0, wall + 1.0), which exceeds the wall for any
    wall under 2 mm. On a 0.8 mm shell that put the root 1.0 mm past the inner
    surface.
    """
    body_r = 38.98
    root_x = _fin_root_x(dict(
        body_radius_mm=body_r, body_height_mm=78.0, nose_radius_mm=body_r,
        nose_height_mm=27.0, fin_span_mm=99.75, fin_thickness_mm=5.0,
        fin_chord_mm=78.0, wall_thickness_mm=wall))
    inner_r = body_r - wall
    assert root_x > inner_r, (
        f"fin root at {root_x:.3f} is inside the cavity "
        f"(inner surface {inner_r:.3f}), wall {wall}")
    assert root_x < body_r, "fin root must be embedded, not floating outside"


@pytest.mark.parametrize("wall", [0.4, 0.8, 1.5, 3.0])
def test_fin_root_is_embedded_deep_enough_to_union(wall):
    """It must overlap the skin, or the boolean union leaves two solids."""
    body_r = 38.98
    root_x = _fin_root_x(dict(
        body_radius_mm=body_r, body_height_mm=78.0, nose_radius_mm=body_r,
        nose_height_mm=27.0, fin_span_mm=99.75, fin_thickness_mm=5.0,
        fin_chord_mm=78.0, wall_thickness_mm=wall))
    overlap = body_r - root_x
    assert overlap >= 0.25 * wall, (overlap, wall)


def test_a_hollow_finned_part_is_one_watertight_solid():
    """The consequence that matters: it has to be a single body to mesh."""
    cq = pytest.importorskip("cadquery")
    from cadflow.backends import build_from_spec, get_backend
    from scripts.smoke_params_to_assembly import constraints_to_geometry

    backend = get_backend(prefer_real=True)
    if backend.name != "cadquery":
        pytest.skip("real CAD backend unavailable")
    shape = build_from_spec(constraints_to_geometry(dict(
        body_radius_mm=38.98, body_height_mm=77.97, nose_radius_mm=38.98,
        nose_height_mm=27.29, fin_span_mm=99.75, fin_thickness_mm=5.0,
        fin_chord_mm=77.97, wall_thickness_mm=0.8)), backend=backend)
    solid = shape.val() if hasattr(shape, "val") else shape
    assert len(solid.Solids()) == 1
    # the sliver tab added faces; without it this part is simple
    assert len(solid.Faces()) <= 12, len(solid.Faces())
