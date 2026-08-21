"""Nose-cone meridian profiles.

The airframe used to be built entirely from cylinders and boxes, which meant the
"nose cone" was a cylinder of the body radius with a flat forward face. That is
wrong twice over: it has none of the drag behaviour of a real nose, and its flat
face carries load nothing like a curved one, so an FEA on it was answering a
question about a shape that would never fly.

These are the standard rocketry meridian curves, each returned as (radius, z)
points to be revolved about Z. z runs from 0 at the base to `length` at the tip;
the caller re-centres. Every family satisfies r(0) = radius and r(length) = 0.

  ogive       tangent ogive -- the common airframe nose; meets the body tangent
              so there is no slope discontinuity at the joint
  conical     straight taper; simple to make, higher drag
  elliptical  blunt nose, best subsonic, poor transonic
  vonkarman   the Haack series at C=0, the minimum-drag body of revolution for a
              given length and base radius; not tangent at the base, so it has a
              small slope break there

A note on hollow noses: the shell is formed by revolving this profile and
cutting an inner one built with (radius - wall, length - wall). That gives an
exactly `wall` thickness at the base, where the surface is near-axial and where
the hoop and axial stresses actually live, and somewhat less than `wall` normal
thickness up near the tip where the surface is inclined. The tip is a solid plug
either way, which is what real nose cones are.
"""

from __future__ import annotations

import math
from typing import Sequence

NOSE_SHAPES = ("ogive", "conical", "elliptical", "vonkarman")


def nose_profile(
    radius: float,
    length: float,
    shape: str = "ogive",
    n: int = 24,
) -> list[tuple[float, float]]:
    """Meridian of a nose cone as (radius, z), base at z=0, tip at z=length."""
    r = float(radius)
    ell = float(length)
    if r <= 0.0 or ell <= 0.0:
        raise ValueError(f"nose_profile needs positive radius/length, got {r}/{ell}")
    shape = str(shape).lower()
    if shape not in NOSE_SHAPES:
        raise ValueError(f"unknown nose shape {shape!r}, expected one of {NOSE_SHAPES}")
    n = max(8, int(n))

    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        z = ell * i / n
        x = ell - z  # distance from the tip, which is how the curves are defined
        if shape == "conical":
            y = r * x / ell
        elif shape == "elliptical":
            # r(z) = R sqrt(1 - (z/L)^2)
            y = r * math.sqrt(max(0.0, 1.0 - (z / ell) ** 2))
        elif shape == "ogive":
            # rho is the ogive radius: the circle through tip and base that is
            # tangent to the body at the base.
            rho = (r * r + ell * ell) / (2.0 * r)
            y = math.sqrt(max(0.0, rho * rho - z * z)) + r - rho
        else:  # vonkarman
            # theta = arccos(1 - 2x/L);  y = R/sqrt(pi) * sqrt(theta - sin(2 theta)/2)
            theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * x / ell)))
            y = (r / math.sqrt(math.pi)) * math.sqrt(
                max(0.0, theta - math.sin(2.0 * theta) / 2.0)
            )
        pts.append((max(0.0, y), z))

    # Pin the ends exactly; the closed forms are within rounding of these but the
    # revolve wants the base radius to match the body it joins, to the micron.
    pts[0] = (r, 0.0)
    pts[-1] = (0.0, ell)
    return pts


def profile_volume(profile: Sequence[tuple[float, float]]) -> float:
    """Exact volume of the solid of revolution of a piecewise-linear profile.

    Each segment sweeps a conical frustum, so this is exact for a polyline and a
    good check on a spline fit through the same points.
    """
    total = 0.0
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        dz = abs(z1 - z0)
        total += math.pi * dz * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
    return total


def centred(profile: Sequence[tuple[float, float]], length: float) -> list[tuple[float, float]]:
    """Shift a base-at-zero profile so it is centred on z=0, like the primitives."""
    half = float(length) / 2.0
    return [(r, z - half) for r, z in profile]


def wetted_area(profile: Sequence[tuple[float, float]]) -> float:
    """Lateral surface area of the solid of revolution of a polyline meridian.

    Each segment sweeps a conical frustum of area pi (r0 + r1) * slant, so this
    is exact for a polyline and converges quickly for a sampled curve. Shape
    reaches the trajectory through this: a blunt elliptical nose wets far more
    skin than a cone of the same length and base, and skin friction scales with
    wetted area.
    """
    total = 0.0
    for (r0, z0), (r1, z1) in zip(profile, profile[1:]):
        slant = math.hypot(r1 - r0, z1 - z0)
        total += math.pi * (r0 + r1) * slant
    return total


def fin_planform(
    span: float,
    root_chord: float,
    taper_ratio: float = 0.5,
    sweep_frac: float = 0.6,
) -> list[tuple[float, float]]:
    """Trapezoidal fin planform as (span_station, axial) points.

    Fins were boxes: constant chord, no taper, no sweep. A box fin has its whole
    area out at the tip where it does least for stability and most for root
    bending, and a square leading edge that no supersonic fin has. This is the
    standard trapezoid -- root chord at the body, a shorter tip chord swept aft.

    x runs radially outward from the root, y runs aft from the root leading
    edge. The caller extrudes through the thickness and turns it into place.
    """
    s = float(span)
    cr = float(root_chord)
    if s <= 0.0 or cr <= 0.0:
        raise ValueError(f"fin needs positive span/chord, got {s}/{cr}")
    ct = cr * max(0.05, min(1.0, float(taper_ratio)))
    sweep = cr * max(0.0, float(sweep_frac))
    return [(0.0, 0.0), (0.0, cr), (s, sweep + ct), (s, sweep)]


def planform_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace area of a planform polygon."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + list(points[:1])):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0
