"""The whole vehicle as one solid.

Components have been designed, meshed, analysed and mass-propertied one at a
time. The intent names "individual parts *and whole assemblies*", and the whole
assembly has never existed: there was no geometry for the vehicle, only a list
of geometries for its pieces and an arithmetic combination of their masses.

Building it needs the sculpting layer. A rocket is a nose cone (a surface of
revolution), a stack of tanks (prisms), transitions between stages of different
diameter (lofts, which are neither), fins (extrusions turned into place), and a
nozzle (a surface of revolution about a curve fitted to angle constraints). Only
after loft and a working shell did all five exist.

Everything here is at flight scale, in millimetres, positioned nose-forward with
z increasing toward the nose so it matches the stability convention used
elsewhere -- centre of pressure aft of centre of gravity is a positive margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Circle discretisation for lofted sections. 64 sides costs 0.161% of the area
#: of the circle it inscribes, which is below every other error in the chain.
CIRCLE_POINTS = 64


@dataclass
class Part:
    """One piece of the vehicle, with where it sits and what it is."""
    name: str
    kind: str
    station_z_mm: float
    length_mm: float
    radius_mm: float
    solid: object = None
    mass_kg: float = 0.0


@dataclass
class VehicleAssembly:
    parts: list[Part] = field(default_factory=list)
    total_length_mm: float = 0.0
    max_radius_mm: float = 0.0

    @property
    def mass_kg(self) -> float:
        return sum(p.mass_kg for p in self.parts)

    def summary(self) -> list[dict]:
        return [{"name": p.name, "kind": p.kind,
                 "station_mm": p.station_z_mm, "length_mm": p.length_mm,
                 "radius_mm": p.radius_mm, "mass_kg": p.mass_kg}
                for p in self.parts]


#: The drag model and the geometry module disagree on one name: `wave_drag`
#: and the CFD corpus call it "cone", `profiles.nose_profile` calls it
#: "conical". Passing one to the other raises, which is the good outcome --
#: silently defaulting to an ogive would have built the wrong body while every
#: number in the packet said otherwise.
_SHAPE_ALIASES = {"cone": "conical"}


#: How much each stage narrows relative to the one below it.
#:
#: A module constant rather than a bare default, so the structural checks can
#: reconstruct the same radii the CAD draws. Two places needing this number and
#: only one of them owning it is how a check comes to describe a vehicle that
#: was never built.
TAPER_PER_STAGE = 0.92


def interstage_length(radius, next_radius):
    """How long the interstage between two stages is: 1.5 |dr| + 0.5 r.

    A shared definition rather than a formula written twice. The buckling check
    in ``cadflow.dry_structure`` needs this length -- the column mode goes as
    1/L^2 and the Batdorf parameter as L^2, so a guessed length is not a small
    error -- and the only thing worse than the check not having it would be the
    check having its own copy that drifts from the geometry actually drawn.
    """
    return 1.5 * abs(radius - next_radius) + 0.5 * radius


def _geometry_shape_name(shape: str) -> str:
    """Translate a drag-model shape name into the geometry module's spelling."""
    from cadflow.profiles import NOSE_SHAPES

    key = str(shape).lower()
    key = _SHAPE_ALIASES.get(key, key)
    if key not in NOSE_SHAPES:
        raise ValueError(
            f"nose shape {shape!r} has no geometry; known shapes are "
            f"{NOSE_SHAPES} (aliases: {_SHAPE_ALIASES})")
    return key


def build_vehicle(stages, payload_kg: float, body_radius_m: float,
                  wall_mm: float = 3.0, nose_fineness: float = 2.0,
                  fin_span_m: float = 0.0, fin_root_chord_m: float = 0.0,
                  n_fins: int = 4, nozzle=None, backend=None,
                  skin_density: float = 2700.0,
                  taper_per_stage: float = TAPER_PER_STAGE,
                  nose_shape: str = "ogive") -> VehicleAssembly:
    """Assemble the complete vehicle from the planner's stack.

    Stage lengths come from propellant volume at the loaded bulk density, the
    same way the mass-properties model computes them, so the geometry and the
    mass budget describe one vehicle rather than two.

    Upper stages are drawn narrower than lower ones, which is what makes the
    transitions necessary and is why this could not be built before: a change of
    diameter between two cylinders is a loft, and there was no loft.
    """
    from cadflow.backends import get_backend
    from cadflow.profiles import centred, nose_profile
    from cadflow.sculpt import transition_sections

    b = backend or get_backend(prefer_real=True)
    scale = 1000.0
    r_base = float(body_radius_m) * scale
    wall = float(wall_mm)
    if r_base <= 0.0 or wall <= 0.0:
        raise ValueError("body radius and wall thickness must be positive")

    asm = VehicleAssembly()
    z = 0.0                                   # aft end of the stack

    def add(name, kind, solid, length, radius, station=None):
        mass = 0.0
        if solid is not None:
            try:
                mass = b.mass_properties(solid, skin_density)["mass_kg"]
            except Exception:  # noqa: BLE001
                mass = 0.0
        asm.parts.append(Part(name=name, kind=kind,
                              station_z_mm=z if station is None else station,
                              length_mm=length, radius_mm=radius,
                              solid=solid, mass_kg=mass))
        asm.max_radius_mm = max(asm.max_radius_mm, radius)

    # nozzle first, hanging off the aft end
    if nozzle is not None:
        from cadflow.sculpt import nozzle_solid

        noz = nozzle_solid(nozzle, wall / scale / 2.0, backend=b)
        noz = b.translate(noz, 0.0, 0.0, -nozzle.length_m * scale)
        # Record the station it was translated to, not the station the loop
        # happens to be at.
        #
        # add() defaults to z, which is still zero here, so the table reported
        # the nozzle at 0.000 m while the solid sits between -0.324 and 0. On
        # paper that is a 324 mm interference with the base of the stage 1
        # tank; in the geometry there is none. The CAD was right and the report
        # invented an overlap.
        add("nozzle", "revolve", noz, nozzle.length_m * scale,
            nozzle.exit_radius_m * scale,
            station=-nozzle.length_m * scale)

    # stages, aft to forward, each narrower than the one below it
    radii = [r_base * (taper_per_stage ** i) for i in range(len(stages))]
    for i, st in enumerate(stages):
        r = radii[i]
        volume_m3 = float(st.prop_mass_kg) / 1020.0
        length = max(0.2, volume_m3 / (math.pi * (r / scale) ** 2)) * scale

        tube = b.cylinder(r, length)
        tube = b.boolean_cut(tube, b.cylinder(r - wall, length * 1.05))
        tube = b.translate(tube, 0.0, 0.0, z + length / 2.0)
        add(f"stage {i+1} tank", "shell", tube, length, r)
        z += length

        if i + 1 < len(stages):
            r_next = radii[i + 1]
            t_len = interstage_length(r, r_next)
            sections = transition_sections(r / scale, r_next / scale,
                                           t_len / scale, points=CIRCLE_POINTS)
            solid = b.loft_sections([(zz + z, ring) for zz, ring in sections])
            # An interstage is a skin, not a billet. Left solid it came out at
            # 114 kg against the 43 kg of the tank below it, which is absurd
            # for a shorter part of the same diameter.
            inner = b.loft_sections([
                (zz + z, [(x * (1 - wall / max(r, r_next)),
                           y * (1 - wall / max(r, r_next))) for x, y in ring])
                for zz, ring in sections])
            solid = b.boolean_cut(solid, inner)
            add(f"interstage {i+1}/{i+2}", "loft", solid, t_len, max(r, r_next))
            z += t_len

    # nose cone on top, at the topmost stage's radius.
    #
    # The shape is an argument because it was hardcoded to "ogive" while the
    # design loop had already begun choosing between shapes on measured drag.
    # The vehicle was being sized for a von Karman nose -- 59.9 kg lighter --
    # and then built with an ogive, so the mass and drag numbers in the packet
    # described a body the exported STEP was not.
    shape = _geometry_shape_name(nose_shape)
    r_top = radii[-1]
    nose_len = 2.0 * r_top * float(nose_fineness)
    prof = centred(nose_profile(r_top, nose_len, shape), nose_len)
    nose = b.revolve_profile(prof)
    # Hollow it the same way the coupon geometry does: the same curve one wall
    # in, pushed below the base so the aft end opens. Left solid, the nose came
    # out at 339 kg -- more than the rest of the vehicle put together, for the
    # lightest-loaded part on it.
    if r_top - wall > 1.0 and nose_len - wall > wall:
        inner = centred(nose_profile(r_top - wall, nose_len - wall, shape),
                        nose_len - wall)
        pad = max(1.0, min(4.0, 0.05 * r_top))
        cut = b.revolve_profile(inner)
        cut = b.translate(cut, 0.0, 0.0,
                          -nose_len / 2.0 - pad + (nose_len - wall) / 2.0)
        nose = b.boolean_cut(nose, cut)
    nose = b.translate(nose, 0.0, 0.0, z + nose_len / 2.0)
    add("nose cone", "revolve", nose, nose_len, r_top)
    z += nose_len

    # fins at the aft end, if the stability solve sized any
    if fin_span_m > 0.0 and fin_root_chord_m > 0.0:
        from cadflow.profiles import fin_planform

        span = float(fin_span_m) * scale
        chord = float(fin_root_chord_m) * scale
        thickness = max(3.0, 0.02 * span)
        for k in range(int(n_fins)):
            planform = fin_planform(span, chord)
            fin = b.extrude_profile(planform, thickness)
            fin = b.rotate(fin, "x", 90.0)
            fin = b.rotate(fin, "z", 360.0 * k / max(1, int(n_fins)))
            angle = 2 * math.pi * k / max(1, int(n_fins))
            fin = b.translate(fin,
                              (r_base - wall / 2) * math.cos(angle),
                              (r_base - wall / 2) * math.sin(angle),
                              chord * 0.25)
            add(f"fin {k+1}", "extrude", fin, chord, r_base + span,
                station=chord * 0.25)

    asm.total_length_mm = z
    return asm


def fuse_assembly(assembly: VehicleAssembly, backend=None):
    """Boolean-union every part into a single solid.

    Slow and occasionally fragile on thin shells that only just touch, which is
    why the assembly keeps its parts separately and fuses on demand: mass
    properties, export and inspection all work on the parts, and only a single
    watertight body needs the fuse.
    """
    from cadflow.backends import get_backend

    b = backend or get_backend(prefer_real=True)
    solids = [p.solid for p in assembly.parts if p.solid is not None]
    if not solids:
        raise ValueError("assembly has no solids to fuse")
    out = solids[0]
    for s in solids[1:]:
        out = b.boolean_union(out, s)
    return out


def export_assembly(assembly: VehicleAssembly, out_dir, backend=None,
                    fuse: bool = False) -> dict:
    """Write the assembly to STEP and STL.

    Parts are exported individually as well as fused, because a manufacturer
    wants the pieces and a viewer wants the whole thing.
    """
    from pathlib import Path

    from cadflow.backends import get_backend

    b = backend or get_backend(prefer_real=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for part in assembly.parts:
        if part.solid is None:
            continue
        stem = part.name.replace(" ", "_").replace("/", "-")
        try:
            written[stem] = str(b.export_step(part.solid, out / f"{stem}.step"))
        except Exception:  # noqa: BLE001
            pass
    if fuse:
        try:
            written["assembly"] = str(
                b.export_step(fuse_assembly(assembly, b), out / "assembly.step"))
        except Exception as exc:  # noqa: BLE001
            written["assembly_error"] = str(exc)
    return written


def mass_closure(assembly: VehicleAssembly, budget_kg: float,
                 liftoff_thrust_n: float = 0.0,
                 engine_twr: float = 60.0,
                 pressurisation_kg: float = 0.0) -> dict:
    """Does the mass budget have room for the geometry plus its engine?

    An independent check on the structural coefficient, arriving from the
    opposite direction. The fixed-point solve says the coefficient should be
    0.25 where the design asserts 0.14; this says the same thing from geometry,
    which knows nothing about that solve: the skin that was actually drawn plus
    an engine sized from the thrust already exceeds the allowance.

    For 25 kg to 4,000 km the drawn skin is 97.4 kg, an engine at T/W 60 is
    85.3 kg, and the budget is 155.6 kg -- so the vehicle is 27 kg short of
    being able to contain itself. Two independent routes agreeing that a number
    is optimistic is worth more than either alone.

    ``pressurisation_kg`` is the helium and its bottle, from
    ``cadflow.pressurization``. It belongs here rather than in a second budget
    of its own: a pump-fed stage cannot start without it, so it is structure in
    every sense that matters to a mass statement, and this is the one place that
    asks whether the vehicle can contain itself. Leaving it out let the packet
    report the shortfall in prose while the closure arithmetic beside it
    disagreed -- which is the same shape as every other defect this project has
    found, two parts each knowing and neither telling the other.
    """
    skin = assembly.mass_kg
    engine = (float(liftoff_thrust_n) / (9.80665 * float(engine_twr))
              if liftoff_thrust_n else 0.0)
    press = float(pressurisation_kg)
    accounted = skin + engine + press
    return {
        "skin_kg": skin,
        "engine_kg": engine,
        "pressurisation_kg": press,
        "accounted_kg": accounted,
        "budget_kg": float(budget_kg),
        "slack_kg": float(budget_kg) - accounted,
        "closes": accounted <= float(budget_kg),
        "parts": len(assembly.parts),
    }
