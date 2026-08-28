"""Why no large launch vehicle is a monocoque, and what this project loses by being one.

A payload sweep at fixed apogee (``artifacts/verification/payload_scaling.json``)
found something this project had not been able to see. The solved structural
coefficient does not improve as the vehicle grows -- it drifts slightly *worse*,
0.2577 at 1.9 t gross to 0.2818 at 131 t -- while flown stages span 0.036 to
0.118. A 131-tonne vehicle at 0.28 is not a small-vehicle artefact and it is not
a bad mission. It is the wall.

The monocoque scaling is easy to see once written down. A cylinder in axial
compression buckles at sigma_cr = gamma 0.605 E t / r and carries
sigma = P / (2 pi r t), and setting those equal gives

    t = sqrt( P / (3.80 gamma E) )

with no radius in it at all. Thickness is set by the load alone, so the mass,
rho 2 pi r t L, grows *linearly with radius* for the same load. Making a
monocoque vehicle bigger costs structural efficiency, always. That is the
opposite of what stiffened structure does, and it is why every flown launch
vehicle above a few tonnes is stringer-stiffened, isogrid, or sandwich.

What this module does
---------------------
Sizes a stringer-stiffened cylinder and reports both buckling modes it must
survive, against the same load the monocoque sizing sees:

  * **General instability**, the whole shell going unstable. The classical
    result for an axially compressed cylinder is

        N_cr = 2 sqrt(D C) / r

    where C is extensional stiffness per unit width and D bending stiffness per
    unit width. This is not an approximation bolted on for stiffened shells: put
    the isotropic values C = E t and D = E t^3 / (12 (1 - nu^2)) into it and it
    returns 0.607 E t / r, the classical monocoque stress. That reduction is
    asserted in the tests, so the general form cannot drift from the special
    case it must contain.

  * **Local skin buckling**, the panel between two stringers going first. A flat
    plate simply supported on four edges buckles at
    k pi^2 E / (12 (1 - nu^2)) (t/b)^2 with k = 4 for a long panel. A stiffened
    shell whose skin buckles between stringers has not gained what the stringers
    promised, so both modes are reported and the lower governs.

What it does not model, and says so: stringer crippling, the local collapse of
the stiffener's own flange, which for deep thin blades can govern before either
mode here. A blade slenderness beyond what this checks is flagged rather than
silently sized through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Poisson's ratio for the metals this project selects.
POISSON = 0.3

#: Plate buckling coefficient for a long panel simply supported on four edges.
#:
#: 4.0 is the classical minimum of the k(a/b) curve, reached for aspect ratios
#: above about 3 and never far above it after. Skin bays between stringers on a
#: launch vehicle are long and narrow, so this is the right end of the curve --
#: and it is the conservative end, since clamped edges would give 6.97.
PLATE_K = 4.0

#: Blade height-to-thickness ratio beyond which crippling is not covered.
#:
#: A deep thin blade collapses locally before the panel or the shell does, and
#: this module has no crippling model. Rather than size through that silently,
#: anything past this is reported. 12 is a common design limit for unflanged
#: blade stiffeners in aluminium and steel alike.
MAX_BLADE_SLENDERNESS = 12.0


@dataclass
class StiffenedWall:
    """A stringer-stiffened cylinder wall, and what it takes to buckle it."""

    skin_m: float
    stringer_count: int
    stringer_height_m: float
    stringer_thickness_m: float
    radius_m: float
    #: Smeared extensional thickness, t_skin + A_stringer / spacing
    t_extensional_m: float
    #: Bending stiffness per unit width, N m
    bending_stiffness_nm: float
    spacing_m: float
    applied_mpa: float
    general_mpa: float
    local_mpa: float
    mass_per_area_kg_m2: float
    notes: list[str] = field(default_factory=list)

    @property
    def allowable_mpa(self) -> float:
        """The lower of the two modes. A shell is as strong as its worst."""
        return min(self.general_mpa, self.local_mpa)

    @property
    def governs(self) -> str:
        return "local skin" if self.local_mpa < self.general_mpa else "general"

    @property
    def margin(self) -> float:
        return self.allowable_mpa / max(self.applied_mpa, 1e-9)

    @property
    def passes(self) -> bool:
        return self.margin >= 1.0

    def as_dict(self) -> dict:
        return {"skin_m": self.skin_m, "stringer_count": self.stringer_count,
                "stringer_height_m": self.stringer_height_m,
                "stringer_thickness_m": self.stringer_thickness_m,
                "radius_m": self.radius_m,
                "t_extensional_m": self.t_extensional_m,
                "bending_stiffness_nm": self.bending_stiffness_nm,
                "spacing_m": self.spacing_m, "applied_mpa": self.applied_mpa,
                "general_mpa": self.general_mpa, "local_mpa": self.local_mpa,
                "allowable_mpa": self.allowable_mpa, "governs": self.governs,
                "margin": self.margin, "passes": self.passes,
                "mass_per_area_kg_m2": self.mass_per_area_kg_m2,
                "notes": list(self.notes)}


def section_properties(skin_m: float, spacing_m: float, height_m: float,
                       thickness_m: float, poisson: float = POISSON) -> dict:
    """Extensional and bending stiffness per unit width of one skin-stringer bay.

    Areas and first moments about the skin mid-plane, then the parallel-axis
    shift to the combined neutral axis. All exact for the idealised section: a
    flat skin strip of width ``spacing_m`` with one rectangular blade standing
    on it.

    Returned per unit width so the results drop straight into the shell
    relations, which are written per unit width and do not care how the material
    is distributed as long as the stiffnesses are right.
    """
    if spacing_m <= 0 or skin_m <= 0:
        raise ValueError("skin thickness and stringer spacing must be positive")

    a_skin = skin_m * spacing_m
    a_str = height_m * thickness_m
    area = a_skin + a_str

    # Centroid measured from the skin mid-plane, outward positive.
    y_str = skin_m / 2.0 + height_m / 2.0
    y_bar = (a_str * y_str) / area if area > 0 else 0.0

    # Second moment about the combined neutral axis.
    i_skin = spacing_m * skin_m ** 3 / 12.0 + a_skin * y_bar ** 2
    i_str = (thickness_m * height_m ** 3 / 12.0
             + a_str * (y_str - y_bar) ** 2)
    inertia = i_skin + i_str

    return {
        "area_m2": area,
        "t_extensional_m": area / spacing_m,
        "neutral_axis_m": y_bar,
        "inertia_m4": inertia,
        # Per unit width, with the plate factor so the isotropic case reduces
        # exactly to E t^3 / (12 (1 - nu^2)).
        "inertia_per_width_m3": inertia / spacing_m,
        "poisson": poisson,
    }


def general_instability_mpa(*, bending_stiffness_nm: float,
                            extensional_stiffness_n_m: float, radius_m: float,
                            t_extensional_m: float, knockdown: float) -> float:
    """N_cr = 2 sqrt(D C) / r, converted to a stress on the smeared wall.

    The classical axially compressed cylinder result. Substituting the isotropic
    C = E t and D = E t^3 / (12 (1 - nu^2)) returns 0.607 E t / r, which is the
    monocoque stress this project already uses -- so the general form contains
    the special case rather than approximating it.
    """
    if radius_m <= 0 or t_extensional_m <= 0:
        raise ValueError("radius and smeared thickness must be positive")
    n_cr = 2.0 * math.sqrt(max(0.0, bending_stiffness_nm
                               * extensional_stiffness_n_m)) / radius_m
    return knockdown * n_cr / t_extensional_m / 1e6


def local_skin_buckling_mpa(*, skin_m: float, spacing_m: float,
                            youngs_pa: float, poisson: float = POISSON,
                            k: float = PLATE_K) -> float:
    """The panel between two stringers, as a long simply supported plate.

    A stiffened shell whose skin buckles between its stringers has not gained
    what the stringers promised, so this is a real limit and not a detail.
    """
    if spacing_m <= 0 or skin_m <= 0:
        raise ValueError("skin thickness and spacing must be positive")
    return (k * math.pi ** 2 * youngs_pa / (12.0 * (1.0 - poisson ** 2))
            * (skin_m / spacing_m) ** 2) / 1e6


def analyse(*, axial_load_n: float, radius_m: float, skin_m: float,
            stringer_count: int, stringer_height_m: float,
            stringer_thickness_m: float, youngs_pa: float,
            density_kg_m3: float, poisson: float = POISSON,
            knockdown: float | None = None) -> StiffenedWall:
    """Both buckling modes for one stiffened cylinder under axial compression."""
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    n = max(0, int(stringer_count))
    spacing = 2.0 * math.pi * radius_m / n if n > 0 else 2.0 * math.pi * radius_m

    sec = section_properties(skin_m, spacing, stringer_height_m,
                             stringer_thickness_m, poisson)
    t_ext = sec["t_extensional_m"]

    # Applied stress on the smeared wall: the load is shared by skin and
    # stringers in proportion to area, which is what the smeared thickness is.
    applied = abs(axial_load_n) / (2.0 * math.pi * radius_m * t_ext) / 1e6

    if knockdown is None:
        # The same SP-8007 correlation the monocoque path uses, on the smeared
        # thickness. Reaching for a different knockdown here would make the two
        # structures incomparable, which is the one thing this module exists to
        # do.
        from cadflow.shell_buckling import knockdown_compression

        knockdown = knockdown_compression(radius_m, t_ext)

    d = youngs_pa * sec["inertia_per_width_m3"] / (1.0 - poisson ** 2)
    c = youngs_pa * t_ext
    general = general_instability_mpa(
        bending_stiffness_nm=d, extensional_stiffness_n_m=c,
        radius_m=radius_m, t_extensional_m=t_ext, knockdown=knockdown)
    # An unstiffened cylinder has no panel to buckle between stringers it does
    # not have: its local mode *is* the shell mode, already counted above.
    # Applying the flat-plate formula across the full circumference returns a
    # stress near zero and would report every monocoque as failed.
    local = (local_skin_buckling_mpa(skin_m=skin_m, spacing_m=spacing,
                                     youngs_pa=youngs_pa, poisson=poisson)
             if n > 0 else float("inf"))

    notes = []
    if stringer_thickness_m > 0:
        slender = stringer_height_m / stringer_thickness_m
        if slender > MAX_BLADE_SLENDERNESS:
            notes.append(
                f"blade slenderness {slender:.0f} exceeds {MAX_BLADE_SLENDERNESS:.0f}: "
                f"stringer crippling is not modelled here and may govern before "
                f"either mode reported")
    if local < general:
        notes.append(
            f"the skin buckles between stringers at {local:.0f} MPa before the "
            f"shell goes unstable at {general:.0f} MPa -- the stringers are "
            f"carrying more than this skin can feed them, so closer spacing or "
            f"a thicker skin buys more than a taller blade")

    return StiffenedWall(
        skin_m=skin_m, stringer_count=n, stringer_height_m=stringer_height_m,
        stringer_thickness_m=stringer_thickness_m, radius_m=radius_m,
        t_extensional_m=t_ext, bending_stiffness_nm=d, spacing_m=spacing,
        applied_mpa=applied, general_mpa=general, local_mpa=local,
        mass_per_area_kg_m2=density_kg_m3 * t_ext, notes=notes)


def monocoque_thickness_m(axial_load_n: float, youngs_pa: float,
                          knockdown: float = 0.35) -> float:
    """t = sqrt(P / (3.80 gamma E)) -- the radius-free monocoque result.

    Worth its own function because the absence of radius is the whole point.
    Thickness follows from the load alone, so monocoque mass grows linearly with
    radius for the same load, and a bigger vehicle is structurally worse.
    """
    if youngs_pa <= 0 or knockdown <= 0:
        raise ValueError("modulus and knockdown must be positive")
    return math.sqrt(abs(axial_load_n) / (3.80 * knockdown * youngs_pa))


def compare_to_monocoque(wall: StiffenedWall, *, youngs_pa: float,
                         poisson: float = POISSON) -> dict:
    """At the same wall mass, which carries more?

    The comparison has to be at equal mass, not equal applied stress. Sizing a
    monocoque to whatever stress the stiffened wall happens to be carrying
    compares an over-strong wall against a thin one and reports the stiffening
    as a mass penalty -- which was the first version of this function, and it
    said stiffeners made the vehicle 62% heavier.

    Equal mass means equal smeared thickness, since both are the same alloy. Put
    that same material into a plain cylinder and ask what it buckles at. The
    ratio is the whole argument for stiffened structure: the stringers move
    material away from the mid-surface, which raises bending stiffness as the
    cube of the offset while extensional stiffness is unchanged.
    """
    from cadflow.shell_buckling import knockdown_compression

    t = wall.t_extensional_m
    gamma = knockdown_compression(wall.radius_m, t)
    mono_mpa = gamma * 0.605 * youngs_pa * t / wall.radius_m / 1e6
    return {
        "equal_mass_thickness_m": t,
        "monocoque_allowable_mpa": mono_mpa,
        "stiffened_allowable_mpa": wall.allowable_mpa,
        "capability_ratio": (wall.allowable_mpa / mono_mpa
                             if mono_mpa > 0 else float("inf")),
        "governs": wall.governs,
    }


def size_for_stress(*, required_mpa: float, radius_m: float, youngs_pa: float,
                    density_kg_m3: float, material_allowable_mpa: float,
                    poisson: float = POISSON) -> dict:
    """The lightest stiffened wall that carries ``required_mpa``, and its limit.

    Searched rather than solved, over stringer count, skin gauge and blade
    proportions, keeping the lightest configuration whose worse buckling mode
    clears the requirement.

    The search is bounded by things this module can defend, not by the edge of
    a grid. Left to run, the optimum walks to ever-finer stringer spacing --
    7 mm, 4 mm, whatever the sweep allows -- because within this physics closer
    stringers always raise local buckling and nothing here pushes back. Two
    things push back in reality and only one of them is modelled:

      * **Material strength.** A buckling allowable above the alloy's design
        allowable is not reachable: the wall yields first. This is the real
        ceiling and it is enforced, so the answer is capped at something
        physical.
      * **Stringer crippling and manufacturability.** Not modelled. Blade
        slenderness is checked against MAX_BLADE_SLENDERNESS and the result says
        when a configuration is trusting geometry this module cannot verify.

    So the useful claim is bounded and honest: stiffening moves a wall from
    buckling-limited to strength-limited, and past that point more stiffening
    buys nothing.
    """
    target = min(float(required_mpa), float(material_allowable_mpa))
    best = None
    for n in (12, 16, 24, 32, 48, 64, 80, 96, 120):
        spacing = 2.0 * math.pi * radius_m / n
        for skin in (0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016):
            for height in (0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020):
                for th in (0.0006, 0.0008, 0.0010, 0.0012, 0.0016):
                    if height / th > MAX_BLADE_SLENDERNESS:
                        continue
                    w = analyse(axial_load_n=1.0, radius_m=radius_m,
                                skin_m=skin, stringer_count=n,
                                stringer_height_m=height,
                                stringer_thickness_m=th, youngs_pa=youngs_pa,
                                density_kg_m3=density_kg_m3, poisson=poisson)
                    if w.allowable_mpa < target:
                        continue
                    if best is None or (w.mass_per_area_kg_m2
                                        < best.mass_per_area_kg_m2):
                        best = w
    if best is None:
        return {"found": False,
                "note": (f"no stiffened wall in the searched range reaches "
                         f"{target:.0f} MPa at {1000*radius_m:.0f} mm radius")}

    from cadflow.shell_buckling import knockdown_compression

    t = best.t_extensional_m
    mono = (knockdown_compression(radius_m, t) * 0.605 * youngs_pa * t
            / radius_m / 1e6)
    return {
        "found": True,
        "wall": best,
        "target_mpa": target,
        "strength_capped": float(required_mpa) > float(material_allowable_mpa),
        "monocoque_at_same_mass_mpa": mono,
        "capability_ratio": best.allowable_mpa / mono if mono > 0 else float("inf"),
        "note": (
            f"{best.stringer_count} stringers at {1000*best.spacing_m:.0f} mm, "
            f"skin {1000*best.skin_m:.2f} mm, blade "
            f"{1000*best.stringer_height_m:.0f}x{1000*best.stringer_thickness_m:.1f} mm, "
            f"{best.mass_per_area_kg_m2:.2f} kg/m2. The same material as a plain "
            f"cylinder buckles at {mono:.0f} MPa, so the stiffening is worth "
            f"{best.allowable_mpa / mono:.1f}x at equal mass. Crippling is not "
            f"modelled; blade slenderness is held under "
            f"{MAX_BLADE_SLENDERNESS:.0f} rather than verified"),
    }
