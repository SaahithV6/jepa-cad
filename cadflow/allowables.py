"""Design allowables and factors of safety for the structural gate.

Until now the structural gate was the module constant ``ALLOWABLE_MPA = 200.0``
in ``plan_and_verify``. One number, no derivation, no material attached, no
factor of safety written down anywhere. Two things follow from that, and both
are wrong in ways that matter.

The first is that the gate did not know what the part was made of. The
verification FEA ran every component in Al 6061 at E = 70 GPa and checked it
against 200 MPa, while ``autodesign`` was free to select Inconel 718 for the
skin. The vehicle then paid Inconel's density in the mass budget -- 8,190
against 2,700 kg/m3 -- while the gate never learned it could carry three times
the stress. The mass penalty was real and the strength benefit was discarded.

The second is that 276 MPa for 6061-T6, and every other number in the
catalogue, is a *typical* handbook value: the middle of a distribution, which
roughly half of all real coupons fall below. Design allowables are lower
tolerance bounds on that distribution -- A-basis is the value 99% of the
population exceeds with 95% confidence, B-basis 90% with 95%. Those require the
underlying population statistics. This project does not have them, so this
module does not pretend to: it refuses to report a statistical basis it cannot
support, and says so in the provenance it returns.

What it does instead is apply an explicit, cited factor of safety to a value it
labels honestly as non-statistical. That is weaker than a certifiable
allowable and stronger than an undocumented 200.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Yield factor of safety for uncrewed launch vehicle primary structure,
#: NASA-STD-5001B Table 1 (structures qualified by test). Ultimate is 1.4 over
#: the same limit load; this module gates on yield because the solver reports
#: von Mises against a yield strength, not a rupture criterion.
FOS_YIELD = 1.25
FOS_ULTIMATE = 1.4

#: Additional knockdown applied because catalogue strengths are typical values
#: rather than statistical allowables. This is a placeholder for real A- or
#: B-basis data and is deliberately not called one: it is a judgement, not a
#: tolerance bound, and it is applied so that a typical value is never used
#: directly as though it were an allowable. Removing it without substituting
#: sourced allowables would make every margin in this project optimistic.
TYPICAL_TO_DESIGN_KNOCKDOWN = 0.85

#: Below this the catalogue value is treated as room-temperature data.
REFERENCE_TEMP_K = 295.0


@dataclass(frozen=True)
class Allowable:
    """A design allowable with everything needed to argue about it."""

    material_id: str
    allowable_mpa: float
    source_strength_mpa: float
    strength_basis: str
    factor_of_safety: float
    knockdown: float
    temperature_k: float
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def certifiable(self) -> bool:
        """Whether this allowable could support a flight qualification case.

        Always False today, and it should stay a computed property rather than
        a constant so that supplying real basis data is what changes it.
        """
        return self.strength_basis in ("A-basis", "B-basis")

    def as_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "allowable_mpa": round(self.allowable_mpa, 2),
            "source_strength_mpa": self.source_strength_mpa,
            "strength_basis": self.strength_basis,
            "factor_of_safety": self.factor_of_safety,
            "knockdown": self.knockdown,
            "temperature_k": self.temperature_k,
            "certifiable": self.certifiable,
            "caveats": list(self.caveats),
        }


def design_allowable(material_id: str, *, temperature_k: float = REFERENCE_TEMP_K,
                     fos: float = FOS_YIELD) -> Allowable:
    """Design allowable for ``material_id``, with its provenance attached.

    Raises rather than falling back to a default when the material has no yield
    strength. A silent default is how the fixed 200 MPa survived as long as it
    did: it applied to everything, so nothing ever looked wrong.
    """
    from cadflow.space_materials import iter_materials

    mat = next((m for m in iter_materials() if m.material_id == material_id), None)
    if mat is None:
        raise KeyError(f"unknown material {material_id!r}")
    if not mat.yield_mpa:
        raise ValueError(
            f"{material_id} carries no yield strength; it cannot be sized against")

    caveats = [
        "Catalogue strength is a typical value, not an A- or B-basis allowable. "
        "A knockdown stands in for the statistical basis and is a judgement, "
        "not a tolerance bound.",
    ]

    # No material in the catalogue carries a yield-versus-temperature curve, so
    # the only defensible thing to do above room temperature is to say that the
    # number is being used outside the conditions it was measured under. The
    # material gate elsewhere keeps service temperature under
    # max_service_temp_k, which bounds how wrong this can be but does not make
    # it right -- 6061-T6 has lost most of its strength well before its service
    # limit, and a room-temperature allowable would not show that.
    if temperature_k > REFERENCE_TEMP_K + 25.0:
        caveats.append(
            f"Allowable is a room-temperature value applied at {temperature_k:.0f} K. "
            f"No yield-versus-temperature data is available for this material, so "
            f"the real hot allowable is lower by an unquantified amount.")
    if mat.max_service_temp_k < temperature_k:
        caveats.append(
            f"Service temperature {temperature_k:.0f} K exceeds the catalogue "
            f"limit of {mat.max_service_temp_k:.0f} K for this material.")

    allowable = mat.yield_mpa * TYPICAL_TO_DESIGN_KNOCKDOWN / fos
    return Allowable(
        material_id=material_id,
        allowable_mpa=allowable,
        source_strength_mpa=float(mat.yield_mpa),
        strength_basis="typical (non-statistical)",
        factor_of_safety=fos,
        knockdown=TYPICAL_TO_DESIGN_KNOCKDOWN,
        temperature_k=temperature_k,
        caveats=tuple(caveats),
    )


def elastic_properties(material_id: str) -> tuple[float, float]:
    """Young's modulus in Pa and Poisson's ratio, for the FEA deck.

    The catalogue carries no Poisson's ratio, so 0.33 is used throughout. That
    is close for aluminium and titanium and about 0.30 for steels and nickel
    superalloys; the resulting stiffness error is a couple of percent and far
    below the discretisation error measured for C3D4, so it is not the thing to
    fix first. It is recorded here rather than buried as a literal in a deck
    writer so that it is visible when it does become the limiting term.
    """
    from cadflow.space_materials import iter_materials

    mat = next((m for m in iter_materials() if m.material_id == material_id), None)
    if mat is None:
        raise KeyError(f"unknown material {material_id!r}")
    if not mat.youngs_modulus_gpa:
        raise ValueError(f"{material_id} carries no Young's modulus")
    return float(mat.youngs_modulus_gpa) * 1e9, 0.33
