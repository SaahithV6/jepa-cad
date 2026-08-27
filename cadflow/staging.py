"""Stage separation: does the spent stage get out of the way in time?

The trajectory already stages. It drops the spent structure at burnout and
lights the next engine, and the mass bookkeeping is right. What it does not ask
is whether those two events can happen in that order without the two bodies
occupying the same space.

Separation is not instantaneous and it is not free. Springs or retro motors push
the stages apart at a metre or two per second, and the upper stage cannot ignite
until the spent one is clear of its plume -- typically one to two vehicle
diameters, because a plume expanding into near-vacuum spreads far wider than the
nozzle. Wait too long and the coast costs velocity and control authority; wait
too little and the exhaust hits a structure still attached by inches.

Two things drive the outcome and both are already in the design. Relative
velocity follows from the separation impulse divided by the two masses, and the
lighter the spent stage the faster it goes. And in atmosphere the spent stage
decelerates harder than the upper one, because it has the worse ballistic
coefficient, which helps -- an effect that vanishes entirely in vacuum, where
most separations happen.

What this does not model: tip-off, where an asymmetric push rotates the stages
into each other; plume impingement loads on the spent stage; and the structural
transient of the release itself. All three are real. This answers the narrower
question of whether the gap opens fast enough, which is the one that can be
answered from the numbers the design already has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Plume clearance, in vehicle diameters, before the upper stage may light. A
#: vacuum plume expands far beyond the nozzle exit, so this is not a nozzle
#: dimension.
PLUME_CLEARANCE_DIAMETERS = 1.5

#: Typical separation impulse per unit of separated mass, N s per kg. Spring
#: separation systems are sized to give a metre or two per second of relative
#: velocity; this is the low end of that.
DEFAULT_IMPULSE_PER_KG = 1.2


@dataclass
class Separation:
    stage_index: int
    spent_mass_kg: float
    upper_mass_kg: float
    relative_velocity_m_s: float
    coast_s: float
    clearance_m: float
    required_clearance_m: float
    clears: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "spent_mass_kg": round(self.spent_mass_kg, 1),
            "upper_mass_kg": round(self.upper_mass_kg, 1),
            "relative_velocity_m_s": round(self.relative_velocity_m_s, 3),
            "coast_s": round(self.coast_s, 2),
            "clearance_m": round(self.clearance_m, 3),
            "required_clearance_m": round(self.required_clearance_m, 3),
            "clears": self.clears,
            "notes": list(self.notes),
        }


def relative_velocity_m_s(impulse_n_s: float, spent_mass_kg: float,
                          upper_mass_kg: float) -> float:
    """Closing rate the separation system produces, from momentum.

    The impulse acts on both bodies, so the relative velocity uses the reduced
    mass: J (1/m_spent + 1/m_upper). Dividing by one mass alone -- the obvious
    mistake -- understates it, and understating separation velocity is the
    direction that reports a recontact which would not happen.
    """
    if spent_mass_kg <= 0 or upper_mass_kg <= 0:
        raise ValueError("both masses must be positive")
    if impulse_n_s <= 0:
        raise ValueError("separation impulse must be positive")
    return float(impulse_n_s) * (1.0 / float(spent_mass_kg)
                                 + 1.0 / float(upper_mass_kg))


def check_separation(*, stage_index: int, spent_mass_kg: float,
                     upper_mass_kg: float, body_diameter_m: float,
                     coast_s: float, impulse_n_s: float | None = None,
                     differential_decel_m_s2: float = 0.0) -> Separation:
    """Does the gap reach plume clearance before the upper stage lights?

    ``differential_decel_m_s2`` is how much harder the spent stage decelerates
    than the upper one, which in atmosphere adds to the gap quadratically. It
    defaults to zero because most separations happen high enough that there is
    no air to provide it, and assuming help that is not there is the wrong way
    to be wrong.
    """
    if body_diameter_m <= 0:
        raise ValueError("body diameter must be positive")
    if coast_s < 0:
        raise ValueError("coast time cannot be negative")

    impulse = (float(impulse_n_s) if impulse_n_s is not None
               else DEFAULT_IMPULSE_PER_KG * float(spent_mass_kg))
    v_rel = relative_velocity_m_s(impulse, spent_mass_kg, upper_mass_kg)
    gap = v_rel * float(coast_s) + 0.5 * float(differential_decel_m_s2) * coast_s ** 2
    required = PLUME_CLEARANCE_DIAMETERS * float(body_diameter_m)

    notes: list[str] = []
    if impulse_n_s is None:
        notes.append(
            f"Separation impulse assumed at {DEFAULT_IMPULSE_PER_KG} N s per kg "
            f"of spent stage, the low end of spring separation practice. A real "
            f"design specifies the hardware; this is a screen.")
    if differential_decel_m_s2 <= 0.0:
        notes.append(
            "No differential drag credited. In atmosphere the spent stage "
            "decelerates harder and the gap opens faster, but most separations "
            "are high enough that the help is not there.")
    if not (gap >= required):
        notes.append(
            f"The gap reaches {gap:.2f} m against {required:.2f} m needed. "
            f"Either the coast is too short, the separation system too weak, or "
            f"the upper stage must light later -- which costs velocity.")
    notes.append(
        "Tip-off, plume impingement on the spent stage, and the structural "
        "transient of release are not modelled. This answers only whether the "
        "gap opens fast enough.")

    return Separation(
        stage_index=stage_index, spent_mass_kg=float(spent_mass_kg),
        upper_mass_kg=float(upper_mass_kg), relative_velocity_m_s=v_rel,
        coast_s=float(coast_s), clearance_m=gap, required_clearance_m=required,
        clears=bool(gap >= required), notes=tuple(notes))


def coast_for_clearance_s(*, spent_mass_kg: float, upper_mass_kg: float,
                          body_diameter_m: float,
                          impulse_n_s: float | None = None) -> float:
    """Shortest coast that reaches plume clearance, in seconds.

    The design lever when a separation does not clear: how long the upper stage
    must wait. Reported so the cost is visible, since every second of coast is
    velocity the vehicle does not gain and attitude it holds on cold gas.
    """
    impulse = (float(impulse_n_s) if impulse_n_s is not None
               else DEFAULT_IMPULSE_PER_KG * float(spent_mass_kg))
    v_rel = relative_velocity_m_s(impulse, spent_mass_kg, upper_mass_kg)
    return PLUME_CLEARANCE_DIAMETERS * float(body_diameter_m) / v_rel
