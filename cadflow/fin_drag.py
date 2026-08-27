"""Drag of the fins, which the trajectory has been flying without.

The vehicle's drag coefficient is a function of nose shape and fineness. That
describes a body of revolution. It does not describe the vehicle this program
actually builds, which carries four fins whose combined planform is three and a
half times the body's frontal area.

Those fins exist because the stability model needs them, and they are sized by
the static margin the design targets -- so the same loop that adds them has
never charged the trajectory for them. The apogee reported in every packet so
far is the apogee of a finless rocket.

Two terms matter and both are computable from the fin geometry already in the
packet. Skin friction acts on the wetted area, which is twice the planform and
therefore large. Wave drag appears above Mach 1 and depends on how thick the
section is relative to its chord; it is what makes a thick fin expensive to fly
supersonically, and a launch vehicle spends most of its dynamic pressure there.

Interference drag at the fin root is not modelled. It is real and positive, so
what follows is a floor rather than an estimate, and the packet says so instead
of implying the number is complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Thickness-to-chord ratio of the fin section. The fins this project builds are
#: flat plates of a given thickness, so this is derived per design rather than
#: assumed, but a value is needed when chord is unavailable.
DEFAULT_THICKNESS_RATIO = 0.06

#: Turbulent flat-plate skin friction at the Reynolds numbers a launch vehicle
#: sees through max-Q. Varies by roughly a factor of two across the ascent; a
#: single representative value is used and its crudeness is reported rather
#: than hidden behind a correlation that would imply more precision than the
#: rest of this estimate has.
REPRESENTATIVE_CF = 0.003


@dataclass
class FinDrag:
    cd_friction: float
    cd_wave: float
    cd_total: float
    planform_m2: float
    wetted_m2: float
    reference_m2: float
    mach: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "cd_friction": round(self.cd_friction, 5),
            "cd_wave": round(self.cd_wave, 5),
            "cd_total": round(self.cd_total, 5),
            "planform_m2": round(self.planform_m2, 4),
            "wetted_m2": round(self.wetted_m2, 4),
            "reference_m2": round(self.reference_m2, 4),
            "mach": self.mach,
            "notes": list(self.notes),
        }


def planform_area_m2(*, span_m: float, root_chord_m: float,
                     tip_chord_m: float, n_fins: int) -> float:
    """Total planform area of the fin set, one side."""
    if span_m <= 0 or root_chord_m <= 0 or n_fins < 1:
        raise ValueError("span, chord and fin count must be positive")
    return n_fins * 0.5 * (float(root_chord_m) + float(tip_chord_m)) * float(span_m)


def wave_drag_coefficient(mach: float, thickness_ratio: float) -> float:
    """Zero-lift wave drag of a thin double-wedge section, on planform area.

        Cd = 4 (t/c)^2 / sqrt(M^2 - 1)

    Zero below Mach 1, where there is no wave drag to have. The singularity at
    exactly Mach 1 is a known failure of linearised theory rather than a
    physical infinity, so the transonic region is clamped: the formula is not
    valid there and returning a huge number would be worse than returning a
    large one.
    """
    m = float(mach)
    if m <= 1.0:
        return 0.0
    beta = math.sqrt(max(m * m - 1.0, 0.04))     # clamp: |M-1| < ~0.02
    return 4.0 * float(thickness_ratio) ** 2 / beta


def fin_drag(*, span_m: float, root_chord_m: float, tip_chord_m: float,
             thickness_m: float, n_fins: int, body_radius_m: float,
             mach: float) -> FinDrag:
    """Fin drag coefficient referenced to body frontal area.

    Referenced to the body so it adds directly to the vehicle's Cd, which is
    what the trajectory integrates. Reporting it on fin area instead would be
    the more natural aerodynamic convention and the easier one to add to the
    wrong denominator.
    """
    r = float(body_radius_m)
    if r <= 0:
        raise ValueError("body radius must be positive")
    ref = math.pi * r * r
    planform = planform_area_m2(span_m=span_m, root_chord_m=root_chord_m,
                                tip_chord_m=tip_chord_m, n_fins=n_fins)
    wetted = 2.0 * planform
    mean_chord = 0.5 * (float(root_chord_m) + float(tip_chord_m))
    t_over_c = (float(thickness_m) / mean_chord if mean_chord > 0
                else DEFAULT_THICKNESS_RATIO)

    cd_f = REPRESENTATIVE_CF * wetted / ref
    cd_w = wave_drag_coefficient(mach, t_over_c) * planform / ref

    notes = [
        "Interference drag where the fins meet the body is not included. It is "
        "real and positive, so this is a floor on fin drag rather than an "
        "estimate of it.",
        f"Skin friction uses a single representative coefficient of "
        f"{REPRESENTATIVE_CF}; the true value varies by about a factor of two "
        f"across an ascent.",
    ]
    if t_over_c > 0.12:
        notes.append(
            f"Section is {100*t_over_c:.0f}% thick, beyond where thin-airfoil "
            f"wave drag theory applies; the wave term is understated.")
    return FinDrag(cd_friction=cd_f, cd_wave=cd_w, cd_total=cd_f + cd_w,
                   planform_m2=planform, wetted_m2=wetted, reference_m2=ref,
                   mach=float(mach), notes=tuple(notes))
