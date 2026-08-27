"""Propellant slosh, and whether it lands on top of anything that matters.

The bending-mode section of the design packet says slosh is not modelled and
lives in the same frequency range. This is that gap.

Liquid in a partly full tank has its own lateral modes. They carry a real
fraction of the vehicle's mass -- most of it, in a shallow tank -- and they are
very lightly damped, so a mode that coincides with a structural bending
frequency or with control bandwidth couples, and the coupling is what destroyed
several early launch vehicles. It is not a small effect nor an exotic one; it is
a standard design check that this program did not have.

The frequency depends on axial acceleration, not on gravity, and that
distinction matters more than it looks. A vehicle under 4.5 g has slosh
frequencies roughly twice what a ground calculation at 1 g would give, so using
9.81 here would put the modes in the wrong place by a factor of two and could
report clearance from a bending mode that in flight sits right on top of it.

The eigenvalue is computed rather than quoted. 1.8412 appears in every
reference, but this project has already been burned by a remembered constant, so
it is obtained as the first zero of J1' at import and checked against the
textbook value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

#: Standard gravity, only for reporting a ground-test comparison.
G0 = 9.80665


@lru_cache(maxsize=1)
def first_eigenvalue() -> float:
    """First zero of J1'(x), the eigenvalue of the first lateral slosh mode.

    Solved rather than written down. The textbook value 1.8412 is correct, and
    checking it costs one root find.
    """
    from scipy.optimize import brentq
    from scipy.special import jvp

    root = brentq(lambda x: jvp(1, x), 1.0, 3.0, xtol=1e-13)
    if abs(root - 1.8412) > 1e-3:
        raise RuntimeError(
            f"first zero of J1' solved as {root}, which disagrees with the "
            f"textbook 1.8412; the special-function library is not behaving")
    return float(root)


@dataclass
class SloshMode:
    tank: str
    frequency_hz: float
    slosh_mass_kg: float
    slosh_mass_fraction: float
    fill_ratio: float
    axial_accel_m_s2: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "tank": self.tank,
            "frequency_hz": round(self.frequency_hz, 4),
            "slosh_mass_kg": round(self.slosh_mass_kg, 2),
            "slosh_mass_fraction": round(self.slosh_mass_fraction, 4),
            "fill_ratio": round(self.fill_ratio, 3),
            "axial_accel_m_s2": round(self.axial_accel_m_s2, 2),
            "notes": list(self.notes),
        }


def frequency_hz(radius_m: float, fill_depth_m: float,
                 axial_accel_m_s2: float) -> float:
    """First lateral slosh frequency in a cylindrical tank.

    w^2 = (lambda g_eff / R) tanh(lambda h / R)

    ``axial_accel_m_s2`` is the acceleration the propellant actually feels,
    which under thrust is several g. Passing 9.81 would be a ground-test
    number and would place every mode a factor of two too low.
    """
    r, h = float(radius_m), float(fill_depth_m)
    a = float(axial_accel_m_s2)
    if r <= 0:
        raise ValueError("tank radius must be positive")
    if h <= 0:
        raise ValueError("fill depth must be positive; an empty tank has no mode")
    if a <= 0:
        raise ValueError(
            "axial acceleration must be positive; in free fall the propellant "
            "does not settle and this model does not apply")
    lam = first_eigenvalue()
    omega_sq = (lam * a / r) * math.tanh(lam * h / r)
    return math.sqrt(omega_sq) / (2.0 * math.pi)


def mass_fraction(radius_m: float, fill_depth_m: float) -> float:
    """Fraction of the propellant that participates in the first slosh mode.

    From the equivalent-mechanical-model result for a cylindrical tank
    (Abramson, NASA SP-106):

        m1/m = 2 tanh(lam h/R) / ((lam h/R)(lam^2 - 1))

    Both limits are physical and are asserted in the tests rather than assumed:
    a shallow pan sloshes almost entirely, tending to 2/(lam^2 - 1) = 0.837,
    while in a deep tank only a surface layer moves and the fraction falls off
    as R/h.
    """
    r, h = float(radius_m), float(fill_depth_m)
    if r <= 0 or h <= 0:
        raise ValueError("radius and fill depth must be positive")
    lam = first_eigenvalue()
    x = lam * h / r
    return 2.0 * math.tanh(x) / (x * (lam * lam - 1.0))


def tank_mode(name: str, *, radius_m: float, propellant_kg: float,
              bulk_density: float, fill_ratio: float,
              axial_accel_m_s2: float) -> SloshMode:
    """Slosh mode of one tank at a given fill state.

    ``fill_ratio`` is how full the tank is now; the tank itself is sized for the
    full load, since it does not shrink as it drains. Slosh is worst part way
    through a burn, not at liftoff, which is why the fill state is an argument
    rather than assumed full.
    """
    if not 0.0 < fill_ratio <= 1.0:
        raise ValueError("fill ratio must be within (0, 1]")
    r = float(radius_m)
    full_volume = float(propellant_kg) / float(bulk_density)
    full_depth = full_volume / (math.pi * r * r)
    depth = full_depth * float(fill_ratio)
    f = frequency_hz(r, depth, axial_accel_m_s2)
    frac = mass_fraction(r, depth)
    liquid_now = float(propellant_kg) * float(fill_ratio)

    notes: list[str] = []
    if depth < 0.25 * r:
        notes.append(
            f"fill depth {depth:.2f} m is under a quarter of the tank radius; "
            f"a shallow layer slings most of its mass and the equivalent model "
            f"is least reliable here")
    return SloshMode(
        tank=name, frequency_hz=f, slosh_mass_kg=liquid_now * frac,
        slosh_mass_fraction=frac, fill_ratio=float(fill_ratio),
        axial_accel_m_s2=float(axial_accel_m_s2), notes=tuple(notes))


def separation_from(mode: SloshMode, structural_hz: float) -> dict:
    """How close a slosh mode sits to a structural frequency.

    Slosh is very lightly damped -- a fraction of a percent without baffles --
    so proximity is the whole question. A ratio near one is a coupling risk
    whatever the amplitudes look like in isolation.
    """
    if structural_hz <= 0:
        raise ValueError("structural frequency must be positive")
    ratio = mode.frequency_hz / structural_hz
    if 0.8 <= ratio <= 1.25:
        verdict = ("coincident: the slosh mode sits within 25% of the "
                   "structural mode and the two will couple")
    elif 0.5 <= ratio <= 2.0:
        verdict = ("close: within an octave, which needs a coupled analysis "
                   "rather than a frequency comparison")
    else:
        verdict = "separated by more than an octave"
    return {"ratio": round(ratio, 3), "verdict": verdict,
            "coupled": ratio >= 0.5 and ratio <= 2.0}
