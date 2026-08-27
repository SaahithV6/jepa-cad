"""Slosh baffles, and the damping the control bandwidth actually needs.

The control-authority check leaves one failure the fins cannot fix. Trading
static margin moved the vehicle's pitch mode away from the slosh mode and bought
gimbal authority, but the bandwidth window stayed shut: flying the vehicle needs
several hertz, and an undamped slosh mode a few hertz up caps bandwidth well
below that.

That cap is not a fact about slosh, it is a fact about *undamped* slosh. The
five-to-one separation the window check applies is the rule for a mode with
essentially no damping, which is what a bare cylindrical tank has -- a fraction
of one percent. Ring baffles raise it by an order of magnitude, and a mode with
real damping can be approached far more closely without being excited.

So the repair for a shut bandwidth window is not a different vehicle. It is
hardware in the tank, and the question this module answers is how much.

Two cautions travel with the numbers. The damping correlation is Miles'
empirical fit for ring baffles and carries real scatter -- it is a design
estimate, not a measurement, and slosh damping is normally confirmed by test
because it depends on amplitude in a way no closed form captures. And baffles
are not free: they add mass, they complicate the tank, and this module prices
the mass so the design loop cannot treat them as costless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Damping of the first lateral mode in a bare cylindrical tank. Small enough
#: that it is normally treated as zero.
BARE_TANK_DAMPING = 0.0005

#: Separation a control system needs from a flexible mode, as a function of how
#: damped that mode is. An undamped mode has to be avoided by a wide margin; a
#: well damped one can be approached. These are the endpoints of the usual
#: engineering range, interpolated between.
SEPARATION_UNDAMPED = 5.0
SEPARATION_WELL_DAMPED = 1.5
WELL_DAMPED_RATIO = 0.05


@dataclass
class BaffleDesign:
    n_baffles: int
    width_m: float
    width_ratio: float
    damping_ratio: float
    mass_kg: float
    achieved_separation: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "n_baffles": self.n_baffles,
            "width_m": round(self.width_m, 4),
            "width_ratio": round(self.width_ratio, 4),
            "damping_ratio": round(self.damping_ratio, 5),
            "mass_kg": round(self.mass_kg, 2),
            "achieved_separation": round(self.achieved_separation, 3),
            "notes": list(self.notes),
        }


def required_separation(damping_ratio: float) -> float:
    """How far control bandwidth must stay below a mode with this damping.

    Linear between the undamped and well-damped endpoints. The point is not the
    precise interpolation -- it is that the separation requirement is a function
    of damping at all. Treating it as a fixed factor of five is what makes a
    bandwidth window look impossible when it is merely un-baffled.
    """
    z = max(0.0, float(damping_ratio))
    if z >= WELL_DAMPED_RATIO:
        return SEPARATION_WELL_DAMPED
    frac = z / WELL_DAMPED_RATIO
    return SEPARATION_UNDAMPED + frac * (SEPARATION_WELL_DAMPED
                                         - SEPARATION_UNDAMPED)


def ring_baffle_damping(*, width_ratio: float, depth_ratio: float,
                        n_baffles: int = 1) -> float:
    """Damping ratio from ring baffles, by Miles' empirical correlation.

        zeta = 2.83 (w/R)^1.5 exp(-4.6 d/R)

    per baffle, with ``d`` the depth of the baffle below the liquid surface.
    The exponential is the important part physically: a baffle only works where
    the fluid is moving, and lateral slosh motion decays fast with depth, so a
    baffle a tank radius down contributes almost nothing.

    Multiple baffles are summed. That over-credits a stack of them, since they
    interact and the deeper ones sit in fluid the shallower ones have already
    quietened, so the result is capped well below unity and flagged.
    """
    if width_ratio <= 0 or n_baffles < 1:
        raise ValueError("baffle width and count must be positive")
    per = 2.83 * (float(width_ratio) ** 1.5) * math.exp(-4.6 * float(depth_ratio))
    return min(0.25, per * int(n_baffles))


def size_baffles(*, tank_radius_m: float, fill_depth_m: float,
                 slosh_hz: float, required_bandwidth_hz: float,
                 wall_density_kg_m3: float = 2700.0,
                 baffle_thickness_m: float = 0.0015,
                 max_width_ratio: float = 0.20) -> BaffleDesign | None:
    """Smallest ring baffles that open the control bandwidth window.

    Returns None when no baffle within the geometric limit provides enough
    damping. That is a real outcome -- it means the slosh mode is simply too
    close to the bandwidth the vehicle needs, and the answer is a notch filter
    or a different tank, not more baffle.
    """
    r = float(tank_radius_m)
    if r <= 0 or fill_depth_m <= 0:
        raise ValueError("tank radius and fill depth must be positive")
    if slosh_hz <= 0 or required_bandwidth_hz <= 0:
        raise ValueError("frequencies must be positive")

    # Two quite different failures both end in "no baffle works", and calling
    # them the same thing hides the one that matters.
    #
    # If the bandwidth the vehicle needs is *above* the slosh frequency, the
    # mode is not something the control system can stay below -- it sits inside
    # the band. Damping does not move a frequency, so no baffle of any size
    # changes that, and the answer is a notch filter at the slosh frequency or
    # a tank geometry that puts the mode somewhere else. Returning a bare None
    # here would read as "needs bigger baffles".
    if float(required_bandwidth_hz) >= float(slosh_hz):
        raise ValueError(
            f"the control bandwidth this vehicle needs ({required_bandwidth_hz:.2f} "
            f"Hz) is above its slosh frequency ({slosh_hz:.2f} Hz), so the mode "
            f"lies inside the control band rather than above it. Baffles add "
            f"damping but do not move the frequency, so no baffle closes this. "
            f"It needs a notch filter at the slosh frequency, a tank that "
            f"sloshes elsewhere, or an autopilot designed to fly through the "
            f"mode")

    # Baffles are placed near the surface, where the fluid actually moves. One
    # at a fifth of a radius down, the next a fifth deeper, and so on.
    for n in (1, 2, 3, 4):
        for step in range(1, 41):
            width_ratio = max_width_ratio * step / 40.0
            depths = [0.2 * (i + 1) for i in range(n)]
            zeta = sum(
                ring_baffle_damping(width_ratio=width_ratio, depth_ratio=d)
                for d in depths)
            zeta = min(0.25, zeta)
            sep = required_separation(zeta)
            if slosh_hz / sep < required_bandwidth_hz:
                continue          # still shuts the window
            width = width_ratio * r
            # Annular rings of the given width and thickness.
            area = math.pi * (r ** 2 - (r - width) ** 2)
            mass = n * area * float(baffle_thickness_m) * float(wall_density_kg_m3)
            notes = [
                "Damping is Miles' empirical correlation for ring baffles. It "
                "carries real scatter and depends on slosh amplitude in a way "
                "no closed form captures; flight programmes confirm slosh "
                "damping by test rather than by analysis.",
            ]
            if n > 1:
                notes.append(
                    f"{n} baffles are summed, which over-credits them: the "
                    f"deeper rings sit in fluid the shallower ones have already "
                    f"quietened. Treat the damping as an upper estimate.")
            return BaffleDesign(
                n_baffles=n, width_m=width, width_ratio=width_ratio,
                damping_ratio=zeta, mass_kg=mass,
                achieved_separation=sep, notes=tuple(notes))
    return None
