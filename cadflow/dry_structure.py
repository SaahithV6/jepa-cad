"""The structure that carries the load with nothing inside it.

Every stage's tank is pressurised, and pressure is a structural asset: it puts
the wall in hoop tension and, through the end domes, in axial tension that works
against the flight compression. ``cadflow.pressurization`` computes that and
finds this vehicle's tank walls in net tension at flight load, with no
compressive buckling mode to go unstable in.

The interstages have none of that. They are dry tubes that transmit the entire
thrust of the stage below into everything above, at the moment of that stage's
highest acceleration, with no internal pressure to stabilise them. They are the
most buckling-critical elements on the vehicle and this project has only ever
analysed them as coupons: 42 mm test articles at a clamped radius, checked
against a yield allowable, for a failure mode a thin shell does not have.

Two modes, not one
------------------
``cadflow.shell_buckling`` gives the shell mode -- the wall folding into
diamonds -- and takes no length, because the classical result

    sigma_cr = E t / (r sqrt(3 (1 - nu^2)))

is the long-cylinder asymptote and genuinely has no length in it. That is only
the right answer over a range, and the Batdorf parameter says where:

    Z = L^2 / (R t) * sqrt(1 - nu^2)

Below Z of about 100 a shell is short enough that its ends stiffen it and the
long-cylinder value is conservative -- safe, but by an unstated margin. Far
above it the asymptote holds.

The mode the shell check cannot see is the tube buckling as a *column*: not the
wall folding locally but the whole cylinder bowing sideways, Euler's

    P_cr = pi^2 E I / (k L)^2,   I = pi r^3 t

which depends on length as strongly as the shell mode ignores it. A long thin
tube fails this way at loads where its wall is nowhere near local instability,
and no amount of checking the shell mode would ever reveal it. Both are computed
here and the governing one is reported, because a structure is only as good as
its worst mode and nothing had been looking for this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Poisson's ratio for the metals this project selects.
POISSON = 0.3

#: Batdorf parameter below which a cylinder is short enough that its end
#: restraint matters and the long-cylinder buckling stress understates it.
#:
#: The transition is gradual, not a cliff. 100 is the conventional place to stop
#: trusting the asymptote, and being below it means the shell check is
#: conservative rather than wrong -- which is worth saying, because an unstated
#: conservatism gets spent twice.
SHORT_SHELL_Z = 100.0

#: End-fixity for the column mode.
#:
#: An interstage is bolted to substantial ring frames at both ends, which is
#: nearer clamped than pinned, and clamped-clamped would give k = 0.5 and four
#: times the critical load. Pinned-pinned at k = 1.0 is the conservative
#: assumption and is used because this project does not design the joint. If the
#: column mode ever governs, the joint is the first thing to look at.
COLUMN_END_FIXITY = 1.0


@dataclass
class DrySection:
    """One unpressurised load-carrying section, checked in both modes."""

    name: str
    length_m: float
    radius_m: float
    wall_m: float
    axial_load_n: float
    axial_stress_mpa: float
    #: Shell mode, from cadflow.shell_buckling
    shell_allowable_mpa: float
    shell_margin: float
    #: Column mode
    euler_load_n: float
    column_margin: float
    #: Batdorf parameter, which says whether the shell asymptote applies
    batdorf_z: float
    governs: str
    notes: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return min(self.shell_margin, self.column_margin)

    @property
    def passes(self) -> bool:
        return self.margin >= 1.0

    def as_dict(self) -> dict:
        return {"name": self.name, "length_m": self.length_m,
                "radius_m": self.radius_m, "wall_m": self.wall_m,
                "axial_load_n": self.axial_load_n,
                "axial_stress_mpa": self.axial_stress_mpa,
                "shell_allowable_mpa": self.shell_allowable_mpa,
                "shell_margin": self.shell_margin,
                "euler_load_n": self.euler_load_n,
                "column_margin": self.column_margin,
                "batdorf_z": self.batdorf_z, "governs": self.governs,
                "margin": self.margin, "passes": self.passes,
                "notes": list(self.notes)}


def batdorf_z(length_m: float, radius_m: float, wall_m: float,
              poisson: float = POISSON) -> float:
    """L^2 / (R t) sqrt(1 - nu^2): how long a cylinder is, structurally.

    Length alone does not say whether a shell is long. A two-metre tube is long
    at 0.8 mm wall and short at 20 mm, because what matters is length against
    the geometric mean of radius and thickness.
    """
    if radius_m <= 0 or wall_m <= 0:
        raise ValueError("radius and wall must be positive")
    return (length_m ** 2) / (radius_m * wall_m) * math.sqrt(1.0 - poisson ** 2)


def euler_buckling_load_n(length_m: float, radius_m: float, wall_m: float,
                          youngs_pa: float,
                          end_fixity: float = COLUMN_END_FIXITY) -> float:
    """pi^2 E I / (k L)^2 for a thin tube, with I = pi r^3 t.

    The second moment of a thin-walled circular tube about a diameter is
    pi r^3 t exactly in the thin-wall limit -- the wall's own thickness
    contributes nothing to second order. This is the mode where the tube bows as
    a whole rather than the wall folding locally, and it is invisible to any
    check that does not know the length.
    """
    if length_m <= 0:
        raise ValueError("length must be positive")
    inertia = math.pi * radius_m ** 3 * wall_m
    return (math.pi ** 2) * youngs_pa * inertia / ((end_fixity * length_m) ** 2)


def check_section(*, name: str, length_m: float, radius_m: float, wall_m: float,
                  axial_load_n: float, youngs_pa: float,
                  bending_mpa: float = 0.0) -> DrySection:
    """Both buckling modes for one dry section, and which one governs."""
    from cadflow.shell_buckling import check as shell_check

    area = 2.0 * math.pi * radius_m * wall_m
    stress_mpa = abs(axial_load_n) / area / 1e6

    bk = shell_check(axial_mpa=stress_mpa, bending_mpa=abs(bending_mpa),
                     radius_m=radius_m, wall_m=wall_m, youngs_pa=youngs_pa)

    p_euler = euler_buckling_load_n(length_m, radius_m, wall_m, youngs_pa)
    col_margin = p_euler / max(abs(axial_load_n), 1e-9)

    z = batdorf_z(length_m, radius_m, wall_m)
    notes = []
    if z < SHORT_SHELL_Z:
        notes.append(
            f"Batdorf Z = {z:.0f}, below {SHORT_SHELL_Z:.0f}: this shell is "
            f"short enough that its end restraint raises the real buckling "
            f"stress above the long-cylinder value used here. The shell margin "
            f"is conservative by an amount this does not quantify")
    else:
        notes.append(
            f"Batdorf Z = {z:.0f}, comfortably above {SHORT_SHELL_Z:.0f}: the "
            f"long-cylinder buckling stress applies")

    governs = "shell" if bk.margin <= col_margin else "column"
    if governs == "column":
        notes.append(
            f"the column mode governs at {col_margin:.2f} against the shell's "
            f"{bk.margin:.2f}. The whole tube bows before its wall folds, which "
            f"no shell check can see -- it takes no length. End fixity is "
            f"assumed pinned; clamped ends would give four times this load, so "
            f"the joint is the first place to look")

    return DrySection(
        name=name, length_m=length_m, radius_m=radius_m, wall_m=wall_m,
        axial_load_n=axial_load_n, axial_stress_mpa=stress_mpa,
        shell_allowable_mpa=bk.allowable_compression_mpa,
        shell_margin=bk.margin, euler_load_n=p_euler,
        column_margin=col_margin, batdorf_z=z, governs=governs, notes=notes)


def interstage_loads(stack, payload_kg: float, axial_g_by_stage) -> list[dict]:
    """What each interstage carries, and when.

    Interstage n/n+1 sits above stage n and transmits stage n's thrust into
    everything above it. The load is therefore the mass above the interstage
    times the acceleration stage n reaches -- and the worst instant is the *end*
    of stage n's burn, when the vehicle is lightest and pulling hardest, not
    liftoff.

    Getting that wrong is a factor of several. This vehicle's first stage lifts
    off near 3 g and burns out above 12, so an interstage sized at liftoff
    acceleration would be under-designed fourfold at the moment it matters most.
    """
    out = []
    n = len(stack)
    for i in range(n - 1):
        # everything above interstage i/i+1: stages i+1.. plus payload
        above = float(payload_kg) + sum(
            float(s.prop_mass_kg) + float(s.struct_mass_kg)
            for s in stack[i + 1:])
        g = float(axial_g_by_stage[i]) if i < len(axial_g_by_stage) else 1.0
        out.append({"name": f"interstage {i+1}/{i+2}",
                    "supported_kg": above,
                    "axial_g": g,
                    "load_n": above * 9.80665 * g})
    return out


def check_interstages(stack, payload_kg: float, axial_g_by_stage, *,
                      radius_m: float, wall_m: float, youngs_pa: float,
                      lengths_m=None) -> list[DrySection]:
    """Every interstage on the vehicle, in both buckling modes."""
    loads = interstage_loads(stack, payload_kg, axial_g_by_stage)
    out = []
    for i, rec in enumerate(loads):
        length = (lengths_m[i] if lengths_m and i < len(lengths_m)
                  else 1.5 * radius_m)
        out.append(check_section(
            name=rec["name"], length_m=length, radius_m=radius_m,
            wall_m=wall_m, axial_load_n=rec["load_n"], youngs_pa=youngs_pa))
        out[-1].notes.insert(
            0, f"carries {rec['supported_kg']:.0f} kg at "
               f"{rec['axial_g']:.1f} g, the peak of the stage below it")
    return out


def failures(sections) -> list[DrySection]:
    return [s for s in sections if not s.passes]


def bending_mpa_at(loads, station_m: float, radius_m: float,
                   wall_m: float) -> float:
    """Bending stress at one station, from the solved moment distribution.

    Stations run from the AFT end, positive forward -- stage 1 starts at zero
    and the payload sits at the top. That is what ``section_extents`` uses and
    therefore what the loads solution is indexed by, whatever the comment on
    ``PointLoad`` used to claim. Reading this backwards returns the moment from
    the mirror-image station: a plausible number, and the wrong one.

    Interpolated rather than nearest-neighbour. The moment distribution has a
    kink at every point load, and snapping to the nearest of 401 stations would
    quietly move a section up to 6 mm along a curve that is steepest exactly
    where the interstages sit.
    """
    xs = list(loads.stations_m)
    ms = list(loads.moment_nm)
    if not xs:
        return 0.0
    x = float(station_m)
    if x <= xs[0]:
        m = ms[0]
    elif x >= xs[-1]:
        m = ms[-1]
    else:
        # bisect by hand to avoid assuming uniform spacing
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        span = xs[hi] - xs[lo]
        f = 0.0 if span <= 0 else (x - xs[lo]) / span
        m = ms[lo] + f * (ms[hi] - ms[lo])
    section_modulus = math.pi * radius_m ** 2 * wall_m
    return abs(m) / section_modulus / 1e6
