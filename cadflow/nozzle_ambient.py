"""Is each nozzle sized for the air it actually flies in?

A rocket nozzle only works over a band of ambient pressures. Expand too far for
the air outside and the flow peels off the wall -- it separates -- and the
engine stops behaving like the one that was designed. The largest expansion
ratio that stays attached is set by ambient alone, so it is a property of *where
the stage burns*, not of the stage.

This project's planner already knew that; its own comment says "expansion ratio
rises with stage number: lower stages fight ambient", and it picks 12 / 30 / 60
/ 80 for a four-stage vehicle. Then it sized every one of those throats against
sea-level ambient, including the stages that ignite above 40 km.

That is not a small bookkeeping error, because of what the nozzle model does
with an over-expanded ratio. Told to evaluate eps=30, 60 or 80 at 101 kPa, it
correctly reports the flow as separated and truncates to the effective ratio the
separation plane implies -- about 15 at sea level, for all three. So the sizing
step saw the *same* nozzle three times. The ladder the planner deliberately
chose had no effect whatsoever on the throat areas, and every upper stage came
out with a throat sized for a nozzle it does not have.

The consequence is a miscalibrated design knob rather than a wrong trajectory.
The integrator flies the real ambient at every step, so thrust, apogee and the
loads that size the structure were all computed correctly. What was wrong is the
thrust-to-weight each stage was *asked* for: an oversized throat delivers more
thrust than intended once it reaches vacuum, so a stage asked for 2.2 flies at
2.79. Thrust-to-weight is the lever the design loop pulls to hold peak axial
acceleration down, and a lever that moves 27% further than the dial says is one
the loop cannot aim with.

Nothing here re-derives the separation criterion; it uses the same
``separation_limited_ratio`` the trajectory integrator uses, so the check and
the thing it checks cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from generate_propulsion_trajectory_corpus import (
    G0, nozzle_performance, separation_limited_ratio)

#: Ambient pressure below which a stage is treated as igniting in vacuum, Pa.
#:
#: 100 Pa is roughly 48 km, two and a half orders of magnitude below sea level.
#: The pressure-thrust term it can still contribute is under a tenth of a
#: percent, so calling it vacuum costs nothing and avoids sizing an upper stage
#: against a number that is noise.
VACUUM_AMBIENT_PA = 100.0

#: Separation margin below which a nozzle is reported as marginal.
#:
#: The Summerfield criterion is itself approximate -- real separation pressure
#: ratios scatter between about 0.3 and 0.4 of ambient depending on wall
#: contour and boundary layer state -- so a nozzle sitting at 1.02x the limit is
#: not meaningfully attached. It is inside the uncertainty of the criterion.
MARGINAL_BELOW = 1.15


@dataclass(frozen=True)
class StageNozzle:
    """One stage's nozzle judged against its own ignition ambient."""

    stage: int
    expansion_ratio: float
    ignition_ambient_pa: float
    ignition_altitude_m: float
    eps_max: float
    #: expansion_ratio / eps_max; below 1 is attached
    utilisation: float
    attached: bool
    marginal: bool
    #: Thrust-to-weight the throat was sized for, and what it flies at
    twr_sized: float | None = None
    twr_flown: float | None = None
    #: The ambient pressure the throat was sized against, Pa
    sized_at_ambient_pa: float = 101325.0

    @property
    def twr_error_pct(self) -> float | None:
        if self.twr_sized is None or self.twr_flown is None:
            return None
        if self.twr_sized <= 0:
            return None
        return (self.twr_flown - self.twr_sized) / self.twr_sized * 100.0

    def as_dict(self) -> dict:
        d = {"stage": self.stage, "expansion_ratio": self.expansion_ratio,
             "ignition_ambient_pa": self.ignition_ambient_pa,
             "ignition_altitude_m": self.ignition_altitude_m,
             "eps_max": self.eps_max, "utilisation": self.utilisation,
             "attached": self.attached, "marginal": self.marginal,
             "twr_sized": self.twr_sized, "twr_flown": self.twr_flown,
             "sized_at_ambient_pa": self.sized_at_ambient_pa}
        d["twr_error_pct"] = self.twr_error_pct
        return d


def sizing_ambient_pa(ignition_ambient_pa: float) -> float:
    """The pressure a stage's throat should be sized against.

    Its own ignition ambient, floored at vacuum. This is the whole fix: the
    planner used sea level for every stage because the trajectory did not hand
    back the pressure each stage lights at, and a number that is not available
    gets replaced by one that is.
    """
    p = float(ignition_ambient_pa)
    if not math.isfinite(p) or p < VACUUM_AMBIENT_PA:
        return 0.0
    return p


def check_stage(*, stage: int, expansion_ratio: float, chamber_pressure_pa: float,
                gamma: float, ignition_ambient_pa: float,
                ignition_altitude_m: float = float("nan"),
                chamber_temp_k: float | None = None,
                mol_mass: float | None = None,
                twr_sized: float | None = None,
                sized_at_ambient_pa: float = 101325.0) -> StageNozzle:
    """Judge one nozzle against the ambient pressure it is lit at.

    ``twr_sized`` is the thrust-to-weight the throat was sized for and
    ``sized_at_ambient_pa`` the pressure it was sized at; given the chamber
    conditions, this reports the ratio the stage actually flies at. The gap
    between the two is the miscalibration this module exists to surface.

    The sizing pressure is an argument and not a constant on purpose. It was a
    hardcoded 101325 for one revision, which was true of the planner at the time
    and stopped being true the moment the planner was fixed -- so the check went
    on reporting a 27% error against a defect that no longer existed. A check
    that carries its own copy of what another module does is a check that
    reports history.
    """
    p_amb = float(ignition_ambient_pa)
    if not math.isfinite(p_amb) or p_amb < 0.0:
        p_amb = 0.0
    # Below the vacuum threshold, say vacuum. The criterion is continuous, so a
    # denormal ambient produces a finite limit of order 1e11 -- true, useless,
    # and it reads as a real number in a report. The module already defines what
    # counts as vacuum for sizing; using a different definition here would be
    # the same module disagreeing with itself.
    eps_max = (float("inf") if p_amb < VACUUM_AMBIENT_PA
               else separation_limited_ratio(chamber_pressure_pa, p_amb, gamma))
    eps = float(expansion_ratio)
    util = eps / eps_max if math.isfinite(eps_max) and eps_max > 0 else 0.0
    attached = eps <= eps_max
    marginal = attached and eps_max < MARGINAL_BELOW * eps

    twr_flown = None
    if (twr_sized is not None and chamber_temp_k is not None
            and mol_mass is not None):
        # The throat was sized so that F(at sizing ambient) = twr * W. The same
        # throat in vacuum makes F(vacuum), so the ratio of thrust coefficients
        # is the ratio of thrust-to-weights, and the throat area cancels.
        kw = dict(chamber_pressure=chamber_pressure_pa,
                  chamber_temp=chamber_temp_k, expansion_ratio=eps,
                  throat_area=1.0, gamma=gamma, mol_mass=mol_mass)
        f_sized = nozzle_performance(
            ambient_pressure=max(0.0, float(sized_at_ambient_pa)), **kw)["thrust"]
        f_flown = nozzle_performance(ambient_pressure=p_amb, **kw)["thrust"]
        if f_sized > 0:
            twr_flown = float(twr_sized) * f_flown / f_sized

    return StageNozzle(stage=stage, expansion_ratio=eps,
                       ignition_ambient_pa=p_amb,
                       ignition_altitude_m=float(ignition_altitude_m),
                       eps_max=eps_max, utilisation=util, attached=attached,
                       marginal=marginal, twr_sized=twr_sized,
                       twr_flown=twr_flown,
                       sized_at_ambient_pa=float(sized_at_ambient_pa))


def check_stack(stages, *, ignition_ambient_pa, ignition_altitude_m=None,
                twr_by_stage=None, sized_at_ambient_pa=None) -> list[StageNozzle]:
    """Judge every stage's nozzle against the ambient it is lit at.

    ``stages`` is the planner's stage list; each entry needs
    ``expansion_ratio``, ``chamber_pressure_pa``, ``gamma`` and, for the
    thrust-to-weight comparison, ``chamber_temp`` and ``mol_mass``.
    """
    alts = list(ignition_altitude_m or [])
    out = []
    for i, st in enumerate(stages):
        p = ignition_ambient_pa[i] if i < len(ignition_ambient_pa) else float("nan")
        out.append(check_stage(
            stage=i + 1,
            expansion_ratio=float(st.expansion_ratio),
            chamber_pressure_pa=float(st.chamber_pressure_pa),
            gamma=float(st.gamma),
            ignition_ambient_pa=p,
            ignition_altitude_m=alts[i] if i < len(alts) else float("nan"),
            chamber_temp_k=getattr(st, "chamber_temp", None),
            mol_mass=getattr(st, "mol_mass", None),
            twr_sized=(twr_by_stage[i] if twr_by_stage
                       and i < len(twr_by_stage) else None),
            # From the stage itself when it carries one -- the planner records
            # what it sized against -- and only then from the argument.
            sized_at_ambient_pa=(
                sized_at_ambient_pa[i]
                if sized_at_ambient_pa and i < len(sized_at_ambient_pa)
                else float(getattr(st, "sized_at_ambient_pa", 101325.0)))))
    return out


def findings(checks) -> list[str]:
    """Plain statements of anything wrong, empty when the stack is clean."""
    out = []
    for c in checks:
        if not c.attached:
            out.append(
                f"stage {c.stage} nozzle separates at ignition: eps {c.expansion_ratio:.0f} "
                f"against a limit of {c.eps_max:.1f} at "
                f"{c.ignition_ambient_pa / 1000:.1f} kPa. The flow leaves the wall "
                f"and side-loads the engine and thrust structure, and the "
                f"effective expansion ratio is {c.eps_max:.1f}, not "
                f"{c.expansion_ratio:.0f}")
        elif c.marginal:
            out.append(
                f"stage {c.stage} nozzle is marginally attached: eps "
                f"{c.expansion_ratio:.0f} against a limit of {c.eps_max:.1f}, "
                f"a {c.utilisation:.2f} utilisation. The separation criterion is "
                f"itself good to about 25%, so this is inside its uncertainty")
        err = c.twr_error_pct
        if err is not None and abs(err) > 5.0:
            out.append(
                f"stage {c.stage} was sized for a thrust-to-weight of "
                f"{c.twr_sized:.2f} and flies at {c.twr_flown:.2f}, "
                f"{err:+.0f}%. The throat was sized at "
                f"{c.sized_at_ambient_pa / 1000:.2f} kPa for a stage that "
                f"ignites at {c.ignition_altitude_m / 1000:.0f} km, where "
                f"ambient is {c.ignition_ambient_pa / 1000:.2f} kPa")
    return out


def recover_twr_sized(stages, payload_kg: float) -> list[float]:
    """The thrust-to-weight each throat was sized for, from the stages alone.

    The planner solves ``A_t = twr * m_supported * g0 / F(A_t=1)``, so the ratio
    comes straight back out:

        twr = A_t * F(A_t=1, at the sizing ambient) / (m_supported * g0)

    Recovered rather than passed in. This project has been bitten six times by a
    reported number that its own subject could not reproduce -- a structural
    coefficient that read a module default, a static margin left behind by the
    fin trade -- and every one of them was a value carried alongside the thing it
    described instead of derived from it. A list threaded through three call
    sites is the same arrangement.
    """
    masses = []
    running = float(payload_kg)
    for st in reversed(list(stages)):
        running += float(st.prop_mass_kg) + float(st.struct_mass_kg)
        masses.append(running)
    masses.reverse()

    out = []
    for st, supported in zip(stages, masses):
        f_unit = nozzle_performance(
            chamber_pressure=float(st.chamber_pressure_pa),
            chamber_temp=float(st.chamber_temp),
            expansion_ratio=float(st.expansion_ratio),
            throat_area=1.0, gamma=float(st.gamma),
            mol_mass=float(st.mol_mass),
            ambient_pressure=max(0.0, float(
                getattr(st, "sized_at_ambient_pa", 101325.0))))["thrust"]
        out.append(float(st.throat_area_m2) * f_unit / (supported * G0))
    return out
