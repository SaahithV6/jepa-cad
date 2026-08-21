"""Close the loop: evaluate a vehicle across disciplines, then repair it.

The packet could find violations and report them. It could not act on them, so
a design that ran its payload at 16 g, or could not cool its chamber, came out
as a well-documented failure rather than as a design.

This is the iterative design/test cycle the project is for. Each constraint has
a knob that moves it, each knob has a direction that was measured rather than
reasoned about, and the loop applies them until the vehicle either satisfies
every constraint or demonstrably cannot.

The directions, all checked before being used
---------------------------------------------
Raising chamber pressure *improves* regenerative cooling. That is the opposite
of the intuition -- higher pressure means more heat -- and the intuition is
wrong because coolant flow rises linearly with pressure while heat load rises
as pressure^0.8. Measured on LOX/RP-1, coolant rise falls from 368 K at 20 bar
to 249 K at 200 bar.

Raising chamber pressure also raises throat heat flux, 14.8 MW/m^2 at 20 bar to
98.7 at 200. So the two thermal constraints push in opposite directions and
between them define a window of feasible chamber pressures rather than a bound.
A design loop that only knew one of them would walk straight out of the other.

Lowering thrust-to-weight lowers peak axial acceleration nearly in proportion --
16.3 g to 10.0 g for [4.5, 3.0] against [2.5, 1.8] -- and costs gross mass to
gravity losses, 1137 kg to 1351 kg for the same mission. That is the trade, and
the loop pays it only when the acceleration limit demands it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Design limits. Each is a real constraint with a real consequence, and each
#: has a knob below that moves it.
DEFAULT_LIMITS = {
    #: axial acceleration a payload is typically qualified to
    "payload_g": 10.0,
    #: coolant must leave the jacket below its own limit, which the coolant
    #: table sets per fuel; this is the margin demanded on top
    "coolant_margin_k": 25.0,
    #: what a well-cooled copper throat survives
    "throat_flux_mw_m2": 90.0,
    #: aluminium stops holding useful strength somewhere here
    "skin_temp_k": 450.0,
    #: static margin in calibers
    "static_margin_min": 1.0,
}


@dataclass(frozen=True)
class Knobs:
    """The design variables this loop is allowed to move."""
    chamber_bar: float = 55.0
    twr_by_stage: tuple[float, ...] = (4.5, 3.0, 2.2, 2.0)
    propellant: str = "lox_rp1"


@dataclass(frozen=True)
class Violation:
    discipline: str
    quantity: str
    value: float
    limit: float
    remedy: str

    def __str__(self) -> str:
        return (f"{self.discipline}: {self.quantity} = {self.value:.4g} "
                f"against {self.limit:.4g} ({self.remedy})")


@dataclass
class Evaluation:
    knobs: Knobs
    plan: object = None
    peak_g: float = 0.0
    skin_temp_k: float = 0.0
    throat_flux_mw_m2: float = 0.0
    coolant_margin_k: float = 0.0
    coolant_outlet_k: float = 0.0
    gross_kg: float = 0.0
    stages: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.violations


def evaluate(payload_kg: float, apogee_km: float, knobs: Knobs,
             limits: dict | None = None) -> Evaluation:
    """Design the vehicle at these knob settings and check every discipline."""
    from cadflow.planner import plan

    lim = dict(DEFAULT_LIMITS)
    if limits:
        lim.update(limits)

    p = plan(apogee_km, payload_kg, propellant=knobs.propellant,
             chamber_bar=knobs.chamber_bar,
             twr_by_stage=list(knobs.twr_by_stage))
    if p is None:
        return Evaluation(knobs=knobs, violations=[Violation(
            "architecture", "closure", 0.0, 1.0,
            "no architecture closes this mission")])

    ev = Evaluation(knobs=knobs, plan=p, gross_kg=p.gross_kg, stages=p.stages)
    ev.peak_g = float(p.trajectory.get("max_axial_g") or 0.0)
    ev.skin_temp_k = float(p.trajectory.get("max_skin_temp_k") or 0.0)

    try:
        from cadflow.combustion import COMBINATIONS, REFERENCE, chamber_equilibrium
        from cadflow.thermal import chamber_heat_load, regenerative_cooling

        if knobs.propellant in COMBINATIONS:
            of = REFERENCE[knobs.propellant][0]
            state = chamber_equilibrium(knobs.propellant, of,
                                        knobs.chamber_bar * 1e5)
            area = p.stack[0].throat_area_m2
            mdot = state.pressure_pa * area / state.c_star_m_s
            load = chamber_heat_load(state, area, mdot)
            cool = regenerative_cooling(load["q_total_w"], mdot / (1.0 + of),
                                        COMBINATIONS[knobs.propellant][1])
            ev.throat_flux_mw_m2 = load["throat"].heat_flux_mw_m2
            ev.coolant_margin_k = cool["margin_k"]
            ev.coolant_outlet_k = cool["outlet_temp_k"]
    except Exception:  # noqa: BLE001 - thermal is an addition, not a gate
        pass

    if ev.peak_g > lim["payload_g"]:
        ev.violations.append(Violation(
            "loads", "peak axial acceleration", ev.peak_g, lim["payload_g"],
            "lower thrust-to-weight"))
    if ev.throat_flux_mw_m2 > lim["throat_flux_mw_m2"]:
        ev.violations.append(Violation(
            "thermal", "throat heat flux", ev.throat_flux_mw_m2,
            lim["throat_flux_mw_m2"], "lower chamber pressure"))
    if ev.coolant_margin_k and ev.coolant_margin_k < lim["coolant_margin_k"]:
        ev.violations.append(Violation(
            "thermal", "coolant margin", ev.coolant_margin_k,
            lim["coolant_margin_k"], "raise chamber pressure"))
    if ev.skin_temp_k > lim["skin_temp_k"]:
        ev.violations.append(Violation(
            "aeroheating", "peak skin temperature", ev.skin_temp_k,
            lim["skin_temp_k"], "needs thermal protection, no knob here"))
    return ev


#: Aim this far inside each limit rather than exactly at it. The planner picks
#: an architecture discretely, so gross mass and peak acceleration move in small
#: jumps as it switches between candidates -- landing exactly on a limit means
#: the next iteration can hop back over it. Five percent is enough to sit still.
_TARGET_MARGIN = 0.95

#: Damping on each correction. The knob-to-constraint relationships are close to
#: the power laws used here, so most of the correction can be taken at once; the
#: exponent below one keeps a mis-estimated relationship from overshooting.
_DAMPING = 0.8


def remedy(knobs: Knobs, violations: list[Violation]) -> Knobs:
    """Move the knobs in the directions the violations call for.

    Each correction is the inverse of the physical scaling that connects the
    knob to the constraint, damped slightly. An earlier version took quarter
    steps toward the target, which near convergence is a one percent move: it
    walked peak acceleration from 16.3 g to 10.4 g and then stalled there for
    five iterations without crossing the limit.
    """
    chamber = knobs.chamber_bar
    twr = list(knobs.twr_by_stage)

    for v in violations:
        target = v.limit * _TARGET_MARGIN
        if v.remedy == "lower thrust-to-weight":
            # peak acceleration scales close to linearly with liftoff T/W
            factor = (target / max(v.value, 1e-9)) ** _DAMPING
            twr = [max(1.2, t * factor) for t in twr]
        elif v.remedy == "lower chamber pressure":
            # flux goes as pressure^0.8
            factor = (target / max(v.value, 1e-9)) ** (_DAMPING / 0.8)
            chamber = max(5.0, chamber * factor)
        elif v.remedy == "raise chamber pressure":
            # coolant rise goes as pressure^-0.2: a very gentle lever, so it
            # needs a large multiplier to move the constraint at all
            shortfall = max(1.05, (v.limit + 1.0) / max(v.value, 1.0))
            chamber = min(300.0, chamber * min(2.0, shortfall ** 2.0))

    return replace(knobs, chamber_bar=chamber, twr_by_stage=tuple(twr))


def _detect_conflict(violations: list[Violation]) -> dict | None:
    """Are two violations asking for the same knob to move both ways?

    An over-constrained problem is a design finding in its own right, and a
    more useful one than a failed search: it says the requirements cannot all
    be met at once and names the pair that cannot coexist.
    """
    remedies = {v.remedy for v in violations}
    for knob in ("chamber pressure", "thrust-to-weight"):
        if f"raise {knob}" in remedies and f"lower {knob}" in remedies:
            opposed = [v for v in violations if knob in v.remedy]
            return {
                "knob": knob,
                "constraints": [str(v) for v in opposed],
                "message": (
                    f"{opposed[0].quantity} and {opposed[1].quantity} both "
                    f"depend on {knob} and pull it in opposite directions. No "
                    f"value satisfies both, so one requirement has to move or "
                    f"the design needs a mechanism this loop does not have -- "
                    f"film cooling, a different wall material, or a different "
                    f"propellant."),
            }
    return None


def autodesign(payload_kg: float, apogee_km: float,
               knobs: Knobs | None = None, limits: dict | None = None,
               max_iters: int = 12) -> dict:
    """Iterate design and repair until every constraint is satisfied.

    Returns the history as well as the answer, because how it got there is the
    interesting part: which constraint bound first, what it cost to satisfy, and
    whether anything had to be traded against anything else.
    """
    knobs = knobs or Knobs()
    history = []
    best = None

    for i in range(max_iters):
        ev = evaluate(payload_kg, apogee_km, knobs, limits)
        history.append({
            "iteration": i,
            "chamber_bar": knobs.chamber_bar,
            "twr": list(knobs.twr_by_stage),
            "gross_kg": ev.gross_kg,
            "stages": ev.stages,
            "peak_g": ev.peak_g,
            "throat_flux_mw_m2": ev.throat_flux_mw_m2,
            "coolant_margin_k": ev.coolant_margin_k,
            "skin_temp_k": ev.skin_temp_k,
            "violations": [str(v) for v in ev.violations],
        })
        if best is None or (len(ev.violations) < len(best.violations)):
            best = ev
        if ev.feasible:
            return {"converged": True, "iterations": i + 1, "evaluation": ev,
                    "knobs": knobs, "history": history}

        actionable = [v for v in ev.violations if "no knob" not in v.remedy]
        if not actionable:
            break

        # Opposing remedies on the same knob mean the constraints are
        # incompatible, not that the loop needs more iterations. Raising
        # chamber pressure improves the coolant margin and worsens throat flux;
        # lowering it does the reverse. When both are violated at once there is
        # no pressure that satisfies both, and iterating just oscillates -- an
        # earlier version bounced between 167 and 172 bar until it ran out of
        # iterations and reported "not converged", which is true but says
        # nothing about why.
        conflict = _detect_conflict(actionable)
        if conflict:
            return {"converged": False, "iterations": i + 1, "evaluation": ev,
                    "knobs": knobs, "history": history, "conflict": conflict}

        knobs = remedy(knobs, actionable)

    return {"converged": False, "iterations": len(history), "evaluation": best,
            "knobs": knobs, "history": history, "conflict": None}
