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

import math
from dataclasses import dataclass, field, replace

#: Design limits. Each is a real constraint with a real consequence, and each
#: has a knob below that moves it.
#: A limit at or above this is not a limit. No structural alloy survives 1e8 K
#: and no vehicle reaches it, so a caller passing one is saying "do not gate on
#: this", which several tests do to isolate the constraint they are about.
_NO_GATE_K = 1e8

DEFAULT_LIMITS = {
    #: axial acceleration a payload is typically qualified to
    "payload_g": 10.0,
    #: coolant must leave the jacket below its own limit, which the coolant
    #: table sets per fuel; this is the margin demanded on top
    "coolant_margin_k": 25.0,
    #: what a well-cooled copper throat survives
    "throat_flux_mw_m2": 90.0,
    #: Not a limit any more -- the skin material provides the physical one, and
    #: this stays only so a caller can impose something *stricter* or switch the
    #: gate off entirely. It was 450 K (aluminium) from before materials were
    #: modelled, and left at that it capped every alloy at aluminium's rating:
    #: Inconel 718 was reported as surviving 450 K rather than 920 K, and the
    #: loop bought thermal protection for a skin that did not need it.
    #:
    #: None, not _NO_GATE_K. Those are two different intentions -- "impose no
    #: extra cap, the material governs" and "switch the gate off entirely" --
    #: and collapsing them onto one sentinel disabled the gate by default.
    #: plan_and_verify calls autodesign with no limits, so every packet this
    #: project has produced ran with no skin-temperature gate: the loop returned
    #: aluminium 6061-T6, rated to 420 K, on a skin at 895 K, reporting
    #: feasible with an empty violation list. Aluminium is nearly molten there.
    #: The 3x density difference against Inconel then decided whether the
    #: vehicle could afford its own tankage, so a disabled thermal gate was
    #: setting a structural verdict.
    "skin_temp_k": None,
    #: how far the structural coefficient the loads demand may exceed the one
    #: the vehicle was sized at before the design is called unclosed. Two
    #: percent, because the planner picks architectures discretely and the
    #: implied coefficient moves in small jumps as it switches between them.
    "struct_closure_tol": 0.02,
    #: static margin in calibers
    "static_margin_min": 1.0,
}


@dataclass(frozen=True)
class Knobs:
    """The design variables this loop is allowed to move."""
    chamber_bar: float = 55.0
    twr_by_stage: tuple[float, ...] = (4.5, 3.0, 2.2, 2.0)
    propellant: str = "lox_rp1"
    #: Structural mass fraction the vehicle is designed at. None means the
    #: planner's asserted constant. This is a design variable because the
    #: constant is a guess: for 25 kg to 4,000 km the vehicle's own loads demand
    #: 0.29 where the source asserts 0.14, and a loop that cannot move this
    #: number can only ever report that failure, never fix it.
    struct_coeff: float | None = None
    #: Skin material. A design variable because the alternative is the loop
    #: reporting "needs thermal protection, no knob here" and stopping -- which
    #: it did for 25 kg to 4,000 km, where the skin reaches 854 K and aluminium
    #: holds 420. The catalogue already carried titanium to 670 K and Inconel
    #: to 920; nothing was missing except the loop's permission to use them.
    skin_material: str = "al-6061-t6"
    #: Nose shape. A design variable because it was hardcoded: the planner held
    #: NOSE_SHAPE = "ogive" and the loop had no way to consider anything else,
    #: despite plan() accepting the argument and the drag model supporting three
    #: shapes. CFD now prices all three on an open-base vehicle, so the choice
    #: can be made on measured drag rather than left at a default.
    nose_shape: str = "ogive"
    #: Oxidiser-to-fuel mass ratio. None takes the propellant catalogue's value.
    #:
    #: A design variable for the same reason the others are, and a conspicuous
    #: omission: the loop could move chamber pressure, thrust-to-weight, the
    #: structural coefficient, the skin alloy, the nose shape, the stage count
    #: and whether to carry a blanket -- but not the single most important
    #: propulsion parameter. planner.plan has accepted an of_ratio throughout;
    #: nothing was passing one.
    #:
    #: The interesting part is which optimum applies. Peak specific impulse for
    #: lox/rp1 sits at 2.208 and peak *density* impulse near 2.56. The usual
    #: answer is to maximise Isp, but this vehicle's tank mass is set by surface
    #: area at minimum gauge rather than by pressure, so denser propellant means
    #: smaller tanks means less structure -- and tank ends are what decide
    #: whether its smallest stage is affordable at all.
    of_ratio: float | None = None
    #: Protect the structure with a blanket instead of skinning it in an alloy
    #: that survives the airflow.
    #:
    #: A design variable for the same reason the others are. The loop treats
    #: thermal protection as a last resort -- sized only when no alloy survives
    #: at all -- so it has no way to notice that a blanket over a light alloy can
    #: weigh less than a bare heavy one. For this vehicle the skin reaches 885 K,
    #: which forces Inconel at 8190 kg/m3, and Inconel's density is what makes
    #: the smallest stage unable to afford its tank ends. Aerogel over aluminium
    #: is about 11.8 kg of blanket against roughly 40 kg of tankage.
    #:
    #: False by default: a bare structure is simpler, and TPS is an ablator or a
    #: blanket with inspection and refurbishment costs this project does not
    #: model. The loop turns it on only when it pays.
    use_tps: bool = False
    #: Cap on stage count. None lets the planner choose freely.
    #:
    #: A design variable for the same reason struct_coeff is one. Architecture
    #: selection in the planner is by gross mass alone, and gross mass cannot
    #: see whether a stage can afford the tank it needs: structure scales with
    #: propellant while minimum gauge scales with nothing, so below some size a
    #: stage's tank ends cost more than its entire structural allowance. The
    #: planner will pick that architecture every time, because every number it
    #: reads says it is lighter. Without this knob the loop can only report the
    #: failure, never fix it.
    max_stages: int | None = None


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
    struct_coeff_asserted: float = 0.0
    struct_coeff_required: float = 0.0
    skin_material: str = ""
    skin_limit_k: float = 0.0
    #: Thermal protection this design needs, or None if the skin survives bare.
    tps: dict | None = None
    tps_mass_kg: float = 0.0
    violations: list[Violation] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.violations


def _material(material_id: str):
    from cadflow.space_materials import iter_materials

    for m in iter_materials():
        if m.material_id == material_id:
            return m
    raise KeyError(f"unknown material {material_id!r}")


def best_material_for(temp_k: float, protected: bool = False,
                      gauge_limited: bool = False):
    """Lightest catalogue material that survives this temperature.

    "Lightest" depends on what sets the wall, and the two regimes rank the
    catalogue almost in opposite orders.

    A strength-driven wall gets thinner as the alloy gets stronger, so the
    figure of merit is yield over density and maraging steel at 0.212 beats
    aluminium at 0.102 despite being three times as dense.

    A *gauge*-driven wall does not get thinner at all -- it is already as thin
    as the shop can make it -- so its mass is rho times area times a fixed
    minimum thickness, and strength does not enter. There the only thing that
    matters is density, and the ranking inverts: maraging-250 weighs 2.96x an
    aluminium wall of identical geometry.

    Every wall in this project's first stage reads "minimum gauge", so the
    selector was optimising a quantity none of them respond to and would have
    answered maraging steel to a question about tank ends that only density can
    answer.

    Returns the material itself rather than "an upgrade or nothing", so a design
    that has grown cooler can be moved back down. The loop otherwise ratchets:
    it picks Rene 41 for an 1,100 K iterate, the vehicle then grows and reaches
    only 854 K, and it flies the refractory alloy anyway -- 22% more structural
    mass per unit strength than the Inconel that temperature actually calls for.
    """
    from cadflow.space_materials import iter_materials

    # Under a blanket the structure never sees the airflow, so the surface
    # temperature stops being a filter and the choice becomes purely
    # structural. That is the whole point of protecting it: this vehicle is
    # forced onto Inconel at 8190 kg/m3 by 885 K of skin, while Ti-6Al-4V at
    # 0.199 MPa per kg/m3 beats Inconel's 0.126 on strength for weight and is
    # simply not allowed to be considered.
    #
    # The thermal side does not disappear, it moves: the blanket is then sized
    # to hold the backface at whatever the chosen alloy can take, so a colder
    # alloy buys a thicker blanket. The loop pays that in tps_mass_kg and keeps
    # the result only if the total is lighter.
    capable = [m for m in iter_materials()
               if m.yield_mpa
               and (protected or m.max_service_temp_k >= temp_k)
               and m.category in ("aluminum", "titanium", "steel", "superalloy")]
    if not capable:
        return None
    if gauge_limited:
        capable.sort(key=lambda m: m.density_kg_m3)
    else:
        capable.sort(key=lambda m: -(m.yield_mpa / m.density_kg_m3))
    return capable[0]


def coolest_capable_material(temp_k: float, incumbent_id: str):
    """Lightest-per-strength material that survives this temperature.

    Ranked by yield over density, so the choice is the one that costs the least
    mass for the strength it must carry rather than simply the most refractory
    thing in the catalogue -- Inconel survives 920 K but is 3x the density of
    aluminium, and reaching for it when titanium would do is how a design gets
    heavy for no reason.

    Returns None when nothing in the catalogue survives, which is a real answer:
    at that point the vehicle needs ablative or reusable TPS, not a different
    alloy, and that is a different design decision.
    """
    from cadflow.space_materials import iter_materials

    best = best_material_for(temp_k)
    return None if best is None or best.material_id == incumbent_id else best


#: Heating is not sustained for the whole flight. Ascent through the dense
#: atmosphere is the heating pulse, and it is roughly the interval around max-Q
#: rather than the full burn. Taken as a fraction of burnout time -- crude, and
#: stated here because TPS thickness goes as its square root, so a factor of
#: four error in duration is only a factor of two in mass.
_HEATING_FRACTION = 0.35


def _size_tps_for(ev, knobs, backface_limit_k: float) -> dict | None:
    """Thermal protection for a vehicle no alloy can skin, and what it weighs.

    Areal mass becomes vehicle mass through the wetted area, which is taken as
    a cylinder of the planner's own radius and length. That understates a
    finned vehicle and overstates a stubby one; it is the same approximation
    the mass-properties model already makes, so at least the two agree.
    """
    from cadflow.thermal import size_tps

    plan = getattr(ev, "plan", None)
    if plan is None:
        return None
    traj = getattr(plan, "trajectory", {}) or {}
    burnout = float(traj.get("burnout_s") or 0.0)
    if burnout <= 0.0:
        return None

    got = size_tps(ev.skin_temp_k, burnout * _HEATING_FRACTION,
                   backface_limit_k, reusable=False)
    if got is None or not got.get("required"):
        return got

    radius_m = max(0.05, (plan.gross_kg / 1000.0) ** (1.0 / 3.0) * 0.55 / 2.0)
    length_m = 0.0
    for st in plan.stack:
        volume = st.prop_mass_kg / 1020.0
        length_m += max(0.2, volume / (math.pi * radius_m ** 2))
    wetted_m2 = 2.0 * math.pi * radius_m * max(length_m, 2.0 * radius_m)

    # Plus the nose, which this omitted entirely.
    #
    # A bare 2 pi r L covers the barrel and stops at the shoulder, leaving out
    # the one part of the vehicle that most needs a blanket: the nose is where
    # stagnation heating peaks. It is 21% of the wetted area on this airframe,
    # so the thermal protection mass -- which is now charged to the mass budget
    # -- was understated by about two kilograms, in the direction that makes the
    # vehicle look lighter than it is.
    #
    # profiles.wetted_area is exact for a solid of revolution and was sitting
    # unused; the nose shape is a design variable the loop already moves.
    try:
        from cadflow.profiles import nose_profile, wetted_area

        _nose = nose_profile(radius_m, 4.0 * radius_m,
                             getattr(knobs, "nose_shape", "ogive"), 200)
        wetted_m2 += wetted_area(_nose)
    except Exception:  # noqa: BLE001 - the barrel area is still the bulk of it
        pass
    got["wetted_area_m2"] = wetted_m2
    got["mass_kg"] = got["areal_mass_kg_m2"] * wetted_m2
    return got


def _plan_at_coeff(apogee_km: float, payload_kg: float, knobs: "Knobs"):
    """Plan at this iterate's structural coefficient.

    The planner keeps its coefficient in a module global and derives a staging
    limit from it, so both have to move together and both have to be put back.
    `plan_solved` learned this the hard way: its restore once sat outside a
    finally and wrote a hardcoded 0.14, so any exception left every later design
    in the process silently mis-configured.
    """
    from cadflow import planner as _pl

    if knobs.struct_coeff is None:
        return _pl.plan(apogee_km, payload_kg, propellant=knobs.propellant,
                        chamber_bar=knobs.chamber_bar,
                        nose_shape=knobs.nose_shape,
                        twr_by_stage=list(knobs.twr_by_stage),
                        max_stages=knobs.max_stages,
                        of_ratio=knobs.of_ratio)
    saved_coeff, saved_mr = _pl.STRUCT_COEFF, _pl.MAX_STAGE_MR
    try:
        _pl.STRUCT_COEFF = float(knobs.struct_coeff)
        _pl.MAX_STAGE_MR = 1.0 / float(knobs.struct_coeff) * 0.62
        _pl.clear_plan_cache()
        return _pl.plan(apogee_km, payload_kg, propellant=knobs.propellant,
                        chamber_bar=knobs.chamber_bar,
                        nose_shape=knobs.nose_shape,
                        twr_by_stage=list(knobs.twr_by_stage),
                        max_stages=knobs.max_stages,
                        of_ratio=knobs.of_ratio)
    finally:
        _pl.STRUCT_COEFF, _pl.MAX_STAGE_MR = saved_coeff, saved_mr
        _pl.clear_plan_cache()


def evaluate(payload_kg: float, apogee_km: float, knobs: Knobs,
             limits: dict | None = None) -> Evaluation:
    """Design the vehicle at these knob settings and check every discipline."""
    from cadflow.planner import plan

    lim = dict(DEFAULT_LIMITS)
    if limits:
        lim.update(limits)

    p = _plan_at_coeff(apogee_km, payload_kg, knobs)
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

    # Does the structure close? The vehicle is sized at an asserted structural
    # coefficient; its own flown loads imply another. When the second exceeds
    # the first the design is lighter on paper than it can be built, which is
    # the failure the design packet reports for this mission (-27.9 kg of
    # slack) and the one this loop was previously unable to see, let alone act
    # on -- it checked loads, heat flux, coolant margin and skin temperature,
    # and nothing about whether the vehicle could contain itself.
    try:
        from cadflow.planner import STRUCT_COEFF, required_struct_coeff

        asserted = float(knobs.struct_coeff or STRUCT_COEFF)
        required = required_struct_coeff(p, material_id=knobs.skin_material)
        ev.struct_coeff_asserted = asserted
        ev.struct_coeff_required = required
        if required > asserted * (1.0 + lim["struct_closure_tol"]):
            ev.violations.append(Violation(
                "structure", "mass closure", required, asserted,
                "raise structural coefficient"))
    except Exception:  # noqa: BLE001 - closure is a gate, not a crash
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
    # The temperature the skin must survive is a property of the material the
    # skin is made of, not a global constant. It was a constant (450 K, which is
    # Al-2219) and the violation it raised said "no knob here" -- so a vehicle
    # reaching 854 K was declared unfixable while the catalogue held titanium at
    # 670 K and Inconel at 920 K.
    try:
        skin = _material(knobs.skin_material)
        skin_limit = float(skin.max_service_temp_k)
        ev.skin_material = skin.material_id
    except KeyError:
        _cap = lim.get("skin_temp_k")
        skin_limit = float(_cap) if _cap is not None else float(_NO_GATE_K)
    # The physical limit is what the material survives. `limits["skin_temp_k"]`
    # remains meaningful in two ways: a *lower* value is a caller demanding more
    # margin than the alloy needs, and an absurd value disables the gate for
    # callers testing something else. Reading the material and ignoring the
    # limit entirely -- which is what this did at first -- silently took that
    # switch away from every caller that had been using it.
    # None means "no extra cap": the material's own rating governs and the gate
    # is live. Only an explicit _NO_GATE_K disables it, which is what a caller
    # testing something unrelated to heating passes.
    caller_cap = lim.get("skin_temp_k")
    gate_disabled = caller_cap is not None and float(caller_cap) >= _NO_GATE_K
    if caller_cap is not None and not gate_disabled:
        skin_limit = min(skin_limit, float(caller_cap))
    ev.skin_limit_k = skin_limit
    if ev.skin_temp_k > skin_limit and not gate_disabled:
        upgrade = coolest_capable_material(ev.skin_temp_k, knobs.skin_material)
        if knobs.use_tps:
            # The caller has chosen to protect the structure rather than skin it
            # in something that survives bare. Size the blanket to hold the
            # backface at what this alloy can take; if nothing catalogued can,
            # _size_tps_for returns None and the violation stands.
            ev.tps = _size_tps_for(ev, knobs, skin_limit)
            # Covered when a blanket was sized and weighed, or when the sizing
            # says none is needed. Anything else leaves the skin unprotected and
            # the violation has to stand -- a knob that silently succeeds when
            # it cannot do its job is worse than no knob.
            covered = bool(ev.tps) and (
                ev.tps.get("required") is False
                or ev.tps.get("mass_kg") is not None)
            if not covered:
                ev.violations.append(Violation(
                    "aeroheating", "peak skin temperature", ev.skin_temp_k,
                    skin_limit, "no thermal protection covers this"))
        elif upgrade is not None:
            ev.violations.append(Violation(
                "aeroheating", "peak skin temperature", ev.skin_temp_k,
                skin_limit, "upgrade skin material"))
        else:
            # No alloy survives. That used to end the loop with "needs thermal
            # protection, no knob here" -- a correct diagnosis and a dead end.
            # Thermal protection IS a knob: a blanket or tile over a cooler
            # structure, sized by how long the heating lasts, paid for in areal
            # mass that the closure constraint then has to accommodate.
            ev.tps = _size_tps_for(ev, knobs, skin_limit)
            if ev.tps is None:
                ev.violations.append(Violation(
                    "aeroheating", "peak skin temperature", ev.skin_temp_k,
                    skin_limit,
                    "exceeds every alloy and every catalogued TPS; "
                    "needs a different trajectory"))

    # After the gate, because the gate is what sizes it.
    #
    # This assignment used to sit sixty lines earlier, reading ev.tps before
    # anything had set it, so tps_mass_kg was always zero. That is the third
    # instance of the same ordering error in this file: a value read above the
    # code that produces it. Combined with the mass never being charged
    # downstream, thermal protection was free twice over -- never weighed, and
    # never even computed.
    if ev.tps:
        ev.tps_mass_kg = float(ev.tps.get("mass_kg") or 0.0)
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
    struct = knobs.struct_coeff
    material = knobs.skin_material

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
        elif v.remedy == "upgrade skin material":
            # Take the lightest alloy that survives the temperature actually
            # reached. The density it brings lands in the structural sizing, so
            # the next iteration re-checks closure carrying the real penalty --
            # titanium is 15% heavier here, Inconel 48%.
            pick = coolest_capable_material(v.value, knobs.skin_material)
            if pick is not None:
                material = pick.material_id
        elif v.remedy == "raise structural coefficient":
            # v.value is the coefficient the loads demand, v.limit the one the
            # vehicle was sized at. Adopting the demand outright overshoots: a
            # heavier structure needs more propellant, which needs more
            # structure, so the demand itself moves. Step most of the way and
            # let the next iteration re-measure -- this is the same fixed point
            # `plan_solved` walks, driven here one step per design cycle.
            struct = min(0.60, v.limit + _DAMPING * (v.value - v.limit))

    return replace(knobs, chamber_bar=chamber, twr_by_stage=tuple(twr),
                   struct_coeff=struct, skin_material=material)


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


def _choose_nose_shape(payload_kg, apogee_km, knobs, ev, limits):
    """Try each measured nose shape and keep the lightest feasible vehicle.

    Nose shape does not fix a violation, so it has no place in `remedy` -- it
    lowers gross mass, which is what the loop is otherwise silent about. Run
    after convergence for the same reason material right-sizing is: a cheaper
    design that breaks a constraint is not cheaper.

    Only shapes with a measured CFD forebody factor are candidates. The
    closed-body slender-body factor is not used here because a launch vehicle
    has no tail closure, and it refuses cones outright.
    """
    from cadflow.wave_drag import CFD_FOREBODY_FACTOR

    best = (knobs, ev)
    for shape in sorted(CFD_FOREBODY_FACTOR, key=CFD_FOREBODY_FACTOR.get):
        if shape == knobs.nose_shape:
            continue
        trial = replace(knobs, nose_shape=shape)
        try:
            got = evaluate(payload_kg, apogee_km, trial, limits)
        except Exception:  # noqa: BLE001
            continue
        if not got.feasible or got.gross_kg <= 0:
            continue
        if got.gross_kg < best[1].gross_kg * 0.999:
            best = (trial, got)
    return best if best[0] is not knobs else None


def _right_size_material(payload_kg, apogee_km, knobs, ev, limits):
    """Swap down to the lightest alloy this design's own temperature allows.

    Returns None unless the swap both changes something and stays feasible --
    a cheaper design that violates a constraint is not a cheaper design.
    """
    want = best_material_for(ev.skin_temp_k, protected=knobs.use_tps,
                             gauge_limited=_walls_are_gauge_limited(ev, knobs))
    if want is None or want.material_id == knobs.skin_material:
        return None
    try:
        current = _material(knobs.skin_material)
    except KeyError:
        return None

    # Judge the swap by what the vehicle weighs, not by a figure of merit.
    #
    # This gated on yield over density improving, which is the right measure
    # only for a strength-driven wall. Every wall in this vehicle reads
    # "minimum gauge", where mass is density times a fixed thickness and
    # strength does not enter -- so the guard would reject aluminium at 2700 in
    # favour of keeping Inconel at 8190, on a comparison neither wall responds
    # to.
    #
    # It also settles the circularity honestly. A weaker alloy can push a wall
    # out of the gauge regime into strength, making it thicker rather than
    # lighter; flying the trial and comparing charged mass catches that without
    # anyone having to reason about it.
    trial = replace(knobs, skin_material=want.material_id)
    got = evaluate(payload_kg, apogee_km, trial, limits)
    if not got.feasible:
        return None
    if _charged_mass(got) >= _charged_mass(ev) - 1e-9:
        return None
    return (trial, got)


def _charged_mass(ev) -> float:
    """Gross plus the thermal protection, which gross does not include.

    Comparing bare gross across a TPS trade would hand the blanket branch a
    free win: the structure it protects is lighter and the blanket that makes
    that possible would not appear on either side of the comparison.
    """
    return float(ev.gross_kg) + float(getattr(ev, "tps_mass_kg", 0.0) or 0.0)


def _walls_are_gauge_limited(ev, knobs) -> bool:
    """Is the wall as thin as the shop can make it, or as thin as the load allows?

    Decides which way to rank the catalogue, so it has to be read from the
    sizing rather than assumed. Falls back to False -- the strength ranking,
    which is what this did before -- when the sizing is unavailable, because
    that is the behaviour every existing caller was tuned against.
    """
    plan = getattr(ev, "plan", None)
    if plan is None or not getattr(plan, "stack", None):
        return False
    try:
        from cadflow.structural_sizing import stage_structural_mass

        mat = _material(knobs.skin_material)
        _total, parts = stage_structural_mass(
            float(plan.stack[0].prop_mass_kg),
            max(0.10, (ev.gross_kg / 1000.0) ** (1 / 3) * 0.55) / 2.0,
            float(getattr(plan, "trajectory", {}).get("liftoff_thrust_n")
                  or 0.0) or 1.0,
            density_kg_m3=float(mat.density_kg_m3),
            yield_pa=float(mat.yield_mpa) * 1e6 if mat.yield_mpa else None,
            # youngs_modulus_gpa, not youngs_gpa. Written as a getattr with a
            # None default the first time, which would have silently passed no
            # modulus at all and let the buckling term size itself off a
            # default -- a wrong number rather than an error.
            modulus_pa=float(mat.youngs_modulus_gpa) * 1e9
            if mat.youngs_modulus_gpa else None)
    except Exception:  # noqa: BLE001
        return False
    drivers = [str(x.get("driver", "")) for x in parts if "driver" in x]
    if not drivers:
        return False
    return sum(d == "minimum gauge" for d in drivers) > len(drivers) / 2


def _tankage_shares(ev, knobs):
    """What fraction of each stage's structural allowance its tankage costs.

    Returns None when the pressurisation model is unavailable, so this stays an
    addition rather than a gate on designing anything.
    """
    try:
        from cadflow.assembly import TAPER_PER_STAGE as _tap
        from cadflow.pressurization import (
            stage_feasibility, stage_pressurisation)
    except Exception:  # noqa: BLE001
        return None

    stack = getattr(ev.plan, "stack", None)
    if not stack:
        return None
    # _material is this module's own lookup, already used by the material
    # right-sizing step. Reaching for a second one would be how the alloy the
    # loop selected and the alloy the domes are weighed in come to disagree.
    try:
        mat = _material(knobs.skin_material)
        rho = float(mat.density_kg_m3)
        allow = float(mat.yield_mpa) * 1e6 if mat.yield_mpa else 280e6
    except Exception:  # noqa: BLE001
        rho, allow = 2700.0, 280e6

    # The same flight radius and taper the packet draws with. Dome mass goes as
    # R^2 at fixed gauge, so using one radius for every stage would overstate the
    # upper stages and could invent an infeasibility that is not there.
    base_r = max(0.10, (ev.gross_kg / 1000.0) ** (1 / 3) * 0.55) / 2.0
    rows = []
    for i, st in enumerate(stack):
        rows.append(stage_pressurisation(
            stage=i + 1, propellant_mass_kg=float(st.prop_mass_kg),
            radius_m=base_r * (_tap ** i), wall_m=0.0008,
            acceleration_g=1.0, head_height_m=1.0,
            wall_density_kg_m3=rho, wall_allowable_pa=allow))
    return stage_feasibility(stack, rows)


def _afford_tankage(payload_kg, apogee_km, knobs, ev, limits):
    """Drop a stage the vehicle cannot pay for the tank of.

    The planner chooses its architecture by gross mass, and by that measure more
    stages are almost always better. What gross mass cannot see is that a
    structural coefficient is a fraction while minimum gauge is not, so the
    smallest stage on a tall stack can end up owing more for its tank ends alone
    than its whole structural allowance. For 25 kg to 4,000 km the fourth stage
    wants 175% of its allowance before any skin, plumbing or avionics.

    Caps the stack one stage shorter and keeps the result only if it both closes
    the mission and is actually affordable. A lighter design that cannot be
    built is not a lighter design -- the same rule _right_size_material follows.
    """
    shares = _tankage_shares(ev, knobs)
    if not shares or all(f["feasible"] for f in shares):
        return None
    n = len(shares)
    # Try each shorter architecture in turn. Dropping one stage may not be
    # enough, and stopping at the first attempt would report "no fix exists"
    # for a vehicle that a two-stage stack closes comfortably.
    tried = []
    for cap in range(n - 1, 0, -1):
        trial = replace(knobs, max_stages=cap)
        got = evaluate(payload_kg, apogee_km, trial, limits)
        if got.plan is None:
            tried.append(f"{cap} stage(s): no architecture closes the mission")
            continue
        if not got.feasible:
            tried.append(f"{cap} stage(s): closes but violates "
                         f"{len(got.violations)} constraint(s)")
            continue
        trial_shares = _tankage_shares(got, trial)
        if trial_shares and all(f["feasible"] for f in trial_shares):
            return {"fixed": True, "knobs": trial, "evaluation": got,
                    "before": shares, "after": trial_shares}
        worst = max(f["fraction_of_allowance"] for f in trial_shares) \
            if trial_shares else float("nan")
        tried.append(f"{cap} stage(s): closes, but its smallest stage still "
                     f"needs {100*worst:.0f}% of its allowance for tankage")

    # Before giving up: protect the structure instead of shortening the stack.
    #
    # The alloy is forced by the airflow, and its density is what the tank ends
    # cannot afford. A blanket moves the thermal problem off the structure and
    # lets it be aluminium, which is the one lever that acts directly on the
    # quantity failing. It costs mass -- about ten kilos of aerogel here -- so
    # it is tried after the architecture changes that cost nothing, and only
    # when it actually fixes what it was reached for.
    if not knobs.use_tps:
        # Take the lighter alloy in the same move.
        #
        # A blanket over the alloy the airflow forced buys nothing: the whole
        # benefit is that the structure no longer has to survive the airflow, so
        # it can be aluminium. Setting use_tps alone left rene-41 in place and
        # the tankage unchanged at 141%, because evaluate does not re-run
        # material selection -- that happens in _right_size_material, a step
        # this one does not reach.
        # getattr, because this reaches into an object it does not own and the
        # step is an enhancement rather than a gate. A real Evaluation always
        # carries skin_temp_k; a caller passing something simpler should get no
        # thermal repair, not an AttributeError from a repair path.
        _light = best_material_for(
            float(getattr(ev, "skin_temp_k", 0.0) or 0.0), protected=True,
            gauge_limited=_walls_are_gauge_limited(ev, knobs))
        trial = replace(knobs, use_tps=True,
                        skin_material=(_light.material_id if _light
                                       else knobs.skin_material))
        got = evaluate(payload_kg, apogee_km, trial, limits)
        if got.plan is not None and got.feasible:
            trial_shares = _tankage_shares(got, trial)
            if trial_shares and all(f["feasible"] for f in trial_shares):
                # Name what actually changed. "Thermal protection" alone hides
                # the substantive half of this repair: the blanket is only worth
                # its mass because it lets the structure be a lighter alloy, and
                # a reader who is not told the material moved cannot judge the
                # result.
                _tps_kg = float(getattr(got, "tps_mass_kg", 0.0) or 0.0)
                _how = (f"a {_tps_kg:.1f} kg blanket, which frees the structure "
                        f"from {knobs.skin_material} to {trial.skin_material}"
                        if trial.skin_material != knobs.skin_material
                        else f"a {_tps_kg:.1f} kg blanket")
                return {"fixed": True, "knobs": trial, "evaluation": got,
                        "before": shares, "after": trial_shares, "how": _how}
            tried.append(
                "thermal protection: the structure survives but its smallest "
                "stage still cannot afford its tankage")
        else:
            tried.append("thermal protection: no catalogued blanket closes it")

    # No shorter stack works. That is a conclusion, not a silence.
    #
    # Returning None here would leave the loop reporting a converged design
    # whose smallest stage cannot be built, with no record that an alternative
    # was even looked for. The two constraints genuinely oppose: the mission
    # needs this many stages to reach its apogee, and this many stages cannot
    # afford their tanks. Saying so is the useful output -- it tells a reader
    # the mission is over-specified for the technology rather than that the
    # loop gave up.
    worst = max(f["fraction_of_allowance"] for f in shares)
    return {
        "fixed": False,
        "conflict": (
            f"the mission needs {n} stages to reach its apogee and a "
            f"{n}-stage stack cannot afford its own tankage: stage "
            f"{max(shares, key=lambda f: f['fraction_of_allowance'])['stage']} "
            f"needs {100*worst:.0f}% of its structural allowance for tank ends "
            f"alone. Shorter architectures were tried and none closed: "
            + "; ".join(tried)),
        "before": shares,
        "alternatives": tried,
    }


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
            # Feasible is not the same as right-sized. The material knob only
            # ever moved upward, so a design that cooled as it grew keeps the
            # alloy its hottest iterate needed. Try the one this vehicle's
            # actual temperature calls for and keep it only if it still closes.
            trimmed = _right_size_material(payload_kg, apogee_km, knobs, ev, limits)
            if trimmed is not None:
                knobs, ev = trimmed
                history.append({"iteration": i + 1, "note": "material right-sized",
                                "skin_material": knobs.skin_material,
                                "gross_kg": ev.gross_kg, "stages": ev.stages,
                                "violations": []})
            nosed = _choose_nose_shape(payload_kg, apogee_km, knobs, ev, limits)
            if nosed is not None:
                before = ev.gross_kg
                knobs, ev = nosed
                history.append({"iteration": i + 1, "note": "nose shape chosen",
                                "nose_shape": knobs.nose_shape,
                                "gross_kg": ev.gross_kg,
                                "saved_kg": round(before - ev.gross_kg, 1),
                                "stages": ev.stages, "violations": []})
            # An architecture the mass fractions permit and physics does not.
            #
            # Every step above trades mass against mass. This one asks a
            # different question -- whether the smallest stage can pay for its
            # own pressure vessel -- and it is the only one that can reject an
            # architecture outright rather than resize it.
            afforded = _afford_tankage(payload_kg, apogee_km, knobs, ev, limits)
            tankage_conflict = None
            if afforded is not None and afforded["fixed"]:
                before_g, before_n = ev.gross_kg, ev.stages
                knobs, ev = afforded["knobs"], afforded["evaluation"]
                history.append({
                    "iteration": i + 1,
                    "note": (f"tankage repaired by {afforded['how']}"
                             if afforded.get("how") else
                             "stage dropped: smallest stage could not afford "
                             "its tankage"),
                    "stages": ev.stages,
                    "stages_before": before_n,
                    "gross_kg": ev.gross_kg,
                    "cost_kg": round(ev.gross_kg - before_g, 1),
                    "worst_share_before": round(
                        max(f["fraction_of_allowance"]
                            for f in afforded["before"]), 3),
                    "worst_share_after": round(
                        max(f["fraction_of_allowance"]
                            for f in afforded["after"]), 3),
                    "violations": []})
            elif afforded is not None:
                tankage_conflict = afforded["conflict"]
                history.append({
                    "iteration": i + 1,
                    "note": "tankage unaffordable and no shorter stack closes",
                    "stages": ev.stages,
                    # gross_kg on every entry, unchanged here because nothing
                    # was repaired. Callers read history[-1]["gross_kg"] to see
                    # what the loop cost, and an entry missing the field is a
                    # KeyError rather than a smaller number -- a record whose
                    # shape its readers do not expect.
                    "gross_kg": ev.gross_kg,
                    "conflict": tankage_conflict,
                    "alternatives": afforded["alternatives"],
                    "violations": []})

            # `conflict` is present on every return path. It used to appear
            # only when the loop gave up, so a caller had to know which branch
            # ran before it could read the result -- and once the loop gained
            # enough knobs to escape a conflict, four callers that had always
            # taken the giving-up path started raising KeyError.
            return {"converged": True, "iterations": i + 1, "evaluation": ev,
                    "knobs": knobs, "history": history,
                    "conflict": tankage_conflict}

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
