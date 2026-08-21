"""Mission planner: choose the architecture, then size it.

Everything before this took the architecture as given -- two stages, split 0.72,
solved against a precomputed grid of payload and apogee. That answers the
specifications it was built for and nothing else, and it never *decides*
anything: the stage count and split were conventions in the source, not
conclusions from the mission.

This is the agent-planner layer from the statement of intent. Given any payload
and altitude, it:

  1. estimates the delta-v the mission needs,
  2. chooses a stage count from that, escalating when a simpler architecture
     would demand a mass ratio no structure can carry,
  3. optimises the propellant split across stages by search rather than
     convention,
  4. reports why it chose what it chose.

Architecture selection is a real decision here: a single stage is preferred
when it closes, because staging costs structure and complexity, and stages are
added only when the required mass ratio exceeds what the structural coefficient
allows. That is the same reasoning a person would apply, made explicit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cadflow.multistage import Stage, integrate_stack
from cadflow.wave_drag import cd_multiplier
from generate_propulsion_trajectory_corpus import (
    G0, P0, PROPELLANTS, nozzle_performance,
)

STRUCT_COEFF = 0.14
CD = 0.42

#: Nose geometry the reference CD above is taken to describe. Shape reaches the
#: trajectory from here: cd_multiplier returns exactly 1.0 for a tangent ogive,
#: so the default vehicle flies precisely as it always did, and any other nose
#: moves the wave-drag share of CD by the amount slender-body theory predicts.
NOSE_SHAPE = "ogive"
NOSE_FINENESS = 3.0


def drag_coefficient(nose_shape: str = NOSE_SHAPE,
                     nose_fineness: float = NOSE_FINENESS) -> float:
    """Zero-lift CD for a vehicle with the given nose."""
    return CD * cd_multiplier(nose_shape, nose_fineness)
# Above this mass ratio a stage's structure cannot be closed at STRUCT_COEFF:
# dry mass would have to be smaller than the tank holding the propellant.
MAX_STAGE_MR = 1.0 / STRUCT_COEFF * 0.62


@dataclass
class Plan:
    stages: int
    split: list[float]
    gross_kg: float
    achieved_km: float
    stack: list = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)


# Earth gravitational parameter, for the energy form of the apogee relation.
MU_EARTH = 3.986004418e14
R_EARTH_M = 6.371e6


def required_delta_v(apogee_km: float) -> float:
    """Delta-v for a ballistic apogee, including gravity and drag losses.

    This used sqrt(2 g h), which assumes gravity is constant all the way up.
    That is fine low down and badly wrong high up, because g falls as 1/r^2:
    at 12,000 km it overstates the burnout speed by 1.7x and at 40,000 km by
    2.7x. The planner then demanded architectures the mission did not need, and
    declared reachable missions impossible.

    The energy form is exact for a ballistic arc:
        v = sqrt( 2 mu (1/R - 1/(R + h)) )
    """
    h = apogee_km * 1000.0
    v = math.sqrt(2.0 * MU_EARTH * (1.0 / R_EARTH_M - 1.0 / (R_EARTH_M + h)))
    return v * 1.35


def build_stack_n(total_prop: float, fractions: list[float], payload: float,
                  pc: float, prop: str) -> tuple[list[Stage], float, list[tuple]]:
    """Build an N-stage stack given the propellant split."""
    gamma, tc, mol = PROPELLANTS[prop]
    props = [total_prop * f for f in fractions]
    structs = [p * STRUCT_COEFF / (1 - STRUCT_COEFF) for p in props]

    # mass each stage must lift = payload + everything above it
    supported = []
    running = payload
    for i in range(len(props) - 1, -1, -1):
        running += props[i] + structs[i]
        supported.append(running)
    supported = list(reversed(supported))
    gross = supported[0]

    # expansion ratio rises with stage number: lower stages fight ambient
    eps_by_stage = [12.0, 30.0, 60.0, 80.0]
    twr_by_stage = [4.5, 3.0, 2.2, 2.0]

    stages = []
    for i, (p_i, s_i) in enumerate(zip(props, structs)):
        eps = eps_by_stage[min(i, len(eps_by_stage) - 1)]
        twr = twr_by_stage[min(i, len(twr_by_stage) - 1)]
        u = nozzle_performance(chamber_pressure=pc, chamber_temp=tc,
                               expansion_ratio=eps, throat_area=1.0,
                               gamma=gamma, mol_mass=mol, ambient_pressure=P0)
        at = (twr * supported[i] * G0) / max(u["thrust"], 1e-9)
        stages.append(Stage(p_i, s_i, at, pc, eps, gamma, tc, mol))
    return stages, gross, list(zip(props, structs))


def fly_plan(total_prop: float, fractions: list[float], payload: float,
             pc: float, prop: str, pitchover_deg: float = 3.0,
             nose_shape: str = NOSE_SHAPE,
             nose_fineness: float = NOSE_FINENESS):
    stages, gross, split = build_stack_n(total_prop, fractions, payload, pc, prop)
    dia = max(0.10, (gross / 1000.0) ** (1.0 / 3.0) * 0.55)
    r = integrate_stack(stages, payload,
                        cd=drag_coefficient(nose_shape, nose_fineness),
                        ref_area_m2=math.pi * (dia / 2) ** 2, dt=0.2,
                        pitchover_angle=math.radians(pitchover_deg))
    return r["apogee_m"] / 1000.0, gross, stages, split, r


def solve_for(target_km: float, fractions: list[float], payload: float,
              pc: float, prop: str, tol: float = 0.02,
              pitchover_deg: float = 3.0,
              nose_shape: str = NOSE_SHAPE,
              nose_fineness: float = NOSE_FINENESS):
    """Bisect total propellant until the stack flies the target."""
    # Near-escape missions need mass ratios around 95, so the upper bound has
    # to allow a vehicle that heavy or the search reports no solution for a
    # mission that is merely expensive. 40,000 km already needs ~650x payload.
    lo, hi = payload * 0.3, payload * 400000.0
    a_lo, *_ = fly_plan(lo, fractions, payload, pc, prop, pitchover_deg, nose_shape, nose_fineness)
    a_hi, *_ = fly_plan(hi, fractions, payload, pc, prop, pitchover_deg, nose_shape, nose_fineness)
    if not (a_lo <= target_km <= a_hi):
        return None

    # Coarse pre-scan to tighten the bracket before bisecting. The initial
    # bracket spans six orders of magnitude in propellant while, near escape,
    # the band that lands on a given apogee can be a few hundred kg wide:
    # apogee climbs 2,761 -> 12,766 -> 25,228 km and then goes to escape
    # between 1,000 and 2,000 kg. Bisection from the full bracket steps past
    # that band. Walking a log grid first finds the last point below target and
    # the first at or above it, which brackets the crossing tightly whatever
    # its width.
    n_scan = 40
    prev_x, prev_a = lo, a_lo
    for i in range(1, n_scan + 1):
        x = lo * (hi / lo) ** (i / n_scan)
        a, *_ = fly_plan(x, fractions, payload, pc, prop, pitchover_deg, nose_shape, nose_fineness)
        if a >= target_km:
            lo, hi = prev_x, x
            break
        prev_x, prev_a = x, a
    # Keep the closest iterate, not the latest. Bisection probes both sides, and
    # a probe near the top of the bracket can be an escape trajectory with
    # infinite apogee; keeping the last one then throws away a solve that had
    # already converged.
    # Near escape, apogee is extraordinarily sensitive to delta-v: for a 5 kg
    # payload the four-stage bracket runs from 5.9 km to escape, so the band
    # that lands on 40,000 km is very narrow and 26 bisection steps cannot
    # resolve it. Each evaluation is a ~0.1 s trajectory, so more steps are
    # cheap.
    best = None
    best_err = float("inf")
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        a, gross, stages, split, r = fly_plan(mid, fractions, payload, pc, prop,
                                             pitchover_deg, nose_shape, nose_fineness)
        err = abs(a - target_km) / target_km if math.isfinite(a) else float("inf")
        if err < best_err:
            best_err, best = err, (mid, a, gross, stages, split, r)
        if err < tol:
            break
        if a < target_km:
            lo = mid
        else:
            hi = mid
    if best is None or best_err > 0.10:
        return None
    return best


def _splits_for(n: int) -> list[list[float]]:
    """Candidate propellant splits for an n-stage stack.

    Previously this hardcoded one, two and "else three" entries, so a candidate
    of four or five silently built a three-stage vehicle and missions needing
    more than three stages reported no solution. Generalised: each stage takes a
    geometric share of what remains, swept over the ratio, which brackets the
    usual optimum without an exhaustive search.
    """
    if n <= 1:
        return [[1.0]]
    out: list[list[float]] = []
    for ratio in (0.50, 0.58, 0.66, 0.74, 0.82):
        fr: list[float] = []
        remaining = 1.0
        for i in range(n - 1):
            take = remaining * ratio
            fr.append(take)
            remaining -= take
        fr.append(remaining)
        out.append(fr)
    return out


def plan(target_km: float, payload_kg: float, *, propellant: str = "lox_rp1",
         chamber_bar: float = 55.0, nose_shape: str = NOSE_SHAPE,
         nose_fineness: float = NOSE_FINENESS) -> Plan | None:
    """Choose an architecture for this mission and size it."""
    pc = chamber_bar * 1e5
    rationale: list[str] = []
    dv = required_delta_v(target_km)
    gamma, tc, mol = PROPELLANTS[propellant]
    u = nozzle_performance(chamber_pressure=pc, chamber_temp=tc,
                           expansion_ratio=20.0, throat_area=1.0,
                           gamma=gamma, mol_mass=mol, ambient_pressure=0.0)
    isp = u["isp"]
    mr_single = math.exp(dv / (isp * G0))
    rationale.append(
        f"mission needs about {dv/1000:.2f} km/s; at Isp {isp:.0f} s a single "
        f"stage would need mass ratio {mr_single:.2f}")

    if mr_single <= MAX_STAGE_MR:
        rationale.append(
            f"that is within what a stage can close at structural coefficient "
            f"{STRUCT_COEFF} (limit {MAX_STAGE_MR:.2f}), so try a single stage first")
        candidates = [1, 2, 3]
    else:
        n_needed = max(2, math.ceil(math.log(mr_single) / math.log(MAX_STAGE_MR)))
        rationale.append(
            f"that exceeds the {MAX_STAGE_MR:.2f} a stage can close, so at least "
            f"{n_needed} stages are required")
        candidates = [n_needed, n_needed + 1, n_needed + 2]

    best_plan = None
    for n in candidates:
        # search the split; equal-ish splits with more in the booster are the
        # usual optimum, so sample around that rather than exhaustively
        splits = _splits_for(n)
        for fr in splits:
          # The ascent profile is part of the design, not a constant. A 3 deg
          # pitchover suits a sounding rocket; on a high-energy flight it turns
          # the vehicle horizontal early, so beyond some point extra propellant
          # buys downrange instead of altitude and apogee stops responding --
          # non-monotonic, which no amount of bisection resolves. Steeper
          # profiles are searched alongside the mass.
          # Steeper first: high-energy missions need a more vertical profile,
          # and stopping at the first profile that closes keeps the search from
          # multiplying out to thousands of trajectory integrations.
          for pitch in (1.5, 3.0, 6.0):
            got = solve_for(target_km, fr, payload_kg, pc, propellant,
                            pitchover_deg=pitch, nose_shape=nose_shape,
                            nose_fineness=nose_fineness)
            if got is None:
                continue
            tp, achieved, gross, stages, split, r = got
            if best_plan is None or gross < best_plan.gross_kg:
                # stage count comes from the split actually built, not from the
                # candidate n. The split table only generates up to 3 entries,
                # so a candidate of 4 or 5 was being reported against a 3-stage
                # vehicle -- a design packet claiming five stages for a stack
                # that has three.
                best_plan = Plan(stages=len(fr), split=fr, gross_kg=gross,
                                 achieved_km=achieved, stack=stages,
                                 trajectory=r, rationale=list(rationale))
            break   # this profile closed the mission; move to the next split
        if best_plan is not None and n == candidates[0]:
            best_plan.rationale.append(
                f"{best_plan.stages} stage(s) closes the mission at "
                f"{best_plan.gross_kg:.1f} kg "
                f"gross; simpler architectures are preferred so this is selected")
            break

    if best_plan is None:
        rationale.append("no architecture up to 3 stages closes this mission")
        return None
    if not any("selected" in s for s in best_plan.rationale):
        best_plan.rationale.append(
            f"{best_plan.stages} stages selected, split "
            f"{[round(f,2) for f in best_plan.split]}, "
            f"{best_plan.gross_kg:.1f} kg gross")
    return best_plan


def plan_sized(target_km: float, payload_kg: float, *,
               propellant: str = "lox_rp1", chamber_bar: float = 55.0,
               max_iters: int = 8, tol: float = 0.01):
    """Plan a mission with structural mass solved rather than assumed.

    STRUCT_COEFF is a constant asserted in the source. Real structural mass
    depends on the loads the stage carries and the radius it carries them at,
    and those depend on the vehicle's mass -- which depends on the structural
    mass. It is a fixed point, so iterate it: size the vehicle at the current
    coefficient, size the walls from the resulting loads, recompute the
    coefficient, repeat.

    Returns the plan plus the coefficient it converged to, which is the number
    that was previously guessed.
    """
    import math as _m

    from cadflow.structural_sizing import stage_structural_mass

    # Save and restore around the whole solve. The restore used to sit after
    # the loop, outside any finally, and wrote a hardcoded 0.14 rather than
    # whatever had been there -- so an exception anywhere in the iteration left
    # the module permanently mis-configured, and every design made afterwards in
    # the same process silently used the last iterate's coefficient. It happens
    # to restore correctly on the happy path, which is exactly what makes that
    # kind of bug survive.
    _saved_coeff = STRUCT_COEFF
    _saved_mr = MAX_STAGE_MR

    coeff = STRUCT_COEFF
    history = []
    p = None
    try:
      for it in range(max_iters):
          globals()["STRUCT_COEFF"] = coeff
          globals()["MAX_STAGE_MR"] = 1.0 / coeff * 0.62
          p = plan(target_km, payload_kg, propellant=propellant,
                   chamber_bar=chamber_bar)
          if p is None:
              return None, coeff, history

          radius_m = max(0.05, (p.gross_kg / 1000.0) ** (1.0 / 3.0) * 0.55 / 2.0)
          # Load each stage's structure with the acceleration this iterate's own
          # trajectory actually reaches, not an assumed 4.5 g and 3.0 g. Those
          # assumptions understated the load by a factor of three to five -- the
          # flown peaks for a two-stage vehicle are 15.4 g and 16.5 g -- which is
          # why this fixed point converged to a structural coefficient around
          # 0.05 when real sounding rockets sit at 0.10 to 0.25. The structure was
          # coming out light because it was being sized for a flight it does not
          # make.
          flown_g = p.trajectory.get("max_axial_g_by_stage") or []
          total_struct = 0.0
          total_prop = 0.0
          for i, st in enumerate(p.stack):
              supported = payload_kg + sum(s.prop_mass_kg + s.struct_mass_kg
                                           for s in p.stack[i:])
              g_load = flown_g[i] if i < len(flown_g) else (4.5 if i == 0 else 3.0)
              thrust = supported * G0 * g_load
              m_struct, _parts = stage_structural_mass(
                  st.prop_mass_kg, radius_m, thrust)
              total_struct += m_struct
              total_prop += st.prop_mass_kg

          new_coeff = total_struct / max(total_struct + total_prop, 1e-6)
          new_coeff = min(0.60, max(0.03, new_coeff))
          history.append({"iter": it, "coeff_in": coeff, "coeff_out": new_coeff,
                          "gross_kg": p.gross_kg, "stages": p.stages,
                          "struct_kg": total_struct, "prop_kg": total_prop})
          if abs(new_coeff - coeff) / max(coeff, 1e-6) < tol:
              coeff = new_coeff
              break
          # damped update: the map can oscillate when buckling and strength swap
          coeff = 0.5 * coeff + 0.5 * new_coeff

    finally:
        globals()["STRUCT_COEFF"] = _saved_coeff
        globals()["MAX_STAGE_MR"] = _saved_mr
    return p, coeff, history
