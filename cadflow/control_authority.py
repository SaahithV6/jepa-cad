"""Thrust vector control: is there enough authority, and a bandwidth to use it?

The design packet says twice that it does not size a control system -- once
under the bending modes, once under slosh. This is that gap, and it is the one
that ties the others together, because a launch vehicle autopilot is squeezed
from both sides at once.

From below, it has to be fast enough to fly the vehicle: hold attitude against
the aerodynamic moment at maximum dynamic pressure, when a wind gust puts the
body at incidence. From above, it has to stay clear of the structure. Push
bandwidth up toward the first bending mode and the autopilot starts driving the
airframe instead of steering it; push it into a slosh mode and it stirs the
propellant. Both of those frequencies are now computed elsewhere in this
package, so the window between them can finally be checked rather than assumed
to exist.

The authority question is separate and more basic: can the engine gimbal far
enough at all? A statically stable vehicle is not free. Its fins produce a
restoring moment that the engine has to overcome to point anywhere other than
into the wind, and the bigger the fins the more gimbal that costs. Sizing fins
for static margin without checking what they cost in control authority is how a
vehicle ends up stable and unsteerable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Gimbal range typical of a production launch vehicle engine, degrees. Beyond
#: this the actuator, the flex duct and the thrust structure all get expensive,
#: so a requirement above it is a design finding rather than a specification.
TYPICAL_GIMBAL_LIMIT_DEG = 8.0

#: Bandwidth is normally kept this far below the first flexible mode so the
#: autopilot does not excite it.
BENDING_SEPARATION_FACTOR = 5.0

#: And this far above the rigid-body response so it can actually fly.
RIGID_BODY_MARGIN = 3.0


@dataclass
class AuthorityResult:
    aero_moment_nm: float
    tvc_arm_m: float
    thrust_n: float
    required_gimbal_deg: float
    available_gimbal_deg: float
    utilisation: float
    statically_stable: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_authority(self) -> bool:
        return self.required_gimbal_deg <= self.available_gimbal_deg

    def as_dict(self) -> dict:
        return {
            "aero_moment_nm": round(self.aero_moment_nm, 1),
            "tvc_arm_m": round(self.tvc_arm_m, 3),
            "thrust_n": round(self.thrust_n, 1),
            "required_gimbal_deg": round(self.required_gimbal_deg, 2),
            "available_gimbal_deg": round(self.available_gimbal_deg, 2),
            "utilisation": round(self.utilisation, 3),
            "has_authority": self.has_authority,
            "statically_stable": self.statically_stable,
            "notes": list(self.notes),
        }


def aero_moment_nm(*, q_pa: float, reference_area_m2: float, cn_alpha: float,
                   alpha_rad: float, cp_station_m: float, cg_station_m: float
                   ) -> tuple[float, bool]:
    """Aerodynamic moment about the centre of gravity, and its sign.

    Stations are measured from the aft end. A centre of pressure *behind* the
    centre of gravity is the stable arrangement: the moment opposes incidence
    and pushes the nose back into the wind. That is helpful for stability and is
    exactly what the engine has to fight in order to point anywhere else, so the
    magnitude matters either way and the sign only says who is fighting whom.
    """
    normal_n = float(q_pa) * float(reference_area_m2) * float(cn_alpha) * float(alpha_rad)
    arm = float(cp_station_m) - float(cg_station_m)
    stable = arm < 0.0            # centre of pressure aft of centre of gravity
    return abs(normal_n * arm), stable


def check(*, q_pa: float, reference_area_m2: float, cn_alpha: float,
          alpha_rad: float, cp_station_m: float, cg_station_m: float,
          thrust_n: float, gimbal_station_m: float = 0.0,
          available_gimbal_deg: float = TYPICAL_GIMBAL_LIMIT_DEG
          ) -> AuthorityResult:
    """Gimbal deflection needed to trim the vehicle at this flight condition."""
    if thrust_n <= 0:
        raise ValueError("thrust must be positive to have any control authority")
    moment, stable = aero_moment_nm(
        q_pa=q_pa, reference_area_m2=reference_area_m2, cn_alpha=cn_alpha,
        alpha_rad=alpha_rad, cp_station_m=cp_station_m,
        cg_station_m=cg_station_m)
    arm = abs(float(cg_station_m) - float(gimbal_station_m))
    if arm <= 0:
        raise ValueError(
            "the gimbal sits at the centre of gravity, so it produces no "
            "moment and the vehicle cannot be steered by thrust vectoring")

    sin_delta = moment / (float(thrust_n) * arm)
    notes: list[str] = []
    if sin_delta >= 1.0:
        required = 90.0
        notes.append(
            f"no gimbal angle trims this condition: the aerodynamic moment "
            f"{moment:.0f} N m exceeds the {thrust_n * arm:.0f} N m the engine "
            f"can produce at full deflection")
    else:
        required = math.degrees(math.asin(sin_delta))

    if stable:
        notes.append(
            "The vehicle is statically stable, so this deflection is spent "
            "overcoming its own fins rather than correcting an instability. "
            "Larger fins buy static margin and cost control authority.")
    else:
        notes.append(
            "The vehicle is statically unstable; the deflection below is what "
            "keeps it pointed, not what steers it, and losing it is loss of "
            "the vehicle rather than loss of accuracy.")

    util = required / max(float(available_gimbal_deg), 1e-9)
    if util > 1.0:
        notes.append(
            f"Required deflection exceeds the {available_gimbal_deg:.1f} deg "
            f"assumed available. Either the fins come down, the engine gimbals "
            f"further, or the vehicle does not fly this condition.")
    return AuthorityResult(
        aero_moment_nm=moment, tvc_arm_m=arm, thrust_n=float(thrust_n),
        required_gimbal_deg=required,
        available_gimbal_deg=float(available_gimbal_deg),
        utilisation=util, statically_stable=stable, notes=tuple(notes))


#: Static margin below this is not enough to keep a finned vehicle pointed
#: through gusts, whatever it buys in control authority. Trading margin for
#: gimbal has to stop somewhere, and it stops here.
MIN_STATIC_MARGIN_CAL = 0.5


def trade_margin_for_authority(size_fins, *, q_pa: float,
                               reference_area_m2: float, alpha_rad: float,
                               cg_station_m: float, thrust_n: float,
                               body_diameter_m: float,
                               start_margin: float = 1.5,
                               available_gimbal_deg: float = TYPICAL_GIMBAL_LIMIT_DEG,
                               floor: float = MIN_STATIC_MARGIN_CAL) -> dict:
    """Shrink the fins until the engine can steer, or report that it cannot.

    A statically stable vehicle spends gimbal deflection fighting its own fins,
    so static margin and control authority are the same lever pulled in
    opposite directions. Sizing fins for margin alone -- which is what this
    project did -- optimises one end of that trade and discovers the other end
    only if somebody checks.

    ``size_fins`` is a callable taking a target margin and returning the fin
    dictionary for it, so this function does not need to know how fins are
    built or what nose it is behind.

    Searches downward from ``start_margin`` and stops at the first margin that
    fits, keeping as much stability as the engine allows rather than shrinking
    the fins as far as possible. Below ``floor`` it gives up and says so: a
    vehicle that can only be steered by making itself unstable is a different
    design, not a repaired one.
    """
    steps, margin = [], float(start_margin)
    while margin >= floor - 1e-9:
        fins = size_fins(margin)
        res = check(q_pa=q_pa, reference_area_m2=reference_area_m2,
                    cn_alpha=float(fins["cna_total"]), alpha_rad=alpha_rad,
                    cp_station_m=float(fins["cp_z_m"]),
                    cg_station_m=cg_station_m, thrust_n=thrust_n,
                    available_gimbal_deg=available_gimbal_deg)
        steps.append({"margin_cal": round(margin, 3),
                      "span_m": round(float(fins.get("span_m", 0.0)), 4),
                      "cna_total": round(float(fins["cna_total"]), 3),
                      "required_gimbal_deg": round(res.required_gimbal_deg, 2),
                      "has_authority": res.has_authority})
        if res.has_authority:
            return {"converged": True, "margin_cal": margin, "fins": fins,
                    "authority": res, "steps": steps,
                    "note": (f"Fins resized for {margin:.2f} calibers of static "
                             f"margin, down from {start_margin:.2f}, so the "
                             f"engine can trim the vehicle at max-Q within "
                             f"{available_gimbal_deg:.1f} degrees.")}
        margin -= 0.1

    return {"converged": False, "margin_cal": None, "fins": None,
            "authority": res, "steps": steps,
            "note": (f"No static margin between {floor:.2f} and "
                     f"{start_margin:.2f} calibers gives the engine enough "
                     f"authority at max-Q. The fins cannot be traded away out "
                     f"of this: the vehicle needs a larger gimbal range, more "
                     f"thrust, or a shape that does not demand so much fin.")}


def mode_disposition(*, crossover_hz: float, modes: dict) -> dict:
    """How each flexible mode has to be handled, given where it sits.

    A launch vehicle autopilot does not need every flexible mode to be far
    away. It needs to know which side of the crossover each one is on, because
    the two sides get opposite treatments and they are not interchangeable.

    A mode well above crossover is gain stabilised: attenuate it with a notch
    or a rolloff and the loop never excites it. That is cheap and routine, and
    it is what happens to the first bending mode here, which sits an order of
    magnitude up.

    A mode *below* crossover cannot be treated that way, because crossover is
    where the loop needs gain to fly the vehicle -- notching there removes the
    control authority along with the mode. Such a mode has to be phase
    stabilised: the controller models it and closes the loop around it with
    the right phase. That is demanding but entirely standard; Saturn V phase
    stabilised its slosh modes.

    This replaces a "usable bandwidth window" test that reported no band and
    read as a dead end. It was arithmetically right and engineeringly
    misleading: real vehicles fly with slosh near the control frequencies all
    the time. What they do not do is pretend a mode below crossover can be
    notched away.
    """
    cross = float(crossover_hz)
    if cross <= 0:
        raise ValueError("crossover frequency must be positive")

    out, needs_phase = {}, []
    for name, f in modes.items():
        f = float(f)
        if f <= 0:
            continue
        ratio = f / cross
        if ratio >= 3.0:
            how, note = "gain", (
                f"{ratio:.1f}x above crossover; a notch or rolloff keeps the "
                f"loop off it")
        elif ratio > 1.0:
            how, note = "gain, tight", (
                f"only {ratio:.1f}x above crossover, so a notch there eats "
                f"phase margin near the frequency the loop is working at")
        else:
            how, note = "phase", (
                f"below crossover at {ratio:.2f}x, where the loop needs gain "
                f"to fly the vehicle -- it cannot be notched out and the "
                f"controller has to model it")
            needs_phase.append(name)
        out[name] = {"hz": round(f, 3), "ratio_to_crossover": round(ratio, 3),
                     "stabilisation": how, "note": note}

    return {
        "crossover_hz": round(cross, 3),
        "modes": out,
        "requires_phase_stabilisation": needs_phase,
        "verdict": (
            f"{', '.join(needs_phase)} "
            f"{'sits' if len(needs_phase) == 1 else 'sit'} below crossover and "
            f"must be phase stabilised: the control design has to model "
            f"{'it' if len(needs_phase) == 1 else 'them'} rather than filter "
            f"{'it' if len(needs_phase) == 1 else 'them'} out. Standard "
            f"practice, and not something this packet can verify, since it does "
            f"not design a control system."
            if needs_phase else
            "Every flexible mode is above crossover and can be gain "
            "stabilised, so a conventional autopilot with rolloff suffices."),
    }


def rigid_body_pitch_hz(*, q_pa: float, reference_area_m2: float,
                        cn_alpha: float, cp_station_m: float,
                        cg_station_m: float, pitch_inertia_kg_m2: float
                        ) -> float:
    """Weathercock frequency: how fast the vehicle swings back into the wind.

    w^2 = q S CNa |x_cp - x_cg| / I

    Derived rather than assumed, because guessing it is easy and wrong by a lot.
    A first pass at this module took 0.3 Hz as a plausible-sounding value; the
    actual figure for the packet vehicle is 2.2 Hz, seven times higher, and the
    difference decides whether a slosh mode sits on top of it.

    Only meaningful for a statically stable vehicle. An unstable one diverges
    rather than oscillating, and the corresponding number is a time constant.
    """
    arm = float(cp_station_m) - float(cg_station_m)
    if arm >= 0:
        raise ValueError(
            "vehicle is statically unstable, so it has no pitch oscillation "
            "frequency; it diverges, and the relevant quantity is a doubling "
            "time rather than a frequency")
    if pitch_inertia_kg_m2 <= 0:
        raise ValueError("pitch inertia must be positive")
    omega_sq = (float(q_pa) * float(reference_area_m2) * float(cn_alpha)
                * abs(arm) / float(pitch_inertia_kg_m2))
    return math.sqrt(omega_sq) / (2.0 * math.pi)


def bandwidth_window(*, first_bending_hz: float, lowest_slosh_hz: float,
                     rigid_body_hz: float) -> dict:
    """The band a control system may occupy, if one exists.

    Bounded above by the flexible modes -- bending and slosh both -- and below
    by how fast the vehicle itself responds. When the upper bound falls under
    the lower one there is no bandwidth that both flies the vehicle and leaves
    the structure alone, and the answer is a notch filter, baffles, or a
    different vehicle. Reporting the numbers separately would let a reader
    assume a window that is not there.
    """
    upper = min(float(first_bending_hz) / BENDING_SEPARATION_FACTOR,
                float(lowest_slosh_hz) / BENDING_SEPARATION_FACTOR)
    lower = RIGID_BODY_MARGIN * float(rigid_body_hz)

    # Both bounds come from rules of thumb, so a verdict that rests on them is
    # worth less than one that survives them. This re-runs the comparison across
    # the range of factors an engineer might defend -- rigid-body margins from
    # 1.5 to 3, flexible separations from 1.5 (well baffled) to 5 (undamped) --
    # and records whether the answer ever changes.
    #
    # For the vehicle that prompted this it never does, and the reason is
    # sharper than any of the factors: the slosh mode sits at only 1.6 times the
    # rigid-body mode, so no bandwidth dominates one without exciting the other.
    # That is a fact about the vehicle. "Fails under our chosen rule" and "fails
    # under every rule" are different findings and should not read alike.
    _flex = min(float(first_bending_hz), float(lowest_slosh_hz))
    _verdicts = [
        (_flex / sep) > (rigid * float(rigid_body_hz))
        for rigid in (1.5, 2.0, 3.0) for sep in (1.5, 2.5, 5.0)]
    robust = all(v == _verdicts[0] for v in _verdicts)
    mode_ratio = _flex / max(float(rigid_body_hz), 1e-9)
    limiter = ("slosh" if float(lowest_slosh_hz) < float(first_bending_hz)
               else "first bending mode")
    exists = upper > lower
    return {
        "lower_bound_hz": round(lower, 4),
        "upper_bound_hz": round(upper, 4),
        "limited_by": limiter,
        "window_exists": exists,
        "robust_to_heuristics": robust,
        "flexible_over_rigid_ratio": round(mode_ratio, 3),
        "robustness_note": (
            f"The verdict holds across every rigid-body margin from 1.5 to 3 "
            f"and every flexible separation from 1.5 to 5, so it is a property "
            f"of the vehicle rather than of the factors chosen. The lowest "
            f"flexible mode sits at {mode_ratio:.2f} times the rigid-body mode; "
            f"below about 3 there is no bandwidth that dominates one without "
            f"exciting the other."
            if robust else
            f"This verdict depends on the separation factors used. The lowest "
            f"flexible mode is {mode_ratio:.2f} times the rigid-body mode, "
            f"close enough that a defensible change of rule reverses the "
            f"answer, so it should be settled by a coupled analysis rather "
            f"than by a frequency comparison."),
        "note": (
            f"Control bandwidth must sit above {lower:.2f} Hz to fly the "
            f"vehicle and below {upper:.2f} Hz to leave the {limiter} alone."
            if exists else
            # Which remedies apply depends on where the mode sits. If the
            # bandwidth needed is below the mode, damping helps and baffles are
            # a real option. If it is above, the mode is inside the band and
            # baffles are beside the point -- damping does not move a
            # frequency. Listing baffles either way had the packet recommending
            # them one line before explaining they could not work.
            f"No usable band: flying the vehicle needs at least {lower:.2f} Hz "
            f"while the {limiter} caps bandwidth at {upper:.2f} Hz. "
            + (f"The {limiter} lies inside the required band rather than above "
               f"it, so added damping cannot open it: this needs a notch filter "
               f"at that frequency, hardware that moves the mode, or an "
               f"autopilot designed to fly through it."
               if lower >= min(float(first_bending_hz), float(lowest_slosh_hz))
               else f"Damping would relax the separation requirement, so "
                    f"baffles or structural damping may open it; failing that "
                    f"it needs notch filtering or a different configuration.")),
    }


def phase_stabilisation_difficulty(damping_ratio: float) -> dict:
    """How hard a mode below crossover is to phase stabilise, given its damping.

    ``mode_disposition`` answers *whether* a mode needs phase stabilisation, and
    that answer depends only on which side of crossover it sits -- damping does
    not move a frequency. But it says nothing about how demanding the task is,
    and the difference is enormous.

    A lightly damped mode has a resonant peak of Q = 1/(2 zeta). A bare
    propellant tank at zeta = 0.0005 gives Q = 1000: a needle-sharp resonance
    where a few degrees of phase error at one frequency drives the vehicle
    unstable. Ring baffles at zeta = 0.047 give Q = 10.6, a broad low hill that
    a conventional autopilot rolls over.

    Same verdict -- phase stabilisation required -- and two completely different
    engineering problems. Reporting only the verdict tells a reader nothing
    about which one they have, so this project's slosh finding read identically
    before and after baffles were added to the design.
    """
    z = float(damping_ratio)
    if z <= 0:
        raise ValueError("damping ratio must be positive")
    q = 1.0 / (2.0 * z)
    if q > 200:
        band = "needle-sharp"
        note = ("a few degrees of phase error at the resonance is enough to "
                "drive it unstable, so the compensator has to be accurate at "
                "one frequency and stay accurate as the tank drains and the "
                "frequency moves")
    elif q > 30:
        band = "sharp"
        note = ("demanding but tractable: the compensator has to track the mode "
                "as the fill level changes")
    else:
        band = "broad"
        note = ("routine: the peak is low enough that a conventional autopilot "
                "with adequate phase margin rolls over it")
    return {"damping_ratio": z, "peak_amplification": q, "band": band,
            "note": f"peak amplification Q = {q:.0f} ({band}); {note}"}
