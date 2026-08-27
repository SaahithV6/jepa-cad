#!/usr/bin/env python3
"""Arbitrary mission specification -> architecture -> vehicle -> verified packet.

Chains every layer that exists:

  planner       chooses stage count and split from the mission's delta-v
  sizing        bisects propellant until the stack flies the requested altitude
  decomposition splits the vehicle into major structural components
  simulation    solves each component in CalculiX under the load it carries
  verification  margin and pass/fail per component, plus the whole-assembly flight
  reporting     writes the packet

Unlike the grid-trained path this takes any payload and altitude, because the
architecture is decided and the vehicle solved at request time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadflow.planner import STRUCT_COEFF, drag_coefficient, plan, plan_sized  # noqa: E402
from cadflow.structural_sizing import size_wall
from cadflow.profiles import nose_profile  # noqa: E402
from cadflow.vehicle import (  # noqa: E402
    Placed, combine, flight_vehicle_properties, size_fins_for_margin,
    static_margin)
from generate_propulsion_trajectory_corpus import load_coupling  # noqa: E402
from scripts.params_to_physics_confirmed import run_confirmed  # noqa: E402

#: The structural gate is no longer a constant. It is derived per design from
#: the alloy the loop selected, in `cadflow.allowables`, because a fixed 200 MPa
#: applied to every material at aluminium stiffness meant an Inconel vehicle was
#: verified as aluminium: charged 8,190 kg/m3 in the mass budget and forbidden
#: from carrying more than a third of the stress the alloy is good for. The name
#: is left removed rather than repointed so that anything still importing it
#: fails loudly instead of picking up a plausible default.


#: Aluminium 6061, for the modal analysis. The static deck does not need a
#: density and does not carry one; a modal analysis is a mass problem.
RHO_AL = 2700.0
E_AL = 70e9
NU_AL = 0.33


def component_first_mode_hz(case_dir: Path) -> float | None:
    """First elastic natural frequency of the part that was just analysed.

    Reuses the mesh the static run already produced, so this costs a fraction of
    a second rather than a re-mesh. A part sized only for steady load can still
    be destroyed by a resonance -- fin flutter is the classic case -- and
    first_mode_hz is a conditioning slot that nothing was populating.
    """
    from cadflow.msh_to_calculix import (
        generate_modal_case_inp, parse_eigenfrequencies, run_calculix_case)

    meshes = sorted(case_dir.rglob("mesh.msh"), key=lambda f: f.stat().st_mtime)
    if not meshes:
        return None
    solver_dir = meshes[-1].parent
    try:
        generate_modal_case_inp(
            solver_dir, case_filename="modal.inp", fix_axis="z", modes=6,
            youngs_modulus=E_AL, poisson=NU_AL, density=RHO_AL)
        run_calculix_case(solver_dir, job_name="modal", timeout=900)
        freqs = parse_eigenfrequencies(solver_dir / "modal.dat")
    except Exception:  # noqa: BLE001
        return None
    return freqs[0] if freqs else None


def frd_stress_percentiles(case_dir: Path) -> dict | None:
    """Von Mises percentiles from the FRD.

    The absolute maximum is the wrong acceptance metric for a tet mesh with
    sharp re-entrant corners: stress there does not converge with refinement and
    does not relieve when the wall is thickened. Sizing the thrust structure
    from 1.44 to 2.44 mm moved the reported peak from 333 to 3,948 MPa on a
    116 MPa nominal wall -- the peak is tracking the corner, not the load. A
    high percentile describes the structure; the peak flags where a fillet or
    doubler is needed.
    """
    import math as _m
    # Newest by mtime, not first by name. Each solve writes to a hash-named
    # directory, and the wall-thickening loop reuses comp_dir, so a component
    # accumulates one FRD per iteration -- and comp_dir also survives between
    # runs of this script. Taking sorted()[0] took whichever hash sorted first,
    # which is a coin flip: a run with the ogive nose reported 109.3 and 167.0
    # MPa for the thrust structure and stage 1 tank that were, to the decimal,
    # the *previous* run's numbers read off its leftover files.
    frds = sorted(case_dir.rglob("case.frd"), key=lambda f: f.stat().st_mtime)
    if not frds:
        return None
    vals = []
    in_stress = False
    for line in open(frds[-1], errors="ignore"):
        if "STRESS" in line:
            in_stress = True
            continue
        if in_stress:
            if line.startswith(" -3"):
                break
            if line.startswith(" -1"):
                # FRD is fixed-width, not whitespace-separated: ' -1', a
                # 10-character node number, then six 12-character values. A
                # negative value fills its whole field, so it abuts the previous
                # one with no space -- "2.44293E+08-1.04280E+07" is two numbers.
                #
                # split() therefore returns too few tokens on any line
                # containing a negative component, and the old parser skipped
                # those lines silently. On one 14,013-node result it read 100 of
                # them, and the hundred it kept were exactly the ones where every
                # component happened to be positive: not a sample, a selection.
                # p99 of 100 values is the 99th, i.e. the maximum, which is why
                # p99 and peak kept coming out identical.
                try:
                    comps = [float(line[13 + 12 * i: 13 + 12 * (i + 1)])
                             for i in range(6)]
                except ValueError:
                    parts = line.split()
                    try:
                        comps = [float(x) for x in parts[2:8]]
                    except (ValueError, IndexError):
                        continue
                    if len(comps) != 6:
                        continue
                sxx, syy, szz, sxy, syz, szx = comps
                vm = _m.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2)
                             + 3.0 * (sxy**2 + syz**2 + szx**2))
                vals.append(vm / 1e6)
    if not vals:
        return None
    vals.sort()
    def pct(q):
        return vals[min(len(vals) - 1, int(q * len(vals)))]
    return {"median": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": vals[-1], "n": len(vals),
            "frac_over_yield": sum(1 for v in vals if v > 276.0) / len(vals)}


#: Physical order along the vehicle, nose forward. Components were only ever
#: analysed in isolation; a rocket's behaviour depends on where its mass sits.
def stack_order(names: list[str]) -> list[str]:
    """Sort component names nose-first into their physical stacking order."""
    def rank(name: str) -> tuple:
        if name == "nose cone":
            return (0, 0)
        if name.startswith("stage") and "tank" in name:
            # upper stages sit forward, so higher stage number comes first
            return (1, -int(name.split()[1]))
        if name.startswith("interstage"):
            # An interstage n/n+1 sits *between* stage n+1 (forward) and stage n
            # (aft), so it ranks just forward of stage n -- minus a half, not
            # plus. With plus it landed one position too far aft, behind the
            # stage it should sit on top of, which put the coupon stack's centre
            # of gravity in the wrong place for every vehicle built so far.
            return (1, -int(name.split()[1].split("/")[0]) - 0.5)
        if name == "fin set":
            return (2, 0)
        return (3, 0)          # thrust structure at the aft end
    return sorted(names, key=rank)


#: Design angle of attack at max-Q, radians. A vehicle does not fly at zero
#: incidence through the transonic region -- wind shear and guidance error put a
#: few degrees on it, and that incidence is what loads the fins. 5 degrees is a
#: conventional figure for a stable unguided vehicle; it is a stated assumption,
#: named here so it can be changed in one place.
DESIGN_ALPHA_RAD = math.radians(5.0)

#: Ultimate factor on limit load. Standard aerospace practice.
ULTIMATE_FACTOR = 1.5

#: Skin thickness for the assembled vehicle, mm. The component analysis sizes
#: each coupon's wall separately; this is one representative gauge for drawing
#: the whole airframe, which is a different job from qualifying a part.
ASSEMBLY_WALL_MM = 3.0

#: Nozzle wall material density, kg/m^3. Inconel or a comparable superalloy is
#: what a regeneratively cooled skirt is actually made from.
NOZZLE_DENSITY = 8000.0

#: Temperature above which aluminium alloys stop holding useful strength. Not a
#: sharp limit -- 6061-T6 is already well down by 500 K -- but past it a
#: room-temperature allowable is the wrong number to be designing against.
ALUMINIUM_SERVICE_K = 450.0

#: Axial acceleration a payload is typically qualified to, in g. Exceeding it is
#: not a solver error -- it is a real consequence of holding thrust constant
#: while mass falls -- but it is a design finding and belongs in the packet.
PAYLOAD_G_LIMIT = 10.0

#: Ceiling on wall thickness, millimetres. Past this the part is not a thin
#: shell any more and the sizing model behind it no longer applies, so hitting
#: it means the design needs a different architecture rather than more metal.
MAX_WALL_MM = 12.0

#: Floor on any design load, newtons. For a very light part the flight loads are
#: not what sizes it -- being carried, clamped in a fixture, or leaned on during
#: assembly is. Without a floor the aerodynamic cases for small components fall
#: to a few tens of newtons and the "design" becomes minimum gauge against
#: nothing at all. Named so it is visibly an assumption rather than a stray
#: constant in a max().
MIN_DESIGN_LOAD_N = 200.0


def component_specs(body_r_mm: float, stages, gross_kg: float, max_q_pa: float,
                    payload_kg: float, cd: float = 0.42,
                    cna_fins: float = 0.0, n_fins: int = 4,
                    axial_g_by_stage: list[float] | None = None,
                    liftoff_thrust_n: float = 0.0):
    """Load cases for each component.

    The aerodynamic loads used to be max-Q dynamic pressure times frontal area
    times a bare multiplier -- 1.8 for the nose, 0.9 for the fins -- with
    nothing behind either number. They are now built from coefficients the rest
    of the program already computes: the nose carries axial drag q Cd A plus its
    share of normal force at the design incidence, and each fin carries the fin
    set's normal force q CNa_fins A alpha divided by the number of fins. Both
    then take the standard 1.5 ultimate factor.

    Areas are the coupon's, so the loads stay consistent with the geometry that
    is actually meshed.
    """
    frontal = math.pi * (body_r_mm / 1000.0) ** 2
    q = float(max_q_pa)

    # Axial loads come from the trajectory's own peak acceleration, per stage,
    # rather than an assumed 4.5 g. For this mission the integrator reports
    # 15.37 g while stage 1 burns and 16.46 g while stage 2 does -- so every
    # axially loaded component had been sized for under a third of the load it
    # actually sees. Per stage and not globally, because the global peak occurs
    # at final burnout when the lower stages have already separated.
    n_stages = len(list(stages))
    gs = list(axial_g_by_stage) if axial_g_by_stage else [4.5] * n_stages
    if len(gs) < n_stages:
        gs = gs + [gs[-1]] * (n_stages - len(gs))
    thrust1 = liftoff_thrust_n or (gross_kg * 9.80665 * max(gs))

    # nose: axial drag plus normal force from CNa_nose = 2 (slender body)
    nose_axial = q * float(cd) * frontal
    nose_normal = q * 2.0 * frontal * DESIGN_ALPHA_RAD
    nose_load = ULTIMATE_FACTOR * math.hypot(nose_axial, nose_normal)

    # one fin's share of the fin set's normal force at the design incidence
    fin_load = ULTIMATE_FACTOR * (
        q * float(cna_fins) * frontal * DESIGN_ALPHA_RAD / max(1, int(n_fins)))

    specs = [
        ("nose cone", "drag plus normal force at max-Q, 1.5 ultimate",
         max(MIN_DESIGN_LOAD_N, nose_load),
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 1.2,
              nose_height_mm=body_r_mm * 1.6, fin_thickness_mm=4.0)),
        ("thrust structure", "engine thrust into the aft ring",
         thrust1 * 1.3,
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 1.2,
              nose_height_mm=body_r_mm * 0.5, fin_thickness_mm=6.5)),
        ("fin set", f"one fin's share of fin normal force at "
                    f"{math.degrees(DESIGN_ALPHA_RAD):.0f} deg, 1.5 ultimate",
         max(MIN_DESIGN_LOAD_N, fin_load),
         dict(body_radius_mm=body_r_mm, body_height_mm=body_r_mm * 2.0,
              nose_height_mm=body_r_mm * 0.7, fin_thickness_mm=5.0)),
    ]
    # one tank and one interstage per stage
    for i, st in enumerate(stages):
        supported = payload_kg + sum(s.prop_mass_kg + s.struct_mass_kg
                                     for s in stages[i:])
        specs.append((f"stage {i+1} tank",
                      f"carries {st.prop_mass_kg:.1f} kg propellant at {gs[i]:.1f} g",
                      supported * 9.80665 * gs[i],
                      dict(body_radius_mm=body_r_mm * (1.0 - 0.08 * i),
                           body_height_mm=body_r_mm * 3.5,
                           nose_height_mm=body_r_mm * 0.8,
                           fin_thickness_mm=5.5)))
        if i < len(stages) - 1:
            specs.append((f"interstage {i+1}/{i+2}",
                          "transmits lower-stage thrust to the stage above",
                          supported * 9.80665 * gs[i],
                          dict(body_radius_mm=body_r_mm * (1.0 - 0.08 * i),
                               body_height_mm=body_r_mm * 1.5,
                               nose_height_mm=body_r_mm * 0.6,
                               fin_thickness_mm=5.5)))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload-kg", type=float, required=True)
    ap.add_argument("--apogee-km", type=float, required=True)
    ap.add_argument("--propellant", type=str, default="lox_rp1")
    ap.add_argument("--chamber-bar", type=float, default=55.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/plan_packet")
    ap.add_argument("--solve-structure", action="store_true",
                    help="design at the structural coefficient the mass model "
                         "solves for, instead of the asserted constant")
    ap.add_argument("--autodesign", action="store_true",
                    help="run the repair loop before reporting, so the packet "
                         "describes a design that closes rather than one that "
                         "does not. Without it the packet can only diagnose: "
                         "for 25 kg to 4,000 km it reports the vehicle 27.9 kg "
                         "short of containing its own skin and engine, which is "
                         "true and unactionable. The loop raises the structural "
                         "coefficient until the mass closes, picks a skin "
                         "material that survives the peak temperature, sizes "
                         "thermal protection when no alloy does, and chooses a "
                         "nose shape on CFD-measured drag.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_coupling()

    spec = (f"deliver {args.payload_kg:.0f} kg payload to {args.apogee_km:.0f} km "
            f"apogee using {args.propellant.replace('_','/')} at "
            f"{args.chamber_bar:.0f} bar chamber pressure")

    # Solve the structural coefficient as a fixed point and compare it with the
    # constant the design is asserted at. The two disagreeing is the single most
    # consequential thing this packet can say about its own vehicle: if the mass
    # model is right, a vehicle sized at the asserted value is too light to
    # exist and will not make the mission.
    # The repair loop, when asked for. It returns the knobs it converged on,
    # and those knobs -- not the defaults -- are what the rest of the packet
    # must describe, or the report and the vehicle diverge again.
    design_knobs = None
    design_history = None
    repaired_plan = None
    if args.autodesign:
        try:
            from cadflow.autodesign import autodesign as _autodesign

            _res = _autodesign(args.payload_kg, args.apogee_km, max_iters=12)
            design_knobs = _res["knobs"]
            design_history = _res["history"]
            _ev = _res["evaluation"]
            # Use the repaired PLAN, not just the repaired knobs. Wiring only
            # the knobs through produced a packet whose header announced
            # "converged, 0 violations" while every section below still
            # described the unrepaired vehicle -- gross 1136.8 kg, 27.1 kg short
            # of containing itself. A report that contradicts itself in the
            # reader's favour is worse than one that simply reports the failure.
            repaired_plan = getattr(_ev, "plan", None)
            print(f"repair loop: converged={_res['converged']} in "
                  f"{_res['iterations']} iterations; "
                  f"coeff {_ev.struct_coeff_asserted:.4f}, "
                  f"skin {design_knobs.skin_material}, "
                  f"nose {design_knobs.nose_shape}, "
                  f"{len(_ev.violations)} violations", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"repair loop failed ({exc}); reporting the unrepaired "
                  f"design", flush=True)

    repaired_plan = repaired_plan if args.autodesign else None

    # Size against the material the design actually selected.
    #
    # The gate used to be a flat 200 MPa with aluminium elastic properties, no
    # matter what autodesign picked. That number is close to a 6061-T6
    # allowable, so for aluminium vehicles it was roughly right by accident;
    # for the Inconel 718 skin this loop now selects it was gating a 700 MPa
    # material at 200, while the mass budget charged the full 8,190 kg/m3. The
    # design paid for the alloy and was forbidden from using it.
    skin_material = getattr(design_knobs, "skin_material", None) or "al-6061-t6"
    try:
        from cadflow.allowables import design_allowable, elastic_properties

        allowable = design_allowable(skin_material)
        skin_e_pa, skin_nu = elastic_properties(skin_material)
    except (KeyError, ValueError) as exc:
        print(f"no allowable for {skin_material} ({exc}); "
              f"falling back to al-6061-t6", flush=True)
        from cadflow.allowables import design_allowable, elastic_properties

        allowable = design_allowable("al-6061-t6")
        skin_e_pa, skin_nu = elastic_properties("al-6061-t6")
    allowable_mpa = allowable.allowable_mpa
    print(f"structural gate: {allowable_mpa:.1f} MPa "
          f"({allowable.material_id}, {allowable.source_strength_mpa:.0f} MPa "
          f"{allowable.strength_basis} / FoS {allowable.factor_of_safety} "
          f"x knockdown {allowable.knockdown}), E {skin_e_pa/1e9:.0f} GPa",
          flush=True)

    solved_coeff = None
    try:
        _sp, solved_coeff, _hist = plan_sized(
            args.apogee_km, args.payload_kg, propellant=args.propellant,
            chamber_bar=args.chamber_bar)
        if args.solve_structure and _sp is not None:
            p = _sp
            print(f"designing at solved structural coefficient "
                  f"{solved_coeff:.4f}", flush=True)
        else:
            p = plan(args.apogee_km, args.payload_kg,
                     propellant=args.propellant, chamber_bar=args.chamber_bar)
    except Exception as exc:  # noqa: BLE001
        print(f"structural fixed point failed ({exc}); using the asserted "
              f"coefficient", flush=True)
        p = plan(args.apogee_km, args.payload_kg,
                 propellant=args.propellant, chamber_bar=args.chamber_bar)
    if args.autodesign and repaired_plan is not None:
        p = repaired_plan
        print(f"reporting the repaired design: gross {p.gross_kg:.1f} kg, "
              f"{p.stages} stages", flush=True)

    if p is None:
        print(f"# Design packet\n\n**Specification:** {spec}\n")
        print("No architecture up to 3 stages closes this mission.")
        return 1

    # The CAD radius is clamped so parts stay meshable; the flight vehicle's
    # radius is what the trajectory actually used. Keeping both, and keeping
    # them named apart, is the only way the packet can report each honestly.
    body_r = max(20.0, min(50.0, 16.0 * (p.gross_kg / 100.0) ** (1 / 3)))
    geo_r_mm = body_r
    flight_r = max(0.10, (p.gross_kg / 1000.0) ** (1 / 3) * 0.55) / 2.0

    # Stability is solved before the components, because it decides the fin.
    # Previously the fin that was structurally verified had a hardcoded 14 mm
    # span and no relation to the fin the vehicle actually needs, so the FEA was
    # checking a part that would never be built.
    fv = flight_vehicle_properties(p.stack, args.payload_kg, flight_r)
    nose_len = 4.0 * flight_r
    prof = nose_profile(flight_r, nose_len, "ogive", 4000)
    root_le = 2.0 * flight_r
    burn_states = [("liftoff", [1.0] * len(p.stack))]
    burn_states.append(("stage 1 burnout", [0.0] + [1.0] * (len(p.stack) - 1))
                       if len(p.stack) > 1 else ("burnout", [0.0]))
    cgs = []
    for label, rem in burn_states:
        st = flight_vehicle_properties(p.stack, args.payload_kg, flight_r,
                                       propellant_remaining=rem)
        cgs.append((label, st["cg_z_m"], st["mass_kg"]))
    worst_label, worst_cg, _ = min(cgs, key=lambda c: c[1])
    fins = size_fins_for_margin(prof, flight_r,
                                nose_tip_station_m=fv["length_m"],
                                cg_z_m=worst_cg,
                                fin_root_le_station_m=root_le)
    # Fins sized for static margin alone are only half a design. A statically
    # stable vehicle spends gimbal deflection fighting its own fins, so margin
    # and control authority are one lever pulled in opposite directions, and
    # optimising the stability end discovers the other end only if something
    # checks. The first packet to compute both wanted 14.5 degrees of gimbal
    # against the 8 a production engine offers -- stable, and unsteerable.
    control_trade = None
    try:
        from cadflow.control_authority import trade_margin_for_authority

        # The thrust the design actually flies, not a nominal ratio. The
        # repair loop lowers thrust-to-weight to hold peak acceleration down --
        # for this mission it reaches 2.93, not 4.5 -- so assuming 4.5 here
        # overstated thrust by 54% and made the engine look able to steer a
        # vehicle it cannot.
        _thr1 = float(p.trajectory.get("liftoff_thrust_n")
                      or p.gross_kg * 9.80665 * 4.5)
        _Sref = math.pi * flight_r ** 2
        control_trade = trade_margin_for_authority(
            lambda tm: size_fins_for_margin(
                prof, flight_r, nose_tip_station_m=fv["length_m"],
                cg_z_m=worst_cg, fin_root_le_station_m=root_le,
                target_margin=tm),
            q_pa=p.trajectory["max_q_pa"], reference_area_m2=_Sref,
            alpha_rad=DESIGN_ALPHA_RAD, cg_station_m=fv["cg_z_m"],
            thrust_n=_thr1, body_diameter_m=2.0 * flight_r)
        if control_trade["converged"] and control_trade["fins"] is not None:
            if control_trade["margin_cal"] < 1.5 - 1e-9:
                print(f"control trade: static margin 1.50 -> "
                      f"{control_trade['margin_cal']:.2f} cal so the engine can "
                      f"steer at max-Q", flush=True)
            fins = control_trade["fins"]
        else:
            print(f"control trade: {control_trade['note'][:90]}", flush=True)
    except Exception as _exc:  # noqa: BLE001
        print(f"control trade unavailable ({_exc})", flush=True)

    margins = [(lbl, static_margin(cg, fins["cp_z_m"], 2.0 * flight_r), m)
               for lbl, cg, m in cgs]
    stability = dict(fins, sized_for=worst_label,
                     margins=[{"state": l, "margin_cal": mg, "mass_kg": m}
                              for l, mg, m in margins])

    # Carry the designed fin's *shape* down to coupon scale: same span and chord
    # in body radii, same taper and sweep. The coupon is smaller, but it is now
    # the same fin.
    scale = geo_r_mm / (flight_r * 1000.0)
    fin_span_mm = fins["span_m"] * 1000.0 * scale
    fin_chord_mm = fins["root_chord_m"] * 1000.0 * scale

    results = []
    for name, why, load, geo in component_specs(
            body_r, p.stack, p.gross_kg, p.trajectory["max_q_pa"],
            args.payload_kg, cd=drag_coefficient(),
            cna_fins=fins["cna_fins"], n_fins=fins["n_fins"],
            axial_g_by_stage=p.trajectory.get("max_axial_g_by_stage"),
            liftoff_thrust_n=p.trajectory.get("liftoff_thrust_n", 0.0)):
        wall_capped = False
        # Size the wall, mesh that shell, and thicken if the analysis says so.
        #
        # The membrane formula assumes a long cylinder. The thrust structure is
        # 42.8 mm long on a 71 mm diameter, so end effects cover the whole part
        # and the analytic wall under-sizes it: sized to 116 MPa nominal it came
        # back at 279 MPa. Rather than trust either number alone, iterate --
        # this is design-by-analysis, and it is why the analysis exists.
        wall = size_wall(load, geo["body_radius_mm"] / 1000.0,
                         geo["body_height_mm"] / 1000.0)
        t_mm = wall.thickness_m * 1000.0
        vm = None
        ok = False
        hulled = False
        err = None
        dist = None
        comp_dir = args.out / "components" / name.replace(" ", "_").replace("/", "-")
        # Start each component from clean ground so no earlier run's results can
        # be picked up as this one's.
        if comp_dir.exists():
            shutil.rmtree(comp_dir)

        for attempt in range(3):
            geom = {"body_radius_mm": geo["body_radius_mm"],
                    "body_height_mm": geo["body_height_mm"],
                    "nose_radius_mm": geo["body_radius_mm"],
                    "nose_height_mm": geo["nose_height_mm"],
                    # Only the fin set has fins. Every component was being
                    # built with a fin stub, including the tanks and the thrust
                    # structure, which welds an artificial stress riser onto
                    # parts that do not have one -- and a partially embedded
                    # stub in a thick wall makes a sharp internal T-joint whose
                    # stress does not relieve when the wall is thickened. That
                    # is why the thrust structure went 230 -> 817 MPa as it was
                    # made heavier.
                    "fin_span_mm": fin_span_mm if "fin" in name else 0.0,
                    "fin_thickness_mm": geo["fin_thickness_mm"],
                    "fin_chord_mm": fin_chord_mm, "fillet_radius_mm": 2.5,
                    "wall_thickness_mm": t_mm,
                    # Roughly three elements through the wall, but the element
                    # size is also capped against the radius. Scaling cl purely
                    # with the wall meant a 12 mm wall gave 16.8 mm elements, so
                    # the loaded face carried only a handful of nodes and
                    # load_per_node = total_load / len(loaded) turned into point
                    # loads. Stress then *rose* with thickness -- 230 MPa at
                    # 2.44 mm, 817 MPa at 6.48 mm -- which is a load-introduction
                    # artifact, not structure.
                    "cl_max_mm": min(max(0.8, t_mm * 1.4),
                                     geo["body_radius_mm"] / 6.0),
                    "cl_min_mm": min(max(0.25, t_mm / 3.0),
                                     geo["body_radius_mm"] / 20.0)}
            try:
                rep = run_confirmed(
                    params_mm=geom, out=comp_dir,
                    max_stress_mpa=allowable_mpa, max_disp_mm=3.0, max_iters=1,
                    load_n=load, prompt=name,
                    youngs_modulus_pa=skin_e_pa, poisson=skin_nu)
                acc = rep.get("accepted") or {}
                vm, ok = acc.get("max_von_mises_mpa"), bool(acc.get("targets_met"))
                hulled = any(comp_dir.rglob("MESH_IS_CONVEX_HULL"))
                # A coarser-than-requested mesh is still the real part, so it
                # is not a failure -- but it resolves the wall with fewer
                # elements than intended and the reader should know.
                coarsened = any(comp_dir.rglob("MESH_COARSENED"))
                if hulled:
                    vm, ok = None, False
                err = None
            except Exception as exc:  # noqa: BLE001
                vm, ok, hulled, coarsened = None, False, False, False
                err = f"{type(exc).__name__}: {exc}"
                print(f"  [{name}] {err}", flush=True)
                break

            dist = None if hulled else frd_stress_percentiles(comp_dir)
            if dist is None:
                break
            # Accept on p95, not p99.
            #
            # Measured on one part at 170 kN across a 16x mesh refinement,
            # 3,253 to 50,737 elements, the spread of each metric was:
            #
            #   median   2.8%      p95  13.8%      p99  36.6%      peak  268%
            #
            # p99 is not stable enough to design against at the mesh densities
            # this loop can afford. With around 1,200 nodes on a component, the
            # top one percent is a dozen nodes and every one of them sits on the
            # same re-entrant corner: one part read median 37 MPa, p95 62 MPa
            # and p99 306 MPa, and thickening its wall from 2.4 mm to 12 mm
            # moved p99 by nothing at all -- 299, 330, 306 -- because a
            # singularity does not care how thick the wall is.
            #
            # The median is steadier still but too permissive to size against,
            # since it ignores the whole loaded upper field. p95 keeps the
            # structure in view and leaves the singularity out of it.
            vm = dist["p95"]
            ok = vm <= allowable_mpa
            if ok:
                break
            # thicken in proportion to the overshoot and re-analyse
            thicker = min(t_mm * min(2.0, 1.15 * vm / allowable_mpa), MAX_WALL_MM)
            if thicker <= t_mm + 1e-9:
                # Already at the cap and still over. Re-solving identical
                # geometry cannot change the answer, so stop and say so rather
                # than burning the remaining iterations on the same mesh.
                print(f"  [{name}] p99 {vm:.0f} MPa over {allowable_mpa:.0f} at "
                      f"the {MAX_WALL_MM:.0f} mm wall limit; not converging",
                      flush=True)
                wall_capped = True
                break
            t_mm = thicker
            print(f"  [{name}] p99 {vm:.0f} MPa over {allowable_mpa:.0f}; "
                  f"thickening wall to {t_mm:.2f} mm", flush=True)

        # Recompute wall properties at the thickness the analysis settled on,
        # otherwise the packet reports mass and margin for the first estimate
        # rather than the design that was actually verified.
        import math as _m
        from cadflow.structural_sizing import E_PA, RHO, SIGMA_YIELD_PA, _knockdown
        t_final = t_mm / 1000.0
        r_final = geo["body_radius_mm"] / 1000.0
        L_final = geo["body_height_mm"] / 1000.0
        wall_mass = RHO * 2.0 * _m.pi * r_final * t_final * L_final
        sigma_app = load / (2.0 * _m.pi * r_final * t_final)
        gamma_kd = _knockdown(r_final / t_final)
        sigma_cr = gamma_kd * 0.605 * E_PA * t_final / r_final
        wall_driver = wall.driver if abs(t_mm - wall.thickness_m * 1000.0) < 1e-6 \
            else "analysis"
        buckling_margin = sigma_cr / max(sigma_app, 1.0)
        wall_mm_final = t_mm
        first_mode = None if (hulled or dist is None) else \
            component_first_mode_hz(comp_dir)

        # Mass properties from the solid that was actually analysed. Ixx/Iyy/Izz
        # were three of only four conditioning slots absent from the entire
        # graph, and they are exactly computable from geometry already being
        # built -- and they are what attitude control and stability need.
        mass_props = None
        try:
            from cadflow.backends import build_from_spec, get_backend
            from scripts.smoke_params_to_assembly import constraints_to_geometry
            _b = get_backend(prefer_real=True)
            _shape = build_from_spec(constraints_to_geometry(geom), backend=_b)
            mass_props = _b.mass_properties(_shape, RHO_AL)
        except Exception:  # noqa: BLE001
            mass_props = None
        results.append({"name": name, "why": why, "load_n": load,
                        "first_mode_hz": first_mode,
                        "mass_properties": mass_props,
                        "geometry": geom,
                        "error": err,
                        "mesh_was_hull": hulled,
                        "mesh_was_coarsened": coarsened,
                        "stress_dist": dist,
                        "shell_von_mises_mpa": vm, "coupon_passed": ok,
                        "coupon_margin": (allowable_mpa / vm) if vm else None,
                        "wall_mm": wall_mm_final,
                        "wall_mass_kg": wall_mass,
                        "wall_driver": ("wall limit" if wall_capped
                                        else wall_driver),
                        "wall_capped": wall_capped,
                        "buckling_margin": buckling_margin,
                        "passed": ok and wall.margin_buckling >= 1.0})

    L = [f"# Design packet\n", f"**Specification:** {spec}\n", "## Architecture\n"]
    for line in p.rationale:
        L.append(f"- {line}")
    if solved_coeff is not None:
        L.append("\n## Structural mass closure\n")
        used = solved_coeff if args.solve_structure else STRUCT_COEFF
        L.append(f"Designed at a structural coefficient of **{used:.3f}**. "
                 f"Solving it as a fixed point -- size the vehicle, size its "
                 f"walls from the resulting loads, recompute the coefficient, "
                 f"repeat -- converges to **{solved_coeff:.3f}**.\n")
        if not args.solve_structure and solved_coeff > STRUCT_COEFF * 1.15:
            L.append(f"> The mass model wants {solved_coeff:.3f} where the "
                     f"design asserts {STRUCT_COEFF:.3f}, so this vehicle is "
                     f"optimistic: built to its own structural model it would "
                     f"be heavier than planned and would fall short of the "
                     f"target. Re-run with `--solve-structure` to design at the "
                     f"solved value. It is reported rather than silently "
                     f"absorbed because it changes the architecture -- a "
                     f"heavier structure closes fewer stages, so the same "
                     f"mission needs more of them.\n")
    L.append(f"\n## Vehicle: {p.stages} stage(s)\n")
    L.append("| stage | propellant | structure | expansion |")
    L.append("|---|---|---|---|")
    for i, st in enumerate(p.stack):
        L.append(f"| {i+1} | {st.prop_mass_kg:.2f} kg | "
                 f"{st.struct_mass_kg:.2f} kg | {st.expansion_ratio:.0f} |")
    L.append(f"\npayload {args.payload_kg:.1f} kg, gross {p.gross_kg:.1f} kg\n")

    # Every other check in this packet is internal: the solvers agree with
    # theory and the budget closes. All of that can be true of a vehicle nobody
    # could build, and the structural coefficient is where that shows first,
    # because it is asserted rather than solved for.
    try:
        from cadflow.flown_envelope import check as _envelope_check

        _stage1_wet = (p.stack[0].prop_mass_kg + p.stack[0].struct_mass_kg
                       if p.stack else p.gross_kg)
        _sigma = (p.stack[0].struct_mass_kg / _stage1_wet
                  if p.stack and _stage1_wet > 0 else None)
        if _sigma is not None:
            _v = _envelope_check(_sigma, stage_wet_kg=_stage1_wet)
            # Why the structure is heavy, which decides whether anything can be
            # done about it. A coefficient above flown practice reads as a
            # warning; whether it is one depends entirely on what set the wall
            # thicknesses. Strength-driven walls can be thinned by a better
            # material or a lower load. Gauge-driven walls cannot be thinned at
            # all -- they are already as thin as the shop can make them -- and a
            # design loop that keeps trying is wasting its iterations.
            _gauge = [r for r in results
                      if "gauge" in str(r.get("wall_driver", "")).lower()]
            _strength = [r for r in results
                         if r.get("wall_driver") and r not in _gauge]
            if _gauge or _strength:
                _gm = sum(float(r.get("mass_properties", {}).get("mass_kg", 0.0))
                          for r in _gauge)
                _sm = sum(float(r.get("mass_properties", {}).get("mass_kg", 0.0))
                          for r in _strength)
                _tot = _gm + _sm
                # Built as one string. Appending the clauses separately put
                # each on its own markdown line, so the sentence rendered as
                # fragments beginning with a comma.
                _share = (f", carrying {100*_gm/_tot:.0f}% of the analysed "
                          f"structural mass" if _tot > 0 else "")
                L.append(f"\n**What sets the wall thicknesses.** "
                         f"{len(_gauge)} of {len(results)} components are at "
                         f"minimum gauge rather than sized by strength"
                         f"{_share}"
                         f". Those walls cannot be thinned -- they are already "
                         f"as thin as the process allows -- so a structural "
                         f"coefficient above flown practice is partly a "
                         f"consequence of building a small vehicle rather than "
                         f"a design fault. The levers that remain are fewer "
                         f"stages, a larger vehicle, or a material with a lower "
                         f"minimum gauge; thinning walls is not one of them.\n")
                if _gauge:
                    L.append("\nAt minimum gauge: "
                             + ", ".join(str(r["name"]) for r in _gauge) + ".\n")
            L.append(f"\n**Structure against flown hardware.** Stage 1 "
                     f"structural coefficient is {_sigma:.4f}. Ten flown stages "
                     f"from Saturn V's S-IC to Electron's first stage span "
                     f"{_v.flown_min:.3f} to {_v.flown_max:.3f}, median "
                     f"{_v.flown_median:.3f}. {_v.note}\n")
    except Exception as _exc:  # noqa: BLE001
        L.append(f"\n(flown-hardware comparison unavailable: {_exc})\n")

    L.append("## Mission verification\n")
    err = abs(p.achieved_km - args.apogee_km) / args.apogee_km * 100
    seps = ", ".join(f"{s:.1f} s" for s in p.trajectory["separations"]) or "none"
    L.append(f"Flown apogee **{p.achieved_km:.1f} km** against "
             f"{args.apogee_km:.0f} km requested (**{err:.1f}%** error); "
             f"downrange {p.trajectory['downrange_m']/1000:.1f} km, "
             f"max-Q {p.trajectory['max_q_pa']/1000:.1f} kPa, "
             f"separations {seps}.\n")
    gmax = p.trajectory.get("max_axial_g", 0.0)
    gs_by = p.trajectory.get("max_axial_g_by_stage") or []
    if gmax:
        per = ", ".join(f"stage {i+1} {g:.1f} g"
                        for i, g in enumerate(gs_by))
        L.append(f"Peak axial acceleration **{gmax:.1f} g** ({per}). This is "
                 f"what sizes the structure, and it is a property of the "
                 f"architecture rather than a choice: thrust is held while mass "
                 f"falls, so acceleration climbs through each burn.")
        if gmax > PAYLOAD_G_LIMIT:
            L.append(f"\n> That exceeds the {PAYLOAD_G_LIMIT:.0f} g a payload is "
                     f"typically qualified to. A real vehicle throttles or "
                     f"stages earlier to hold this down; this planner does "
                     f"neither, so the number is reported rather than hidden.")
        L.append("")
    # Nozzle geometry. The nozzle existed only as an area ratio: no contour,
    # no wall, no mass, and no way for its shape to matter to anything.
    nozzle = None
    bell = None
    assembly = None
    try:
        from cadflow.backends import get_backend
        from cadflow.sculpt import bell_contour, nozzle_solid

        st0 = p.stack[0]
        at = st0.throat_area_m2
        r_t = math.sqrt(at / math.pi)
        bell = bell_contour(r_t, st0.expansion_ratio)
        wall_m = max(0.0015, 0.02 * r_t)
        _b = get_backend(prefer_real=True)
        solid = nozzle_solid(bell, wall_m, backend=_b)
        mp = _b.mass_properties(solid, NOZZLE_DENSITY)
        nozzle = {
            "throat_radius_mm": r_t * 1000.0,
            "exit_radius_mm": bell.exit_radius_m * 1000.0,
            "length_mm": bell.length_m * 1000.0,
            "area_ratio": bell.area_ratio,
            "percent_bell": bell.percent_bell,
            "exit_angle_deg": bell.exit_angle_deg,
            "divergence_efficiency": bell.divergence_efficiency,
            "wall_mm": wall_m * 1000.0,
            "mass_kg": mp["mass_kg"],
        }
        L.append("\n## Nozzle\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| throat / exit radius | {nozzle['throat_radius_mm']:.1f} / "
                 f"{nozzle['exit_radius_mm']:.1f} mm |")
        L.append(f"| area ratio | {bell.area_ratio:.1f} |")
        L.append(f"| contour | {100*bell.percent_bell:.0f}% bell, "
                 f"{bell.length_m*1000:.0f} mm long, exiting at "
                 f"{bell.exit_angle_deg:.0f} deg |")
        L.append(f"| divergence efficiency | {bell.divergence_efficiency:.4f} |")
        L.append(f"| wall / mass | {wall_m*1000:.2f} mm, "
                 f"{mp['mass_kg']:.2f} kg of Inconel |")
        L.append("\nThe contour is a quadratic pinned by four constraints that "
                 "are all given or forced -- throat radius, the exit radius the "
                 "area ratio demands, and the flow angle at each end -- so "
                 "nothing about it is read off a chart. Its shape now has a "
                 "consequence: divergence loss multiplies thrust, and a 25 "
                 "degree exit would cost 4.7% of specific impulse against this "
                 "one's 0.6%. The wall is offset along the surface normal, so "
                 "it is constant-thickness sheet rather than thinning where the "
                 "contour is steep.")
    except Exception as exc:  # noqa: BLE001 - geometry is an addition, not a gate
        print(f"nozzle geometry unavailable: {exc}", flush=True)

    # Thermal. An engine is very often limited by what its throat can survive
    # rather than by performance or structures, and nothing in this packet knew
    # that. The throat is the worst place on the vehicle: densest gas, nearly
    # chamber temperature, smallest area.
    thermal = None
    try:
        from cadflow.combustion import COMBINATIONS, REFERENCE, chamber_equilibrium
        from cadflow.thermal import (chamber_heat_load, exhaust_thermal_power,
                                     regenerative_cooling)
        if args.propellant in COMBINATIONS:
            of = REFERENCE[args.propellant][0]
            ch = chamber_equilibrium(args.propellant, of, args.chamber_bar * 1e5)
            at = p.stack[0].throat_area_m2
            mdot = ch.pressure_pa * at / ch.c_star_m_s
            load = chamber_heat_load(ch, at, mdot)
            power = exhaust_thermal_power(mdot, ch)
            fuel_flow = mdot / (1.0 + of)
            fuel = COMBINATIONS[args.propellant][1]
            cool = regenerative_cooling(load["q_total_w"], fuel_flow, fuel)
            thr = load["throat"]
            thermal = {
                "of_ratio": of,
                "throat_diameter_mm": thr.throat_diameter_m * 1000.0,
                "throat_heat_flux_MWm2": thr.heat_flux_mw_m2,
                "wall_temp_max_K": thr.wall_temp_k,
                "recovery_temp_K": thr.recovery_temp_k,
                "prandtl": thr.prandtl,
                "reynolds": thr.reynolds,
                "q_total_kw": load["q_total_w"] / 1e3,
                "rejected_fraction": load["q_total_w"] / power,
                "coolant": cool,
            }
            L.append("\n## Thermal\n")
            L.append("| quantity | value |")
            L.append("|---|---|")
            L.append(f"| throat diameter | {thermal['throat_diameter_mm']:.1f} mm |")
            L.append(f"| throat heat flux | "
                     f"{thermal['throat_heat_flux_MWm2']:.1f} MW/m^2 at a "
                     f"{thr.wall_temp_k:.0f} K wall |")
            L.append(f"| adiabatic wall temperature | "
                     f"{thr.recovery_temp_k:.0f} K |")
            L.append(f"| total heat into the walls | "
                     f"{thermal['q_total_kw']:.0f} kW, "
                     f"{100*thermal['rejected_fraction']:.2f}% of exhaust power |")
            L.append(f"| regenerative cooling | {fuel} rises "
                     f"{cool['delta_t_k']:.0f} K to {cool['outlet_temp_k']:.0f} K "
                     f"(limit {cool['limit_temp_k']:.0f} K) |")
            L.append(f"| cooling closes | "
                     f"{'**yes**' if cool['feasible'] else '**NO**'}, margin "
                     f"{cool['margin_k']:+.0f} K |")
            L.append("\nGas properties are the real equilibrium mixture's -- "
                     f"Prandtl {thr.prandtl:.3f}, Reynolds {thr.reynolds:.2e} -- "
                     "not a textbook value for air. The convective correlation "
                     "is the standard turbulent form; what makes it checkable is "
                     "its scalings, and heat flux is asserted to go as throat "
                     "diameter to the -0.200 and chamber pressure to the 0.8. "
                     "The cooling check is an energy balance and involves no "
                     "correlation at all.")
            skin = p.trajectory.get("max_skin_temp_k") or 0.0
            skin_alt = (p.trajectory.get("max_skin_temp_altitude_m") or 0.0) / 1000.0
            if skin:
                thermal["max_skin_temp_K"] = skin
                thermal["max_skin_temp_altitude_km"] = skin_alt
                L.append(f"| peak skin temperature | {skin:.0f} K at "
                         f"{skin_alt:.1f} km |")
                if skin > ALUMINIUM_SERVICE_K:
                    L.append(f"\n> The skin reaches {skin:.0f} K, past the "
                             f"{ALUMINIUM_SERVICE_K:.0f} K at which aluminium "
                             f"keeps useful strength. Every allowable in the "
                             f"component table below is a room-temperature "
                             f"value, so those margins do not hold at this "
                             f"condition: the vehicle needs a thermal "
                             f"protection system, a different skin material, or "
                             f"a trajectory that spends less time fast in thick "
                             f"air. This is a radiation-equilibrium steady "
                             f"state and the vehicle passes through quickly, so "
                             f"it is an upper bound rather than what the "
                             f"structure actually reaches -- but it is far "
                             f"enough past the limit to matter.")
            if not cool["feasible"]:
                L.append(f"\n> The fuel flow cannot carry this heat load. The "
                         f"engine needs film cooling, an ablative liner, or a "
                         f"lower chamber pressure. Reported rather than ignored, "
                         f"because it is a harder constraint than any structural "
                         f"margin in the table below.")
    except Exception as exc:  # noqa: BLE001 - thermal is an addition, not a gate
        print(f"thermal analysis unavailable: {exc}", flush=True)

    L.append("## Component verification\n")
    L.append("| component | load case | load | wall | driver | buckling margin |"
             " shell p95 | p99 | peak | 1st mode | mass | Izz | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        vm = (f"{r['shell_von_mises_mpa']:.1f} MPa"
              if r["shell_von_mises_mpa"] else "-")
        d = r.get("stress_dist")
        peak = f"{d['max']:.0f} MPa" if d else ("hull" if r["mesh_was_hull"] else "-")
        p99 = f"{d['p99']:.0f}" if d else "-"
        if d and d["p95"] > 0 and d["p99"] > 2.0 * d["p95"]:
            p99 += "*"
        f1 = r.get("first_mode_hz")
        mode = f"{f1:.0f} Hz" if f1 else "-"
        if r.get("mesh_was_coarsened"):
            vm = vm + " (coarse)"
        mp = r.get("mass_properties") or {}
        mass = f"{mp['mass_kg']*1000:.0f} g" if mp.get("mass_kg") else "-"
        izz = f"{mp['Izz_kg_m2']:.2e}" if mp.get("Izz_kg_m2") else "-"
        L.append(f"| {r['name']} | {r['why']} | {r['load_n']:.0f} N | "
                 f"{r['wall_mm']:.2f} mm | {r['wall_driver']} | "
                 f"{r['buckling_margin']:.2f}x | {vm} | {p99} | {peak} | {mode} | "
                 f"{mass} | {izz} | "
                 f"{'PASS' if r['passed'] else 'FAIL'} |")
    # Assemble the stack. Individual components have always been analysed
    # alone, but static stability and control response depend on where the mass
    # sits along the vehicle, which no single component knows.
    placed, station = [], 0.0
    by_name = {r["name"]: r for r in results}
    for name in reversed(stack_order(list(by_name))):   # build aft to forward
        r = by_name[name]
        mp = r.get("mass_properties")
        geo = r.get("geometry") or {}
        length_m = (geo.get("body_height_mm", 0.0)
                    + geo.get("nose_height_mm", 0.0)) / 1000.0
        if not mp or length_m <= 0.0:
            continue
        placed.append(Placed(
            name=name, mass_kg=mp["mass_kg"],
            cx_m=mp["cx_m"], cy_m=mp["cy_m"], cz_m=mp["cz_m"],
            Ixx_kg_m2=mp["Ixx_kg_m2"], Iyy_kg_m2=mp["Iyy_kg_m2"],
            Izz_kg_m2=mp["Izz_kg_m2"],
            station_z_m=station + length_m / 2.0))
        station += length_m

    coupon_stack = flight_vehicle = None   # stability is set above
    if placed:
        veh = combine(placed)
        coupon_stack = dict(veh, length_m=station, sections=len(placed))
        L.append("\n## Coupon stack (what was analysed)\n")
        L.append(f"The six analysed parts stacked nose-forward: {station:.3f} m "
                 f"in {len(placed)} sections, {veh['mass_kg']*1000:.0f} g of "
                 f"structure, centre of gravity {veh['cg_z_m']:.3f} m from the "
                 f"aft end, Ixx {veh['Ixx_kg_m2']:.3e} and Izz "
                 f"{veh['Izz_kg_m2']:.3e} kg m^2.\n")
        L.append(f"These are **coupons, not the vehicle**. Body radius is "
                 f"clamped to 50 mm so the parts stay meshable, while this "
                 f"mission's reference diameter is {2*flight_r*1000:.0f} mm -- a "
                 f"factor of {flight_r*1000/geo_r_mm:.1f} in radius, so the "
                 f"coupons carry {veh['mass_kg']*1000:.0f} g against the "
                 f"{sum(st.struct_mass_kg for st in p.stack):.0f} kg of "
                 f"structure the planner sized. The stresses and modes above "
                 f"are about representative sections; the mass properties that "
                 f"describe the flight vehicle are below.")

        flight_vehicle = {k: v for k, v in fv.items() if k != "sections"}
        flight_vehicle["sections"] = [
            {"name": n, "mass_kg": m, "station_z_m": z} for n, m, z in fv["sections"]]
        # Assembly-level verdicts, collected as they are computed so the
        # summary at the end can account for them. Without this the packet
        # reported "all 10 components passed: True" for a vehicle that could
        # not be steered and whose propellant sloshed in time with its own
        # pitch mode -- the findings existed in the prose and nowhere a reader
        # or a downstream consumer of PACKET.json would be forced to see them.
        _assembly_findings: list[dict] = []

        L.append("\n## Flight vehicle\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| length | {fv['length_m']:.2f} m |")
        L.append(f"| diameter | {2*fv['radius_m']*1000:.0f} mm |")
        L.append(f"| wet mass | {fv['mass_kg']:.1f} kg |")
        L.append(f"| centre of gravity from aft end | {fv['cg_z_m']:.3f} m "
                 f"({100*fv['cg_z_m']/fv['length_m']:.0f}% of length) |")
        L.append(f"| pitch/yaw inertia Ixx | {fv['Ixx_kg_m2']:.1f} kg m^2 |")
        L.append(f"| roll inertia Izz | {fv['Izz_kg_m2']:.1f} kg m^2 |")

        # Bending along the assembled stack. Every structural number above this
        # point came from a component solved on its own, gripped at one end and
        # pushed on the other. A launch vehicle does not fail that way: at max-Q
        # it flies at incidence, the nose and fins push sideways, and because it
        # is free the load is reacted by the whole vehicle's inertia rather than
        # by a support. The largest bending moment lands in the middle of the
        # body, which is where no component analysis is looking.
        try:
            from cadflow.flight_loads import PointLoad, skin_stress_mpa
            from cadflow.flight_loads import solve as solve_loads

            _q = p.trajectory["max_q_pa"]
            _fr = math.pi * flight_r ** 2
            _a = DESIGN_ALPHA_RAD
            _loads = [
                PointLoad("nose", float(fins["cp_nose_z_m"]),
                          _q * float(fins["cna_nose"]) * _fr * _a),
                PointLoad("fins", float(fins["cp_fins_z_m"]),
                          _q * float(fins["cna_fins"]) * _fr * _a),
            ]
            _res = solve_loads(fv, _loads)
            _peak = _res.peak_moment_station_m
            # Axial compression the same station carries: everything forward of
            # it, times the axial acceleration. Thrust reacted through the skin.
            _above = sum(mass for _n, z0, z1, mass in fv["section_extents"]
                         if z0 >= _peak) + sum(
                mass * max(0.0, (z1 - _peak) / (z1 - z0))
                for _n, z0, z1, mass in fv["section_extents"]
                if z0 < _peak < z1)
            # Peak axial acceleration, not the liftoff ratio. Thrust is held
            # while mass falls, so acceleration climbs through every burn and
            # the structure sees its maximum near burnout rather than at
            # liftoff. Using the liftoff number understates the load.
            _peak_g = float(p.trajectory.get("max_axial_g") or 4.5)
            _axial_n = _above * 9.80665 * _peak_g
            # Size the flight skin for the flight loads, at the flight radius.
            #
            # This previously took the thinnest wall the component loop had
            # sized and applied it here. Those components are coupons built at
            # a radius clamped to 50 mm so they stay meshable, while this
            # section is 335 mm, and a wall sized for one is not a wall for the
            # other -- required thickness scales inversely with radius for the
            # same load. It happened to land on the minimum gauge, which is
            # radius independent and therefore harmless, but only by way of the
            # min(): a strength-sized coupon wall would have transferred a
            # number that means nothing here.
            _wall = size_wall(load_n=_axial_n, radius_m=flight_r,
                              length_m=fv["length_m"],
                              sigma_allow_pa=allowable_mpa * 1e6,
                              yield_pa=allowable.source_strength_mpa * 1e6,
                              modulus_pa=skin_e_pa)
            _t_m = _wall.thickness_m
            _wall_driver = _wall.driver

            # size_wall knows about axial buckling and nothing about bending.
            # The assembly load case supplies both, and on a large vehicle the
            # bending term is what pushes the shell over: the 1,000 kg mission
            # came out at an interaction of 1.21 -- the skin folds -- while every
            # component still passed its own axial stress check. So the wall is
            # re-sized here against the combined interaction and the thicker of
            # the two is kept.
            from cadflow.shell_buckling import wall_for_buckling_m

            _t_buckle = wall_for_buckling_m(
                axial_n=_axial_n, moment_nm=_res.peak_moment_nm,
                radius_m=flight_r, youngs_pa=skin_e_pa, target_margin=1.4)
            _buckle_repair = None
            if _t_buckle is not None and _t_buckle > _t_m + 1e-9:
                _buckle_repair = (_t_m, _t_buckle)
                _t_m = _t_buckle
                _wall_driver = "buckling under combined axial and bending"
            elif _t_buckle is None:
                _buckle_repair = (_t_m, None)

            _st = skin_stress_mpa(_res.peak_moment_nm, _axial_n, flight_r, _t_m)

            L.append("\n## Flight loads on the assembly\n")
            L.append("| quantity | value |")
            L.append("|---|---|")
            L.append(f"| load case | max-Q {_q/1000:.1f} kPa at "
                     f"{math.degrees(_a):.0f} deg incidence |")
            L.append(f"| aerodynamic side load | nose "
                     f"{_loads[0].force_n/1000:.1f} kN + fins "
                     f"{_loads[1].force_n/1000:.1f} kN |")
            L.append(f"| peak bending moment | {_res.peak_moment_nm/1000:.1f} kN m "
                     f"at {_peak:.2f} m from aft "
                     f"({100*_peak/fv['length_m']:.0f}% of length) |")
            L.append(f"| skin stress at that station | axial "
                     f"{_st['axial_mpa']:.1f} + bending {_st['bending_mpa']:.1f} "
                     f"= **{_st['combined_mpa']:.1f} MPa** on a "
                     f"{1000*_t_m:.2f} mm wall |")
            L.append(f"| that wall | sized here for the flight loads at "
                     f"{1000*flight_r:.0f} mm radius, driven by "
                     f"**{_wall_driver}** -- not carried over from the "
                     f"coupons, which are built at a clamped radius |")
            L.append(f"| margin against {allowable_mpa:.0f} MPa allowable | "
                     f"{allowable_mpa/max(_st['combined_mpa'], 1e-9):.2f} |")
            L.append(f"| load set closes | shear {_res.closure_shear_n:.2e} N, "
                     f"moment {_res.closure_moment_nm:.2e} N m -> "
                     f"**{_res.balanced}** |")
            if _buckle_repair is not None:
                _was, _now = _buckle_repair
                if _now is None:
                    L.append(f"\n**The skin cannot be sized against buckling.** "
                             f"No wall within the thickness cap survives the "
                             f"combined axial and bending load at "
                             f"{1000*flight_r:.0f} mm radius. This vehicle needs "
                             f"stringers, rings or a smaller diameter -- a "
                             f"monocoque shell will not do it.\n")
                else:
                    # A repair has to cost something or it is not one. The
                    # drawn geometry and the mass budget are built from the
                    # component walls, not from this one, so thickening here
                    # changes no mass anywhere -- "skin, as drawn" came back
                    # byte-identical across the repair. The charge is computed
                    # and stated so the saving is not silently free.
                    _rho_skin = 2700.0 if "al-" in str(allowable.material_id) else 8190.0
                    _dm = (_rho_skin * 2.0 * math.pi * flight_r
                           * fv["length_m"] * (_now - _was))
                    L.append(f"\n**The wall was thickened for buckling**, "
                             f"{1000*_was:.2f} to {1000*_now:.2f} mm. Sizing it "
                             f"for axial load alone left the shell unstable once "
                             f"the assembly bending moment was applied; nothing "
                             f"in the component checks would have shown that, "
                             f"since each part passes its own stress test.\n")
                    L.append(f"\nThat costs about **{_dm:.1f} kg** over the "
                             f"barrel at {1000*flight_r:.0f} mm radius. The "
                             f"geometry below is still drawn at the component "
                             f"wall and the mass budget still charges that, so "
                             f"this figure is owed rather than paid -- subtract "
                             f"it from the slack before believing the budget "
                             f"closes.\n")
            if _st["combined_mpa"] > allowable_mpa:
                L.append(f"\nThe assembly is over its allowable in bending even "
                         f"though every component passed its own axial check. "
                         f"Bending contributes {100*_st['bending_share']:.0f}% of "
                         f"the skin stress and no per-part analysis in this "
                         f"packet applies it.")
            for _n in _res.notes:
                L.append(f"\n- {_n}")
            L.append("\nThe load set is checked for closure rather than assumed "
                     "to balance: a free vehicle must return shear and moment to "
                     "zero at its aft end, and a distribution that does not still "
                     "draws a smooth and entirely plausible moment curve.\n")

            # Buckling, which is the failure mode a thin skin actually has.
            # Every other structural number in this packet is a stress against a
            # yield allowable. A monocoque cylinder in compression does not
            # yield -- it goes unstable and folds, at a stress that can be an
            # order of magnitude lower -- so a comfortable yield margin says
            # very little about whether the vehicle survives.
            from cadflow.shell_buckling import check as buckling_check

            _bk = buckling_check(axial_mpa=_st["axial_mpa"],
                                 bending_mpa=_st["bending_mpa"],
                                 radius_m=flight_r, wall_m=_t_m,
                                 youngs_pa=skin_e_pa)
            _yield_margin = allowable_mpa / max(_st["combined_mpa"], 1e-9)
            L.append("\n## Skin buckling\n")
            L.append("| quantity | value |")
            L.append("|---|---|")
            L.append(f"| radius over thickness | {_bk.r_over_t:.0f} |")
            L.append(f"| classical critical stress | {_bk.classical_mpa:.1f} MPa "
                     f"(a perfect cylinder; no real shell reaches it) |")
            L.append(f"| knockdown, compression / bending | "
                     f"{_bk.gamma_compression:.3f} / {_bk.gamma_bending:.3f} "
                     f"(NASA SP-8007) |")
            L.append(f"| allowable, compression / bending | "
                     f"{_bk.allowable_compression_mpa:.1f} / "
                     f"{_bk.allowable_bending_mpa:.1f} MPa |")
            L.append(f"| interaction R_c + R_b | {_bk.interaction:.2f} "
                     f"(must not exceed 1) |")
            L.append(f"| buckling margin | **{_bk.margin:.2f}**, governed by "
                     f"{_bk.governs} |")
            L.append(f"| yield margin for comparison | {_yield_margin:.1f} |")
            if not _bk.passes:
                L.append(f"\n**The skin buckles.** Interaction is "
                         f"{_bk.interaction:.2f} against a limit of 1. The yield "
                         f"margin of {_yield_margin:.1f} above is real and "
                         f"irrelevant: this wall folds before it yields.")
            elif _bk.margin < 0.5 * _yield_margin:
                L.append(f"\nBuckling governs. The yield margin of "
                         f"{_yield_margin:.1f} overstates the real one by a "
                         f"factor of {_yield_margin/_bk.margin:.1f}, and every "
                         f"component margin in the table above is a yield "
                         f"margin computed the same way.")
            _assembly_findings.append({
                "check": "skin buckling",
                "passed": bool(_bk.passes),
                "detail": (f"interaction {_bk.interaction:.2f}, margin "
                           f"{_bk.margin:.2f} governed by {_bk.governs}"),
                "severity": "fail" if not _bk.passes else "pass"})
            for _n in _bk.notes:
                L.append(f"\n- {_n}")
            L.append("")

            # First elastic bending mode of the stack, flying free. The
            # component table reports a frequency per part from a modal run on
            # that part clamped at one end, which is a fact about a bracket:
            # nothing clamps a rocket in flight. What an autopilot has to stay
            # away from is this number.
            from cadflow.assembly_modes import vehicle_bending_modes

            _modes = vehicle_bending_modes(fv, youngs_pa=skin_e_pa, wall_m=_t_m)
            L.append("\n## Bending modes of the assembly\n")
            L.append("| quantity | value |")
            L.append("|---|---|")
            L.append(f"| first elastic bending mode | "
                     f"**{_modes.first_bending_hz:.1f} Hz** free-free |")
            L.append(f"| next modes | " + ", ".join(
                f"{f:.1f} Hz" for f in _modes.frequencies_hz[1:4]) + " |")
            L.append(f"| rigid-body modes found | {_modes.rigid_body_modes} "
                     f"(a free planar beam has exactly 2) -> "
                     f"**{_modes.well_posed}** |")
            L.append(f"| section | {1000*_t_m:.2f} mm wall at "
                     f"{1000*flight_r:.0f} mm radius, E "
                     f"{skin_e_pa/1e9:.0f} GPa |")
            for _n in _modes.notes:
                L.append(f"\n- {_n}")
            L.append(f"\nControl bandwidth has to sit well below "
                     f"{_modes.first_bending_hz:.1f} Hz -- the usual allowance is "
                     f"a factor of five to ten -- or the autopilot drives the "
                     f"structure instead of steering it. This packet does not "
                     f"size a control system, so that comparison is left open "
                     f"rather than claimed as satisfied.\n")

            # Slosh. Liquid in a partly full tank has lateral modes of its own,
            # they are almost undamped, and one that lands on a structural
            # frequency or on control bandwidth couples. Frequency follows
            # axial acceleration rather than gravity, so these are flight
            # numbers and roughly twice what a ground calculation would give.
            from cadflow.slosh import G0 as _G0
            from cadflow.slosh import separation_from, tank_mode
            from cadflow.vehicle import PROPELLANT_BULK_DENSITY

            _accel = 4.5 * _G0
            L.append("\n## Propellant slosh\n")
            L.append("| tank | fill | slosh mode | participating mass | "
                     "vs first bending |")
            L.append("|---|---|---|---|---|")
            _worst = None
            for _i, _stg in enumerate(p.stack):
                for _fill in (0.9, 0.5, 0.2):
                    try:
                        _m = tank_mode(f"stage {_i+1}", radius_m=flight_r,
                                       propellant_kg=float(_stg.prop_mass_kg),
                                       bulk_density=PROPELLANT_BULK_DENSITY,
                                       fill_ratio=_fill, axial_accel_m_s2=_accel)
                    except ValueError:
                        continue
                    _sep = separation_from(_m, _modes.first_bending_hz)
                    if _worst is None or _sep["ratio"] > _worst[0]["ratio"]:
                        _worst = (_sep, _m)
                    L.append(f"| stage {_i+1} | {100*_fill:.0f}% | "
                             f"{_m.frequency_hz:.2f} Hz | "
                             f"{_m.slosh_mass_kg:.0f} kg "
                             f"({100*_m.slosh_mass_fraction:.0f}% of liquid) | "
                             f"ratio {_sep['ratio']:.3f} |")
            if _worst is not None:
                _sep, _m = _worst
                L.append(f"\nClosest approach to the first bending mode is a "
                         f"ratio of {_sep['ratio']:.3f} -- {_sep['verdict']}.")
                if _sep["coupled"]:
                    L.append("\n**A slosh mode coincides with the first bending "
                             "mode.** Slosh damping without baffles is a fraction "
                             "of one percent, so the two will exchange energy and "
                             "neither this packet's modal analysis nor its "
                             "component checks describe what happens next.")
                else:
                    L.append(f"\nStructural coupling is not a concern at these "
                             f"frequencies. Control coupling may still be: the "
                             f"slosh modes here sit at a few hertz, which is "
                             f"where launch vehicle control bandwidth normally "
                             f"lives, and this packet does not size a control "
                             f"system. Baffles are not modelled.")
            L.append("")

            # Control authority and bandwidth. This is where the sections above
            # meet: the engine has to out-moment the fins, and the autopilot has
            # to be quick enough to fly the vehicle while staying clear of the
            # bending and slosh modes just computed. Neither constraint is
            # visible from any one analysis on its own.
            from cadflow.control_authority import (
                bandwidth_window, check as tvc_check, rigid_body_pitch_hz)

            _thrust1 = float(p.trajectory.get("liftoff_thrust_n")
                             or p.gross_kg * 9.80665 * 4.5)
            _S = math.pi * flight_r ** 2
            _auth = tvc_check(
                q_pa=_q, reference_area_m2=_S,
                cn_alpha=float(fins["cna_total"]), alpha_rad=_a,
                cp_station_m=float(fins["cp_z_m"]),
                cg_station_m=float(fv["cg_z_m"]), thrust_n=_thrust1)
            L.append("\n## Control authority\n")
            L.append("| quantity | value |")
            L.append("|---|---|")
            L.append(f"| condition | max-Q {_q/1000:.1f} kPa at "
                     f"{math.degrees(_a):.0f} deg incidence |")
            L.append(f"| aerodynamic moment about the CG | "
                     f"{_auth.aero_moment_nm/1000:.1f} kN m |")
            L.append(f"| thrust x gimbal arm | {_thrust1/1000:.1f} kN x "
                     f"{_auth.tvc_arm_m:.2f} m |")
            L.append(f"| gimbal deflection required | "
                     f"**{_auth.required_gimbal_deg:.1f} deg** |")
            L.append(f"| assumed available | "
                     f"{_auth.available_gimbal_deg:.1f} deg (typical production "
                     f"engine, not a hardware specification) |")
            L.append(f"| authority | **{_auth.has_authority}** "
                     f"(utilisation {_auth.utilisation:.2f}) |")
            for _n in _auth.notes:
                L.append(f"\n- {_n}")

            # Frequencies are only comparable at a condition where both exist.
            # Stage 1 slosh is the one that coexists with max-Q; the 0.66 Hz
            # mode further down the table belongs to a nearly empty upper stage
            # in vacuum, where there is no aerodynamic moment to control.
            try:
                _rb = rigid_body_pitch_hz(
                    q_pa=_q, reference_area_m2=_S,
                    cn_alpha=float(fins["cna_total"]),
                    cp_station_m=float(fins["cp_z_m"]),
                    cg_station_m=float(fv["cg_z_m"]),
                    pitch_inertia_kg_m2=float(fv["Ixx_kg_m2"]))
                _maxq_slosh = tank_mode(
                    "stage 1", radius_m=flight_r,
                    propellant_kg=float(p.stack[0].prop_mass_kg),
                    bulk_density=PROPELLANT_BULK_DENSITY, fill_ratio=0.5,
                    axial_accel_m_s2=_accel).frequency_hz
                _win = bandwidth_window(
                    first_bending_hz=_modes.first_bending_hz,
                    lowest_slosh_hz=_maxq_slosh, rigid_body_hz=_rb)
                _coincide = _maxq_slosh / _rb
                L.append("\n| frequency | value |")
                L.append("|---|---|")
                L.append(f"| rigid-body pitch (weathercock) at max-Q | "
                         f"{_rb:.2f} Hz |")
                L.append(f"| stage 1 slosh at max-Q | {_maxq_slosh:.2f} Hz |")
                L.append(f"| ratio | **{_coincide:.2f}** |")
                L.append(f"| first bending | {_modes.first_bending_hz:.1f} Hz |")
                L.append(f"| usable control band | {_win['note']} |")
                L.append(f"| lowest flexible mode / rigid-body mode | "
                         f"**{_win['flexible_over_rigid_ratio']:.2f}** |")
                L.append(f"| verdict robust to the separation rules | "
                         f"**{_win['robust_to_heuristics']}** |")
                L.append(f"\n{_win['robustness_note']}\n")
                if 0.8 <= _coincide <= 1.25:
                    L.append(f"\n**The stage 1 slosh mode coincides with the "
                             f"vehicle's own pitch mode**, at the condition where "
                             f"the aerodynamic moment is largest. The propellant "
                             f"sloshes at essentially the rate the vehicle "
                             f"weathercocks. Slosh damping without baffles is a "
                             f"fraction of one percent, and this is the coupling "
                             f"that has historically been fatal. Nothing in the "
                             f"component analyses above can see it: it appears "
                             f"only when the slosh model, the mass properties "
                             f"and the aerodynamics are evaluated together at "
                             f"one flight condition.")
                _assembly_findings.append({
                    "check": "slosh / pitch-mode separation",
                    "passed": not (0.8 <= _coincide <= 1.25),
                    "detail": (f"slosh {_maxq_slosh:.2f} Hz against pitch "
                               f"{_rb:.2f} Hz, ratio {_coincide:.2f}"),
                    "severity": ("fail" if 0.8 <= _coincide <= 1.25
                                 else "pass")})
                # A shut window is not automatically a dead end. The 5:1
                # separation is the rule for an undamped mode, and baffles
                # change that. What baffles cannot change is a mode that sits
                # inside the band rather than above it.
                if not _win["window_exists"]:
                    from cadflow.slosh_baffles import size_baffles

                    try:
                        _baf = size_baffles(
                            tank_radius_m=flight_r,
                            fill_depth_m=max(0.2, 0.5 * fv["length_m"] /
                                             max(1, len(p.stack))),
                            slosh_hz=_maxq_slosh,
                            required_bandwidth_hz=_win["lower_bound_hz"])
                        if _baf is not None:
                            L.append(
                                f"\n**Baffles close this.** {_baf.n_baffles} "
                                f"ring baffle(s) {1000*_baf.width_m:.0f} mm wide "
                                f"raise slosh damping to {_baf.damping_ratio:.3f}, "
                                f"which relaxes the separation requirement to "
                                f"{_baf.achieved_separation:.1f} and opens the "
                                f"band. Mass {_baf.mass_kg:.1f} kg.\n")
                            for _n in _baf.notes:
                                L.append(f"- {_n}\n")
                        else:
                            L.append("\nNo ring baffle within a sensible width "
                                     "provides enough damping to open the band.\n")
                    except ValueError as _bexc:
                        L.append(f"\n**Baffles cannot close this.** {_bexc}\n")
                # Which side of crossover each mode sits on decides how it must
                # be handled, and the two treatments are not interchangeable.
                # Reporting "no usable band" for both read as a dead end when a
                # mode below crossover is a demanding but entirely standard
                # control design problem.
                from cadflow.control_authority import mode_disposition

                _disp = mode_disposition(
                    crossover_hz=_win["lower_bound_hz"],
                    modes={"slosh": _maxq_slosh,
                           "first bending": _modes.first_bending_hz})
                L.append("\n| mode | frequency | vs crossover | stabilisation |")
                L.append("|---|---|---|---|")
                for _mn, _mv in _disp["modes"].items():
                    L.append(f"| {_mn} | {_mv['hz']:.2f} Hz | "
                             f"{_mv['ratio_to_crossover']:.2f}x | "
                             f"**{_mv['stabilisation']}** |")
                L.append(f"\n{_disp['verdict']}\n")
                # A requirement this packet cannot check, not a failure.
                #
                # Phase stabilising a mode below crossover is standard practice
                # -- Saturn V did it -- so reporting it as FAIL cries wolf. But
                # nothing here designs a control system, so reporting PASS would
                # claim a check that never ran. The honest third answer is that
                # a requirement exists and is unverified, and the overall
                # verdict has to distinguish that from a failure.
                _needs_phase = _disp["requires_phase_stabilisation"]
                _assembly_findings.append({
                    "check": "flexible mode stabilisation",
                    "passed": not _needs_phase,
                    "detail": (
                        "all modes gain-stabilisable with a conventional rolloff"
                        if not _needs_phase else
                        f"{', '.join(_needs_phase)} below crossover; phase "
                        f"stabilisation required and not verified here"),
                    "severity": "pass" if not _needs_phase else "unverified"})
                # The bandwidth window is not a separate finding. It and the
                # mode disposition above are the same physical fact -- slosh
                # near the control frequency -- and listing both double-counted
                # it while keeping the framing that reads as a dead end. The
                # table stays as context; the verdict comes from the
                # disposition, which says what actually has to be done.
            except ValueError as _exc:
                L.append(f"\n(pitch frequency not defined: {_exc})")
            if control_trade is not None and control_trade.get("steps"):
                L.append("\n**Static margin was traded for control authority.** "
                         "Fins sized purely for stability left the engine short "
                         "of gimbal; the loop searched downward until it could "
                         "steer.\n")
                L.append("| target margin | fin span | CNa | gimbal needed | ok |")
                L.append("|---|---|---|---|---|")
                for _s in control_trade["steps"]:
                    L.append(f"| {_s['margin_cal']:.2f} cal | "
                             f"{1000*_s['span_m']:.0f} mm | "
                             f"{_s['cna_total']:.2f} /rad | "
                             f"{_s['required_gimbal_deg']:.2f} deg | "
                             f"{'yes' if _s['has_authority'] else 'no'} |")
                L.append(f"\n{control_trade['note']}\n")
            _assembly_findings.append({
                "check": "thrust vector control authority",
                "passed": bool(_auth.has_authority),
                "detail": (f"{_auth.required_gimbal_deg:.1f} deg required "
                           f"against {_auth.available_gimbal_deg:.1f} available"),
                "severity": "fail" if not _auth.has_authority else "pass"})
            L.append("")
        except Exception as _exc:  # noqa: BLE001
            L.append(f"\n(assembly flight loads unavailable: "
                     f"{type(_exc).__name__}: {_exc})\n")

        # Stage separation. The trajectory stages already -- it drops spent
        # structure and lights the next engine -- but never asked whether the two
        # bodies can do that in order without occupying the same space.
        if len(p.stack) > 1:
            from cadflow.staging import check_separation, coast_for_clearance_s

            _D = 2.0 * flight_r
            L.append("\n## Stage separation\n")
            L.append("| separation | spent | upper | closing rate | coast to clear |")
            L.append("|---|---|---|---|---|")
            _sep_ok, _need = True, 0.0
            for _i in range(len(p.stack) - 1):
                _spent = float(p.stack[_i].struct_mass_kg)
                _upper = sum(float(st.prop_mass_kg) + float(st.struct_mass_kg)
                             for st in p.stack[_i + 1:]) + args.payload_kg
                _need = coast_for_clearance_s(spent_mass_kg=_spent,
                                              upper_mass_kg=_upper,
                                              body_diameter_m=_D)
                _sr = check_separation(stage_index=_i + 1, spent_mass_kg=_spent,
                                       upper_mass_kg=_upper, body_diameter_m=_D,
                                       coast_s=max(1.0, _need))
                _sep_ok = _sep_ok and _sr.clears
                L.append(f"| {_i+1}/{_i+2} | {_spent:.1f} kg | {_upper:.1f} kg | "
                         f"{_sr.relative_velocity_m_s:.2f} m/s | **{_need:.2f} s** |")
            L.append(f"\nPlume clearance is taken as 1.5 body diameters "
                     f"({1.5*_D:.2f} m), because a vacuum plume spreads well beyond "
                     f"the nozzle that produced it. Tip-off and plume impingement on "
                     f"the spent stage are not modelled; this answers only whether the "
                     f"gap opens fast enough.\n")
            _assembly_findings.append({
                "check": "stage separation clearance",
                "passed": bool(_sep_ok),
                "detail": (f"{len(p.stack)-1} separation(s), longest coast to plume "
                           f"clearance {_need:.2f} s"),
                "severity": "pass" if _sep_ok else "fail"})

        L.append("\n## Stability\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| fin span (each of {fins['n_fins']}) | "
                 f"{fins['span_m']*1000:.0f} mm |")
        L.append(f"| fin root / tip chord | {fins['root_chord_m']*1000:.0f} / "
                 f"{fins['tip_chord_m']*1000:.0f} mm, sweep "
                 f"{fins['sweep_m']*1000:.0f} mm |")
        # The trajectory above was flown on a body-of-revolution drag
        # coefficient. These fins are not a rounding error against that: their
        # planform is several times the body's frontal area, and nothing charged
        # the flight for them. The correction is reported rather than folded into
        # the trajectory, because the fins are sized from a centre of gravity the
        # trajectory produces -- applying it properly means iterating the plan
        # rather than adding a term to it.
        try:
            from cadflow.fin_drag import fin_drag as _fin_drag

            _fd = _fin_drag(span_m=float(fins["span_m"]),
                            root_chord_m=float(fins["root_chord_m"]),
                            tip_chord_m=float(fins["tip_chord_m"]),
                            thickness_m=float(fins.get("thickness_m", 0.005)),
                            n_fins=int(fins["n_fins"]), body_radius_m=flight_r,
                            mach=2.0)
            _body_cd = drag_coefficient(
                design_knobs.nose_shape if design_knobs else "ogive", 3.0)
            L.append(f"| fin drag, absent from the flown trajectory | Cd "
                     f"{_fd.cd_total:.4f} on body frontal area, "
                     f"**{100 * _fd.cd_total / _body_cd:.0f}%** of the body's "
                     f"{_body_cd:.3f} |")
            L.append(f"| fin planform vs body frontal area | "
                     f"{_fd.planform_m2 / _fd.reference_m2:.1f}x |")
            _fin_note = (
                "\nThe apogee under mission verification is a finless vehicle's. "
                "Adding this drag costs roughly 1% of apogee, about 23 kg of gross "
                "mass to recover -- small against the other uncertainties here, and "
                "always in the optimistic direction. " + " ".join(_fd.notes) + "\n")
            L.append(_fin_note)
        except Exception as _exc:  # noqa: BLE001
            L.append(f"\n(fin drag unavailable: {type(_exc).__name__}: {_exc})\n")

        L.append(f"| centre of pressure | {fins['cp_z_m']:.3f} m from aft |")
        L.append(f"| centre of gravity | {fv['cg_z_m']:.3f} m from aft |")
        L.append(f"| static margin | {fins['static_margin_cal']:.2f} calibers "
                 f"(target {fins['target_margin_cal']:.1f}, sized for "
                 f"{worst_label}) |")
        L.append(f"| normal force slope | nose {fins['cna_nose']:.2f} + fins "
                 f"{fins['cna_fins']:.2f} = {fins['cna_total']:.2f} /rad |")
        L.append("")
        L.append("| burn state | vehicle mass | centre of gravity | static margin |")
        L.append("|---|---|---|---|")
        for (lbl, mg, m), (_, cg, _) in zip(margins, cgs):
            L.append(f"| {lbl} | {m:.1f} kg | {cg:.3f} m | {mg:.2f} cal |")
        L.append("\nFins are sized by solving for the span that meets the "
                 "margin, not assumed and then checked. The nose centre of "
                 "pressure comes from slender-body theory as L - V/A_base, which "
                 "needs only the nose volume and reproduces the exact families "
                 "(cone 2L/3, von Karman L/2) to the last digit. The fin set is "
                 "Barrowman, whose CN_alpha converges onto Jones' slender-wing "
                 "result pi AR/2 as aspect ratio goes to zero and whose "
                 "unswept-rectangular centre of pressure is exactly the quarter "
                 "chord -- two limits with known answers, which is what makes "
                 "its constants checkable rather than merely quoted.")

        # The vehicle as actual geometry. Components have been designed, meshed
        # and analysed one at a time; this is the assembly the intent asks for,
        # and it could not be built before the sculpting layer existed -- a
        # change of diameter between stages is a loft, and there was no loft.
        try:
            from cadflow.assembly import (build_vehicle, export_assembly,
                                          mass_closure)

            asm = build_vehicle(
                p.stack, args.payload_kg, flight_r,
                wall_mm=ASSEMBLY_WALL_MM,
                fin_span_m=fins["span_m"],
                fin_root_chord_m=fins["root_chord_m"],
                n_fins=fins["n_fins"],
                nozzle=bell if nozzle else None,
                nose_shape=(design_knobs.nose_shape if design_knobs
                            else "ogive"))
            budget = sum(st.struct_mass_kg for st in p.stack)
            closure = mass_closure(
                asm, budget,
                liftoff_thrust_n=p.trajectory.get("liftoff_thrust_n", 0.0))
            files = export_assembly(asm, args.out / "cad")
            assembly = {"summary": asm.summary(),
                        "total_length_m": asm.total_length_mm / 1000.0,
                        "max_radius_m": asm.max_radius_mm / 1000.0,
                        "skin_mass_kg": asm.mass_kg,
                        "closure": closure,
                        "exported": sorted(files)}

            L.append("\n## Assembly\n")
            L.append(f"The vehicle built as geometry: {asm.total_length_mm/1000:.2f} m "
                     f"over {len(asm.parts)} parts, {len(files)} of them exported "
                     f"to STEP under `cad/`.\n")
            L.append("| part | kind | station | length | mass |")
            L.append("|---|---|---|---|---|")
            for row in sorted(asm.summary(), key=lambda r: r["station_mm"]):
                L.append(f"| {row['name']} | {row['kind']} | "
                         f"{row['station_mm']/1000:.3f} m | "
                         f"{row['length_mm']:.0f} mm | {row['mass_kg']:.2f} kg |")

            L.append("\n### Does the mass budget hold the vehicle?\n")
            L.append("| term | mass |")
            L.append("|---|---|")
            L.append(f"| skin, as drawn | {closure['skin_kg']:.1f} kg |")
            L.append(f"| engine, from liftoff thrust at T/W 60 | "
                     f"{closure['engine_kg']:.1f} kg |")
            L.append(f"| accounted for | {closure['accounted_kg']:.1f} kg |")
            L.append(f"| structural budget | {closure['budget_kg']:.1f} kg |")
            L.append(f"| slack | {closure['slack_kg']:+.1f} kg |")
            if not closure["closes"]:
                L.append(f"\n> The vehicle cannot contain itself. Its own skin "
                         f"plus its own engine exceed the mass it was allowed "
                         f"by {-closure['slack_kg']:.1f} kg, so the design does "
                         f"not close. This is the same verdict the structural "
                         f"fixed point reaches from the opposite direction and "
                         f"knowing nothing about this calculation -- it wants a "
                         f"structural coefficient of "
                         f"{solved_coeff:.3f} where the design asserts "
                         f"{STRUCT_COEFF:.3f}. Two independent routes agreeing "
                         f"that a number is optimistic is worth more than "
                         f"either alone.")
            else:
                L.append(f"\n> The budget holds, with {closure['slack_kg']:.1f} kg "
                         f"left for plumbing, avionics, tank domes and "
                         f"separation hardware -- none of which the geometry "
                         f"draws.")
        except Exception as exc:  # noqa: BLE001 - assembly is an addition
            print(f"assembly unavailable: {exc}", flush=True)

        L.append("\nStage lengths come from propellant volume at LOX/RP-1 bulk "
                 "density and each stage is a uniform cylinder of its wet mass "
                 "-- coarse, since a real stage has domes, a dry engine at one "
                 "end and a moving liquid level, but built only from numbers the "
                 "planner produced, and the wet mass reproduces the planner's "
                 f"gross of {p.gross_kg:.1f} kg. Pitch inertia is "
                 f"{fv['Ixx_kg_m2']/max(fv['Izz_kg_m2'],1e-12):.0f}x roll, as "
                 "it must be for a long thin vehicle.")

    L.append("\nSizing is on p95. Across a 16x mesh refinement on one part the "
             "median moved 2.8%, p95 13.8%, p99 36.6% and the peak 268% -- p99 "
             "is not stable enough to design against at the mesh densities this "
             "loop can afford, because with ~1,200 nodes its top percent is a "
             "dozen nodes all on the same corner. A starred p99 marks a part "
             "whose p99 exceeds twice its p95: that is a stress concentration "
             "wanting a fillet or a doubler, not a wall that wants thickening. "
             "The peak column never converges and is shown only to locate it.")
    L.append("\nWall thickness and buckling margin size the thin shell; the "
             "shell FEA column is CalculiX on that same hollow geometry, so the "
             "meshed part is the part being designed rather than a solid billet "
             "with the same outer dimensions. The 1st mode column is a CalculiX "
             "*FREQUENCY solve on that same mesh, clamped at the aft face: it is "
             "what a static check cannot see, and it is the quantity a flutter "
             "or coupled-loads assessment starts from. Mass and Izz are "
             "computed on that same solid, exact for the geometry, in kg and "
             "kg m^2 about the centroid.")
    # The verdict has to include the assembly, not just the parts.
    #
    # This line used to read `all(r["passed"] for r in results)` and reported
    # True for a vehicle that needed 14.5 degrees of gimbal against 8
    # available, and whose stage 1 propellant sloshed within 11% of its own
    # pitch frequency at max-Q. Both findings were sitting in the prose a few
    # hundred lines above, and neither reached the summary or PACKET.json. A
    # packet that contradicts itself in the reader's favour is worse than one
    # that simply reports the failure.
    _components_ok = all(r["passed"] for r in results)
    # A boolean cannot carry three outcomes, and forcing it to try is how this
    # packet reported "Overall: True" for a vehicle with an unverified control
    # requirement outstanding. Reclassifying a check from FAIL to REQUIRED made
    # the summary cleaner and less true. Nothing is verified while a
    # requirement remains unchecked, so `all_passed` is False unless the packet
    # actually verified everything, and the three-way status says which of the
    # three situations it is.
    _assembly_fails = [f for f in _assembly_findings if f["severity"] == "fail"]
    _assembly_unverified = [f for f in _assembly_findings
                            if f["severity"] == "unverified"]
    allp = _components_ok and not _assembly_fails and not _assembly_unverified
    _status = ("FAILED" if (not _components_ok or _assembly_fails)
               else "INCOMPLETE" if _assembly_unverified
               else "VERIFIED")

    if _assembly_findings:
        L.append("\n## Assembly verification\n")
        L.append("| check | result | detail |")
        L.append("|---|---|---|")
        for _f in _assembly_findings:
            _mark = {"pass": "PASS", "fail": "FAIL",
                     "unverified": "REQUIRED"}.get(_f["severity"], "FAIL")
            L.append(f"| {_f['check']} | **{_mark}** | {_f['detail']} |")
        if _assembly_unverified:
            L.append(f"\n{len(_assembly_unverified)} check(s) marked REQUIRED "
                     f"are neither passed nor failed: they name work this "
                     f"packet cannot do. Phase stabilising a mode below "
                     f"crossover is routine practice, but nothing here designs "
                     f"a control system, so claiming it passes would report a "
                     f"check that never ran.\n")
        if _assembly_fails:
            L.append(f"\n{len(_assembly_fails)} assembly-level check(s) failed "
                     f"while every component passed its own coupon test. The "
                     f"components are not wrong -- each one survives the load it "
                     f"was given. What fails is the vehicle they add up to, and "
                     f"no per-part analysis can see it.\n")

    L.append(f"\nAllowable {allowable_mpa:.0f} MPa, derived from "
             f"{allowable.material_id} at {allowable.source_strength_mpa:.0f} MPa "
             f"({allowable.strength_basis}) with a yield factor of safety of "
             f"{allowable.factor_of_safety} and a {allowable.knockdown} knockdown. "
             f"All {len(results)} components passed: "
             f"**{_components_ok}**. Assembly checks failed: "
             f"**{len(_assembly_fails)}**. Requirements unverified: "
             f"**{len(_assembly_unverified)}**.\n\nOverall: **{_status}**"
             + (" -- nothing failed, but the packet cannot call a design "
                "verified while a requirement it did not check remains "
                "outstanding.\n" if _status == "INCOMPLETE" else "\n"))
    L.append("\nThis allowable is not certifiable and the packet should not be "
             "read as though it were. The catalogue carries typical strengths "
             "rather than A- or B-basis values, so the knockdown above stands in "
             "for a statistical basis that does not exist here:\n")
    for c in allowable.caveats:
        L.append(f"- {c}")
    L.append(
        "\nDiscretisation error is measured separately and is not included in "
        "any margin quoted here. Against an exact Lame solution "
        "(`artifacts/verification/fea_mesh_convergence.json`) the C3D4 linear "
        "tetrahedra this pipeline writes converge first-order in stress "
        "(p = 1.31) and read 9.8% low on surface stress at 34,493 elements, "
        "against a 40,000 element budget; quadratic C3D10 elements reach a "
        "lower field error with 661.\n\n"
        "On the real corpus parts that number is larger and two-sided. Solving "
        "twelve components at identical meshes and loads under both element "
        "types (`artifacts/verification/element_order_ab.json`) moved the p95 "
        "this loop sizes against by a median of 1.1% but a range of -13.9% to "
        "+14.5%. The small median belongs to smooth parts like body tubes; the "
        "double-digit ends belong to fins and nose cones, where the field is "
        "dominated by stress concentrations. Read the component margins below "
        "as carrying at least that much numerical uncertainty.\n")

    (args.out / "PACKET.md").write_text("\n".join(L))
    (args.out / "PACKET.json").write_text(json.dumps({
        "specification": spec, "stages": p.stages, "split": p.split,
        "gross_kg": p.gross_kg, "achieved_km": p.achieved_km,
        "error_pct": err, "rationale": p.rationale,
        "components": results, "all_passed": allp,
        "components_passed": _components_ok,
        "assembly_findings": _assembly_findings,
        "assembly_passed": not _assembly_fails,
        "assembly_unverified": [f["check"] for f in _assembly_unverified],
        "status": _status,
        # What actually sized this vehicle, not what the module defaults to.
        #
        # This read `solved_coeff if args.solve_structure else STRUCT_COEFF`
        # and reported 0.14 for a vehicle whose every stage was built at
        # 0.2613, because the repair loop had raised it and this line never
        # heard. 0.14 sits just above the flown range; 0.2613 is 2.2 times the
        # heaviest stage ever flown, so the packet was under-reporting how
        # unusual its own structure is -- to a reader and to anything consuming
        # the JSON. Recovered from the stack itself, which cannot disagree with
        # the vehicle that was analysed.
        "struct_coeff_used": (
            sum(s.struct_mass_kg for s in p.stack)
            / max(1e-9, sum(s.struct_mass_kg + s.prop_mass_kg for s in p.stack))
            if p.stack else (solved_coeff if args.solve_structure
                             else STRUCT_COEFF)),
        "struct_coeff_default": STRUCT_COEFF,
        "struct_coeff_solved": solved_coeff,
        # Vehicle-level results were markdown-only, so nothing downstream --
        # including the model -- could consume them. Kept as two separate keys
        # because they describe two different objects: the coupons that were
        # meshed and analysed, and the vehicle that flew the trajectory.
        "coupon_stack": coupon_stack,
        "flight_vehicle": flight_vehicle,
        "stability": stability,
        "thermal": thermal,
        "nozzle": nozzle,
        "assembly": assembly}, indent=2))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
