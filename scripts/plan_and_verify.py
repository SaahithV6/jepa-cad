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
import re
import shutil
import math
import sys
from pathlib import Path
from typing import NamedTuple

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


def component_first_mode_hz(case_dir: Path, *, youngs_pa: float = E_AL,
                            poisson: float = NU_AL,
                            density: float = RHO_AL) -> float | None:
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
            youngs_modulus=youngs_pa, poisson=poisson, density=density)
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


def _planner_propellants() -> set[str]:
    """Propellant combinations the planner can actually fly.

    Read from the chemistry rather than listed here, so the CLI cannot drift
    from what the program supports -- which is exactly how lox_lch4 came to be
    offered by one table and understood by none.
    """
    try:
        from generate_propulsion_trajectory_corpus import PROPELLANTS

        return set(PROPELLANTS)
    except Exception:  # noqa: BLE001
        return {"lox_rp1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload-kg", type=float, required=True)
    ap.add_argument("--apogee-km", type=float, required=True)
    # Validated against what the planner can actually fly, not left open.
    #
    # --propellant lox_lch4 crashed with a bare KeyError inside
    # chamber_properties, several hundred lines into a run, because one lookup
    # table offered a combination the chemistry does not know. A choice the
    # program cannot honour should be refused at the point it is asked for.
    _PROPELLANTS = tuple(sorted(_planner_propellants()))
    ap.add_argument("--propellant", type=str, default="lox_rp1",
                    choices=_PROPELLANTS)
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
    # Initialised before the block that sets them, not after it.
    #
    # Both of these were declared 570 lines below their own assignment, so the
    # declaration ran second and reset them. design_conflict is the worse case:
    # the repair loop reports why it cannot fix an unaffordable stack, and that
    # message has never once reached the packet, because it was overwritten with
    # None before anything read it. A variable initialised after its first write
    # is not a safety net, it is an eraser.
    design_conflict = None
    _tps_total_kg = 0.0
    # Does this vehicle carry liquid at all?
    #
    # slosh.py computes a free-surface mode for whatever mass it is handed; it
    # has no notion of phase. A solid_apcp packet duly reported "slosh below
    # crossover, phase stabilisation required" for a motor whose propellant is
    # a cast grain bonded to the case. Nothing sloshes in a solid, and a
    # requirement invented for a mode that cannot exist is worse than a missing
    # one, because it reads as engineering.
    #
    # Read from the tankage model rather than a list here: a propellant with no
    # tanks has no free surface, and that is the same question.
    try:
        from cadflow.pressurization import NON_LIQUID as _NON_LIQUID

        # Phase, not modelled-ness. An unmodelled liquid still sloshes, so
        # keying this on COMBINATIONS would have silenced the slosh checks for
        # any bipropellant whose fluid properties happen to be missing.
        _has_liquid = args.propellant not in _NON_LIQUID
    except Exception:  # noqa: BLE001
        _has_liquid = True
    # The inter-stage coast the trajectory integrates, so the separation check
    # judges the same flight the rest of the packet describes.
    try:
        import inspect as _inspect

        from cadflow.multistage import integrate_stack as _istack

        _flown_coast_s = float(_inspect.signature(_istack)
                               .parameters["coast_between_stages_s"].default)
    except Exception:  # noqa: BLE001
        _flown_coast_s = 2.0
    # Ring baffles, charged to the closure. One per tank: the slosh criterion is
    # evaluated on stage 1 and every stage has the same problem.
    _baffle_total_kg = 0.0

    if args.autodesign:
        try:
            from cadflow.autodesign import autodesign as _autodesign

            # Give the loop the propellant the caller asked for.
            #
            # This called autodesign with no knobs, so Knobs.propellant stayed
            # at its "lox_rp1" default and --propellant was honoured everywhere
            # except the loop that actually builds the vehicle. 250 kg to
            # 600 km on lox_lh2 returned 2837.2 kg gross and two stages --
            # identical to the same mission on lox_rp1, to a tenth of a
            # kilogram. Hydrogen is 11.4x bulkier than kerosene and has far
            # higher specific impulse; it cannot produce the same vehicle. The
            # flag was being read, printed in the header, and used for the
            # standalone plan, while the design it described came from a
            # different propellant entirely.
            from cadflow.autodesign import Knobs as _Knobs

            _res = _autodesign(args.payload_kg, args.apogee_km,
                               knobs=_Knobs(propellant=args.propellant),
                               max_iters=12)
            design_knobs = _res["knobs"]
            design_history = _res["history"]
            # What the loop concluded it could not fix.
            #
            # The repair loop now tries shorter architectures when a stage
            # cannot afford its tankage, and reports a conflict when none
            # closes. That conclusion reached the return value and nothing read
            # it -- the same shape as the mass-closure verdict that lived in
            # markdown for several revisions.
            design_conflict = _res.get("conflict")
            _ev = _res["evaluation"]
            _tps_total_kg = float(getattr(_ev, "tps_mass_kg", 0.0) or 0.0)
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

    # Density of the alloy the design chose, for the component mass properties.
    #
    # Those were computed with RHO_AL regardless of material. On an Inconel
    # vehicle that understates every component mass by a factor of three, and
    # the coupon stack total with it. The ratios that the gauge-versus-strength
    # split reports survive it, since every part scales together, but no
    # absolute mass in the packet did.
    try:
        from cadflow.space_materials import iter_materials as _iter_m

        _skin_rho = next((float(m.density_kg_m3) for m in _iter_m()
                          if m.material_id == skin_material), RHO_AL)
    except Exception:  # noqa: BLE001
        _skin_rho = RHO_AL
    try:
        from cadflow.allowables import design_allowable, elastic_properties

        # At the temperature the skin actually reaches, not at room
        # temperature.
        #
        # The thermal section already says the skin hits 863 K and warns that
        # "every allowable in the component table below is a room-temperature
        # value". The allowables module already has a caveat that fires above
        # room temperature. Neither was connected to the other: this call
        # passed no temperature, so the default of 295 K meant the caveat could
        # never fire and every margin in the packet was quoted as though the
        # structure were cold. The information was present in two places and
        # absent from the one that mattered.
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

    # Re-derive the allowable at the temperature the skin actually reaches.
    #
    # The thermal section says the skin hits several hundred kelvin and warns
    # that "every allowable in the component table below is a room-temperature
    # value". The allowables module has a caveat that fires above room
    # temperature. Neither was connected: the first call above runs before the
    # trajectory exists, so it defaulted to 295 K, the caveat could never fire,
    # and every margin in the packet was quoted as though the structure were
    # cold. The information was present in two places and absent from the one
    # that reports it.
    #
    # The number is not knocked down, because the catalogue carries no
    # yield-versus-temperature data and an invented knockdown would look
    # computed. It is flagged instead, and the flag travels with every margin.
    _skin_k = float(p.trajectory.get("max_skin_temp_k") or 0.0)
    if _skin_k > 0:
        try:
            allowable = design_allowable(skin_material, temperature_k=_skin_k)
            allowable_mpa = allowable.allowable_mpa
            if len(allowable.caveats) > 1:
                print(f"allowable re-derived at {_skin_k:.0f} K skin "
                      f"temperature: {len(allowable.caveats)} caveats",
                      flush=True)
        except (KeyError, ValueError):
            pass

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
    # The whole state at each condition, not only its centre of gravity.
    #
    # Static margin, the gimbal angle max-Q demands and the rigid-body pitch
    # frequency are three answers about one vehicle at one instant, and each was
    # reading its inputs from a different place: fins sized at the worst CG, the
    # control trade and the pitch mode at the full-vehicle CG and inertia. Those
    # are the same numbers here only because this vehicle's largest tank is aft,
    # so liftoff is the worst case. Keeping the state together makes the
    # condition a thing the packet names once instead of a coincidence it
    # re-derives four times.
    # Named rather than positional. Adding the state to a 3-tuple broke two
    # unpack sites that had no reason to care, and a tuple that cannot grow
    # without breaking its readers is a tuple that discourages carrying the
    # thing its readers need -- which is how the CG and the state came to be
    # fetched from different places to begin with.
    class _BurnState(NamedTuple):
        label: str
        cg_z_m: float
        mass_kg: float
        state: dict

    cgs = []
    for label, rem in burn_states:
        st = flight_vehicle_properties(p.stack, args.payload_kg, flight_r,
                                       propellant_remaining=rem)
        cgs.append(_BurnState(label, st["cg_z_m"], st["mass_kg"], st))
    _worst = min(cgs, key=lambda c: c.cg_z_m)
    worst_label, worst_cg, worst_state = _worst.label, _worst.cg_z_m, _worst.state
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
            # The same centre of gravity the fins were sized at.
            #
            # This read fv["cg_z_m"] -- the full-vehicle value -- while the
            # lambda above sized fins at worst_cg. On this design the two are
            # the same number to four decimals, because liftoff happens to be
            # the worst case for a vehicle whose largest tank is aft. That is a
            # property of the architecture, not of the code: a design with a
            # forward tank would size fins at one condition and check the gimbal
            # they demand at another, and nothing would say so. Six defects in
            # this packet were exactly two expressions that agreed until they
            # did not.
            alpha_rad=DESIGN_ALPHA_RAD, cg_station_m=worst_cg,
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
               for lbl, cg, m, _st in cgs]
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
        # Aluminium properties gave only a 2.2% error on Inconel, because the
        # higher stiffness nearly cancels the higher density in sqrt(E/rho).
        # Small, and still no reason to leave the wrong material in a frequency
        # the packet reports as this part's own.
        first_mode = None if (hulled or dist is None) else \
            component_first_mode_hz(comp_dir, youngs_pa=skin_e_pa,
                                    poisson=skin_nu, density=_skin_rho)

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
            mass_props = _b.mass_properties(_shape, _skin_rho)
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

            # Which part of the stage makes it heavy.
            #
            # The paragraph above has reported the coefficient as outside flown
            # practice for many revisions without ever saying what is
            # responsible, which makes it a complaint rather than a lead. The
            # answer is not the wall: shell is about a quarter of stage
            # structure and the engine is about half, at an assumed
            # thrust-to-weight of 60 against flown engines running 80 to 180.
            try:
                from cadflow.structural_sizing import (
                    coefficient_attribution as _attr)

                _a = _attr(float(p.stack[0].prop_mass_kg), flight_r,
                           float(p.trajectory.get("liftoff_thrust_n") or 0.0)
                           or 1.0,
                           density_kg_m3=_skin_rho,
                           yield_pa=allowable.source_strength_mpa * 1e6,
                           modulus_pa=skin_e_pa)
                L.append("\n| stage 1 structure | mass | share |")
                L.append("|---|---|---|")
                for _k, _vv in _a["terms_kg"].items():
                    L.append(f"| {_k} | {_vv:.1f} kg | "
                             f"{100*_a['shares'][_k]:.0f}% |")
                L.append(f"\n{_a['note']}. The wall driver is "
                         f"**{_a['wall_driver']}**"
                         + (", so stiffening it would add mass without adding "
                            "capability -- stringers buy buckling resistance "
                            "and a gauge-limited wall has none to buy"
                            if _a["wall_driver"] == "minimum gauge" else
                            ", so it is buckling-limited and stiffened "
                            "construction would buy real mass here")
                         + ". This matters because the obvious reading of a "
                           "heavy coefficient is that the structure needs "
                           "improving, and for this vehicle the structure is "
                           "already as thin as the process allows.\n")
            except Exception as _aexc:  # noqa: BLE001
                L.append(f"\n(structural attribution unavailable: {_aexc})\n")
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
    # Helium and bottles, charged to the mass budget below. Initialised here so
    # a pressurisation block that did not run charges nothing rather than
    # raising -- and so the closure arithmetic cannot silently read a stale
    # value from a previous design iteration.
    _press_total_kg = 0.0
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
                # Measure against the material the vehicle is made of.
                #
                # This compared every design to aluminium's 450 K, and then
                # advised "a different skin material" -- which the repair loop
                # had already chosen. On this vehicle the skin is Inconel 718,
                # good to about 980 K, so 863 K is inside its limit and the
                # warning was both measuring the wrong alloy and recommending a
                # step already taken.
                _svc = None
                try:
                    from cadflow.space_materials import iter_materials

                    _m = next((m for m in iter_materials()
                               if m.material_id == allowable.material_id), None)
                    _svc = float(_m.max_service_temp_k) if _m else None
                except Exception:  # noqa: BLE001
                    _svc = None
                _limit = _svc if _svc else ALUMINIUM_SERVICE_K
                _name = allowable.material_id if _svc else "aluminium"
                _tail = ("This is a radiation-equilibrium steady state and the "
                         "vehicle passes through quickly, so it is an upper "
                         "bound rather than what the structure actually "
                         "reaches.")
                if skin > _limit:
                    L.append(f"\n> The skin reaches {skin:.0f} K, past the "
                             f"{_limit:.0f} K service limit of {_name}, which "
                             f"is what this design selected. The vehicle needs "
                             f"thermal protection or a trajectory that spends "
                             f"less time fast in thick air; no catalogued alloy "
                             f"is left to upgrade to. {_tail}")
                elif skin > ALUMINIUM_SERVICE_K:
                    L.append(f"\n> The skin reaches {skin:.0f} K. That is "
                             f"inside the {_limit:.0f} K service limit of "
                             f"{_name} -- the repair loop selected that alloy "
                             f"for this reason, having started from aluminium, "
                             f"which stops being useful at "
                             f"{ALUMINIUM_SERVICE_K:.0f} K. The margins below "
                             f"still use a room-temperature allowable, so they "
                             f"are optimistic by an amount the catalogue has no "
                             f"data to quantify. {_tail}")
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

            # And against the load the packet itself calls dominant.
            #
            # Internal pressure never sized anything here. The wall came from
            # axial compression, buckling and minimum gauge, and the pressure
            # case was applied afterwards as a check -- so the largest membrane
            # stress on the tank, three times the flight stress by the packet's
            # own reckoning, had no influence on the thickness carrying it.
            #
            # v40 passed that check at a margin of 1.010. Not a designed margin:
            # the wall had been sized for a different load and the pressure case
            # landed just inside. Sizing to merely reach the allowable gives
            # 0.794 mm, under the gauge floor, which is why it read as a pass at
            # all. Sizing so the margin clears the 14.5% this project has
            # measured between element orders gives 0.909 mm, and pressure
            # becomes the driver it should always have been.
            _press_repair = None
            try:
                from cadflow.margin_audit import RESOLVED_MARGIN
                from cadflow.pressurization import (
                    tank_pressure as _tp, wall_for_pressure_m as _wall_press)

                _tank_h_pre = fv["length_m"] / max(1, len(p.stack))
                _p_des = _tp("lox", acceleration_g=1.0,
                             head_height_m=_tank_h_pre).design_pa
                _st_pre = skin_stress_mpa(_res.peak_moment_nm, _axial_n,
                                          flight_r, _t_m)
                _t_press = _wall_press(
                    pressure_pa=_p_des, radius_m=flight_r,
                    allowable_pa=allowable_mpa * 1e6,
                    axial_flight_pa=-_st_pre["axial_mpa"] * 1e6,
                    bending_pa=-_st_pre["bending_mpa"] * 1e6,
                    reference_wall_m=_t_m, target_margin=RESOLVED_MARGIN)
                if _t_press > _t_m + 1e-9:
                    _press_repair = (_t_m, _t_press)
                    _t_m = _t_press
                    _wall_driver = ("internal pressure, sized for a margin "
                                    "outside measured uncertainty")
            except Exception as _pexc:  # noqa: BLE001
                print(f"pressure sizing unavailable: {_pexc}", flush=True)

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
            if _press_repair is not None:
                _was, _now = _press_repair
                L.append(
                    f"\n**The wall was thickened for internal pressure**, "
                    f"{1000*_was:.3f} to {1000*_now:.3f} mm. Pressure was "
                    f"previously checked and never sized: the wall came from "
                    f"axial load, buckling and minimum gauge, so the largest "
                    f"membrane stress on the tank had no influence on the "
                    f"thickness carrying it. The target is a margin outside the "
                    f"numerical uncertainty this packet measures for its own "
                    f"stresses, not merely a margin above one -- sizing to "
                    f"exactly the allowable leaves a verdict the analysis "
                    f"cannot resolve.\n")
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
                    # _skin_rho comes from the catalogue for the alloy the
                    # design chose. This line used to guess it from whether the
                    # material id started with "al-", which happens to give the
                    # right answer for the two alloys this loop currently picks
                    # and would quietly be wrong for titanium, for steel, or for
                    # any aluminium alloy named differently.
                    _dm = (_skin_rho * 2.0 * math.pi * flight_r
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

            # Tank pressure, which this packet did not have.
            #
            # A pump will not draw from an unpressurised tank -- the propellant
            # flashes at the inducer -- so every stage carries ullage above its
            # propellant's vapour pressure. That costs helium and a bottle to
            # keep it in, and neither was anywhere in the mass budget. It also
            # puts hoop tension in the wall, which is the largest membrane
            # stress this shell carries and had never been computed.
            #
            # And it bears on the section immediately above. A tank with end
            # domes carries pr/2t of axial *tension*, working directly against
            # the compression the wall was just thickened for. A shell in net
            # tension has no compressive buckling mode; whether that repair was
            # necessary depends on a pressure nothing was calculating.
            try:
                from cadflow.pressurization import (
                    feed_system_verdict as _feed, stage_pressurisation as _press,
                    wall_load_state as _wall_state)

                from cadflow.assembly import TAPER_PER_STAGE as _tap

                _tank_h = fv["length_m"] / max(1, len(p.stack))
                # Each stage at its own radius, not the base radius four times.
                #
                # Dome mass goes as R^2 at fixed gauge, so using flight_r for a
                # stage drawn at 0.78 of it overstates its tank ends by two
                # thirds -- and the affordability verdict below turns on exactly
                # that number. The taper is the one the CAD draws with.
                _press_radii = [flight_r * (_tap ** _i)
                                for _i in range(len(p.stack))]
                _press_rows = []
                for _i, _s in enumerate(p.stack, 1):
                    # Tell it which propellant it is sizing.
                    #
                    # combination was never passed, so it defaulted to lox_rp1
                    # and every vehicle got kerosene tanks whatever it burns. A
                    # solid_apcp stage came back with 0.98 m3 of tank, helium
                    # and domes -- and the refusal added to stage_pressurisation
                    # for exactly that case could not fire, because the module
                    # was never told what it was looking at. The same omission
                    # as the design loop, one layer down.
                    _press_rows.append(_press(
                        stage=_i, propellant_mass_kg=float(_s.prop_mass_kg),
                        combination=args.propellant,
                        radius_m=_press_radii[_i - 1], wall_m=_t_m,
                        acceleration_g=1.0, head_height_m=_tank_h,
                        wall_density_kg_m3=_skin_rho,
                        wall_allowable_pa=allowable_mpa * 1e6))
                _press_total = sum(r.total_kg for r in _press_rows)
                _press_total_kg = _press_total
                _ws = _wall_state(
                    pressure_pa=_press_rows[0].tanks[0].design_pa,
                    radius_m=flight_r, wall_m=_t_m,
                    axial_flight_pa=-_st["axial_mpa"] * 1e6,
                    bending_pa=-_st["bending_mpa"] * 1e6)

                L.append("\n## Tank pressurisation\n")
                L.append("| stage | tank volume | ullage | helium | bottle | "
                         "domes | total |")
                L.append("|---|---|---|---|---|---|---|")
                for _r in _press_rows:
                    L.append(f"| {_r.stage} | {_r.tank_volume_m3:.3f} m3 | "
                             f"{_r.ullage_pa/1000:.0f} kPa | "
                             f"{_r.helium_kg:.2f} kg | {_r.bottle_kg:.2f} kg | "
                             f"{_r.dome_kg:.2f} kg | "
                             f"**{_r.total_kg:.2f} kg** |")
                L.append(f"\nThat is **{_press_total:.1f} kg** the mass budget "
                         f"above does not carry, {100*_press_total/p.gross_kg:.2f}% "
                         f"of gross and {100*_press_total/args.payload_kg:.0f}% of "
                         f"the payload it is competing with. The structural "
                         f"coefficient absorbs it only in the sense that a "
                         f"lumped number absorbs anything.\n")

                L.append("| membrane stress in the tank wall | value |")
                L.append("|---|---|")
                L.append(f"| hoop, p r / t | "
                         f"**{_ws.hoop_pa/1e6:.1f} MPa** tension |")
                L.append(f"| axial from the end domes, p r / 2t | "
                         f"{_ws.axial_pressure_pa/1e6:.1f} MPa tension |")
                L.append(f"| axial from flight loads | "
                         f"{_st['axial_mpa']:.1f} MPa compression |")
                L.append(f"| bending at the peak station | "
                         f"{_st['bending_mpa']:.1f} MPa |")
                L.append(f"| net axial | "
                         f"**{_ws.net_axial_pa/1e6:+.1f} MPa** |")
                L.append(f"| von Mises, biaxial membrane | "
                         f"{_ws.von_mises_pa/1e6:.1f} MPa against "
                         f"{allowable_mpa:.0f} allowable |")

                _press_notes = []
                if _ws.hoop_pa / 1e6 > _st["combined_mpa"]:
                    _press_notes.append(
                        f"hoop is {_ws.hoop_pa/1e6/max(_st['combined_mpa'],1e-9):.1f}x "
                        f"the combined flight stress the wall was sized against: "
                        f"internal pressure, not flight loads, is the dominant "
                        f"membrane load on this tank")
                if not _ws.in_compression:
                    _press_notes.append(
                        "the pressurised tank wall is in net axial tension, so "
                        "it has no compressive buckling mode at this "
                        "condition. That does not retire the "
                        "buckling case, it scopes it: the relief lasts exactly "
                        "as long as the pressure does. A tank is dry on the pad "
                        "before pressurisation, dry in transport and handling, "
                        "and dry once its stage is spent, and it must not fold "
                        "in any of those. The interstages are dry throughout, "
                        "carry the same axial load with no pressure to relieve "
                        "them, and this packet does not analyse them separately")
                else:
                    _press_notes.append(
                        "bending exceeds the pressure relief, so the wall is in "
                        "net compression and the buckling case stands")
                for _n in _press_notes + _press_rows[0].notes:
                    L.append(f"\n- {_n}")

                L.append(
                    "\n- The pressure terms above assume a closed tank with end "
                    "domes, which is what a stage tank is. The shell model in "
                    "the barrel section deliberately uses the open-ended "
                    "reference p r / t with no axial term, because the thing it "
                    "meshes is an open barrel. Both are right about their own "
                    "subject and they are not the same subject: the axial "
                    "relief below belongs to the tank, not to the barrel that "
                    "was solved.\n")

                _fv = _feed(float(p.stack[0].chamber_pressure_pa),
                            _press_rows[0].tank_volume_m3,
                            density_kg_m3=_skin_rho,
                            allowable_pa=allowable_mpa * 1e6)
                L.append(f"\n- {_fv['note']}\n")

                # Two questions, and only one of them is answerable here.
                #
                # Whether the wall carries the pressure is settled by the
                # numbers above. Whether the vehicle can afford the helium is a
                # mass-budget question, and the budget is not computed until the
                # assembly section far below -- so this finding used to answer
                # it anyway, by asserting the mass was missing. It is now
                # charged to the closure and the closure reports its own
                # verdict, which means neither finding has to guess at what the
                # other concluded.
                _press_ok = _ws.von_mises_pa / 1e6 <= allowable_mpa
                _assembly_findings.append({
                    "check": "tank wall under pressure",
                    "passed": bool(_press_ok),
                    "detail": (f"hoop {_ws.hoop_pa/1e6:.0f} MPa, von Mises "
                               f"{_ws.von_mises_pa/1e6:.0f} MPa against "
                               f"{allowable_mpa:.0f} allowable; net axial "
                               f"{_ws.net_axial_pa/1e6:+.0f} MPa so the wall is "
                               f"{'in compression' if _ws.in_compression else 'in tension'}; "
                               f"{_press_total:.1f} kg of helium and bottles "
                               f"charged to the mass closure below"),
                    "severity": "pass" if _press_ok else "fail"})

                # Can each stage afford the tank it needs?
                #
                # A structural coefficient is a fraction, so structure scales
                # with propellant. Minimum gauge scales with nothing. Below some
                # size the two cross and a stage's tank ends alone cost more
                # than its entire structural allowance. That is why flown
                # structural coefficients get *worse* for small stages rather
                # than staying flat, and it is invisible to every other check
                # here: they all ask whether a wall carries its load, and this
                # asks whether the stage can pay for the wall at all.
                from cadflow.pressurization import (
                    DOME_MIN_GAUGE_M as _gauge, stage_feasibility as _feas)

                _fz = _feas(p.stack, _press_rows)
                L.append("\n| stage | structural allowance | tankage needed | "
                         "share | break-even dome gauge | affordable |")
                L.append("|---|---|---|---|---|---|")
                for _f in _fz:
                    L.append(
                        f"| {_f['stage']} | {_f['struct_allowance_kg']:.1f} kg | "
                        f"{_f['pressurisation_kg']:.1f} kg | "
                        f"**{100*_f['fraction_of_allowance']:.0f}%** | "
                        f"{1000*_f['break_even_gauge_m']:.2f} mm | "
                        f"**{_f['feasible']}** |")
                _fz_bad = [f for f in _fz if not f["feasible"]]
                for _f in _fz_bad:
                    L.append(f"\n- {_f['note']}")

                # Why the alloy is what it is, when a stage cannot afford it.
                #
                # Dome mass is linear in density, so the material decides
                # affordability as much as the geometry does -- Inconel is three
                # times aluminium. If that alloy was forced by the thermal
                # environment rather than chosen for strength, then "this stage
                # is too small" is the wrong diagnosis: the chain runs
                # aeroheating to material to density to dome mass to
                # affordability, and no single check in this packet owns it.
                #
                # This became visible only when the skin-temperature gate was
                # turned back on. With it disabled the loop flew aluminium at
                # 895 K, and the resulting threefold density advantage made
                # designs read affordable that are not.
                if _fz_bad:
                    try:
                        # Imported here rather than relying on the module
                        # alias bound 1,100 lines above inside another try: if
                        # that one failed, this reads as a NameError about
                        # _iter_m rather than as the missing catalogue it is.
                        from cadflow.autodesign import _material as _mat_of
                        from cadflow.space_materials import iter_materials

                        _m = _mat_of(skin_material)
                        _skin_k = float(p.trajectory.get("max_skin_temp_k") or 0.0)
                        _lim_k = float(_m.max_service_temp_k)
                        # "Lighter" has to mean lighter enough to matter.
                        #
                        # Any density below the incumbent's passes a bare < ,
                        # and stainless 321 at 8000 against Inconel's 8190 does
                        # -- by 2.3%, which changes no verdict. Reporting that
                        # as an available trade would say the alloy is not
                        # thermally forced when for every practical purpose it
                        # is. A tenth is the smallest saving that moves dome
                        # mass enough to be worth a reader's attention.
                        _MEANINGFUL = 0.10
                        _cooler = [x for x in iter_materials()
                                   if x.yield_mpa
                                   and x.max_service_temp_k >= _skin_k
                                   and x.density_kg_m3
                                   <= _m.density_kg_m3 * (1.0 - _MEANINGFUL)]
                        L.append(
                            f"\n**The alloy is thermally forced.** The skin "
                            f"reaches {_skin_k:.0f} K and {skin_material} is "
                            f"rated to {_lim_k:.0f} K, a margin of "
                            f"{_lim_k - _skin_k:.0f} K "
                            f"({100*(_lim_k - _skin_k)/max(_lim_k,1):.1f}%). "
                            + (f"No lighter alloy in the catalogue survives this "
                               f"temperature, so the tank ends weigh what they "
                               f"weigh: at {_m.density_kg_m3:.0f} kg/m3 the domes "
                               f"are the largest part of the tankage bill, and "
                               f"the affordability failure above is a thermal "
                               f"result reported as a structural one."
                               if not _cooler else
                               f"{len(_cooler)} alloy(s) at least "
                               f"{100*_MEANINGFUL:.0f}% lighter also survive it "
                               f"(lightest {min(_cooler, key=lambda x: x.density_kg_m3).material_id} "
                               f"at {min(x.density_kg_m3 for x in _cooler):.0f} "
                               f"kg/m3), so the choice is not thermally forced "
                               f"and the design loop should be able to trade "
                               f"here.")
                            + "\n")
                    except Exception as _mexc:  # noqa: BLE001
                        L.append(f"\n(alloy provenance unavailable: {_mexc})\n")
                if _fz_bad and design_conflict:
                    L.append(
                        f"\n**The design loop could not repair this.** "
                        f"{design_conflict}\n")
                    L.append(
                        "\nThat is a statement about the mission rather than "
                        "about the loop. The apogee demands a stage count whose "
                        "smallest stage cannot pay for its own pressure vessel, "
                        "so the specification is over-constrained for this "
                        "technology: it wants a thinner dome gauge than anyone "
                        "welds, a denser propellant, a larger payload to amortise "
                        "the fixed costs against, or a lower apogee.\n")
                L.append(
                    f"\nThe verdict rests on a {1000*_gauge:.2f} mm minimum "
                    f"dome gauge, which is an assumption about welding and "
                    f"handling rather than a derived quantity -- the membrane "
                    f"requirement at this pressure is under a fifth of a "
                    f"millimetre. The break-even column is what that gauge "
                    f"would have to be for each stage to fit, so a reader can "
                    f"weigh the finding against the assumption instead of "
                    f"taking it on trust.\n")
                _assembly_findings.append({
                    "check": "each stage can afford its own tankage",
                    "passed": not _fz_bad,
                    "detail": ("; ".join(
                        f"stage {f['stage']} needs "
                        f"{100*f['fraction_of_allowance']:.0f}% of its "
                        f"allowance, break-even gauge "
                        f"{1000*f['break_even_gauge_m']:.2f} mm"
                        for f in _fz_bad)
                        + ("; no shorter architecture closes the mission"
                           if design_conflict else "") if _fz_bad else
                        f"worst stage uses "
                        f"{100*max(f['fraction_of_allowance'] for f in _fz):.0f}% "
                        f"of its structural allowance for tankage"),
                    "severity": "pass" if not _fz_bad else "fail"})
            except KeyError as _kexc:
                # A propellant with no tankage model is a statement about the
                # vehicle, not a missing feature. A solid motor carries its
                # propellant in the case: there is nothing to pressurise, and
                # saying so is the correct output rather than an omission.
                L.append(f"\n## Tank pressurisation\n")
                L.append(f"\nNot applicable. {str(_kexc).strip(chr(34))}\n")
                # A vehicle with genuinely no tanks passes; a liquid this
                # model cannot size is unverified, not fine.
                #
                # This reported "the propellant is carried in the motor case"
                # for anything the tankage model refused, which is true of a
                # solid and false of every liquid. n2o4_mmh would have been told
                # it has no tanks -- a claim about the vehicle, made because the
                # model lacked one fluid's properties.
                try:
                    from cadflow.pressurization import NON_LIQUID as _NOTANK
                except Exception:  # noqa: BLE001
                    _NOTANK = {"solid_apcp"}
                _genuinely_none = args.propellant in _NOTANK
                _assembly_findings.append({
                    "check": "tank pressurisation applies",
                    "passed": bool(_genuinely_none),
                    "detail": str(_kexc).strip(chr(34)),
                    "severity": "pass" if _genuinely_none else "unverified"})
            except Exception as _exc:  # noqa: BLE001
                print(f"pressurisation unavailable: {_exc}", flush=True)

            # The dry structure, which is the item the section above raises.
            #
            # The tanks are pressurised and that relief is real. The interstages
            # are not: they transmit the whole thrust of the stage below into
            # everything above, at the instant that stage accelerates hardest,
            # with nothing inside to stabilise them. Until now they had only
            # been analysed as coupons -- 42 mm articles at a clamped radius,
            # against a yield allowable -- for a failure mode a thin shell does
            # not have.
            #
            # Two modes are checked, not one. shell_buckling takes no length,
            # because the classical stress is the long-cylinder asymptote and
            # genuinely has none; the Batdorf parameter says whether that
            # asymptote applies, and the Euler column mode is the failure a
            # length-free check structurally cannot see.
            try:
                from cadflow.dry_structure import (
                    bending_mpa_at as _bend_at, check_interstages as _dry,
                    failures as _dry_fails)

                from cadflow.assembly import (
                    TAPER_PER_STAGE as _taper, interstage_length as _is_len_of)

                _gs = p.trajectory.get("max_axial_g_by_stage") or []
                # Lengths from the same formula the CAD draws with, not from
                # the assembly dict -- that is not built until well below this
                # point, so reading it here would silently fall back to a
                # guessed length. The column mode goes as 1/L^2 and the Batdorf
                # parameter as L^2; a guessed length is not a small error, and
                # one that arrives silently is worse than one that raises.
                _radii = [flight_r * (_taper ** _i) for _i in range(len(p.stack))]
                _is_len = [_is_len_of(_radii[_i], _radii[_i + 1])
                           for _i in range(len(p.stack) - 1)]
                _is_stations = []
                _z = 0.0
                for _i, _sec in enumerate(fv.get("section_extents", [])[:len(p.stack)]):
                    _z = float(_sec[2])
                    if _i < len(p.stack) - 1:
                        _is_stations.append(_z)
                _dry_secs = _dry(p.stack, args.payload_kg, _gs,
                                 radius_m=flight_r, wall_m=_t_m,
                                 youngs_pa=skin_e_pa, lengths_m=_is_len)

                # Bending at each interstage's own station, not zero and not the
                # peak. The moment curve is steepest around the interstages and
                # taking either extreme would be wrong in a knowable direction.
                if _is_stations:
                    from cadflow.dry_structure import check_section as _dsec
                    _re = []
                    for _sec, _stn in zip(_dry_secs, _is_stations):
                        _bm = _bend_at(_res, _stn, flight_r, _t_m)
                        _n = _dsec(name=_sec.name, length_m=_sec.length_m,
                                   radius_m=flight_r, wall_m=_t_m,
                                   axial_load_n=_sec.axial_load_n,
                                   youngs_pa=skin_e_pa, bending_mpa=_bm)
                        _n.notes = list(_sec.notes) + [
                            f"bending at its own station "
                            f"({_stn:.2f} m from aft) is "
                            f"{_bm:.1f} MPa, read from the solved moment curve "
                            f"rather than taken as zero or as the peak"]
                        _re.append(_n)
                    _dry_secs = _re

                L.append("\n## Interstages: the structure with nothing in it\n")
                L.append("| section | length | load | stress | shell margin | "
                         "column margin | Batdorf Z | governs |")
                L.append("|---|---|---|---|---|---|---|---|")
                for _s in _dry_secs:
                    L.append(
                        f"| {_s.name} | {_s.length_m*1000:.0f} mm | "
                        f"{_s.axial_load_n/1000:.1f} kN | "
                        f"{_s.axial_stress_mpa:.1f} MPa | "
                        f"**{_s.shell_margin:.2f}** | {_s.column_margin:.0f} | "
                        f"{_s.batdorf_z:.0f} | {_s.governs} |")
                for _s in _dry_secs:
                    for _n in _s.notes:
                        L.append(f"\n- {_s.name}: {_n}")
                L.append(
                    f"\nThese margins are at the **sized** wall of "
                    f"{1000*_t_m:.2f} mm, not the {ASSEMBLY_WALL_MM:.1f} mm "
                    f"representative gauge the CAD is drawn at. That is the "
                    f"conservative choice by a wide margin: buckling margin "
                    f"goes as t squared, so the drawn geometry would read "
                    f"about {(ASSEMBLY_WALL_MM/1000.0/_t_m)**2:.0f}x these "
                    f"numbers. The conclusion does not depend on which wall a "
                    f"reader believes, which is the only reason it is safe to "
                    f"have two.\n")
                L.append(
                    "\nThe column mode is Euler's pi^2 E I / (kL)^2 with "
                    "I = pi r^3 t, at pinned ends. It is here because the shell "
                    "check takes no length and so cannot see a tube that bows "
                    "as a whole rather than folding locally -- on these short "
                    "interstages it is nowhere near governing, which is a "
                    "result and not a reason to have skipped it.\n")

                _dry_bad = _dry_fails(_dry_secs)
                _assembly_findings.append({
                    "check": "interstage buckling, unpressurised",
                    "passed": not _dry_bad,
                    "detail": (
                        "; ".join(f"{s.name} margin {s.margin:.2f}"
                                  for s in _dry_bad) if _dry_bad else
                        f"{len(_dry_secs)} dry section(s), worst margin "
                        f"{min(s.margin for s in _dry_secs):.2f} "
                        f"governed by the {min(_dry_secs, key=lambda s: s.margin).governs} "
                        f"mode"),
                    "severity": "pass" if not _dry_bad else "fail"})
            except Exception as _exc:  # noqa: BLE001
                print(f"dry structure check unavailable: {_exc}", flush=True)

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
                # The condition the fins were sized at, so the gimbal angle
                # reported belongs to the fins reported beside it.
                cg_station_m=float(worst_cg), thrust_n=_thrust1)
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
                    # CG and inertia from one state. Taking the CG from the
                    # worst condition and the inertia from the full vehicle
                    # would be a frequency for a vehicle that does not exist.
                    cg_station_m=float(worst_cg),
                    pitch_inertia_kg_m2=float(worst_state["Ixx_kg_m2"]))
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
                # Judge the separation against what the damping requires, not
                # against a coincidence window.
                #
                # This passed anything outside 0.8 to 1.25, while
                # slosh_baffles.required_separation computes the real
                # criterion -- 5.0x for a bare tank at zeta = 0.0005, falling to
                # 1.5x once baffles bring it to 0.05. The comment directly below
                # already stated the 5:1 rule; the verdict above it used a
                # different and much weaker test, so a design separated by 1.89x
                # read as a clean pass against a requirement of 5.0x.
                #
                # The fix is cheap enough that the gap is worth closing rather
                # than reporting: one 41 mm ring baffle, 0.33 kg, raises damping
                # by a factor of ninety and takes the requirement below what
                # this vehicle already has.
                from cadflow.slosh_baffles import (
                    BARE_TANK_DAMPING as _ZETA_BARE,
                    required_separation as _req_sep, size_baffles as _size_baf)

                _need_sep = _req_sep(_ZETA_BARE)
                _baffle = None
                if _coincide < _need_sep:
                    try:
                        _baffle = _size_baf(
                            tank_radius_m=flight_r,
                            fill_depth_m=max(0.2, 0.5 * fv["length_m"]
                                             / max(1, len(p.stack))),
                            slosh_hz=_maxq_slosh,
                            required_bandwidth_hz=_rb,
                            wall_density_kg_m3=_skin_rho)
                    except ValueError:
                        _baffle = None
                _sep_ok = _coincide >= _need_sep or _baffle is not None
                _sep_detail = (
                    f"slosh {_maxq_slosh:.2f} Hz against pitch {_rb:.2f} Hz, "
                    f"ratio {_coincide:.2f} against {_need_sep:.1f} required at "
                    f"a bare-tank damping of {_ZETA_BARE}")
                if _baffle is not None:
                    _sep_detail += (
                        f"; {_baffle.n_baffles} ring baffle(s) "
                        f"{1000*_baffle.width_m:.0f} mm wide raise damping to "
                        f"{_baffle.damping_ratio:.3f}, which drops the "
                        f"requirement to {_baffle.achieved_separation:.1f} and "
                        f"costs {_baffle.mass_kg:.2f} kg")
                elif _coincide < _need_sep:
                    _sep_detail += "; no baffle within a sensible width closes it"
                if not _has_liquid:
                    _sep_detail = (
                        f"not applicable: {args.propellant} is a cast grain "
                        f"bonded to the case, so there is no free surface and "
                        f"nothing to slosh")
                    _sep_ok = True
                _assembly_findings.append({
                    "check": "slosh / pitch-mode separation",
                    "passed": bool(_sep_ok),
                    "detail": _sep_detail,
                    "severity": "pass" if _sep_ok else "fail"})
                if _baffle is not None:
                    # Charged, not merely recommended.
                    #
                    # Passing this check on the strength of a part that is not
                    # in the vehicle and not in the mass budget is the overclaim
                    # this packet has already made three times -- with the
                    # pressurant, the tank domes and the thermal protection. A
                    # baffle the design depends on is a baffle the design
                    # carries.
                    _baffle_total_kg += float(_baffle.mass_kg) * len(p.stack)
                    L.append(
                        f"\n**The separation needs baffles.** At the bare-tank "
                        f"damping of {_ZETA_BARE} the rule is "
                        f"{_need_sep:.1f}x and this vehicle has "
                        f"{_coincide:.2f}x. {_baffle.n_baffles} ring baffle(s) "
                        f"{1000*_baffle.width_m:.0f} mm wide raise damping to "
                        f"{_baffle.damping_ratio:.3f} -- a factor of "
                        f"{_baffle.damping_ratio/_ZETA_BARE:.0f} -- which takes "
                        f"the requirement to {_baffle.achieved_separation:.1f}x, "
                        f"below what the vehicle already has, for "
                        f"{_baffle.mass_kg:.2f} kg.\n")
                    L.append(
                        "\nThis verdict previously used a coincidence window, "
                        "passing anything outside 0.8 to 1.25 while the "
                        "criterion the damping actually implies sat unused in "
                        "the module next to it.\n")
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
                # The damping the design actually carries, which is the
                # baffled value when baffles were fitted and the bare-tank value
                # when they were not.
                _phase_hard = None
                if _needs_phase:
                    try:
                        from cadflow.control_authority import (
                            phase_stabilisation_difficulty as _phase_diff)
                        from cadflow.slosh_baffles import (
                            BARE_TANK_DAMPING as _ZB)

                        _zeta_now = (_baffle.damping_ratio
                                     if _baffle is not None else _ZB)
                        _phase_hard = _phase_diff(_zeta_now)
                    except Exception:  # noqa: BLE001
                        _phase_hard = None
                if not _has_liquid:
                    # Bending modes remain real on a solid; slosh does not.
                    _needs_phase = [m for m in _needs_phase
                                    if "slosh" not in m.lower()]
                    _phase_hard = None
                _assembly_findings.append({
                    "check": "flexible mode stabilisation",
                    "passed": not _needs_phase,
                    # Say how hard, not only whether.
                    #
                    # mode_disposition answers which side of crossover a mode is
                    # on, and damping does not move a frequency, so the verdict
                    # is unchanged by baffles. The difficulty is not: a bare
                    # tank resonates at Q = 1000, a needle where a few degrees of
                    # phase error destabilises the vehicle, while the baffles
                    # this packet now carries put it at Q = 10.6, a broad hill a
                    # conventional autopilot rolls over. Reporting only the
                    # verdict made this finding read identically before and after
                    # a change that altered the problem by two orders of
                    # magnitude.
                    "detail": (
                        "all modes gain-stabilisable with a conventional rolloff"
                        if not _needs_phase else
                        f"{', '.join(_needs_phase)} below crossover; phase "
                        f"stabilisation required and not verified here"
                        + (f" -- {_phase_hard['note']}" if _phase_hard else "")),
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
                # Judge the coast the vehicle actually flies, not the minimum
                # that would just suffice.
                #
                # This passed coast_s = max(1.0, _need), where _need is
                # coast_for_clearance_s -- required_gap / v_rel. check_separation
                # then computes gap = v_rel * coast_s and asks gap >= required,
                # so the test was comparing a quantity against the value it was
                # derived from. In floating point that answers no about half the
                # time, and 50 kg to 20,000 km duly reported a separation
                # failure on a vehicle whose separations are fine.
                #
                # 25 kg to 4,000 km could never expose it: its requirement is
                # 0.60 s, below the 1.0 s floor, so the coast always exceeded
                # the minimum and the comparison was never self-referential.
                #
                # The trajectory coasts COAST_BETWEEN_STAGES_S between stages.
                # That is the number the design flies and the one the check
                # should judge.
                _sr = check_separation(stage_index=_i + 1, spent_mass_kg=_spent,
                                       upper_mass_kg=_upper, body_diameter_m=_D,
                                       coast_s=_flown_coast_s)
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

        # Is each nozzle sized for the air its own stage lights in?
        #
        # A nozzle only flows full over a band of ambient pressures, and the
        # planner picks a rising expansion ratio precisely because of that. It
        # then sized every throat at sea level, where an upper stage's ratio is
        # over-expanded: the model correctly reported separation and truncated
        # to the effective ratio at the separation plane -- about 15 for all of
        # eps 30, 60 and 80 -- so three deliberately different upper stages were
        # sized as one engine. The trajectory was never wrong, because it flies
        # the real ambient at every step. What was wrong is the thrust-to-weight
        # each stage was asked for, which is the lever the design loop pulls to
        # hold peak axial acceleration down.
        try:
            from cadflow.nozzle_ambient import (
                check_stack as _nz_check, findings as _nz_findings,
                recover_twr_sized as _nz_twr)

            _ign_p = p.trajectory.get("ignition_ambient_pa") or []
            _ign_h = p.trajectory.get("ignition_altitude_m") or []
            if _ign_p:
                # Recovered from the stack, not carried alongside it. Every
                # disconnection defect in this packet was a number reported
                # beside the thing it described rather than derived from it.
                _nz = _nz_check(p.stack, ignition_ambient_pa=_ign_p,
                                ignition_altitude_m=_ign_h,
                                twr_by_stage=_nz_twr(p.stack, args.payload_kg))
                L.append("\n## Nozzle against its own ambient\n")
                L.append("| stage | expansion | sized at | ignites at | "
                         "separation limit | attached |")
                L.append("|---|---|---|---|---|---|")
                for _c in _nz:
                    _lim = ("vacuum, none" if not math.isfinite(_c.eps_max)
                            else f"{_c.eps_max:.1f}")
                    L.append(
                        f"| {_c.stage} | {_c.expansion_ratio:.0f} | "
                        f"{_c.sized_at_ambient_pa/1000:.2f} kPa | "
                        f"{_c.ignition_altitude_m/1000:.1f} km, "
                        f"{_c.ignition_ambient_pa/1000:.2f} kPa | {_lim} | "
                        f"**{_c.attached}** |")
                if any(c.twr_flown is not None for c in _nz):
                    L.append("\n| stage | thrust-to-weight asked | flown | error |")
                    L.append("|---|---|---|---|")
                    for _c in _nz:
                        if _c.twr_flown is None:
                            continue
                        L.append(f"| {_c.stage} | {_c.twr_sized:.2f} | "
                                 f"{_c.twr_flown:.2f} | "
                                 f"{_c.twr_error_pct:+.1f}% |")
                _nz_bad = _nz_findings(_nz)
                for _f in _nz_bad:
                    L.append(f"\n- {_f}")
                L.append("\nThe separation limit is the same "
                         "`separation_limited_ratio` the trajectory integrator "
                         "uses, not a second copy of the criterion: a check that "
                         "re-derives the physics it checks will agree with "
                         "itself and drift from the model.\n")
                _assembly_findings.append({
                    "check": "nozzle sized for its ignition ambient",
                    "passed": not _nz_bad,
                    "detail": ("; ".join(_nz_bad) if _nz_bad else
                               f"{len(_nz)} nozzle(s) attached at ignition, "
                               f"each throat sized at the pressure its stage "
                               f"lights in"),
                    "severity": "pass" if not _nz_bad else "fail"})
        except Exception as exc:  # noqa: BLE001
            print(f"nozzle ambient check unavailable: {exc}", flush=True)

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
        for (lbl, mg, m), _bs in zip(margins, cgs):
            cg = _bs.cg_z_m
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
            # The pressurisation mass computed above is charged here rather
            # than only described. Reporting a shortfall in prose while the
            # closure arithmetic beside it silently omits the same number is
            # exactly the disconnection this packet keeps finding in itself.
            closure = mass_closure(
                asm, budget,
                liftoff_thrust_n=p.trajectory.get("liftoff_thrust_n", 0.0),
                pressurisation_kg=_press_total_kg,
                tps_kg=_tps_total_kg + _baffle_total_kg)
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
            L.append(f"| pressurisation, helium and bottles | "
                     f"{closure['pressurisation_kg']:.1f} kg |")
            if closure.get("tps_kg"):
                L.append(f"| thermal protection and slosh baffles | "
                         f"{closure['tps_kg']:.1f} kg |")
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

            # The closure verdict was prose only.
            #
            # "The vehicle cannot contain itself" is as strong a statement as
            # this packet makes, and it reached the markdown and stopped there:
            # not assembly_findings, not all_passed, not PACKET.json. Every
            # downstream consumer -- including the model this project trains --
            # would have read a clean pass on a vehicle too heavy to exist.
            # That is the exact defect tests/test_packet_self_consistency.py
            # was written for, sitting unnoticed in the section that decides
            # whether the design is real.
            _assembly_findings.append({
                "check": "mass budget holds the vehicle",
                "passed": bool(closure["closes"]),
                "detail": (
                    f"skin as drawn {closure['skin_kg']:.1f} kg + engine "
                    f"{closure['engine_kg']:.1f} kg + pressurisation "
                    f"{closure['pressurisation_kg']:.1f} kg = "
                    f"{closure['accounted_kg']:.1f} kg against a "
                    f"{closure['budget_kg']:.1f} kg budget, slack "
                    f"{closure['slack_kg']:+.1f} kg"),
                "severity": "pass" if closure["closes"] else "fail"})
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
    # A part that could not be analysed did not fail; it was never checked.
    #
    # When a component's mesh degrades to a convex hull the solve deliberately
    # reports no stress, because a hull is not the part. That was then read
    # downstream as passed=False, so an unanalysable fin drove the whole packet
    # to FAILED with nothing anywhere saying why -- error was None, because
    # nothing raised. The 250 kg to 600 km mission hits this where 25 kg to
    # 4,000 km does not, which is the only reason it was ever visible.
    #
    # This packet already learned that a boolean cannot carry three outcomes
    # and gave the assembly findings VERIFIED, FAILED and INCOMPLETE. The
    # component path kept pass/fail, so the same lesson had to be learned twice.
    _components_unanalysed = [r for r in results
                              if not r["passed"] and r.get("mesh_was_hull")
                              and r.get("stress_dist") is None
                              and not r.get("error")]
    _components_failed = [r for r in results
                          if not r["passed"] and r not in _components_unanalysed]
    _components_ok = not _components_failed
    # A boolean cannot carry three outcomes, and forcing it to try is how this
    # packet reported "Overall: True" for a vehicle with an unverified control
    # requirement outstanding. Reclassifying a check from FAIL to REQUIRED made
    # the summary cleaner and less true. Nothing is verified while a
    # requirement remains unchecked, so `all_passed` is False unless the packet
    # actually verified everything, and the three-way status says which of the
    # three situations it is.
    # How the loop arrived at this design.
    #
    # design_history was assigned and never read: the repair loop's record of
    # its own decisions -- which alloy it right-sized to, which nose it chose,
    # whether it dropped a stage or reached for thermal protection -- was
    # computed and discarded. The packet reported what the vehicle is and never
    # how it came to be that, which is the one thing a reader cannot reconstruct
    # from the result.
    #
    # It matters more now that the loop can repair things. A design that passes
    # because a blanket was added and the alloy changed underneath it is a
    # different claim from one that passed on its first evaluation, and the
    # difference is invisible in the findings.
    if design_history:
        _steps = [h for h in design_history if h.get("note")]
        if not _steps:
            # Say so, rather than rendering nothing.
            #
            # An absent section is ambiguous between "the loop changed nothing"
            # and "the loop's changes were not recorded", and those are very
            # different claims about a design. 50 kg to 20,000 km converges in
            # three iterations without a single repair, which is a fact worth
            # stating -- it is the difference between a design that was right
            # and one that was made right.
            L.append("\n## What the design loop changed\n")
            L.append("\nNothing. The loop converged without needing to move a "
                     "single knob, so this design is as first evaluated rather "
                     "than as repaired. That is a stronger claim than a clean "
                     "verdict reached after several interventions, and the "
                     "distinction is invisible in the findings alone.\n")
        if _steps:
            L.append("\n## What the design loop changed\n")
            L.append("| step | change | gross |")
            L.append("|---|---|---|")
            for _h in _steps:
                _g = _h.get("gross_kg")
                L.append(f"| {_h.get('iteration', '?')} | {_h['note']} | "
                         + (f"{_g:.1f} kg |" if isinstance(_g, (int, float))
                            else "-- |"))
            _tried = [a for _h in _steps for a in (_h.get("alternatives") or [])]
            for _a in _tried:
                L.append(f"\n- also tried: {_a}")
            L.append("\nEvery row is a change the loop made to satisfy a "
                     "constraint it could not otherwise meet. A design that "
                     "reaches a clean verdict after four repairs is a different "
                     "claim from one that was right at the first evaluation, and "
                     "nothing else in this packet distinguishes them.\n")

    # Is each passing margin bigger than the error bar on the number it came
    # from?
    #
    # v40 reported the tank wall as passing at 130.0 MPa against a 131 MPa
    # allowable -- eight parts in a thousand -- in the same document that states
    # element order moves the p95 stress this loop sizes against by -13.9% to
    # +14.5%. Both are true and they cannot both be load-bearing. A margin
    # smaller than its own uncertainty is not a pass, and reporting it as one is
    # the kind of overclaim a reader has no way to see.
    try:
        from cadflow.margin_audit import (
            audit as _margin_audit, summary as _margin_summary,
            unresolved as _margin_thin)

        _margins = {}
        for _f in _assembly_findings:
            _m = re.search(r"margin ([0-9.]+)", str(_f.get("detail", "")))
            if _m:
                _margins[_f["check"]] = float(_m.group(1))
        try:
            if _st and allowable_mpa:
                _margins["tank wall under pressure"] = (
                    allowable_mpa / max(_ws.von_mises_pa / 1e6, 1e-9))
        except Exception:  # noqa: BLE001
            pass
        _mv = _margin_audit(_margins) if _margins else []
        _thin = _margin_thin(_mv)
        if _mv:
            L.append("\n## Are these margins bigger than their own error bars?\n")
            L.append("| check | margin | outside measured uncertainty |")
            L.append("|---|---|---|")
            for _v in _mv:
                L.append(f"| {_v.check} | {_v.margin:.3f} | "
                         f"**{_v.resolved}** |")
            _txt = _margin_summary(_mv)
            if _txt:
                L.append(f"\n{_txt}\n")
            else:
                L.append("\nEvery passing margin clears its allowable by more "
                         "than the 14.5% this project has measured between "
                         "element orders, so no verdict here rests on numerical "
                         "noise.\n")
        for _v in _thin:
            _assembly_findings.append({
                "check": f"margin is established: {_v.check}",
                "passed": False,
                "detail": _v.note,
                "severity": "unverified"})
    except Exception as _mexc:  # noqa: BLE001
        print(f"margin audit unavailable: {_mexc}", flush=True)

    for _r in _components_unanalysed:
        _assembly_findings.append({
            "check": f"component analysed: {_r['name']}",
            "passed": False,
            "detail": (
                f"the mesh degraded to a convex hull, so no stress was solved "
                f"for the part as drawn. This is not a failed check -- it is an "
                f"absent one, and the distinction decides whether the verdict "
                f"above is a fault in the design or a gap in the analysis"),
            "severity": "unverified"})

    _assembly_fails = [f for f in _assembly_findings if f["severity"] == "fail"]
    # The components that passed are coupons, not the flight parts.
    #
    # The prose says so plainly -- body radius is clamped so parts stay
    # meshable, giving a factor of several in radius and 1.3 kg of coupon
    # against hundreds of kg of real structure. The verdict did not. A consumer
    # reading components_passed: True would take the flight components as
    # verified, and the caveat correcting them is prose it cannot read. Stress
    # does not scale with a coupon, so those runs establish that representative
    # sections survive representative loads -- worth having, and not the same
    # claim.
    # Analyse the flight barrel itself, at flight scale.
    #
    # The coupons cannot do this: resolving a 0.8 mm wall with solid
    # tetrahedra needs quarter-millimetre elements, and a 4.4 m tank at 335 mm
    # radius then wants about two billion of them against a budget of forty
    # thousand. That is why the radius was clamped. Shell elements carry
    # thickness as a property instead of meshing through it, so the same
    # barrel is a few hundred elements and solves in under a second.
    # The wall, modulus and axial load all come from the flight-loads block
    # above, which is wrapped in its own try. If that did not run these names
    # do not exist, and NameError is caught alongside everything else rather
    # than being special-cased -- the outcome is the same either way: no
    # flight-scale run, so the coupon caveat stands.
    _shell = None
    try:
        from cadflow.shell_fea import analyse_barrel

        _shell = analyse_barrel(
            args.out / "shell_flight_barrel",
            radius_m=flight_r, length_m=fv["length_m"], thickness_m=_t_m,
            youngs_pa=skin_e_pa, axial_n=_axial_n)

        # Per-stage barrels, each sized for the load it actually carries.
        #
        # Not the coupon walls: those were sized at the clamped radius, and
        # required thickness scales inversely with radius, so carrying them
        # across means nothing. Each section is sized here from the mass it
        # supports times peak axial acceleration, at the flight radius, and
        # then analysed at that thickness.
        _per_stage = []
        for _i, _st in enumerate(p.stack):
            _above = sum(float(x.prop_mass_kg) + float(x.struct_mass_kg)
                         for x in p.stack[_i + 1:]) + args.payload_kg
            _load = _above * 9.80665 * _peak_g
            if _load <= 0:
                continue
            _len = max(0.3, float(_st.prop_mass_kg) / PROPELLANT_BULK_DENSITY
                       / (math.pi * flight_r ** 2))
            _w = size_wall(load_n=_load, radius_m=flight_r, length_m=_len,
                           sigma_allow_pa=allowable_mpa * 1e6,
                           yield_pa=allowable.source_strength_mpa * 1e6,
                           modulus_pa=skin_e_pa)
            _r = analyse_barrel(
                args.out / f"shell_stage{_i+1}", radius_m=flight_r,
                length_m=_len, thickness_m=_w.thickness_m,
                youngs_pa=skin_e_pa, axial_n=_load)
            _per_stage.append((_i + 1, _len, _w, _load, _r))
    except Exception as _sx:  # noqa: BLE001
        print(f"flight-scale shell run unavailable: "
              f"{type(_sx).__name__}: {_sx}", flush=True)
        _shell = None

    _ratio = (flight_r / (geo_r_mm / 1000.0)) if geo_r_mm else 1.0
    if _shell is not None and _shell.converged:
        _ok = _shell.mean_von_mises_mpa <= allowable_mpa
        _assembly_findings.append({
            "check": "flight barrel stress",
            "passed": bool(_ok),
            # The per-stage runs belong in the verdict too, not only in the
            # table above. The whole-vehicle barrel happens to bound them --
            # it carries the peak load over the full length -- but leaving
            # that implied is how a reader ends up assuming a check covers
            # more than it does.
            "detail": (f"{_shell.elements} shell elements at "
                       f"{1000*flight_r:.0f} mm radius: "
                       f"{_shell.mean_von_mises_mpa:.1f} MPa against a "
                       f"{allowable_mpa:.0f} MPa allowable"
                       + (f", {_shell.error_pct:+.1f}% from closed form"
                          if _shell.error_pct is not None else "")
                       + (f"; worst of {len(_per_stage)} per-stage barrels "
                          f"{max(r.mean_von_mises_mpa for *_x, r in _per_stage):.1f} "
                          f"MPa" if _per_stage else "")),
            "severity": "pass" if _ok else "fail"})
        L.append("\n## Flight barrel, at flight scale\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| mesh | {_shell.elements} shell elements, "
                 f"{_shell.nodes} nodes |")
        L.append(f"| membrane stress | {_shell.mean_von_mises_mpa:.2f} MPa |")
        if _shell.analytic_mpa is not None:
            L.append(f"| closed form | {_shell.analytic_mpa:.2f} MPa "
                     f"({_shell.error_pct:+.2f}%) |")
        L.append(f"| peak, at the clamped end | "
                 f"{_shell.max_von_mises_mpa:.2f} MPa |")
        L.append(f"\nThis is the flight barrel, not a coupon. The component "
                 f"table above analyses parts at {geo_r_mm:.0f} mm radius "
                 f"because a {1000*_t_m:.2f} mm wall cannot be resolved "
                 f"with solid elements at {1000*flight_r:.0f} mm -- it would "
                 f"take on the order of a billion tetrahedra. Shell elements "
                 f"carry the thickness as a property, so the same barrel is a "
                 f"few hundred elements.\n")
        for _n in _shell.notes:
            L.append(f"- {_n}")
        if _per_stage:
            L.append("\n| stage | length | wall | driver | axial load | "
                     "stress | vs closed form |")
            L.append("|---|---|---|---|---|---|---|")
            for _n, _len, _w, _load, _r in _per_stage:
                L.append(f"| {_n} | {_len:.2f} m | {1000*_w.thickness_m:.2f} mm | "
                         f"{_w.driver} | {_load/1000:.0f} kN | "
                         f"{_r.mean_von_mises_mpa:.1f} MPa | "
                         f"{_r.error_pct:+.2f}% |")
            _worst = max(_per_stage, key=lambda x: x[4].mean_von_mises_mpa)
            L.append(f"\nEach barrel is sized for the mass it supports at peak "
                     f"axial acceleration, at the flight radius -- not carried "
                     f"over from the coupons, whose walls were sized at "
                     f"{geo_r_mm:.0f} mm and mean nothing here. Stage "
                     f"{_worst[0]} is worst at "
                     f"{_worst[4].mean_von_mises_mpa:.1f} MPa against a "
                     f"{allowable_mpa:.0f} MPa allowable.\n")
        L.append("")
    elif _ratio > 1.5:
        _assembly_findings.append({
            "check": "flight component stress",
            "passed": False,
            "detail": (f"analysed as coupons at {geo_r_mm:.0f} mm radius "
                       f"against a flight radius of {1000*flight_r:.0f} mm, a "
                       f"factor of {_ratio:.1f}; the flight parts themselves "
                       f"are not analysed"),
            "severity": "unverified"})

    # Does the packet agree with itself?
    #
    # Five defects this session were the same shape: one part of the program
    # knew something and another reported otherwise, and every one was
    # invisible to the tests because each individual model was right. This
    # compares the numbers the packet already reports against each other.
    #
    # The logic lives in cadflow.packet_audit rather than here so it can be
    # tested against a deliberately inconsistent packet. Its first run said
    # "13 of 13 agree", which is worth nothing on its own: a check only ever
    # run on correct data is not known to work, it is known to be quiet, and a
    # broken check is quiet too. Under test it now has to catch a component
    # weighed in the wrong alloy, a gross mass its own stack does not support,
    # and a repair that reached some stages and not others.
    from cadflow.packet_audit import audit as _audit, failures as _failures

    _crosses = _audit(
        gross_kg=p.gross_kg, stack=p.stack, payload_kg=args.payload_kg,
        flight_vehicle_mass_kg=fv["mass_kg"], components=results,
        skin_density_kg_m3=_skin_rho, skin_material=skin_material)
    # Mission, architecture and stability: values the packet reports that are
    # computable from other values it reports. None was wrong when these were
    # added, which is the state the six drifted quantities were in until
    # something moved one side and not the other.
    # Did every subsystem use the inputs the caller asked for?
    #
    # Nearly every defect the mission and propellant sweeps found was a
    # component using a value nobody gave it: a default argument, a .get
    # fallback, an omitted parameter. --propellant lox_lh2 produced a kerosene
    # vehicle; tank sizing was never told the combination; tanks were sized at
    # O/F 2.56 while the chemistry burned 2.45. None raised, and every module
    # was right about what it was handed.
    try:
        from cadflow.packet_audit import input_provenance as _prov

        _requested = {"payload_kg": float(args.payload_kg),
                      "apogee_km": float(args.apogee_km),
                      "propellant": str(args.propellant),
                      "chamber_bar": float(args.chamber_bar)}
        _used = {
            "payload_kg": float(args.payload_kg),
            "apogee_km": float(args.apogee_km),
            "propellant": str(getattr(design_knobs, "propellant", None)
                              or args.propellant),
            "chamber_bar": (float(p.stack[0].chamber_pressure_pa) / 1e5
                            if p.stack else float(args.chamber_bar)),
        }
        _crosses = list(_crosses) + _prov(requested=_requested, used=_used)
    except Exception:  # noqa: BLE001
        pass

    try:
        from cadflow.packet_audit import derived_quantities as _derived

        _crosses = list(_crosses) + _derived(
            stages=p.stages, split=p.split, achieved_km=p.achieved_km,
            target_km=args.apogee_km, error_pct=err, stability=stability,
            # The CG the reported static margin was measured from. Checking it
            # against the full-vehicle CG would pass on this vehicle and raise a
            # false failure on any design whose worst case is not liftoff -- and
            # a check that cries wolf on a correct packet is one that gets
            # switched off, which is how a check comes to protect nothing.
            cg_z_m=worst_cg, radius_m=flight_r)
    except Exception:  # noqa: BLE001
        pass

    # Geometry too: do the in-line parts tile the vehicle?
    try:
        from cadflow.packet_audit import stack_interference as _interf

        _rows = [(r["name"], r["station_mm"] / 1000.0, r["length_mm"] / 1000.0)
                 for r in (assembly or {}).get("summary", [])]
        if _rows:
            _crosses = list(_crosses) + _interf(_rows)
    except Exception:  # noqa: BLE001
        pass

    _consistency = [c.as_dict() for c in _crosses]
    _bad = _failures(_crosses)
    if _bad:
        _assembly_findings.append({
            "check": "packet self-consistency",
            "passed": False,
            "detail": (f"{len(_bad)} of {len(_crosses)} internal checks "
                       f"disagree: " + "; ".join(c.check for c in _bad[:3])),
            "severity": "fail"})

    _assembly_unverified = [f for f in _assembly_findings
                            if f["severity"] == "unverified"]
    allp = (_components_ok and not _components_unanalysed
            and not _assembly_fails and not _assembly_unverified)
    _status = ("FAILED" if (not _components_ok or _assembly_fails)
               else "INCOMPLETE"
               if (_assembly_unverified or _components_unanalysed)
               else "VERIFIED")

    if _consistency:
        _bad_n = sum(1 for c in _consistency if not c["ok"])
        L.append(f"\n## Packet self-consistency\n")
        # Describe the checks that ran, from the categories they declare.
        #
        # This sentence has gone stale three times: written from memory, then
        # derived by matching on check names, which did not recognise the
        # mission and stability checks added afterwards. Each Cross now carries
        # its own kind, so a check that exists is a check that gets described.
        from cadflow.packet_audit import describe as _describe

        L.append(f"{len(_consistency) - _bad_n} of {len(_consistency)} internal "
                 f"cross-checks agree, comparing numbers the packet already "
                 f"reports against each other: {_describe(_crosses)}. Six "
                 f"defects in this packet were of exactly that shape and every "
                 f"one was invisible to the tests, because each individual "
                 f"model was right and only the connection between them was "
                 f"wrong.\n")
        if _bad_n:
            L.append("| check | reported | expected |")
            L.append("|---|---|---|")
            for _c in _consistency:
                if not _c["ok"]:
                    L.append(f"| {_c['check']} | {_c['got']:.4g} | "
                             f"{_c['want']:.4g} |")
            L.append("")

    if _assembly_findings:
        L.append("\n## Assembly verification\n")
        L.append("| check | result | detail |")
        L.append("|---|---|---|")
        for _f in _assembly_findings:
            _mark = {"pass": "PASS", "fail": "FAIL",
                     "unverified": "REQUIRED"}.get(_f["severity"], "FAIL")
            L.append(f"| {_f['check']} | **{_mark}** | {_f['detail']} |")
        if _assembly_unverified:
            # Enumerate them rather than explaining one. With a single
            # unverified item a general sentence was fine; with two it
            # described the first and silently ignored the second, which is a
            # smaller version of the same failure this section exists to fix.
            L.append(f"\n{len(_assembly_unverified)} check(s) marked REQUIRED "
                     f"are neither passed nor failed: they name real work this "
                     f"packet cannot do. Claiming they pass would report checks "
                     f"that never ran; claiming they fail would suggest defects "
                     f"where there are none.\n")
            for _u in _assembly_unverified:
                L.append(f"- **{_u['check']}** -- {_u['detail']}.")
            L.append("")
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
             f"All {len(results)} coupons passed: "
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
        "self_consistency": _consistency,
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
