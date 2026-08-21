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

ALLOWABLE_MPA = 200.0


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
                    max_stress_mpa=ALLOWABLE_MPA, max_disp_mm=3.0, max_iters=1,
                    load_n=load, prompt=name)
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
            vm = dist["p99"]
            ok = vm <= ALLOWABLE_MPA
            if ok:
                break
            # thicken in proportion to the overshoot and re-analyse
            thicker = min(t_mm * min(2.0, 1.15 * vm / ALLOWABLE_MPA), MAX_WALL_MM)
            if thicker <= t_mm + 1e-9:
                # Already at the cap and still over. Re-solving identical
                # geometry cannot change the answer, so stop and say so rather
                # than burning the remaining iterations on the same mesh.
                print(f"  [{name}] p99 {vm:.0f} MPa over {ALLOWABLE_MPA:.0f} at "
                      f"the {MAX_WALL_MM:.0f} mm wall limit; not converging",
                      flush=True)
                wall_capped = True
                break
            t_mm = thicker
            print(f"  [{name}] p99 {vm:.0f} MPa over {ALLOWABLE_MPA:.0f}; "
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
                        "coupon_margin": (ALLOWABLE_MPA / vm) if vm else None,
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
             " shell p99 | peak | 1st mode | mass | Izz | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        vm = (f"{r['shell_von_mises_mpa']:.1f} MPa"
              if r["shell_von_mises_mpa"] else "-")
        d = r.get("stress_dist")
        peak = f"{d['max']:.0f} MPa" if d else ("hull" if r["mesh_was_hull"] else "-")
        f1 = r.get("first_mode_hz")
        mode = f"{f1:.0f} Hz" if f1 else "-"
        if r.get("mesh_was_coarsened"):
            vm = vm + " (coarse)"
        mp = r.get("mass_properties") or {}
        mass = f"{mp['mass_kg']*1000:.0f} g" if mp.get("mass_kg") else "-"
        izz = f"{mp['Izz_kg_m2']:.2e}" if mp.get("Izz_kg_m2") else "-"
        L.append(f"| {r['name']} | {r['why']} | {r['load_n']:.0f} N | "
                 f"{r['wall_mm']:.2f} mm | {r['wall_driver']} | "
                 f"{r['buckling_margin']:.2f}x | {vm} | {peak} | {mode} | "
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
        L.append("\n## Stability\n")
        L.append("| quantity | value |")
        L.append("|---|---|")
        L.append(f"| fin span (each of {fins['n_fins']}) | "
                 f"{fins['span_m']*1000:.0f} mm |")
        L.append(f"| fin root / tip chord | {fins['root_chord_m']*1000:.0f} / "
                 f"{fins['tip_chord_m']*1000:.0f} mm, sweep "
                 f"{fins['sweep_m']*1000:.0f} mm |")
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

        L.append("\nStage lengths come from propellant volume at LOX/RP-1 bulk "
                 "density and each stage is a uniform cylinder of its wet mass "
                 "-- coarse, since a real stage has domes, a dry engine at one "
                 "end and a moving liquid level, but built only from numbers the "
                 "planner produced, and the wet mass reproduces the planner's "
                 f"gross of {p.gross_kg:.1f} kg. Pitch inertia is "
                 f"{fv['Ixx_kg_m2']/max(fv['Izz_kg_m2'],1e-12):.0f}x roll, as "
                 "it must be for a long thin vehicle.")

    L.append("\nWall thickness and buckling margin size the thin shell; the "
             "shell FEA column is CalculiX on that same hollow geometry, so the "
             "meshed part is the part being designed rather than a solid billet "
             "with the same outer dimensions. The 1st mode column is a CalculiX "
             "*FREQUENCY solve on that same mesh, clamped at the aft face: it is "
             "what a static check cannot see, and it is the quantity a flutter "
             "or coupled-loads assessment starts from. Mass and Izz are "
             "computed on that same solid, exact for the geometry, in kg and "
             "kg m^2 about the centroid.")
    allp = all(r["passed"] for r in results)
    L.append(f"\nAllowable {ALLOWABLE_MPA:.0f} MPa. "
             f"All {len(results)} components passed: **{allp}**\n")

    (args.out / "PACKET.md").write_text("\n".join(L))
    (args.out / "PACKET.json").write_text(json.dumps({
        "specification": spec, "stages": p.stages, "split": p.split,
        "gross_kg": p.gross_kg, "achieved_km": p.achieved_km,
        "error_pct": err, "rationale": p.rationale,
        "components": results, "all_passed": allp,
        "struct_coeff_used": (solved_coeff if args.solve_structure
                              else STRUCT_COEFF),
        "struct_coeff_solved": solved_coeff,
        # Vehicle-level results were markdown-only, so nothing downstream --
        # including the model -- could consume them. Kept as two separate keys
        # because they describe two different objects: the coupons that were
        # meshed and analysed, and the vehicle that flew the trajectory.
        "coupon_stack": coupon_stack,
        "flight_vehicle": flight_vehicle,
        "stability": stability,
        "thermal": thermal}, indent=2))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
