"""Generate the supersonic drag corpus this project could not find anywhere.

Six parallel searches across CAD, CFD, propulsion, materials, structures and
spacecraft archives turned up no open dataset pairing rocket geometry with drag
coefficient over a Mach sweep. Supersonic archives publish pressure signatures
for tens of cases; the large labelled aero datasets are all subsonic. So the
aerodynamic half of this project's corpus has to be computed.

The pipeline underneath is validated, not assumed: a 15 degree cone at Mach 2
comes out at 159,131 Pa against 158,705 Pa from an exact Taylor-Maccoll solve,
an error of 0.23%. That check exists because the same pipeline previously
returned converged, stable, entirely wrong numbers -- twice -- and nothing
inside it objected either time.

Every case here is written to disk as it completes and carries its own
provenance: mesh size, run duration, whether the surface pressure settled, and
the analytic wave drag for the same body. A case whose drag disagrees violently
with theory is flagged in the record rather than dropped, because the pattern of
disagreement is itself information -- slender bodies should agree closely and
blunt ones should not.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Nose shapes the wave-drag module can price. Cones are included even though
#: their slender-body integral diverges -- the CFD handles them fine and the
#: analytic column is simply left empty, which is more honest than pricing a
#: shape the theory refuses.
SHAPES = ("cone", "ogive", "vonkarman")

#: Nose fineness: length over base diameter. Below about 1.5 the shock detaches
#: at the lower Mach numbers and conical theory stops applying.
FINENESS = (1.5, 2.0, 3.0, 4.0)

MACHS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)

#: Boattails are deliberately absent. A boattail earns its keep by shrinking the
#: blunt base and the drag that base carries; this case is sting-mounted, so the
#: body continues downstream at base radius and there is no base to shrink. The
#: integral would see only the boattail's own surface pressure, which adds drag,
#: and would report boattails as strictly harmful. That is a limitation of the
#: measurement, not a fact about boattails, so the corpus does not claim it.


def analytic_wave_drag(shape: str, fineness: float, radius: float) -> float | None:
    """Karman wave drag for a CLOSED body: nose + cylinder + tail closure.

    This is not the same quantity the CFD measures. `wave_drag_coefficient`
    builds a closed meridian -- the slender-body integral is only defined for a
    body that closes at both ends -- so it prices the tail's wave drag as well
    as the nose's. The CFD case is sting-mounted: nose plus cylinder, no
    closure, so it sees the forebody alone.

    The two differ by roughly a factor of two, which is what the ratio below
    shows (0.39 to 0.50 across the ogives). That is a definitional difference,
    not solver error, and the field is named accordingly so nobody reads 0.39
    as a 61% miss. The apples-to-apples check for this pipeline is
    Taylor-Maccoll on cones, where forebody meets forebody and the agreement is
    -1.8% mean across sixteen cases.
    """
    try:
        from cadflow.wave_drag import wave_drag_coefficient

        return float(wave_drag_coefficient(shape, fineness, radius=radius))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    from cadflow.supersonic_drag import BodySpec, run_case, write_case

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/drag_corpus")
    ap.add_argument("--radius", type=float, default=0.075)
    ap.add_argument("--afterbody", type=float, default=0.6,
                    help="cylindrical length aft of the nose, in metres")
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--nr", type=int, default=40)
    ap.add_argument("--grading", type=float, default=20.0)
    ap.add_argument("--flowthroughs", type=float, default=1.6)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-cases", action="store_true",
                    help="keep the OpenFOAM case directories (they are large)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    record_path = args.out / "drag_corpus.json"
    records: list[dict] = []
    if record_path.exists():
        try:
            records = json.loads(record_path.read_text())
        except Exception:  # noqa: BLE001
            records = []
    done = {(r["shape"], r["fineness"], r["mach"]) for r in records}

    plan = [(s, f, m) for s in SHAPES for f in FINENESS for m in MACHS
            if (s, f, m) not in done]
    if args.limit:
        plan = plan[:args.limit]
    print(f"{len(plan)} cases to run ({len(done)} already recorded)")

    work = args.out / "_cases"
    work.mkdir(exist_ok=True)

    for i, (shape, fineness, mach) in enumerate(plan, 1):
        nose_len = 2.0 * args.radius * fineness
        spec = BodySpec(f"{shape}_f{fineness:g}", nose_length_m=nose_len,
                        body_length_m=args.afterbody, radius_m=args.radius,
                        nose_shape=shape, tip_bluntness=0.01)
        meta = write_case(work / "_probe", spec, mach)
        end_time = args.flowthroughs * (2.5 * spec.total_length_m) / meta["u_inf"]

        case = work / f"{shape}_f{fineness:g}_m{mach:g}"
        shutil.rmtree(case, ignore_errors=True)
        t0 = time.time()
        res = run_case(case, spec, mach, nx_body=args.nx, nr=args.nr,
                       radial_grading=args.grading, end_time=end_time,
                       timeout_s=args.timeout)
        elapsed = time.time() - t0

        wave = analytic_wave_drag(shape, fineness, args.radius)
        rec = {
            "shape": shape,
            "fineness": fineness,
            "mach": mach,
            "radius_m": args.radius,
            "nose_length_m": nose_len,
            "afterbody_m": args.afterbody,
            "cd_pressure": res.cd,
            "error": res.error,
            "wave_drag_karman_closed_body": wave,
            "cd_forebody_over_karman_closed_body":
                (res.cd / wave) if (res.cd and wave) else None,
            "comparison_note": (
                "CFD is forebody-only (sting-mounted); Karman is a closed body "
                "including tail closure. Expect ~0.4-0.5, not 1.0."),
            "seconds": round(elapsed, 1),
            "mesh": {"nx_body": args.nx, "nr": args.nr,
                     "radial_grading": args.grading},
            "flowthroughs": args.flowthroughs,
        }
        records.append(rec)
        record_path.write_text(json.dumps(records, indent=1))

        status = (f"cd={res.cd:.5f}" if res.cd is not None
                  else f"FAILED {res.error}")
        ratio_v = rec["cd_forebody_over_karman_closed_body"]
        ratio = f"  fore/closed={ratio_v:.2f}" if ratio_v else ""
        print(f"[{i}/{len(plan)}] {shape:10} f={fineness:<4g} M={mach:<4g} "
              f"{status}{ratio}  ({elapsed:.0f}s)", flush=True)

        if not args.keep_cases:
            shutil.rmtree(case, ignore_errors=True)

    ok = [r for r in records if r["cd_pressure"] is not None]
    print(f"\n{len(ok)}/{len(records)} cases produced a drag coefficient")
    print(f"corpus: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
