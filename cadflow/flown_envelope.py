"""Does this vehicle's structure resemble anything that has flown?

Every other check in this project is internal. The solvers agree with theory,
the mass budget closes, the components pass their allowables -- and all of that
could be true of a vehicle no one could build. The structural coefficient is
where that shows first, because it is the single number that decides how much
of a stage is tankage and how much is propellant, and it is asserted here
rather than derived from any solver.

So this module compares it against flown hardware. Ten stages, from Saturn V's
S-IC to Electron's first stage, span 0.036 to 0.118. This project's planner
constant is 0.140 and the repair loop's fixed point reached 0.261 -- above and
well above that range.

Two honest caveats, both of which the verdict carries.

The reference figures are secondary-source. Manufacturers do not publish stage
mass statements for most of these vehicles, and where they do the numbers move
with what is counted as interstage. This is a regime check, not a validation.

And structural coefficient is strongly size-dependent. A tank's mass scales with
its surface area while its contents scale with volume, so small stages are
always worse: Electron, the smallest here at roughly 10 tonnes of stage, is also
among the heaviest fractions. This project designs vehicles of one or two
tonnes, below anything in the reference set, and a coefficient above the flown
range is therefore expected rather than wrong. What would be wrong is not
saying so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOWN_DATA = ROOT / "data/flown_stages/flown_stage_masses.json"

#: Smallest stage in the reference set, by wet mass. Below this the comparison
#: is an extrapolation and the verdict says so rather than reporting a
#: percentage that reads like a defect.
SMALLEST_REFERENCE_WET_KG = 10_200.0


@dataclass(frozen=True)
class EnvelopeVerdict:
    struct_coeff: float
    flown_min: float
    flown_max: float
    flown_median: float
    inside: bool
    extrapolating: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "struct_coeff": round(self.struct_coeff, 4),
            "flown_min": round(self.flown_min, 4),
            "flown_max": round(self.flown_max, 4),
            "flown_median": round(self.flown_median, 4),
            "inside_flown_envelope": self.inside,
            "extrapolating_below_reference_sizes": self.extrapolating,
            "note": self.note,
        }


@lru_cache(maxsize=1)
def flown_coefficients() -> tuple[tuple[str, float, str], ...]:
    """(label, structural coefficient, confidence) for each reference stage."""
    payload = json.loads(FLOWN_DATA.read_text())
    out = []
    for s in payload["stages"]:
        dry, prop = float(s["dry_kg"]), float(s["propellant_kg"])
        if dry <= 0 or prop <= 0:
            raise ValueError(f"non-physical masses for {s.get('stage')!r}")
        out.append((f"{s['vehicle']} {s['stage']}", dry / (dry + prop),
                    str(s.get("confidence", "unknown"))))
    return tuple(sorted(out, key=lambda r: r[1]))


def check(struct_coeff: float, *, stage_wet_kg: float | None = None
          ) -> EnvelopeVerdict:
    """Place ``struct_coeff`` against flown hardware and say what that means."""
    coeffs = [c for _lbl, c, _conf in flown_coefficients()]
    lo, hi = min(coeffs), max(coeffs)
    mid = sorted(coeffs)[len(coeffs) // 2]

    extrapolating = (stage_wet_kg is not None
                     and stage_wet_kg < SMALLEST_REFERENCE_WET_KG)
    inside = lo <= struct_coeff <= hi

    if inside:
        note = (f"Within the flown range {lo:.3f}-{hi:.3f}; "
                f"{struct_coeff / mid:.2f}x the flown median.")
    elif struct_coeff > hi and extrapolating:
        # The expected direction for a small vehicle, so this is reported as a
        # consequence of scale rather than as a finding against the design.
        note = (f"Above the flown range {lo:.3f}-{hi:.3f}, by "
                f"{100.0 * (struct_coeff / hi - 1.0):.0f}%. The reference set "
                f"contains no stage below {SMALLEST_REFERENCE_WET_KG:.0f} kg "
                f"wet and this one is {stage_wet_kg:.0f} kg, so a heavier "
                f"fraction is expected: tank mass follows area while propellant "
                f"follows volume. The comparison is an extrapolation and does "
                f"not by itself indicate an error.")
    elif struct_coeff > hi:
        note = (f"Above the flown range {lo:.3f}-{hi:.3f}, by "
                f"{100.0 * (struct_coeff / hi - 1.0):.0f}%, at a stage size the "
                f"reference set does cover. This vehicle would be heavier than "
                f"any comparable stage that has flown.")
    else:
        note = (f"Below the flown minimum of {lo:.3f}. No stage in the "
                f"reference set has achieved this, so the mass budget is "
                f"claiming a structure lighter than flight-proven practice.")

    if any(conf == "low" for _lbl, _c, conf in flown_coefficients()):
        note += (" Reference figures are secondary-source and unverified "
                 "against primary mass statements; treat as a regime check, "
                 "not a validation.")

    return EnvelopeVerdict(struct_coeff, lo, hi, mid, inside, extrapolating, note)
