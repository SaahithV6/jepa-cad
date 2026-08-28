"""Engine thrust-to-weight, against engines that have flown.

``structural_sizing`` sizes the engine as thrust / (g0 * 60), and that 60 turned
out to be the single largest term in stage structure -- about half of it, against
a quarter for the whole shell. A number carrying that much of the answer should
not be a bare constant, and the project already has the pattern for fixing that:
``flown_envelope`` places the structural coefficient against ten flown stages
rather than asserting a limit. This does the same for the engine.

Two things fall out of the table that are worth having written down.

The range is 66 to 183 with a median of 82, so 60 is *modestly* conservative
rather than wildly so. An earlier note in this session claimed flown engines run
"80 to 180" and used that to argue the model was badly pessimistic. The bottom of
the real range is 66 -- the RS-25 -- and the RD-180 is 71, so the claim was
wrong at the low end and the argument built on it was too strong.

And thrust-to-weight does not scale with thrust. Merlin 1D at 845 kN reaches 183
while the RD-180 at 3830 kN sits at 71; Rutherford at 24 kN manages 70, better
than either. The scatter is driven by engine cycle, propellant and design era,
not size. That matters because it refutes an appealing explanation for something
this project measured: the solved structural coefficient does not improve as the
vehicle grows, and a thrust-scaled engine model would have explained that
neatly. It would also have been wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Flown engines: (name, vehicle, thrust N, dry mass kg, propellant, cycle).
#:
#: Thrust is sea-level for boosters and vacuum for upper stages, matching how
#: each is usually quoted; the ratio is not sensitive enough to the choice to
#: change any conclusion drawn here.
FLOWN_ENGINES = [
    ("Rutherford", "Electron", 24_000.0, 35.0, "lox/rp1", "electric pump"),
    ("Vikas", "PSLV / GSLV", 725_000.0, 900.0, "n2o4/udmh", "gas generator"),
    ("Merlin 1D", "Falcon 9", 845_000.0, 470.0, "lox/rp1", "gas generator"),
    ("RS-25", "Space Shuttle", 2_280_000.0, 3527.0, "lox/lh2",
     "staged combustion"),
    ("Raptor 2", "Starship", 2_300_000.0, 1600.0, "lox/lch4",
     "full-flow staged"),
    ("RD-180", "Atlas V", 3_830_000.0, 5480.0, "lox/rp1", "staged combustion"),
    ("F-1", "Saturn V", 6_770_000.0, 8400.0, "lox/rp1", "gas generator"),
]

G0 = 9.80665


@dataclass(frozen=True)
class EngineVerdict:
    assumed: float
    flown_min: float
    flown_max: float
    flown_median: float
    inside: bool
    #: Where the assumption sits in the flown range, 0 at min and 1 at max
    percentile: float
    note: str

    def as_dict(self) -> dict:
        return {"assumed": self.assumed, "flown_min": self.flown_min,
                "flown_max": self.flown_max, "flown_median": self.flown_median,
                "inside": self.inside, "percentile": self.percentile,
                "note": self.note}


def thrust_to_weight(thrust_n: float, dry_mass_kg: float) -> float:
    if dry_mass_kg <= 0:
        raise ValueError("dry mass must be positive")
    return thrust_n / (G0 * dry_mass_kg)


def flown_ratios() -> list[float]:
    return [thrust_to_weight(t, m) for _n, _v, t, m, _p, _c in FLOWN_ENGINES]


def check(assumed_twr: float) -> EngineVerdict:
    """Place an assumed engine thrust-to-weight against flown hardware.

    Below the flown minimum is not a fault -- a conservative engine model is a
    defensible choice -- but it is a fact about the design that should be
    visible, because at roughly half of stage structure the engine is where a
    conservative assumption costs the most.
    """
    ratios = sorted(flown_ratios())
    lo, hi = ratios[0], ratios[-1]
    mid = ratios[len(ratios) // 2]
    a = float(assumed_twr)
    inside = lo <= a <= hi
    pct = (a - lo) / (hi - lo) if hi > lo else 0.0

    if a < lo:
        note = (f"{a:.0f} is below every engine in the table; the lowest flown "
                f"is the RS-25 at {lo:.0f}. The engine is about half of stage "
                f"structure, so this is the most expensive conservatism in the "
                f"mass budget")
    elif a > hi:
        note = (f"{a:.0f} exceeds every engine flown; the best is Merlin 1D at "
                f"{hi:.0f}, and claiming better than that needs an argument "
                f"this project does not have")
    else:
        note = (f"{a:.0f} sits inside the flown range {lo:.0f} to {hi:.0f} "
                f"(median {mid:.0f}), at the {100*pct:.0f}th percentile of it")
    return EngineVerdict(assumed=a, flown_min=lo, flown_max=hi,
                         flown_median=mid, inside=inside, percentile=pct,
                         note=note)


def scales_with_thrust() -> dict:
    """Does a bigger engine do better? Measured, because it looks like it should.

    A thrust-scaled engine model would neatly explain why this project's solved
    structural coefficient fails to improve as the vehicle grows. The table says
    no: the correlation between thrust and thrust-to-weight across flown engines
    is weak, and the two highest ratios belong to a 845 kN engine and a 2.3 MN
    one while the largest engine in the list sits near the bottom.

    Returned as a coefficient rather than a verdict so a reader can judge the
    strength rather than take the word "weak" on trust.
    """
    xs = [t for _n, _v, t, _m, _p, _c in FLOWN_ENGINES]
    ys = flown_ratios()
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else 0.0
    return {
        "pearson_r": r,
        "supports_scaling": abs(r) > 0.7,
        "note": (f"correlation between thrust and thrust-to-weight across "
                 f"{n} flown engines is r = {r:+.2f}. Cycle, propellant and "
                 f"design era drive the scatter, not size"),
    }
