"""Shell buckling of the vehicle skin, which is what actually sizes it.

Every structural check in this project compares a stress against a yield
allowable. For a thin-walled cylinder in compression that is the wrong failure
mode, and not by a little. A monocoque skin does not yield and tear; it goes
unstable and folds, at a stress that can be an order of magnitude below yield.

The packet that prompted this module reported 67.8 MPa of combined skin stress
against a 700 MPa yield allowable -- a margin of ten, comfortable by any
reading. The same wall, 0.80 mm at 335 mm radius, buckles at about 102 MPa. The
real margin is 1.5, and at the larger radius the sizing loop had been
considering it is 0.83, meaning the vehicle folds on the pad.

Two things drive that. The classical critical stress falls with t/r, so a
slender thin shell is intrinsically weak in compression. And real cylinders
buckle far below the classical value because they are not perfect cylinders: a
dent of a fraction of the wall thickness is enough. The knockdown factors here
are the empirical lower bounds from NASA SP-8007, fitted to a large body of test
data precisely because theory alone is unconservative by factors of two to five.

Bending is treated separately from uniform compression, and less harshly. Under
bending only part of the circumference is in compression and the peak stress
occupies a small arc, so an imperfection is less likely to sit where it does
harm. The two are then combined by their interaction rule rather than added.

What this module does not model: internal pressure, which stabilises a shell and
raises its buckling stress substantially -- a pressurised tank is much stronger
than this reports. Leaving it out is conservative and is stated rather than
quietly assumed, because claiming pressure stabilisation for an unpressurised
interstage would be the dangerous direction to be wrong in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Above this radius-to-thickness ratio the shell is thin enough that the
#: knockdown correlations apply. Below it the wall is thick and the failure
#: mode drifts back toward material yield.
THIN_SHELL_MIN_R_OVER_T = 20.0

#: SP-8007 fits the knockdown over roughly this range of R/t. Outside it the
#: correlation is an extrapolation and says so.
CORRELATION_MAX_R_OVER_T = 1500.0


@dataclass
class BucklingResult:
    r_over_t: float
    classical_mpa: float
    gamma_compression: float
    gamma_bending: float
    allowable_compression_mpa: float
    allowable_bending_mpa: float
    interaction: float
    margin: float
    governs: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passes(self) -> bool:
        return self.interaction <= 1.0

    def as_dict(self) -> dict:
        return {
            "r_over_t": round(self.r_over_t, 1),
            "classical_mpa": round(self.classical_mpa, 2),
            "gamma_compression": round(self.gamma_compression, 4),
            "gamma_bending": round(self.gamma_bending, 4),
            "allowable_compression_mpa": round(self.allowable_compression_mpa, 2),
            "allowable_bending_mpa": round(self.allowable_bending_mpa, 2),
            "interaction": round(self.interaction, 4),
            "margin": round(self.margin, 3),
            "governs": self.governs,
            "passes": self.passes,
            "notes": list(self.notes),
        }


def classical_axial_stress_pa(youngs_pa: float, radius_m: float, wall_m: float,
                              poisson: float = 0.33) -> float:
    """Classical buckling stress of a perfect cylinder in axial compression.

    sigma = E t / (r sqrt(3 (1 - nu^2))). This is an upper bound no real shell
    reaches; it exists here to be knocked down, and is reported alongside the
    knockdown so the size of that correction is visible rather than buried.
    """
    r, t = float(radius_m), float(wall_m)
    if r <= 0 or t <= 0:
        raise ValueError("radius and wall thickness must be positive")
    return float(youngs_pa) * t / (r * math.sqrt(3.0 * (1.0 - poisson ** 2)))


def knockdown_compression(radius_m: float, wall_m: float) -> float:
    """SP-8007 empirical lower bound for uniform axial compression."""
    phi = (1.0 / 16.0) * math.sqrt(float(radius_m) / float(wall_m))
    return 1.0 - 0.901 * (1.0 - math.exp(-phi))


def knockdown_bending(radius_m: float, wall_m: float) -> float:
    """SP-8007 empirical lower bound for bending.

    Less severe than uniform compression -- 0.731 against 0.901 -- because only
    part of the circumference carries peak compressive stress, so an
    imperfection is less likely to coincide with it.
    """
    phi = (1.0 / 16.0) * math.sqrt(float(radius_m) / float(wall_m))
    return 1.0 - 0.731 * (1.0 - math.exp(-phi))


def check(*, axial_mpa: float, bending_mpa: float, radius_m: float,
          wall_m: float, youngs_pa: float, poisson: float = 0.33
          ) -> BucklingResult:
    """Buckling check for a monocoque skin under axial load and bending.

    ``axial_mpa`` and ``bending_mpa`` are the compressive stresses already
    computed for the section; they are kept separate rather than summed because
    the two have different allowables and are combined by an interaction rule.
    """
    r, t = float(radius_m), float(wall_m)
    r_over_t = r / t
    classical = classical_axial_stress_pa(youngs_pa, r, t, poisson)
    gc = knockdown_compression(r, t)
    gb = knockdown_bending(r, t)
    allow_c = gc * classical / 1e6
    allow_b = gb * classical / 1e6

    notes: list[str] = []
    if r_over_t < THIN_SHELL_MIN_R_OVER_T:
        notes.append(
            f"R/t is {r_over_t:.0f}, below {THIN_SHELL_MIN_R_OVER_T:.0f}. The "
            f"wall is thick enough that buckling may not govern and material "
            f"yield should be checked as well.")
    if r_over_t > CORRELATION_MAX_R_OVER_T:
        notes.append(
            f"R/t is {r_over_t:.0f}, beyond the range the SP-8007 knockdown was "
            f"fitted over; the allowable below is an extrapolation.")

    # Linear interaction, R_c + R_b <= 1. The conservative choice among the
    # forms in use, and the right one absent test data for this specific shell.
    rc = max(0.0, float(axial_mpa)) / max(allow_c, 1e-9)
    rb = max(0.0, float(bending_mpa)) / max(allow_b, 1e-9)
    interaction = rc + rb
    margin = 1.0 / interaction if interaction > 0 else float("inf")
    governs = "bending" if rb > rc else "compression"

    notes.append(
        "Internal pressure is not credited. A pressurised tank buckles at a "
        "substantially higher stress than this, so the result is conservative "
        "for a tank and correct as written for an unpressurised interstage.")

    return BucklingResult(
        r_over_t=r_over_t, classical_mpa=classical / 1e6,
        gamma_compression=gc, gamma_bending=gb,
        allowable_compression_mpa=allow_c, allowable_bending_mpa=allow_b,
        interaction=interaction, margin=margin, governs=governs,
        notes=tuple(notes))


def wall_for_buckling_m(*, axial_n: float, moment_nm: float, radius_m: float,
                        youngs_pa: float, poisson: float = 0.33,
                        target_margin: float = 1.4,
                        max_wall_m: float = 0.05) -> float | None:
    """Thinnest wall that survives buckling at ``target_margin``.

    Bisection rather than a closed form: the knockdown depends on the very
    thickness being solved for, so the allowable moves as the wall moves and
    there is no explicit inverse.
    """
    lo, hi = 1e-5, float(max_wall_m)

    def interaction_at(t: float) -> float:
        area = 2.0 * math.pi * radius_m * t
        modulus = math.pi * radius_m * radius_m * t
        res = check(axial_mpa=(axial_n / area) / 1e6,
                    bending_mpa=(abs(moment_nm) / modulus) / 1e6,
                    radius_m=radius_m, wall_m=t, youngs_pa=youngs_pa,
                    poisson=poisson)
        return res.interaction

    if interaction_at(hi) > 1.0 / target_margin:
        return None            # even the thickest allowed wall buckles
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if interaction_at(mid) > 1.0 / target_margin:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return hi
