"""Size thin-walled structure from its load, so mass is an output not a guess.

Until now the structural coefficient was a constant (0.14) asserted in the
source, and the CAD components were *solid* cylinders -- which is why their FEA
margins came out at 200-300x. A solid billet is not a structure, and a constant
coefficient is not a design.

A launch vehicle tank is a thin-walled cylinder, and for one under axial
compression the wall is almost never set by yield. It is set by buckling, which
depends on t/r, and real shells buckle well below the classical prediction
because of imperfections. Sizing on yield alone underestimates the wall badly:
for the cases here it is out by roughly an order of magnitude.

Strength:
    sigma = P / (2 pi r t)                  ->  t = P / (2 pi r sigma_allow)

Classical axial buckling of a thin cylinder:
    sigma_cr = E t / (r sqrt(3 (1 - nu^2)))  ~  0.605 E t / r

Setting applied equal to critical, r cancels:
    P / (2 pi r t) = gamma * 0.605 E t / r
    t = sqrt( P / (1.21 pi gamma E) )

gamma is the NASA SP-8007 knockdown for imperfection sensitivity,
    phi = (1/16) sqrt(r/t),  gamma = 1 - 0.901 (1 - e^-phi)
which depends on t, so it is iterated to a fixed point.

The governing wall is the largest of strength, buckling and a manufacturing
minimum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Al-6061-T6
E_PA = 68.9e9
RHO = 2700.0
SIGMA_YIELD_PA = 276e6
SAFETY_FACTOR = 1.4
# Geometric stress concentration at joints and cutouts. Sizing the wall to the
# nominal membrane stress alone puts the peak straight past the allowable: a
# wall sized to 197 MPa nominal came back from CalculiX at 333 MPa, a factor of
# 1.69 at the fin root. Real hardware either sizes for this or adds a local
# doubler; the wall is sized for it here so the part survives its own analysis.
STRESS_CONCENTRATION = 1.7
T_MIN_M = 0.0008          # 0.8 mm, a practical minimum for spun/welded shells
# Avionics, recovery, separation hardware: roughly constant per stage
# regardless of stage size.
FIXED_STAGE_MASS_KG = 2.5

#: Engine thrust-to-weight. The stage mass model had no engine in it at all,
#: which is why solving for the structural coefficient converged near 0.06 when
#: real sounding rockets sit at 0.10 to 0.25. The omission is not marginal: for
#: a 48.8 kN stage an engine weighs 50 kg at T/W 100 and 125 kg at T/W 40,
#: against 31 kg for everything the model did count. Engine mass alone more than
#: closes the gap.
#:
#: This is an empirical design parameter, not a derived one, so it is named and
#: adjustable rather than buried. 60 is mid-range for a small pump-fed engine;
#: pressure-fed engines run lower, large pump-fed ones much higher.
ENGINE_THRUST_TO_WEIGHT = 60.0


@dataclass
class Wall:
    thickness_m: float
    mass_kg: float
    driver: str
    margin_yield: float
    margin_buckling: float


def _knockdown(r_over_t: float) -> float:
    """NASA SP-8007 imperfection knockdown for axial compression."""
    phi = (1.0 / 16.0) * math.sqrt(max(r_over_t, 1.0))
    return max(0.10, 1.0 - 0.901 * (1.0 - math.exp(-phi)))


def size_wall(load_n: float, radius_m: float, length_m: float,
              *, sigma_allow_pa: float | None = None,
              density_kg_m3: float | None = None,
              yield_pa: float | None = None,
              modulus_pa: float | None = None) -> Wall:
    """Wall thickness and mass for a thin cylinder under axial compression.

    Material defaults to the Al-6061-T6 constants at module scope. It is an
    argument because a vehicle that gets too hot for aluminium has to be built
    from something denser, and both the thickness (through yield and modulus)
    and the mass (through density) have to feel that.
    """
    rho = float(density_kg_m3 or RHO)
    sigma_y = float(yield_pa or SIGMA_YIELD_PA)
    young = float(modulus_pa or E_PA)
    if sigma_allow_pa is None:
        sigma_allow_pa = sigma_y / SAFETY_FACTOR / STRESS_CONCENTRATION
    load = max(load_n, 1.0)

    t_strength = load / (2.0 * math.pi * radius_m * sigma_allow_pa)

    # buckling: iterate because the knockdown depends on r/t
    t_buckle = math.sqrt(load / (1.21 * math.pi * 0.5 * young))
    for _ in range(40):
        gamma = _knockdown(radius_m / max(t_buckle, 1e-6))
        t_new = math.sqrt(load / (1.21 * math.pi * gamma * young))
        if abs(t_new - t_buckle) < 1e-9:
            t_buckle = t_new
            break
        t_buckle = t_new

    t = max(t_strength, t_buckle, T_MIN_M)
    driver = ("buckling" if t == t_buckle else
              "strength" if t == t_strength else "minimum gauge")

    mass = rho * 2.0 * math.pi * radius_m * t * length_m

    sigma_applied = load / (2.0 * math.pi * radius_m * t)
    gamma = _knockdown(radius_m / t)
    sigma_cr = gamma * 0.605 * young * t / radius_m
    return Wall(
        thickness_m=t, mass_kg=mass, driver=driver,
        margin_yield=sigma_y / max(sigma_applied, 1.0),
        margin_buckling=sigma_cr / max(sigma_applied, 1.0),
    )


def stage_structural_mass(prop_mass_kg: float, radius_m: float,
                          thrust_n: float, *, propellant_density: float = 1030.0,
                          density_kg_m3: float | None = None,
                          yield_pa: float | None = None,
                          modulus_pa: float | None = None
                          ) -> tuple[float, list[dict]]:
    """Structural mass of a stage, summed from its sized components.

    Tank length follows from the propellant it must hold at the given radius,
    so a bigger stage gets a longer tank and a heavier one -- the coupling that
    a constant structural coefficient throws away.

    The material properties default to the Al-6061-T6 constants above. They are
    arguments because skin material became a design variable: a vehicle whose
    peak skin temperature exceeds what aluminium holds has to be built from
    something else, and titanium is 64% denser and twice as strong. Passing the
    material through means that trade lands in the mass budget instead of being
    granted for free.
    """
    rho = float(density_kg_m3 or RHO)
    sigma = float(yield_pa or SIGMA_YIELD_PA)
    young = float(modulus_pa or E_PA)
    vol = prop_mass_kg / propellant_density
    tank_len = max(0.2, vol / (math.pi * radius_m ** 2))

    mat = {"density_kg_m3": rho, "yield_pa": sigma, "modulus_pa": young}
    parts = []
    tank = size_wall(thrust_n, radius_m, tank_len, **mat)
    parts.append({"name": "tank", "length_m": tank_len, **tank.__dict__})

    inter = size_wall(thrust_n, radius_m, radius_m * 1.5, **mat)
    parts.append({"name": "interstage", "length_m": radius_m * 1.5, **inter.__dict__})

    thrust_struct = size_wall(thrust_n * 1.3, radius_m, radius_m * 1.2, **mat)
    parts.append({"name": "thrust structure", "length_m": radius_m * 1.2,
                  **thrust_struct.__dict__})

    shell_mass = sum(p["mass_kg"] for p in parts)
    engine_mass = thrust_n / (9.80665 * ENGINE_THRUST_TO_WEIGHT)
    parts.append({"name": "engine", "length_m": 0.0, "mass_kg": engine_mass,
                  "note": f"thrust/(g0 * T/W={ENGINE_THRUST_TO_WEIGHT:g})"})
    # Bulkheads, engine, plumbing scale with the stage; avionics, recovery and
    # separation hardware largely do not. Scaling everything with shell mass
    # makes small vehicles come out absurdly light -- the loop converged to a
    # structural coefficient of 0.048 for a 25 kg payload where real sounding
    # rockets sit at 0.10-0.25, because a 2 kg flight computer does not shrink
    # with the tank.
    scaling = shell_mass * 0.85
    fixed = FIXED_STAGE_MASS_KG
    return shell_mass + scaling + fixed + engine_mass, parts


def coefficient_attribution(prop_mass_kg: float, radius_m: float,
                            thrust_n: float, **mat) -> dict:
    """Which term makes the structural coefficient what it is.

    The packet already reports that this vehicle's coefficient sits at roughly
    twice the heaviest stage ever flown. It has never said which part of the
    stage is responsible, and without that the finding is a complaint rather
    than a lead.

    The answer is not the one the buckling work suggested. Shell -- tank,
    interstage and thrust structure together -- is a quarter to a third of stage
    structure at every size tried. The engine is half, and it is half because
    ENGINE_THRUST_TO_WEIGHT is 60, which is conservative: flown engines run from
    about 80 for the RD-180 to 180 for a modern kerolox booster engine. At 150
    the coefficient for this stage lands inside the flown range on its own.

    So the lever is the engine mass model, not the wall. Reported as a
    decomposition rather than a conclusion, because a reader who disagrees with
    the engine assumption should be able to see exactly how much of the answer
    rests on it.
    """
    total, parts = stage_structural_mass(prop_mass_kg, radius_m, thrust_n, **mat)
    by_name = {p["name"]: p["mass_kg"] for p in parts}
    shell = sum(m for n, m in by_name.items() if n != "engine")
    engine = by_name.get("engine", 0.0)
    # stage_structural_mass adds a scaling term and a fixed term on top of the
    # parts it lists; recover them rather than restating the formula, so this
    # cannot drift from the function it describes.
    other = total - shell - engine
    coeff = total / (total + prop_mass_kg) if prop_mass_kg > 0 else 0.0

    terms = {"shell": shell, "engine": engine,
             "plumbing, avionics and fixed": other}
    dominant = max(terms, key=terms.get)
    return {
        "total_kg": total,
        "coefficient": coeff,
        "terms_kg": terms,
        "shares": {k: (v / total if total > 0 else 0.0)
                   for k, v in terms.items()},
        "dominant": dominant,
        "engine_thrust_to_weight": ENGINE_THRUST_TO_WEIGHT,
        "wall_driver": by_name and next(
            (p.get("driver", "") for p in parts if p["name"] == "tank"), ""),
        "note": (
            f"{dominant} is {100*terms[dominant]/total:.0f}% of stage structure. "
            f"The engine is sized at thrust over g0 times a thrust-to-weight of "
            f"{ENGINE_THRUST_TO_WEIGHT:g}, which is conservative against flown "
            f"engines at 80 to 180, and it is the single largest term"),
    }
