"""What it costs to keep the tanks pressurised.

Nothing in this project had a tank pressure. The barrel was sized for axial load
and bending and checked against buckling; the mass budget carried structure and
propellant and nothing else. A launch vehicle cannot fly like that, for two
reasons that pull in opposite directions.

A pump will not draw from an unpressurised tank. Below a certain head the
propellant flashes at the inducer and the pump cavitates, so every stage carries
ullage pressure above the propellant's vapour pressure -- the net positive
suction head. That pressure has to come from somewhere, and on a pump-fed stage
it comes from a bottle of helium that weighs something, in a bottle that weighs
considerably more.

The same pressure then loads the tank wall in hoop, and hoop from internal
pressure is a *tension*, at a stress the wall may or may not have been sized to
carry. This packet sizes its wall against compression and bending, where the
governing failure is buckling. Those are different failure modes reading the
same thickness, and until both are computed nothing knows which one wins.

What is derived here and what is assumed
----------------------------------------
Derived, exactly: the mass of a pressure vessel. For a thin-walled sphere the
membrane thickness is t = pR/(2 sigma), so

    m = rho * 4 pi R^2 * pR / (2 sigma) = 2 pi rho p R^3 / sigma

and with V = (4/3) pi R^3 this collapses to

    m = (3/2) * (rho / sigma) * p * V

which is independent of radius. Mass per unit stored volume is set by the
pressure and by the material's strength-to-density ratio, and by nothing else.
The same relation with a factor of 3 rather than 3/2 holds for a cylinder, which
is why pressurant bottles are spherical wherever packaging allows.

Derived: helium mass from the ideal gas law over the volume it must fill, with a
collapse factor for gas cooling against cold walls and colder propellant.

Assumed, and labelled as such: the net positive suction head a turbopump
requires. That is a property of a pump this project does not design. The default
is taken from flown practice for LOX/kerosene stages and the resulting tank
pressure is reported, so a reader can see what it rests on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G0 = 9.80665

#: Specific gas constant for helium, J/(kg K).
R_HELIUM = 2077.1

#: Storage pressure of a composite overwrapped pressurant bottle, Pa.
#:
#: 350 bar is ordinary for flight COPVs. It matters less than it looks: bottle
#: mass by the relation above is (3/2)(rho/sigma) p V, and the helium a tank
#: needs fixes p*V, so storage pressure cancels out of the bottle mass almost
#: entirely. What it sets is the bottle's *size*, not its weight.
BOTTLE_STORAGE_PA = 350e5

#: Ullage collapse factor.
#:
#: Helium entering a tank of cryogenic oxygen does not stay at the temperature it
#: was heated to. It cools against the walls and the liquid surface and loses
#: pressure, so more of it is needed than an isothermal calculation says. Flown
#: stages run between about 1.6 and 2.5 depending on how much of the tank wall
#: is wetted; 2.0 is mid-range and is an assumption, not a derivation.
COLLAPSE_FACTOR = 2.0

#: Temperature helium is delivered at after the heat exchanger, K.
PRESSURANT_TEMP_K = 250.0

#: Factor of safety on tank proof pressure, NASA-STD-5001B for pressurised
#: structure that is also load-bearing.
TANK_FOS = 1.5

#: Net positive suction head a turbopump inducer requires, metres of propellant.
#:
#: ASSUMED, not derived: this is a property of a pump this project does not
#: design. Flown LOX/kerosene stages sit between roughly 10 and 30 m of head at
#: the inlet. 20 m is mid-range. The tank pressure that follows is reported
#: explicitly so the assumption is visible rather than buried.
NPSH_REQUIRED_M = 20.0

#: Vapour pressures at storage temperature, Pa, and densities, kg/m3.
#:
#: Cryogens are stored at their boiling point at one atmosphere, so their vapour
#: pressure *is* one atmosphere -- which is why a LOX tank needs meaningful
#: ullage pressure before a pump will draw from it at all, and a kerosene tank
#: needs almost none.
PROPELLANTS = {
    "lox": {"vapour_pa": 101325.0, "density": 1141.0, "name": "liquid oxygen"},
    "rp1": {"vapour_pa": 500.0, "density": 810.0, "name": "RP-1"},
    "lh2": {"vapour_pa": 101325.0, "density": 71.0, "name": "liquid hydrogen"},
    "lch4": {"vapour_pa": 101325.0, "density": 422.0, "name": "liquid methane"},
    "n2o4": {"vapour_pa": 96_000.0, "density": 1440.0, "name": "N2O4"},
}

#: Which propellants a combination uses, oxidiser first.
#:
#: Only combinations the planner can actually fly. lox_lch4 was listed here and
#: nowhere else in the program: not in planner.PROPELLANTS, not in
#: combustion.REFERENCE. Asking for it died with a bare KeyError deep inside
#: chamber_properties, because this table advertised a capability the rest of
#: the code does not have. A lookup that offers more than the system supports is
#: a promise made by whoever edited the table last.
#:
#: Methane is spelled lox_ch4, which is what the chemistry calls it. This table
#: said lox_lch4 -- a name nothing else in the program uses -- so a real,
#: supported combination was unreachable through it while appearing to be
#: offered. The CLI now validates against the chemistry's own key set, so a
#: name that exists only here cannot be advertised again.
COMBINATIONS = {
    "lox_rp1": ("lox", "rp1"),
    "lox_lh2": ("lox", "lh2"),
    "lox_ch4": ("lox", "lch4"),
}


@dataclass(frozen=True)
class TankPressure:
    """The pressure one tank must hold, and why."""

    propellant: str
    vapour_pa: float
    npsh_pa: float
    ullage_pa: float
    #: Design pressure including the factor of safety
    design_pa: float
    note: str = ""

    def as_dict(self) -> dict:
        return {"propellant": self.propellant, "vapour_pa": self.vapour_pa,
                "npsh_pa": self.npsh_pa, "ullage_pa": self.ullage_pa,
                "design_pa": self.design_pa, "note": self.note}


@dataclass(frozen=True)
class Pressurisation:
    """What pressurising a stage costs in mass."""

    stage: int
    tank_volume_m3: float
    ullage_pa: float
    helium_kg: float
    bottle_kg: float
    #: Tank end closures. Not an extra: the axial relief wall_load_state
    #: credits is pr/2t, which a tank only has because it has ends.
    dome_kg: float = 0.0
    #: Hoop stress the ullage pressure puts in the tank wall, Pa
    hoop_pa: float = 0.0
    tanks: list[TankPressure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_kg(self) -> float:
        return self.helium_kg + self.bottle_kg + self.dome_kg

    def as_dict(self) -> dict:
        return {"stage": self.stage, "tank_volume_m3": self.tank_volume_m3,
                "ullage_pa": self.ullage_pa, "helium_kg": self.helium_kg,
                "bottle_kg": self.bottle_kg, "dome_kg": self.dome_kg,
                "total_kg": self.total_kg,
                "hoop_pa": self.hoop_pa,
                "tanks": [t.as_dict() for t in self.tanks],
                "notes": list(self.notes)}


def pressure_vessel_mass_kg(pressure_pa: float, volume_m3: float, *,
                            density_kg_m3: float, allowable_pa: float,
                            spherical: bool = True) -> float:
    """Minimum membrane mass of a pressure vessel: (3/2) (rho/sigma) p V.

    Derived, not tabulated. For a thin sphere t = pR/(2 sigma), so
    m = rho 4 pi R^2 t = 2 pi rho p R^3 / sigma, and substituting
    V = (4/3) pi R^3 removes the radius entirely. A cylinder of the same volume
    carries twice the membrane stress for the same wall and comes out at 3 rho p
    V / sigma -- exactly double -- which is why pressurant bottles are spheres
    wherever anything else does not force the shape.

    This is a floor. It counts membrane material and nothing else: no bosses, no
    liner, no overwrap resin, no weld lands. A real bottle is heavier.
    """
    if pressure_pa <= 0 or volume_m3 <= 0:
        return 0.0
    if allowable_pa <= 0:
        raise ValueError("allowable stress must be positive")
    shape = 1.5 if spherical else 3.0
    return shape * (density_kg_m3 / allowable_pa) * pressure_pa * volume_m3


def tank_pressure(propellant: str, *, npsh_m: float = NPSH_REQUIRED_M,
                  fos: float = TANK_FOS,
                  acceleration_g: float = 0.0,
                  head_height_m: float = 0.0) -> TankPressure:
    """The ullage pressure a pump-fed tank needs, and the wall's design pressure.

    Ullage must exceed the propellant's vapour pressure by the suction head the
    pump requires, less whatever the vehicle's own acceleration supplies: under
    thrust the column of propellant above the inlet is itself a head, and on a
    tall first stage at several g that term is not small.

    That last part is why this is worth computing rather than assuming a number.
    A stage pulling 4 g with two metres of propellant above the outlet generates
    tens of kPa of head for free, and a design that ignores it carries helium it
    does not need.
    """
    key = propellant.lower()
    if key not in PROPELLANTS:
        raise KeyError(f"unknown propellant {propellant!r}; "
                       f"known: {sorted(PROPELLANTS)}")
    prop = PROPELLANTS[key]
    rho = prop["density"]

    npsh_pa = rho * G0 * float(npsh_m)
    # Acceleration head, which the vehicle supplies for free while it burns.
    accel_pa = rho * G0 * float(acceleration_g) * float(head_height_m)
    ullage = prop["vapour_pa"] + max(0.0, npsh_pa - accel_pa)

    note = ""
    if accel_pa > 0:
        note = (f"acceleration supplies {accel_pa/1000:.0f} kPa of the "
                f"{npsh_pa/1000:.0f} kPa suction head")
    if accel_pa >= npsh_pa:
        note = (f"acceleration alone ({accel_pa/1000:.0f} kPa) covers the "
                f"{npsh_pa/1000:.0f} kPa suction head; ullage holds vapour "
                f"pressure only")
    return TankPressure(propellant=key, vapour_pa=prop["vapour_pa"],
                        npsh_pa=npsh_pa, ullage_pa=ullage,
                        design_pa=ullage * float(fos), note=note)


def helium_mass_kg(ullage_pa: float, volume_m3: float, *,
                   temperature_k: float = PRESSURANT_TEMP_K,
                   collapse_factor: float = COLLAPSE_FACTOR) -> float:
    """Helium needed to hold ``ullage_pa`` over a tank of ``volume_m3``.

    Ideal gas over the volume the propellant vacates, which by the end of the
    burn is the whole tank. The collapse factor covers the gas cooling against
    cold walls and colder liquid, and is an assumption drawn from flown
    practice rather than a derivation -- helium entering a LOX tank does not
    stay at the temperature the heat exchanger delivered it at.
    """
    if ullage_pa <= 0 or volume_m3 <= 0:
        return 0.0
    return (ullage_pa * volume_m3 / (R_HELIUM * temperature_k)
            * float(collapse_factor))


def hoop_stress_pa(pressure_pa: float, radius_m: float, wall_m: float) -> float:
    """Thin-shell hoop, p r / t. Tension, unlike everything else on this wall."""
    if wall_m <= 0:
        raise ValueError("wall thickness must be positive")
    return pressure_pa * radius_m / wall_m


def stage_pressurisation(*, stage: int, propellant_mass_kg: float,
                         combination: str = "lox_rp1",
                         of_ratio: float | None = None,
                         radius_m: float, wall_m: float,
                         bottle_density_kg_m3: float = 4430.0,
                         bottle_allowable_pa: float = 620e6,
                         acceleration_g: float = 0.0,
                         head_height_m: float = 0.0,
                         npsh_m: float = NPSH_REQUIRED_M,
                         wall_density_kg_m3: float = 8190.0,
                         wall_allowable_pa: float = 700e6) -> Pressurisation:
    """Everything one stage's pressurisation costs.

    Bottle defaults are Ti-6Al-4V at a 620 MPa design allowable, which is a
    metal-lined bottle without an overwrap. A real COPV does better; this is the
    conservative end and it is stated rather than optimised, because a lighter
    number here would be a claim about hardware this project does not design.
    """
    # The mixture ratio the engine actually burns, not a number written here.
    #
    # This defaulted to a hardcoded 2.56 while the chemistry and trajectory used
    # the catalogue's 2.45 for lox/rp1, so the tanks were sized for a 1.3%
    # different oxidiser fraction than the engine consumes -- both tank volumes
    # wrong, in opposite directions, on a vehicle whose tank ends are the thing
    # that decides whether a stage is affordable.
    if of_ratio is None:
        try:
            from cadflow.combustion import REFERENCE

            of_ratio = float(REFERENCE[combination][0])
        except Exception:  # noqa: BLE001
            of_ratio = 2.45
    ox_key, fuel_key = COMBINATIONS.get(combination, ("lox", "rp1"))
    ox = PROPELLANTS[ox_key]
    fuel = PROPELLANTS[fuel_key]

    # Split the propellant by mixture ratio, then by density into volumes.
    m_ox = propellant_mass_kg * of_ratio / (1.0 + of_ratio)
    m_fuel = propellant_mass_kg - m_ox
    v_ox = m_ox / ox["density"]
    v_fuel = m_fuel / fuel["density"]
    volume = v_ox + v_fuel

    t_ox = tank_pressure(ox_key, npsh_m=npsh_m, acceleration_g=acceleration_g,
                         head_height_m=head_height_m)
    t_fuel = tank_pressure(fuel_key, npsh_m=npsh_m,
                           acceleration_g=acceleration_g,
                           head_height_m=head_height_m)

    he = (helium_mass_kg(t_ox.ullage_pa, v_ox)
          + helium_mass_kg(t_fuel.ullage_pa, v_fuel))

    # The bottle holds that helium at storage pressure. Its volume follows from
    # the gas law at ambient bottle temperature; its mass from the membrane
    # relation. Storage pressure cancels almost entirely: p*V is fixed by the
    # helium mass, so a higher-pressure bottle is smaller, not lighter.
    v_bottle = (he * R_HELIUM * 293.0 / BOTTLE_STORAGE_PA) if he > 0 else 0.0
    bottle = pressure_vessel_mass_kg(
        BOTTLE_STORAGE_PA * TANK_FOS, v_bottle,
        density_kg_m3=bottle_density_kg_m3,
        allowable_pa=bottle_allowable_pa, spherical=True)

    # Hoop in the tank wall, from the higher of the two ullage pressures.
    worst = max(t_ox, t_fuel, key=lambda t: t.ullage_pa)
    hoop = hoop_stress_pa(worst.design_pa, radius_m, wall_m)

    # The ends, without which there is no axial relief to credit. Two tanks per
    # stage, two domes each -- oxidiser and fuel are separate vessels.
    dome = dome_mass_kg(pressure_pa=worst.design_pa, radius_m=radius_m,
                        density_kg_m3=wall_density_kg_m3,
                        allowable_pa=wall_allowable_pa, domes=4)

    notes = [
        f"tanks sized by volume: {v_ox:.3f} m3 {ox['name']} + "
        f"{v_fuel:.3f} m3 {fuel['name']} at O/F {of_ratio:.2f}",
        f"helium is {he:.2f} kg in a {bottle:.2f} kg bottle, "
        f"{100*(he+bottle)/max(propellant_mass_kg, 1e-9):.2f}% of the "
        f"propellant it pressurises",
        f"tank ends are {dome['mass_kg']:.2f} kg across {dome['domes']} domes at "
        f"{1000*dome['thickness_used_m']:.2f} mm"
        + (f", set by minimum gauge rather than by pressure -- the membrane "
           f"requirement is only {1000*dome['thickness_membrane_m']:.3f} mm"
           if dome["gauge_limited"] else ", membrane-limited"),
        f"net positive suction head of {npsh_m:.0f} m is ASSUMED from flown "
        f"practice, not derived -- this project designs no pump",
    ]
    if t_ox.note:
        notes.append(f"{ox['name']}: {t_ox.note}")

    return Pressurisation(stage=stage, tank_volume_m3=volume,
                          ullage_pa=worst.ullage_pa, helium_kg=he,
                          bottle_kg=bottle, dome_kg=dome["mass_kg"],
                         hoop_pa=hoop,
                          tanks=[t_ox, t_fuel], notes=notes)


def pressure_fed_tank_pressure_pa(chamber_pressure_pa: float,
                                  injector_drop_fraction: float = 0.2,
                                  line_loss_fraction: float = 0.1) -> float:
    """What the tanks would need with no pumps at all.

    Chamber pressure plus the injector drop that keeps combustion stable plus
    line and valve losses. There is no free lunch: a pressure-fed stage carries
    its chamber pressure in its tank walls.
    """
    return chamber_pressure_pa * (1.0 + injector_drop_fraction
                                  + line_loss_fraction)


def feed_system_verdict(chamber_pressure_pa: float, tank_volume_m3: float, *,
                        density_kg_m3: float, allowable_pa: float) -> dict:
    """Pump-fed or pressure-fed, decided by what the tanks would weigh.

    Not a preference. At 55 bar chamber pressure a pressure-fed stage needs
    about 72 bar in its tanks, and the membrane relation says what that weighs
    for a given volume. Comparing it against the propellant it holds settles the
    architecture without anyone having to assert it.
    """
    p_fed = pressure_fed_tank_pressure_pa(chamber_pressure_pa)
    m_fed = pressure_vessel_mass_kg(p_fed * TANK_FOS, tank_volume_m3,
                                    density_kg_m3=density_kg_m3,
                                    allowable_pa=allowable_pa, spherical=False)
    p_pump = tank_pressure("lox").design_pa
    m_pump = pressure_vessel_mass_kg(p_pump, tank_volume_m3,
                                     density_kg_m3=density_kg_m3,
                                     allowable_pa=allowable_pa,
                                     spherical=False)
    return {
        "pressure_fed_tank_pa": p_fed,
        "pressure_fed_tank_mass_kg": m_fed,
        "pump_fed_tank_pa": p_pump,
        "pump_fed_tank_mass_kg": m_pump,
        "ratio": m_fed / m_pump if m_pump > 0 else float("inf"),
        "verdict": "pump-fed" if m_fed > m_pump else "pressure-fed",
        "note": (f"pressure-fed tanks would need {p_fed/1e5:.0f} bar and weigh "
                 f"{m_fed:.0f} kg against {m_pump:.0f} kg pump-fed, "
                 f"{m_fed/max(m_pump,1e-9):.0f}x -- the architecture is decided "
                 f"by the tank, not chosen"),
    }


@dataclass(frozen=True)
class WallLoadState:
    """The membrane stresses in a pressurised tank wall, all of them.

    A pressurised tank is a different structure from the empty cylinder this
    packet was sizing. Internal pressure adds hoop tension -- which is the
    largest membrane stress on the wall by some margin -- and, because a tank has
    end domes, an axial tension of pr/2t that works directly against the
    compression the flight loads apply.
    """

    hoop_pa: float
    axial_pressure_pa: float
    axial_flight_pa: float
    bending_pa: float

    @property
    def net_axial_pa(self) -> float:
        """Compression negative. Pressure pushes this toward tension."""
        return self.axial_pressure_pa - abs(self.axial_flight_pa) - abs(self.bending_pa)

    @property
    def in_compression(self) -> bool:
        """Can this wall buckle at all?

        A shell in net axial tension has no compressive buckling mode to go
        unstable in. This is the question the packet never asked: it sized the
        wall against buckling under a compression the tank may never see.
        """
        return self.net_axial_pa < 0.0

    @property
    def von_mises_pa(self) -> float:
        """Biaxial membrane, no shear: sqrt(s1^2 - s1 s2 + s2^2)."""
        s1, s2 = self.hoop_pa, self.net_axial_pa
        return math.sqrt(s1 * s1 - s1 * s2 + s2 * s2)

    def as_dict(self) -> dict:
        return {"hoop_pa": self.hoop_pa,
                "axial_pressure_pa": self.axial_pressure_pa,
                "axial_flight_pa": self.axial_flight_pa,
                "bending_pa": self.bending_pa,
                "net_axial_pa": self.net_axial_pa,
                "in_compression": self.in_compression,
                "von_mises_pa": self.von_mises_pa}


def wall_load_state(*, pressure_pa: float, radius_m: float, wall_m: float,
                    axial_flight_pa: float = 0.0,
                    bending_pa: float = 0.0) -> WallLoadState:
    """Every membrane stress in a pressurised tank wall at one station.

    Hoop is pr/t and the axial term from the end domes is pr/2t -- exactly half,
    which is the whole reason a pressurised cylinder fails along its length
    rather than around it.

    The result answers a question the structural chain never asked. This packet
    sizes its wall for buckling under combined axial and bending load, and
    thickens it when the margin falls short. A tank at three bar is in net axial
    *tension* at those load levels, and a shell in tension has no compressive
    buckling mode. Whether the repair was necessary depends on a pressure that
    was not being computed.
    """
    if wall_m <= 0:
        raise ValueError("wall thickness must be positive")
    return WallLoadState(
        hoop_pa=pressure_pa * radius_m / wall_m,
        axial_pressure_pa=pressure_pa * radius_m / (2.0 * wall_m),
        axial_flight_pa=float(axial_flight_pa),
        bending_pa=float(bending_pa))


#: Minimum gauge a tank dome is actually built at, m.
#:
#: The membrane relation puts a dome at this vehicle's pressure and radius at
#: 0.11 mm, which is not a thing anyone welds. Real domes are set by handling,
#: weld lands, manhole reinforcement and forming, not by the pressure they hold.
#: Stated as a floor rather than folded in silently, because the membrane number
#: is the derivable one and this is not.
#:
#: Taken from structural_sizing rather than chosen here. This was an independent
#: 1.0 mm for one revision, while the wall sizing next to it used 0.8 mm for
#: "spun/welded shells" -- and the packet's own wall driver reads "minimum
#: gauge", so the two constants were describing the same manufacturing limit on
#: the same vehicle in the same alloy and disagreeing by 25%. A dome is a spun
#: and welded shell. One process, one number, in one place.
try:
    from cadflow.structural_sizing import T_MIN_M as DOME_MIN_GAUGE_M
except Exception:  # noqa: BLE001
    DOME_MIN_GAUGE_M = 0.0008


def dome_mass_kg(*, pressure_pa: float, radius_m: float,
                 density_kg_m3: float, allowable_pa: float,
                 min_gauge_m: float = DOME_MIN_GAUGE_M,
                 domes: int = 2) -> dict:
    """Mass of the tank end closures, which this module had been assuming free.

    ``wall_load_state`` credits the tank wall with pr/2t of axial tension, and
    that credit is what puts the wall in net tension with no compressive
    buckling mode. It is entirely real -- and it exists only because the tank
    has ends. The packet's own mass closure lists "tank domes" among the things
    the geometry does not draw, so the relief was being taken without the part
    that provides it ever being weighed.

    Taking a structural credit and not paying its mass is the overclaim
    direction. Both halves belong in the same place.

    A hemisphere is 2 pi R^2 of surface at t = pR/(2 sigma), so the membrane
    mass is rho pi p R^3 / sigma -- the same relation as the pressurant bottle,
    since a dome *is* half a pressure vessel.
    """
    t_membrane = pressure_pa * radius_m / (2.0 * allowable_pa)
    t = max(t_membrane, float(min_gauge_m))
    area = 2.0 * math.pi * radius_m ** 2
    each = density_kg_m3 * area * t
    return {
        "thickness_membrane_m": t_membrane,
        "thickness_used_m": t,
        "gauge_limited": t > t_membrane + 1e-12,
        "mass_each_kg": each,
        "domes": int(domes),
        "mass_kg": each * int(domes),
    }


def _break_even_gauge(pr, allowance_kg: float) -> float:
    """The dome gauge at which a stage would just afford its own tankage.

    The infeasibility verdict rests on DOME_MIN_GAUGE_M, which is an assumption
    about welding and handling rather than a derived quantity. A finding that
    depends on one constant has to say what that constant would have to be for
    the finding to go away -- otherwise it is reporting the assumption rather
    than the vehicle.

    Dome mass is linear in thickness once gauge-limited, so this inverts
    directly. Returns 0.0 when even a zero-thickness dome would not fit, which
    means the helium and bottle alone exceed the allowance.
    """
    fixed = float(pr.helium_kg) + float(pr.bottle_kg)
    spare = float(allowance_kg) - fixed
    if spare <= 0 or pr.dome_kg <= 0:
        return 0.0
    return DOME_MIN_GAUGE_M * spare / float(pr.dome_kg)


def stage_feasibility(stages, pressurisations) -> list[dict]:
    """Can each stage afford the tankage it needs?

    A structural coefficient is a fraction, so it scales a stage's structure
    with its propellant. Minimum gauge does not scale with anything. Below some
    size the two cross, and the stage's tank ends alone cost more than its whole
    structural allowance -- which is why flown structural coefficients get
    *worse* for small stages rather than staying flat, and why the flown range
    this project compares against spans 0.036 to 0.118 instead of one number.

    This is the check that catches an architecture the mass fractions permit and
    physics does not. Nothing else in the packet can see it: every other
    structural check asks whether a wall carries its load, and this one asks
    whether the stage can pay for the wall at all.
    """
    out = []
    for st, pr in zip(stages, pressurisations):
        allowance = float(st.struct_mass_kg)
        needed = float(pr.total_kg)
        out.append({
            "stage": pr.stage,
            "struct_allowance_kg": allowance,
            "pressurisation_kg": needed,
            "dome_kg": pr.dome_kg,
            "fraction_of_allowance": needed / max(allowance, 1e-9),
            "feasible": needed < allowance,
            "break_even_gauge_m": _break_even_gauge(pr, allowance),
            "note": (
                f"stage {pr.stage} needs {needed:.1f} kg of tankage against a "
                f"{allowance:.1f} kg structural allowance"
                + ("" if needed < allowance else
                   f" -- its tank ends alone are {pr.dome_kg:.1f} kg, so this "
                   f"stage cannot pay for its own pressure vessel before any "
                   f"skin, plumbing or avionics")),
        })
    return out


def wall_for_pressure_m(*, pressure_pa: float, radius_m: float,
                        allowable_pa: float, axial_flight_pa: float = 0.0,
                        bending_pa: float = 0.0, reference_wall_m: float = 0.0008,
                        target_margin: float = 1.0) -> float:
    """Wall thickness at which the membrane state reaches its allowable.

    Internal pressure never sized anything in this project. The wall came from
    axial compression, buckling and minimum gauge, and the pressure case was
    applied afterwards as a check -- so the dominant membrane load on the tank,
    which ``wall_load_state`` reports as three times the flight stress, had no
    influence on the thickness carrying it.

    That produced a margin of 1.010 in packet v40: not a designed margin but an
    arithmetic coincidence, the wall having been sized for a different load and
    the pressure case landing just inside the allowable.

    Every membrane term here scales as 1/t -- hoop pr/t, dome axial pr/2t, and
    the flight stresses which are load over area -- so von Mises scales as 1/t
    exactly and the required thickness follows by proportion rather than by
    iteration.

    ``target_margin`` defaults to 1.0, meaning "just reaches the allowable".
    Callers that want a margin their analysis can actually resolve should pass
    ``cadflow.margin_audit.RESOLVED_MARGIN``: sizing to 1.0 leaves a verdict
    inside the 14.5% this project has measured between element orders, which is
    a pass that cannot be distinguished from a failure.
    """
    if reference_wall_m <= 0 or allowable_pa <= 0:
        raise ValueError("reference wall and allowable must be positive")
    state = wall_load_state(pressure_pa=pressure_pa, radius_m=radius_m,
                            wall_m=reference_wall_m,
                            axial_flight_pa=axial_flight_pa,
                            bending_pa=bending_pa)
    vm = state.von_mises_pa
    if vm <= 0:
        return 0.0
    return reference_wall_m * vm * float(target_margin) / float(allowable_pa)
