"""Does the packet agree with itself?

Five defects in this project shared one shape: one part of the program knew
something and another reported otherwise. The structural coefficient field read
a module default while the stack was built at 0.2613. The coupon caveat lived in
prose while the verdict said the components passed. The allowable was computed
at room temperature while the thermal section said the skin reaches 863 K.
Component masses were weighed in aluminium on an Inconel vehicle, understating
every one by a factor of three.

All five were invisible to five hundred tests, because each individual model was
correct. Tests check models. Nothing was checking whether the models agreed with
each other, and that is where every one of those defects lived.

These checks compare numbers the packet already reports against each other. The
sharpest is density: a component's mass divided by its volume has to be the
alloy the design selected, and nothing else. That one check would have caught
the three-fold mass error outright, with no inspection at all.

Written as a function rather than inline in the report so it can be tested
against a deliberately inconsistent packet. A self-check that has only ever been
run on correct data is not known to work -- it is known to be quiet, which is
the same thing a broken check does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cross:
    """One cross-check, carrying the category it belongs to.

    ``kind`` exists so the report can say what it checked without matching on
    check names. The description of this audit has gone stale three times: once
    written from memory, once derived by hardcoded string patterns that did not
    recognise checks added later. A category the check declares itself cannot
    lag, because adding a check means naming its kind.
    """

    check: str
    ok: bool
    got: float
    want: float
    detail: str = ""
    kind: str = "other"

    def as_dict(self) -> dict:
        return {"check": self.check, "ok": self.ok, "got": self.got,
                "want": self.want, "detail": self.detail, "kind": self.kind}


#: Human-readable descriptions, keyed by the kind a check declares.
KIND_DESCRIPTIONS = {
    "mass": "gross mass against the stack it lists",
    "coefficient": "one structural coefficient across every stage",
    "density": "each component's mass over its volume against the density of "
               "the alloy the design selected",
    "geometry": "every in-line part meeting its neighbour without overlap or gap",
    "architecture": "the stage count against the split it was built from",
    "mission": "the apogee error against the apogee it is computed from",
    "stability": "the normal force slope, centre of pressure and static margin "
                 "against the stations they are measured between",
}


def describe(crosses) -> str:
    """List the kinds of check that actually ran, in a stable order."""
    seen = []
    for c in crosses:
        if c.kind not in seen and c.kind in KIND_DESCRIPTIONS:
            seen.append(c.kind)
    return "; ".join(KIND_DESCRIPTIONS[k] for k in seen)


def _agree(label: str, got: float, want: float, tol: float,
           detail: str = "", kind: str = "other") -> Cross:
    ok = abs(float(got) - float(want)) <= tol * max(abs(float(want)), 1e-9)
    return Cross(label, ok, float(got), float(want), detail, kind)


def audit(*, gross_kg: float, stack, payload_kg: float,
          flight_vehicle_mass_kg: float, components, skin_density_kg_m3: float,
          skin_material: str) -> list[Cross]:
    """Cross-check the numbers a packet reports against one another.

    ``stack`` is the planner's stage list and ``components`` the analysed part
    records, each optionally carrying ``mass_properties`` with ``mass_kg`` and
    ``volume_m3``. Anything missing is skipped rather than guessed: a check that
    invents an input to stay green is worse than one that does not run.
    """
    out: list[Cross] = []

    stage_total = sum(float(s.prop_mass_kg) + float(s.struct_mass_kg)
                      for s in stack) + float(payload_kg)
    out.append(_agree("gross mass equals the stack it lists",
                      gross_kg, stage_total, 1e-6, kind="mass"))
    out.append(_agree("flight vehicle mass equals gross",
                      flight_vehicle_mass_kg, gross_kg, 1e-6, kind="mass"))

    struct = sum(float(s.struct_mass_kg) for s in stack)
    prop = sum(float(s.prop_mass_kg) for s in stack)
    if struct + prop > 0:
        # Every stage should share one structural coefficient; a stage that
        # drifted from the others would mean the repair loop updated some and
        # not the rest.
        coeffs = [float(s.struct_mass_kg)
                  / max(1e-9, float(s.struct_mass_kg) + float(s.prop_mass_kg))
                  for s in stack]
        # A tenth of a percent, not a millionth.
        #
        # 1e-6 looked rigorous and was wrong: stage masses are reported to two
        # decimals, so coefficients recovered from them differ in the fifth
        # even when the repair reached every stage. The failure this check
        # exists for is a stage left at 0.14 where the others are 0.2613 -- an
        # 87% gap -- so a tolerance that survives rounding still catches it by
        # three orders of magnitude. A check that cries wolf on its own inputs
        # is one that gets switched off.
        out.append(_agree("every stage shares one structural coefficient",
                          max(coeffs), min(coeffs), 1e-3,
                          "a stage out of step means the repair reached some "
                          "and not others", kind="coefficient"))

    for rec in components:
        mp = (rec.get("mass_properties") or {}) if isinstance(rec, dict) else {}
        vol = float(mp.get("volume_m3") or 0.0)
        mass = float(mp.get("mass_kg") or 0.0)
        if vol > 0 and mass > 0:
            name = rec.get("name", "component") if isinstance(rec, dict) else "component"
            out.append(_agree(
                f"{name} weighed in {skin_material}", mass / vol,
                float(skin_density_kg_m3), 1e-3,
                f"{mass / vol:.0f} kg/m3 against {float(skin_density_kg_m3):.0f} "
                f"for the alloy the design selected", kind="density"))
    return out


def derived_quantities(*, stages: int, split, achieved_km: float,
                       target_km: float, error_pct: float, stability: dict,
                       cg_z_m: float, radius_m: float) -> list[Cross]:
    """Values the packet reports that are computable from other values it reports.

    Every one of these is currently correct, which is the same state the six
    drifted quantities were in until something changed one side and not the
    other. A cross-check earns its place by being cheap and by covering a
    dimension the audit otherwise cannot see -- here the mission numbers, the
    architecture counts and the stability derivatives, none of which the mass
    and geometry checks touch.
    """
    out: list[Cross] = []
    out.append(_agree("stage count matches the split it was built from",
                      float(stages), float(len(split)), 1e-9,
                      kind="architecture"))

    if target_km > 0:
        recomputed = abs(float(achieved_km) - float(target_km)) / float(target_km) * 100.0
        out.append(_agree("apogee error matches the apogee it is computed from",
                          float(error_pct), recomputed, 1e-3,
                          f"{achieved_km:.1f} km against {target_km:.0f} requested",
                          kind="mission"))

    nose = float(stability.get("cna_nose") or 0.0)
    fins = float(stability.get("cna_fins") or 0.0)
    total = float(stability.get("cna_total") or 0.0)
    if total > 0:
        out.append(_agree("normal force slope is the sum of its parts",
                          nose + fins, total, 1e-6, kind="stability"))
        cp_n = stability.get("cp_nose_z_m")
        cp_f = stability.get("cp_fins_z_m")
        cp = stability.get("cp_z_m")
        if None not in (cp_n, cp_f, cp) and nose + fins > 0:
            # Centre of pressure is the slope-weighted mean of the two
            # contributors; a cp that drifts from that is a cp belonging to
            # different fins than the ones reported beside it.
            weighted = (nose * float(cp_n) + fins * float(cp_f)) / (nose + fins)
            out.append(_agree("centre of pressure is the weighted mean of its "
                              "contributors", weighted, float(cp), 1e-4, kind="stability"))

    margin = stability.get("static_margin_cal")
    cp = stability.get("cp_z_m")
    if margin is not None and cp is not None and radius_m > 0:
        recomputed = (float(cg_z_m) - float(cp)) / (2.0 * float(radius_m))
        out.append(_agree("static margin matches the stations it is measured "
                          "between", float(margin), recomputed, 1e-3,
                          "the fin trade moves this; a stale margin means the "
                          "trade reached the fins and not the report", kind="stability"))
    return out


def stack_interference(parts) -> list[Cross]:
    """Do the in-line parts tile the vehicle without overlapping or gapping?

    ``parts`` is a sequence of (name, station_m, length_m). Fins are excluded:
    they attach to the outside of a tank and share its axial span by design, so
    including them would report every finned vehicle as broken.

    This exists because the assembly table showed the nozzle at 0.000 m
    overlapping 324 mm of the stage 1 tank. The geometry was correct -- the
    solid is translated to negative z, hanging off the aft end -- and the
    station recorded in the report was not. Nothing compared the two, so a
    report that described an impossible vehicle read exactly like one that did
    not.
    """
    inline = [(n, float(z), float(l)) for n, z, l in parts
              if "fin" not in str(n).lower()]
    inline.sort(key=lambda r: r[1])
    out: list[Cross] = []
    for (n1, z1, l1), (n2, z2, _l2) in zip(inline, inline[1:]):
        end = z1 + l1
        # A tolerance in metres rather than relative: these are stations on a
        # vehicle a few metres long, and a millimetre is the right scale for
        # "these parts meet" regardless of how far up the stack they sit.
        out.append(Cross(f"{n1} meets {n2} without overlap or gap",
                         abs(end - z2) <= 1e-3, end, z2,
                         f"{n1} ends at {end:.3f} m, {n2} starts at {z2:.3f} m",
                         kind="geometry"))
    return out


def failures(crosses) -> list[Cross]:
    return [c for c in crosses if not c.ok]
