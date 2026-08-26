"""Does the vehicle resemble anything that has flown?

Every other check in this project is internal. The solvers agree with closed
form, the mass budget closes, the components clear their allowables -- and all
of it could hold for a vehicle nobody could build. The structural coefficient is
where that would show first: it decides how much of a stage is tankage and how
much is propellant, and unlike the stresses it is asserted rather than solved.

Ten flown stages put the real range at 0.036 to 0.118. This project's planner
constant is 0.140 and its repair loop reached 0.261. Those are outside, and the
tests here pin both that fact and the two caveats that make it interpretable
rather than damning: the reference figures are secondary-source, and structural
coefficient is size-dependent in a direction that excuses a small vehicle.
"""

import json
from pathlib import Path

import pytest

from cadflow.flown_envelope import (
    SMALLEST_REFERENCE_WET_KG, check, flown_coefficients)

DATA = Path(__file__).resolve().parents[1] / "data/flown_stages/flown_stage_masses.json"


def test_the_reference_set_spans_the_regimes_that_matter():
    """Hydrogen and kerosene, boosters and upper stages, huge and small.

    A reference set of only large kerosene boosters would make any upper stage
    look anomalous. The spread is the point: hydrogen stages are structurally
    heavy because the propellant is not dense, and small stages are heavy
    because tank mass follows area while propellant follows volume.
    """
    coeffs = flown_coefficients()
    assert len(coeffs) >= 8
    values = [c for _lbl, c, _conf in coeffs]
    assert min(values) < 0.05, "no lightweight stage in the reference set"
    assert max(values) > 0.11, "no structurally heavy stage in the reference set"
    labels = " ".join(lbl for lbl, _c, _conf in coeffs)
    assert "Saturn V" in labels and "Electron" in labels


def test_the_project_constant_sits_above_every_flown_stage():
    """0.140 against a flown maximum of 0.118.

    Not necessarily wrong -- it is conservative, and this project designs
    vehicles far smaller than anything in the set -- but it must not be
    reported as though it were ordinary.
    """
    from cadflow.planner import STRUCT_COEFF

    v = check(STRUCT_COEFF, stage_wet_kg=1806.0)
    assert not v.inside
    assert STRUCT_COEFF > v.flown_max
    assert v.extrapolating


def test_a_small_vehicle_is_told_it_is_being_extrapolated():
    """The size caveat has to travel with the verdict.

    Reporting "121% above flown hardware" for a 1.8 tonne vehicle, against a
    reference set whose smallest member is 10 tonnes, states a percentage that
    reads like a defect when it is mostly a scaling law.
    """
    small = check(0.2613, stage_wet_kg=1806.0)
    large = check(0.2613, stage_wet_kg=400_000.0)
    assert small.extrapolating and not large.extrapolating
    assert "extrapolation" in small.note
    assert "does cover" in large.note, (
        "at a size the set covers, the same coefficient must be called out")


def test_a_structure_lighter_than_anything_flown_is_flagged_too():
    """The envelope has a floor as well as a ceiling.

    An optimiser that drives structural mass down until the mission closes will
    happily produce a stage lighter than any ever built, and that is the
    failure mode that produces an impressive packet describing an impossible
    vehicle.
    """
    v = check(0.02, stage_wet_kg=50_000.0)
    assert not v.inside
    assert "lighter than flight-proven" in v.note


def test_a_realistic_coefficient_passes():
    """The check must be capable of saying yes.

    A gate that never passes is not a gate. 0.08 is close to the flown median
    and has to read as ordinary.
    """
    v = check(0.08, stage_wet_kg=200_000.0)
    assert v.inside
    assert not v.extrapolating
    assert "Within the flown range" in v.note


def test_the_provenance_warning_is_carried_into_every_verdict():
    """These are not primary sources and the verdict must never imply they are.

    Manufacturers do not publish stage mass statements for most of these
    vehicles. Presenting a secondary-source comparison as validation against
    flight data would be the same category of error as calling a typical
    material strength an A-basis allowable.
    """
    payload = json.loads(DATA.read_text())
    assert "SECONDARY-SOURCE" in payload["provenance_warning"]
    assert any(s.get("confidence") == "low" for s in payload["stages"])
    for coeff in (0.05, 0.08, 0.14, 0.30):
        assert "secondary-source" in check(coeff).note


def test_masses_are_physical():
    """A negative or zero mass would silently produce a meaningless ratio."""
    payload = json.loads(DATA.read_text())
    for s in payload["stages"]:
        assert s["dry_kg"] > 0 and s["propellant_kg"] > 0, s
        assert s["dry_kg"] < s["propellant_kg"], (
            f"{s['stage']} is more structure than propellant", s)


def test_the_smallest_reference_matches_the_data():
    """The extrapolation threshold must track the set, not drift from it.

    If a smaller stage is added later and this constant is not updated, the
    check would claim coverage it does not have.
    """
    payload = json.loads(DATA.read_text())
    wets = [s["dry_kg"] + s["propellant_kg"] for s in payload["stages"]]
    assert min(wets) == pytest.approx(SMALLEST_REFERENCE_WET_KG, rel=0.05)
