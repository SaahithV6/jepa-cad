"""The dry structure, and the second buckling mode nobody was looking for.

The packet named this gap itself. Its pressurisation section concluded that the
tank walls sit in net axial tension and so have no compressive buckling mode --
and then said the interstages are dry throughout, carry the same axial load with
no pressure to relieve them, and are not analysed separately. This closes that.

Two things here are new rather than reused. The Batdorf parameter says whether
the long-cylinder buckling stress that ``shell_buckling`` uses applies at all,
which matters because that module takes no length and so cannot tell a short
shell from a long one. And the column mode -- the whole tube bowing rather than
its wall folding -- is a failure a shell check structurally cannot see, for the
same reason.
"""

import math

import pytest

from cadflow.dry_structure import (
    COLUMN_END_FIXITY, SHORT_SHELL_Z, batdorf_z, bending_mpa_at, check_section,
    check_interstages, euler_buckling_load_n, failures, interstage_loads)


class _Stage:
    def __init__(self, prop, struct):
        self.prop_mass_kg = prop
        self.struct_mass_kg = struct


def _stack():
    """The four-stage vehicle this project designs, from packet v31."""
    return [_Stage(1006.6, 349.5), _Stage(261.7, 90.9),
            _Stage(68.0, 23.6), _Stage(23.9, 8.3)]


def test_euler_load_matches_the_hand_calculation():
    """pi^2 E I / (kL)^2 with I = pi r^3 t, built the long way."""
    L, r, t, E = 2.0, 0.338, 0.0008, 200e9
    I = math.pi * r ** 3 * t
    assert euler_buckling_load_n(L, r, t, E) == pytest.approx(
        math.pi ** 2 * E * I / (COLUMN_END_FIXITY * L) ** 2, rel=1e-12)


def test_the_column_mode_falls_as_the_square_of_length():
    """Which is the whole reason it is invisible to the shell check.

    The shell buckling stress has no length in it at all, so a section twice as
    long reads identically there and is four times weaker here.
    """
    kw = dict(radius_m=0.338, wall_m=0.0008, youngs_pa=200e9)
    short = euler_buckling_load_n(1.0, **kw)
    long_ = euler_buckling_load_n(2.0, **kw)
    assert long_ == pytest.approx(short / 4.0, rel=1e-12)


def test_a_long_enough_tube_fails_as_a_column_not_a_shell():
    """The mode the existing check cannot reach.

    A slender tube bows sideways at a load where its wall is nowhere near local
    instability. If this test could not be written the column mode would be
    decoration; it governs somewhere, and the check has to find it there.
    """
    s = check_section(name="long tube", length_m=30.0, radius_m=0.10,
                      wall_m=0.0004, axial_load_n=20_000.0, youngs_pa=70e9)
    assert s.governs == "column"
    assert s.column_margin < s.shell_margin
    assert any("column mode governs" in n for n in s.notes)


def test_a_short_interstage_fails_as_a_shell():
    """And the governing mode has to come out the other way for a stubby one."""
    s = check_section(name="interstage", length_m=0.21, radius_m=0.338,
                      wall_m=0.0008, axial_load_n=34_400.0, youngs_pa=200e9)
    assert s.governs == "shell"


def test_batdorf_says_when_the_long_cylinder_stress_applies():
    """Length alone does not make a shell long, and neither does thickness.

    The first version of this test asserted that two metres is structurally
    short at a 20 mm wall. It is not -- Z comes out at 564, well above the
    threshold -- so the assertion failed and the fixture was what was wrong. At
    this radius nothing of tank length is short whatever its wall; what crosses
    the threshold on this vehicle is the *stubby* sections.

    The scaling is the real content and it is exact: Z goes as L^2 and as 1/t.
    """
    assert batdorf_z(2.0, 0.338, 0.0008) > SHORT_SHELL_Z
    assert batdorf_z(0.15, 0.338, 0.0008) < SHORT_SHELL_Z

    # exact inverse scaling in thickness
    assert (batdorf_z(2.0, 0.338, 0.0008) / batdorf_z(2.0, 0.338, 0.020)
            == pytest.approx(0.020 / 0.0008, rel=1e-12))
    # exact square scaling in length
    assert (batdorf_z(4.0, 0.338, 0.0008) / batdorf_z(2.0, 0.338, 0.0008)
            == pytest.approx(4.0, rel=1e-12))


def test_a_short_shell_says_its_margin_is_conservative():
    """An unstated conservatism gets spent twice.

    Below the Batdorf threshold the end restraint raises the real buckling
    stress above the value used, so the margin is understated. A reader who does
    not know that will treat the number as tight when it is not.
    """
    s = check_section(name="stub", length_m=0.05, radius_m=0.338,
                      wall_m=0.0008, axial_load_n=10_000.0, youngs_pa=200e9)
    assert s.batdorf_z < SHORT_SHELL_Z
    assert any("conservative" in n for n in s.notes)


def test_interstage_load_is_taken_at_the_peak_of_the_stage_below():
    """Not at liftoff, which is a factor of several.

    Thrust is held while mass falls, so a stage's acceleration climbs through
    its burn. This vehicle's first stage ends near 7 g. An interstage sized at
    liftoff acceleration would be under-designed at exactly the moment it
    carries the most.
    """
    loads = interstage_loads(_stack(), 25.0, [7.0, 4.3, 2.9, 2.7])
    assert len(loads) == 3
    first = loads[0]
    # everything above interstage 1/2: stages 2, 3, 4 and the payload
    above = 25.0 + 261.7 + 90.9 + 68.0 + 23.6 + 23.9 + 8.3
    assert first["supported_kg"] == pytest.approx(above)
    assert first["axial_g"] == 7.0
    assert first["load_n"] == pytest.approx(above * 9.80665 * 7.0)


def test_each_interstage_carries_less_than_the_one_below():
    """Mass above falls with every stage, and so must the load."""
    loads = interstage_loads(_stack(), 25.0, [7.0, 4.3, 2.9, 2.7])
    got = [r["load_n"] for r in loads]
    assert got == sorted(got, reverse=True)


def test_the_real_interstages_are_checked_and_pass():
    """The outstanding item, closed.

    They pass, comfortably. That is a result and not a formality: it was
    unknown, the packet said so, and the number that settles it is a margin of
    about five on the most loaded one.
    """
    secs = check_interstages(_stack(), 25.0, [7.0, 4.3, 2.9, 2.7],
                             radius_m=0.338, wall_m=0.0008, youngs_pa=200e9,
                             lengths_m=[0.2096, 0.1928, 0.1774])
    assert len(secs) == 3
    assert not failures(secs)
    assert secs[0].margin == pytest.approx(4.98, abs=0.3)


def test_an_underbuilt_interstage_is_caught():
    """The check must be able to fail, or it establishes nothing.

    Same vehicle at a fifth of the wall. If this still passed, the margin above
    would be reporting the absence of a check rather than the presence of
    structure.
    """
    secs = check_interstages(_stack(), 25.0, [7.0, 4.3, 2.9, 2.7],
                             radius_m=0.338, wall_m=0.00015, youngs_pa=200e9,
                             lengths_m=[0.2096, 0.1928, 0.1774])
    assert failures(secs)


def test_bending_is_read_from_the_aft_end_not_the_nose():
    """The convention that would have returned a plausible wrong number.

    ``section_extents`` puts stage 1 at zero and the payload at the top, so
    stations run from the aft end. The comment on PointLoad said "from the nose
    tip, positive aft" -- the exact opposite -- and reading a moment by it would
    return the value from the mirror-image station.
    """
    class L:
        stations_m = [0.0, 1.0, 2.0, 3.0, 4.0]
        moment_nm = [0.0, 8000.0, 4000.0, 1000.0, 0.0]

    near_aft = bending_mpa_at(L, 1.0, 0.338, 0.0008)
    near_nose = bending_mpa_at(L, 3.0, 0.338, 0.0008)
    assert near_aft > near_nose, "station 1.0 is near the aft end, where the moment is"


def test_bending_interpolates_rather_than_snapping():
    """The moment curve kinks at every point load.

    Nearest-neighbour on 401 stations moves a section up to 6 mm along a curve
    that is steepest exactly where the interstages sit.
    """
    class L:
        stations_m = [0.0, 1.0, 2.0]
        moment_nm = [0.0, 1000.0, 0.0]

    half = bending_mpa_at(L, 0.5, 0.338, 0.0008)
    full = bending_mpa_at(L, 1.0, 0.338, 0.0008)
    assert half == pytest.approx(0.5 * full, rel=1e-9)


def test_bending_outside_the_vehicle_clamps_and_does_not_extrapolate():
    """A station off the end is a caller error, not a licence to invent."""
    class L:
        stations_m = [0.0, 1.0, 2.0]
        moment_nm = [5.0, 1000.0, 7.0]

    assert bending_mpa_at(L, -3.0, 0.338, 0.0008) == pytest.approx(
        bending_mpa_at(L, 0.0, 0.338, 0.0008))
    assert bending_mpa_at(L, 99.0, 0.338, 0.0008) == pytest.approx(
        bending_mpa_at(L, 2.0, 0.338, 0.0008))


def test_bending_makes_a_section_worse_never_better():
    """Adding a load cannot raise a margin.

    Sounds trivial; it is the sign convention, and getting it backwards would
    make every bending-loaded section read safer than it is.
    """
    kw = dict(name="x", length_m=0.21, radius_m=0.338, wall_m=0.0008,
              axial_load_n=34_400.0, youngs_pa=200e9)
    assert check_section(bending_mpa=40.0, **kw).shell_margin < \
        check_section(bending_mpa=0.0, **kw).shell_margin


def test_degenerate_geometry_is_refused():
    with pytest.raises(ValueError):
        batdorf_z(1.0, 0.0, 0.0008)
    with pytest.raises(ValueError):
        euler_buckling_load_n(0.0, 0.338, 0.0008, 200e9)
