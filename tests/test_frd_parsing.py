"""Parsing CalculiX FRD results, which silently threw most of them away.

CalculiX writes FRD in fixed-width columns: " -1", a 10-character node number,
then six 12-character values. A negative value fills its whole field, so it
abuts the previous one with no space -- "2.44293E+08-1.04280E+07" is two
numbers, and str.split() returns six tokens where eight are needed.

The parser caught that and continued, so it skipped every line containing a
negative stress component. On a 14,013-node result it read 100, and those
hundred were exactly the lines where all six components happened to be positive.
p99 of 100 values is the 99th -- the maximum -- which is why p99 and peak came
out identical in every result table until this was found.

These tests use a synthetic FRD built to contain the exact pathology.
"""

import math

import pytest

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from plan_and_verify import frd_stress_percentiles  # noqa: E402


def _row(node, comps):
    """One FRD stress record, written the way CalculiX writes it."""
    return " -1" + f"{node:>10}" + "".join(f"{c:>12.5E}" for c in comps)


def _write_frd(path, rows):
    lines = [
        "    1C",
        "    2C",
        " -4  STRESS      6    1",
        " -5  SXX         1    4    1    1",
    ]
    lines.extend(rows)
    lines.append(" -3")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _von_mises(sxx, syy, szz, sxy, syz, szx):
    return math.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    ) / 1e6


def test_negative_components_do_not_lose_the_row(tmp_path):
    """The bug itself. Every one of these rows has a negative component."""
    case = tmp_path / "solver"
    case.mkdir()
    rows = [
        _row(i, [2.4e7, 1.05e8, 2.44e8, -1.04e7, 1.23e6, -2.52e7])
        for i in range(1, 51)
    ]
    _write_frd(case / "case.frd", rows)
    got = frd_stress_percentiles(tmp_path)
    assert got is not None
    assert got["n"] == 50, "rows with negative components were dropped"


def test_values_match_the_fixed_width_fields(tmp_path):
    case = tmp_path / "solver"
    case.mkdir()
    comps = (2.4e7, 1.05e8, 2.44e8, -1.04e7, 1.23e6, -2.52e7)
    _write_frd(case / "case.frd", [_row(1, comps)])
    got = frd_stress_percentiles(tmp_path)
    assert got["max"] == pytest.approx(_von_mises(*comps), rel=1e-6)


def test_mixed_signs_are_all_read(tmp_path):
    """A realistic result has both, and the split() parser kept only one kind."""
    case = tmp_path / "solver"
    case.mkdir()
    rows = []
    for i in range(1, 101):
        sign = -1.0 if i % 2 else 1.0
        rows.append(_row(i, [sign * 1.0e8, 2.0e7, 3.0e7, sign * 4.0e6,
                             5.0e6, sign * 6.0e6]))
    _write_frd(case / "case.frd", rows)
    got = frd_stress_percentiles(tmp_path)
    assert got["n"] == 100


def test_percentiles_separate_from_the_maximum(tmp_path):
    """What the metric is for.

    One hot node among many must move the maximum and leave p99 alone. With only
    100 rows surviving, p99 *was* the maximum and could never do that.
    """
    case = tmp_path / "solver"
    case.mkdir()
    rows = [_row(i, [5.0e7, -1.0e7, 2.0e7, -3.0e6, 1.0e6, -2.0e6])
            for i in range(1, 1000)]
    rows.append(_row(1000, [3.0e9, -1.0e7, 2.0e7, -3.0e6, 1.0e6, -2.0e6]))
    _write_frd(case / "case.frd", rows)
    got = frd_stress_percentiles(tmp_path)
    assert got["n"] == 1000
    assert got["max"] > 10.0 * got["p99"], (got["p99"], got["max"])
    assert got["median"] == pytest.approx(got["p99"], rel=0.01)


def test_no_frd_returns_nothing(tmp_path):
    assert frd_stress_percentiles(tmp_path) is None


def test_newest_frd_wins(tmp_path):
    """The thickening loop leaves one FRD per iteration in the same tree."""
    import os
    import time

    for name, level in (("old", 1.0e8), ("new", 5.0e7)):
        d = tmp_path / name
        d.mkdir()
        _write_frd(d / "case.frd",
                   [_row(i, [level, -1.0e7, 2.0e7, -3.0e6, 1.0e6, -2.0e6])
                    for i in range(1, 20)])
    old = tmp_path / "old" / "case.frd"
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    got = frd_stress_percentiles(tmp_path)
    # the newer, lower-stress result is the one that counts
    assert got["median"] < 60.0, got
