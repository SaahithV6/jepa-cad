"""Tests for real raw-data parsers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.parsers import ParseError, parse_raw_file
from data.prepare_data import process_sample


def _write_ascii_stl(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "solid box",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 0 0",
                "      vertex 1 1 0",
                "    endloop",
                "  endfacet",
                "  facet normal 0 0 1",
                "    outer loop",
                "      vertex 0 0 0",
                "      vertex 1 1 0",
                "      vertex 0 1 0",
                "    endloop",
                "  endfacet",
                "endsolid box",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_vtk(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# vtk DataFile Version 3.0",
                "test",
                "ASCII",
                "DATASET POLYDATA",
                "POINTS 4 float",
                "0 0 0",
                "1 0 0",
                "1 1 0",
                "0 1 0",
                "POINT_DATA 4",
                "SCALARS pressure float 1",
                "LOOKUP_TABLE default",
                "1.0 2.0 3.0 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_stl_and_resample(tmp_path: Path) -> None:
    stl = tmp_path / "part.stl"
    _write_ascii_stl(stl)
    sample = parse_raw_file(stl, num_points=16, num_fields=3)
    assert sample.points.shape == (16, 3)
    assert sample.fields.shape == (16, 3)
    assert sample.source_format == "stl"


def test_parse_vtk_scalars(tmp_path: Path) -> None:
    vtk = tmp_path / "field.vtk"
    _write_vtk(vtk)
    sample = parse_raw_file(vtk, num_points=4, num_fields=3)
    assert sample.points.shape[0] == 4
    assert sample.fields.shape == (4, 3)
    assert float(sample.fields[:, 0].mean()) > 0


def test_unsupported_format_raises_without_fallback(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_raw_file(bad, allow_synthetic_fallback=False)


def test_prepare_data_process_sample_stl(tmp_path: Path) -> None:
    stl = tmp_path / "part.stl"
    _write_ascii_stl(stl)
    shard = process_sample(stl, num_points=32, num_fields=3)
    assert shard["points"].shape == (32, 3)
    assert shard["fields"].shape == (32, 3)
    assert shard["max_stress"].shape == ()
