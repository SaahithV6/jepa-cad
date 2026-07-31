"""Regression: OpenFOAM segfaults on binary STLs with solid-padded headers."""
from __future__ import annotations

import struct
from pathlib import Path

from cadflow.rocket_cfd_bodyfit import _is_binary_stl, write_stl_meters


def _make_binary_stl_with_solid_header(ntris: int = 2) -> bytes:
    header = b"solid FakeName".ljust(80, b" ")
    body = bytearray(header)
    body.extend(struct.pack("<I", ntris))
    for i in range(ntris):
        # normal + 3 verts + attr
        floats = [0.0, 0.0, 1.0] + [float(i), 0.0, 0.0, float(i) + 1, 1.0, 0.0, float(i), 1.0, 0.0]
        body.extend(struct.pack("<12fH", *floats, 0))
    return bytes(body)


def test_is_binary_stl_detects_solid_padded_header():
    data = _make_binary_stl_with_solid_header()
    assert data[:5] == b"solid"
    assert _is_binary_stl(data)


def test_write_stl_meters_rewrites_solid_binary_header(tmp_path: Path):
    src = tmp_path / "in.STL"
    dst = tmp_path / "body.stl"
    src.write_bytes(_make_binary_stl_with_solid_header(3))
    write_stl_meters(src, dst, 1.0)
    out = dst.read_bytes()
    assert _is_binary_stl(out)
    assert not out.lower().startswith(b"solid")
    assert struct.unpack_from("<I", out, 80)[0] == 3


def test_write_stl_meters_scales_binary(tmp_path: Path):
    src = tmp_path / "in.stl"
    dst = tmp_path / "out.stl"
    src.write_bytes(_make_binary_stl_with_solid_header(1))
    write_stl_meters(src, dst, 0.001)
    out = dst.read_bytes()
    vals = struct.unpack_from("<12fH", out, 84)
    # first vertex x was 0.0 * scale
    assert abs(vals[3] - 0.0) < 1e-9
    assert abs(vals[6] - 0.001) < 1e-9
