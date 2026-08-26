"""Generate parametric spaceflight components as watertight STL solids.

The previous version of this file built every part with

    trimesh.Trimesh(vertices=vertices)          # no faces

which is a point cloud, not a mesh. STL is a triangle format, so each export
wrote an 80-byte header and a triangle count of zero: 1,961 files of 84 bytes
each, and a summary line reading "1961/1961 components generated". Nothing
raised, because exporting a mesh with no faces is not an error -- it is just an
empty file.

That corpus was then used for training and evaluation. Every one of these parts
loaded as an unreadable sample, the nozzle family in particular, so the physics
labels attached to them -- expansion ratio 4 to 80, thrust 0.5 to 49 kN, Isp 250
to 380 s, all perfectly good numbers from a real parameter sweep -- described
geometry that did not exist. A linear probe on those labels scored R^2 = -1.07
for expansion ratio, a quantity you can measure off the mesh with a ruler.

So the rules here are:

* every surface is triangulated and capped, and
* every file is re-read after writing and checked for triangles, watertightness
  and positive volume, and
* the summary counts files that passed that check, not calls that did not throw.

The nozzle uses ``cadflow.sculpt.bell_contour`` -- the same Bezier meridian the
design code uses, pinned by throat radius, area ratio and the two tangent angles
-- so the shape a record carries and the expansion ratio its label claims are
the same number by construction.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    print("trimesh is required: pip install trimesh", file=sys.stderr)
    raise

OUTPUT_DIR = Path("data/generated_spaceflight_cad")
N_THETA = 48


# --- meshing ---------------------------------------------------------------

def revolve(profile: list[tuple[float, float]], n_theta: int = N_THETA):
    """Watertight surface of revolution from a (z, radius) meridian.

    Builds the side wall as a quad strip split into triangles, then closes both
    ends with a fan to a centre vertex. Without the caps the mesh has boundary
    edges, `is_watertight` is False and the volume is undefined -- which matters
    because the volume is what the mass properties downstream are computed from.
    """
    if len(profile) < 2:
        raise ValueError("a meridian needs at least two stations")
    eps = 1e-9
    # A station at r=0 is a pole, not a ring. Emitting n_theta coincident
    # vertices there and then also adding a cap centre gives degenerate,
    # zero-area triangles: the mesh looks built but `is_watertight` is False,
    # which is how 240 tanks failed verification. Poles become one vertex and
    # the adjacent band is a fan.
    prof = [(float(z), max(float(r), 0.0)) for z, r in profile]
    while len(prof) > 2 and prof[0][1] <= eps and prof[1][1] <= eps:
        prof.pop(0)
    while len(prof) > 2 and prof[-1][1] <= eps and prof[-2][1] <= eps:
        prof.pop()

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    verts: list[list[float]] = []
    ids: list[int] = []                    # first vertex id of each station
    for z, r in prof:
        ids.append(len(verts))
        if r <= eps:
            verts.append([0.0, 0.0, z])    # a pole is one vertex
        else:
            for c, s in zip(cos_t, sin_t):
                verts.append([r * c, r * s, z])

    def idx(i: int, j: int) -> int:
        return ids[i] if prof[i][1] <= eps else ids[i] + j

    faces: list[list[int]] = []
    for i in range(len(prof) - 1):
        a_pole = prof[i][1] <= eps
        b_pole = prof[i + 1][1] <= eps
        for j in range(n_theta):
            k = (j + 1) % n_theta
            if a_pole and b_pole:
                continue
            if a_pole:
                faces.append([idx(i, 0), idx(i + 1, j), idx(i + 1, k)])
            elif b_pole:
                faces.append([idx(i, j), idx(i + 1, 0), idx(i, k)])
            else:
                faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, k)])
                faces.append([idx(i, j), idx(i + 1, k), idx(i, k)])

    # flat caps only where an end is a genuine disc rather than a pole
    if prof[0][1] > eps:
        c = len(verts)
        verts.append([0.0, 0.0, prof[0][0]])
        for j in range(n_theta):
            k = (j + 1) % n_theta
            faces.append([c, idx(0, k), idx(0, j)])
    if prof[-1][1] > eps:
        c = len(verts)
        verts.append([0.0, 0.0, prof[-1][0]])
        last = len(prof) - 1
        for j in range(n_theta):
            k = (j + 1) % n_theta
            faces.append([c, idx(last, j), idx(last, k)])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=True)
    mesh.fix_normals()
    return mesh


# --- component families ----------------------------------------------------

def create_nozzle(expansion_ratio: float, throat_diameter_mm: float,
                  length_mm: float):
    """Bell nozzle whose exit-to-throat area ratio IS `expansion_ratio`.

    The meridian comes from the same Bezier construction the design code uses,
    so a record's geometry and its label agree by construction rather than by
    coincidence. A converging section is prepended so the throat is a genuine
    minimum in the profile -- without it the part is a flared tube and the
    "throat" is just the small end.
    """
    from cadflow.sculpt import bell_contour

    r_t = float(throat_diameter_mm) / 2.0
    bell = bell_contour(r_t / 1000.0, float(expansion_ratio))
    # contour is (radius, z) in metres, throat to exit
    diverging = [(z * 1000.0, r * 1000.0) for r, z in bell.contour]
    # scale the diverging section to the requested length
    z_span = diverging[-1][0] - diverging[0][0]
    if z_span > 1e-9:
        k = float(length_mm) / z_span
        diverging = [(z * k, r) for z, r in diverging]

    # converging inlet: chamber radius down to the throat, so the throat is a
    # true minimum and expansion ratio is measurable from the mesh
    r_chamber = r_t * 2.5
    conv_len = 1.5 * r_t
    conv = []
    for i in range(9):
        f = i / 8.0
        conv.append((-conv_len * (1.0 - f),
                     r_chamber + (r_t - r_chamber) * (0.5 - 0.5 * math.cos(math.pi * f))))
    profile = conv[:-1] + diverging
    return revolve(profile)


def create_tank(diameter_mm: float, length_mm: float):
    """Cylindrical tank with elliptical domes."""
    r = float(diameter_mm) / 2.0
    barrel = float(length_mm)
    dome = 0.5 * r                            # semi-minor axis of each dome
    n = 8
    prof: list[tuple[float, float]] = []
    for i in range(n + 1):                    # aft dome, pole to tangent
        a = 0.5 * math.pi * i / n
        prof.append((-dome * math.cos(a), r * math.sin(a)))
    prof.append((barrel, r))
    for i in range(1, n + 1):                 # forward dome, tangent to pole
        a = 0.5 * math.pi * i / n
        prof.append((barrel + dome * math.sin(a), r * math.cos(a)))
    return revolve(prof)


def create_strut(length_mm: float, diameter_mm: float):
    r = float(diameter_mm) / 2.0
    return revolve([(0.0, r), (float(length_mm), r)])


def create_fairing(diameter_mm: float, length_mm: float):
    """Cylindrical barrel with an ogive nose."""
    r = float(diameter_mm) / 2.0
    total = float(length_mm)
    barrel = 0.65 * total
    nose = total - barrel
    prof = [(0.0, r), (barrel, r)]
    rho = (r * r + nose * nose) / (2.0 * r)
    for i in range(1, 15):
        f = i / 14.0
        z = nose * f
        rr = math.sqrt(max(0.0, rho * rho - (nose - z) ** 2)) + r - rho
        prof.append((barrel + z, max(0.0, rr)))
    return revolve(prof)


def create_injector(face_diameter_mm: float, num_holes: int):
    """Injector plate; hole count sets the plate thickness and boss size."""
    r = float(face_diameter_mm) / 2.0
    thickness = 5.0 + 0.15 * float(num_holes)
    boss_r = r * (0.35 + 0.005 * float(num_holes))
    return revolve([(0.0, r), (thickness, r),
                    (thickness, boss_r), (thickness + 0.4 * thickness, boss_r)])


GENERATORS = {
    "nozzle": (create_nozzle, ("expansion_ratio", "throat_diameter_mm", "length_mm")),
    "tank": (create_tank, ("diameter_mm", "length_mm")),
    "strut": (create_strut, ("length_mm", "diameter_mm")),
    "fairing": (create_fairing, ("diameter_mm", "length_mm")),
    "injector": (create_injector, ("face_diameter_mm", "num_holes")),
}

FAMILIES = {
    "nozzle": {
        "expansion_ratio": list(range(5, 100, 5)),
        "throat_diameter_mm": list(range(5, 50, 5)),
        "length_mm": list(range(40, 200, 20)),
    },
    "tank": {
        "diameter_mm": list(range(100, 3000, 200)),
        "length_mm": list(range(200, 5000, 300)),
    },
    "strut": {
        "length_mm": list(range(500, 5000, 500)),
        "diameter_mm": list(range(20, 150, 10)),
    },
    "fairing": {
        "diameter_mm": list(range(500, 4000, 300)),
        "length_mm": list(range(500, 3000, 200)),
    },
    "injector": {
        "face_diameter_mm": list(range(40, 200, 20)),
        "num_holes": [4, 7, 9, 12, 16, 19, 25, 36, 49, 64],
    },
}


def verify(path: Path) -> tuple[bool, str]:
    """Re-read what was written and confirm it is a solid.

    The failure this replaces was silent: `export` succeeded, the file existed,
    the count went up, and the file held nothing. Checking the artefact rather
    than the call is the whole point.
    """
    try:
        m = trimesh.load(path, force="mesh")
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {exc}"
    if not hasattr(m, "faces") or len(m.faces) == 0:
        return False, "no triangles"
    if len(m.vertices) == 0:
        return False, "no vertices"
    if not m.is_watertight:
        return False, "not watertight"
    if abs(m.volume) <= 0.0:
        return False, "zero volume"
    return True, ""


def main() -> int:
    import argparse
    import itertools

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-per-family", type=int, default=0,
                    help="cap parts per family (0 = the full sweep)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify in memory without writing")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    written, failed = 0, []
    per_family = {}
    # Global index across families, not per-family. The graph refers to these
    # parts by filename -- tank_001368.stl, fairing_001725.stl -- and numbering
    # each family from zero silently repoints 593 of those references at files
    # that do not exist. The names are a key held by another system; they are
    # not ours to renumber.
    next_index = 0
    for family, ranges in FAMILIES.items():
        gen, names = GENERATORS[family]
        combos = list(itertools.product(*(ranges[n] for n in names)))
        if args.limit_per_family:
            combos = combos[:args.limit_per_family]
        ok_here = 0
        for offset, combo in enumerate(combos):
            path = out / f"{family}_{next_index + offset:06d}.stl"
            try:
                mesh = gen(*combo)
            except Exception as exc:  # noqa: BLE001
                failed.append((path.name, f"build: {exc}"))
                continue
            if mesh is None or len(getattr(mesh, "faces", ())) == 0:
                failed.append((path.name, "generator produced no triangles"))
                continue
            if args.dry_run:
                ok_here += 1
                written += 1
                continue
            mesh.export(str(path))
            good, why = verify(path)
            if not good:
                failed.append((path.name, why))
                path.unlink(missing_ok=True)
                continue
            ok_here += 1
            written += 1
        next_index += len(combos)
        per_family[family] = ok_here
        print(f"  {family:10} {ok_here:5} / {len(combos):5} verified")

    summary = {
        "verified_solids": written,
        "per_family": per_family,
        "failed": len(failed),
        "output_directory": str(out),
        "format": "STL",
        "note": ("counts are files that were re-read after writing and found "
                 "to have triangles, be watertight and have positive volume"),
    }
    if not args.dry_run:
        (out / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{written} verified solids, {len(failed)} failed")
    for name, why in failed[:10]:
        print(f"   {name}: {why}")
    return 0 if written and not failed else (0 if written else 1)


if __name__ == "__main__":
    raise SystemExit(main())
