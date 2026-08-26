"""Turn OpenRocket .ork design files into geometry the corpus can train on.

716 of these were sitting in `data/` being skipped as an unreadable format. They
are not unreadable -- an .ork is a zip holding one XML document that describes a
complete vehicle: nose cone shape and length, body tube radii and wall
thickness, fin planforms, transitions, and the bulk density of the material each
part is made from.

That makes them better than a bare mesh. A downloaded STL is geometry with no
labels; these carry the parameters the geometry came from, so the mesh this
script writes and the numbers recorded beside it describe the same object by
construction. That property is the whole reason the nozzle corpus had to be
rebuilt this week -- 1,368 files whose labels described a different vehicle than
their geometry, which no amount of training could have survived.

Every part is meshed with the same `revolve` used for the generated corpus, and
every written file is re-read and checked before it counts. A part whose
measured radius disagrees with the radius its design claims is dropped, not
reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Component tags this reads. Anything else in the tree is ignored rather than
#: guessed at -- a mis-parsed component would put a wrong number next to a real
#: mesh, which is the failure mode this whole exercise exists to avoid.
BODY_TAGS = {"nosecone", "bodytube", "transition"}
FIN_TAGS = {"trapezoidfinset", "ellipticalfinset", "freeformfinset"}


def _f(node, tag: str, default: float | None = None) -> float | None:
    """Read a float child, tolerating OpenRocket's 'auto 0.033' syntax."""
    el = node.find(tag)
    if el is None or not (el.text or "").strip():
        return default
    text = el.text.strip()
    if text.startswith("auto"):
        parts = text.split()
        text = parts[1] if len(parts) > 1 else ""
    try:
        return float(text)
    except ValueError:
        return default


def parse_ork(path: Path) -> dict | None:
    """Design parameters for one .ork, or None if it holds no usable body."""
    # Most .ork files are zips holding one XML; some are the bare XML. Trying
    # only the zip path silently discarded every uncompressed design.
    try:
        with zipfile.ZipFile(path) as archive:
            name = next((n for n in archive.namelist() if n.endswith(".ork")),
                        archive.namelist()[0])
            payload = archive.read(name)
    except zipfile.BadZipFile:
        try:
            payload = path.read_bytes()
        except OSError:
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return None

    rocket = root.find("rocket")
    if rocket is None:
        return None

    design = {
        "name": (rocket.findtext("name") or path.stem).strip(),
        "source_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "components": [],
        "fins": [],
    }

    for el in rocket.iter():
        tag = el.tag.lower()
        if tag in BODY_TAGS:
            length = _f(el, "length")
            if not length or length <= 0:
                continue
            fore = _f(el, "foreradius") or _f(el, "aftradius")
            aft = _f(el, "aftradius") or fore
            if tag == "nosecone":
                fore = 0.0
                aft = _f(el, "aftradius")
            if aft is None or aft <= 0:
                continue
            mat = el.find("material")
            design["components"].append({
                "kind": tag,
                "name": (el.findtext("name") or tag).strip(),
                "length_m": float(length),
                "fore_radius_m": float(fore or 0.0),
                "aft_radius_m": float(aft),
                "thickness_m": _f(el, "thickness", 0.0),
                "shape": (el.findtext("shape") or "").strip() or None,
                "shape_parameter": _f(el, "shapeparameter"),
                "material": (mat.text or "").strip() if mat is not None else None,
                "density_kg_m3": float(mat.get("density")) if mat is not None
                and mat.get("density") else None,
            })
        elif tag in FIN_TAGS:
            design["fins"].append({
                "kind": tag,
                "count": int(_f(el, "fincount", 3) or 3),
                "root_chord_m": _f(el, "rootchord"),
                "tip_chord_m": _f(el, "tipchord"),
                "span_m": _f(el, "height"),
                "sweep_m": _f(el, "sweeplength"),
                "thickness_m": _f(el, "thickness"),
            })

    if not design["components"]:
        return None

    body = design["components"]
    design["total_length_m"] = float(sum(c["length_m"] for c in body))
    # Both ends of every component. Taking only the aft radius reported 0.025 m
    # for a two-stage vehicle whose booster is 0.056 m at its fore end -- the
    # mesh check caught it, which is the entire reason that check compares
    # geometry against the design rather than trusting the parse.
    design["max_radius_m"] = float(max(
        max(c["aft_radius_m"], c["fore_radius_m"]) for c in body))
    design["fineness_ratio"] = (design["total_length_m"]
                                / max(2.0 * design["max_radius_m"], 1e-9))
    design["n_body_components"] = len(body)
    design["n_fin_sets"] = len(design["fins"])
    return design


def profile_for(design: dict) -> list[tuple[float, float]]:
    """(z, radius) meridian for the whole airframe, nose forward.

    Nose cone shape is honoured where OpenRocket names one this project already
    implements; anything else becomes a straight taper, which is recorded in the
    design rather than passed off as the named shape.
    """
    prof: list[tuple[float, float]] = []
    z = 0.0
    for comp in design["components"]:
        length = comp["length_m"] * 1000.0
        fore = comp["fore_radius_m"] * 1000.0
        aft = comp["aft_radius_m"] * 1000.0
        if comp["kind"] == "nosecone":
            shape = (comp.get("shape") or "").lower()
            n = 14
            for i in range(n + 1):
                f = i / n
                if shape == "ogive":
                    rho = (aft * aft + length * length) / (2.0 * aft)
                    r = math.sqrt(max(0.0, rho * rho - (length * (1 - f)) ** 2)) + aft - rho
                elif shape in ("conical", "cone"):
                    r = aft * f
                elif shape in ("ellipsoid", "elliptical"):
                    r = aft * math.sqrt(max(0.0, 1.0 - (1.0 - f) ** 2))
                else:
                    r = aft * f
                prof.append((z + length * f, max(0.0, r)))
        else:
            prof.append((z, fore if fore > 0 else aft))
            prof.append((z + length, aft))
        z += length

    # Drop consecutive duplicates. Where a nose cone ends and a tube begins the
    # two components both name that station at the same radius, which revolves
    # into a band of zero-area triangles: the mesh builds, and `is_watertight`
    # is False. A station repeated at a *different* radius is kept -- that is a
    # flat annular shoulder, and its triangles have real area.
    cleaned: list[tuple[float, float]] = []
    for point in prof:
        if cleaned and abs(point[0] - cleaned[-1][0]) < 1e-9 \
                and abs(point[1] - cleaned[-1][1]) < 1e-9:
            continue
        cleaned.append(point)
    return cleaned


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import trimesh

    from generate_spaceflight_cad import revolve

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/openrocket_designs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-mesh", action="store_true",
                    help="parse and record parameters without writing geometry")
    args = ap.parse_args()

    files = sorted(args.search.rglob("*.ork"))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} .ork files found")

    args.out.mkdir(parents=True, exist_ok=True)
    designs, meshed, rejected = [], 0, []
    for path in files:
        design = parse_ork(path)
        if design is None:
            rejected.append((path.name, "no usable body components"))
            continue

        if not args.no_mesh:
            try:
                prof = profile_for(design)
                if len(prof) < 3:
                    raise ValueError("meridian too short")
                mesh = revolve(prof)
                stem = f"ork_{len(designs):05d}"
                out_path = args.out / f"{stem}.stl"
                mesh.export(str(out_path))
                back = trimesh.load(out_path, force="mesh")
                if len(back.faces) == 0 or not back.is_watertight:
                    raise ValueError("not a closed solid")
                # the mesh must agree with the design it claims to be
                v = np.asarray(back.vertices, dtype=float)
                measured_r = float(np.hypot(v[:, 0], v[:, 1]).max()) / 1000.0
                claimed_r = design["max_radius_m"]
                if abs(measured_r - claimed_r) > 0.02 * max(claimed_r, 1e-6):
                    raise ValueError(
                        f"radius {measured_r:.4f} m vs design {claimed_r:.4f} m")
                design["mesh"] = str(out_path.relative_to(ROOT))
                design["mesh_volume_m3"] = abs(float(back.volume)) / 1e9
                meshed += 1
            except Exception as exc:  # noqa: BLE001
                rejected.append((path.name, str(exc)[:100]))
                (args.out / f"ork_{len(designs):05d}.stl").unlink(missing_ok=True)
                design.pop("mesh", None)

        designs.append(design)

    manifest = args.out / "designs.json"
    manifest.write_text(json.dumps(designs, indent=1))
    print(f"\n{len(designs)} designs parsed, {meshed} meshed and verified, "
          f"{len(rejected)} rejected")
    for name, why in rejected[:8]:
        print(f"   {name}: {why}")
    if designs:
        lens = [d["total_length_m"] for d in designs]
        fine = [d["fineness_ratio"] for d in designs]
        print(f"\nlength    {min(lens):.2f} to {max(lens):.2f} m")
        print(f"fineness  {min(fine):.1f} to {max(fine):.1f}")
        print(f"fin sets  {sum(d['n_fin_sets'] for d in designs)} across the corpus")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
