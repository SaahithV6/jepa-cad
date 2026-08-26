"""Rebuild each generated nozzle from the labels its own record carries.

Regenerating the empty STLs from a fresh parameter sweep fixed the geometry and
broke the correspondence: the meshes matched my enumeration order (Pearson
+0.9994) and not the graph's labels (-0.0125). Filename index is not a key. The
original generator's task ordering is not recoverable from what survives, and
reconstructing it would only move the assumption somewhere else.

So nothing here is enumerated. Every file is built from the physics attached to
the record that points at it, which makes the correspondence true by
construction rather than by a shared convention that has already failed once:

* expansion ratio sets the bell contour directly, so the exit-to-throat area
  ratio measured off the mesh IS the label;
* throat area comes from thrust and Isp through the standard relations, so a
  higher-thrust record gets a physically larger throat --

      mdot = F / (Isp * g0)          and       At = mdot * c_star / Pc

  with chamber pressure and characteristic velocity fixed at the values below.
  Those two are assumptions, stated here because they are not free: they set
  the absolute scale of every throat. What they cannot do is scramble the
  ordering -- At stays monotone in thrust whatever they are -- so the labels
  and the geometry agree either way.

Each file is re-read after writing and checked; a mesh whose measured expansion
ratio misses its label is deleted rather than counted.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Chamber pressure and characteristic velocity used to size the throat.
#: Representative of a pressure-fed storable bipropellant; see the note above.
CHAMBER_PRESSURE_PA = 5.0e6
C_STAR_MS = 1800.0
G0 = 9.80665


def throat_radius_m(thrust_kn: float, isp_s: float) -> float:
    """Throat radius implied by thrust and specific impulse."""
    thrust_n = float(thrust_kn) * 1000.0
    mdot = thrust_n / (float(isp_s) * G0)
    area = mdot * C_STAR_MS / CHAMBER_PRESSURE_PA
    return math.sqrt(area / math.pi)


def measured_expansion(mesh, nbins: int = 200) -> float:
    """Exit-to-throat area ratio read off the mesh.

    Binned along z, which is the axis `revolve` builds about -- taking the
    longest bounding-box extent instead breaks for wide, short nozzles, where
    the exit diameter exceeds the length.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    z = v[:, 2]
    r = np.hypot(v[:, 0], v[:, 1])
    edges = np.linspace(z.min(), z.max(), nbins + 1)
    prof = []
    for a, b in zip(edges, edges[1:]):
        sel = (z >= a) & (z <= b)
        if sel.sum() >= 3:
            prof.append(r[sel].max())
    if len(prof) < 4:
        return float("nan")
    prof = np.asarray(prof)
    return float((prof[-1] / prof.min()) ** 2)


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    import trimesh

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path,
                    default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--family", type=str, default="space_cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="max relative error between measured and labelled "
                         "expansion ratio before the file is rejected")
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]
    import train as train_mod
    from generate_spaceflight_cad import revolve
    from utils.config import load_yaml_with_family

    from cadflow.sculpt import bell_contour

    cfg = load_yaml_with_family(str(ROOT / "configs/base.yaml"),
                                family=args.family)
    cfg["data"]["graph_path"] = str(args.graph)
    dataset = train_mod.build_dataloader(cfg, "graph").dataset

    # one entry per file: records agreeing on a label collapse to that label
    targets: dict[Path, dict] = {}
    conflicts = 0
    for i, rec in enumerate(dataset.records):
        if rec.path is None or "generated_spaceflight_cad" not in str(rec.path):
            continue
        if not rec.path.name.startswith("nozzle"):
            continue
        phys = dataset.physics_for(i)
        eps = phys.get("expansion_ratio")
        if eps is None:
            continue
        prev = targets.get(rec.path)
        if prev is not None:
            if abs(prev["eps"] - float(eps)) > 1e-6:
                conflicts += 1
            continue
        targets[rec.path] = {
            "eps": float(eps),
            "thrust_kN": float(phys.get("thrust_kN", 0.0)),
            "isp_vac_s": float(phys.get("isp_vac_s", 0.0)),
        }

    print(f"{len(targets)} nozzle files carry a label"
          f"{f'; {conflicts} records disagreed with their file' if conflicts else ''}")
    items = sorted(targets.items())
    if args.limit:
        items = items[:args.limit]

    written, rejected = 0, []
    for path, spec in items:
        eps = spec["eps"]
        if spec["thrust_kN"] > 0.0 and spec["isp_vac_s"] > 0.0:
            r_t = throat_radius_m(spec["thrust_kN"], spec["isp_vac_s"])
        else:
            r_t = 0.01
        try:
            bell = bell_contour(r_t, eps)
            diverging = [(z * 1000.0, r * 1000.0) for r, z in bell.contour]
            r_t_mm = r_t * 1000.0
            r_chamber = r_t_mm * 2.5
            conv_len = 1.5 * r_t_mm
            conv = [(-conv_len * (1.0 - f),
                     r_chamber + (r_t_mm - r_chamber)
                     * (0.5 - 0.5 * math.cos(math.pi * f)))
                    for f in (i / 8.0 for i in range(9))]
            mesh = revolve(conv[:-1] + diverging)
        except Exception as exc:  # noqa: BLE001
            rejected.append((path.name, f"build: {exc}"))
            continue

        mesh.export(str(path))
        try:
            back = trimesh.load(path, force="mesh")
        except Exception as exc:  # noqa: BLE001
            rejected.append((path.name, f"unreadable: {exc}"))
            path.unlink(missing_ok=True)
            continue
        if not back.is_watertight or len(back.faces) == 0:
            rejected.append((path.name, "not a closed solid"))
            path.unlink(missing_ok=True)
            continue
        got = measured_expansion(back)
        if not np.isfinite(got) or abs(got - eps) / eps > args.tolerance:
            rejected.append((path.name, f"eps {got:.2f} vs label {eps:.2f}"))
            path.unlink(missing_ok=True)
            continue
        written += 1

    print(f"\n{written} nozzles rebuilt from their own labels, "
          f"{len(rejected)} rejected")
    for name, why in rejected[:10]:
        print(f"   {name}: {why}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
