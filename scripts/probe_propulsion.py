"""Probe the propulsion targets, writing each result the moment it exists.

The generic probe reloads a 430 MB graph per invocation, which is 40 s of work
repeated for every target. On a contended box that is the difference between a
run that finishes and one that gets killed with nothing to show -- four attempts
at this measurement died mid-load and produced no output at all.

So: load once, probe every target, and append each result to disk as it lands.
A run that dies halfway still leaves the targets it finished, which matters more
than elegance when the machine is shared.

The measurement itself is unchanged and deliberately strict -- split by mesh so
no geometry is fitted and scored, consensus labels only, paired bootstrap
against random projections, and the spread reported across several splits rather
than one. Those are the corrections that turned an earlier +0.08 into noise; the
point of running this again is that the corpus underneath it is finally sound,
not that the bar has moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGETS = ("expansion_ratio", "thrust_kN", "isp_vac_s", "throat_heat_flux_MWm2")


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "checkpoints/fixed_corpus/step_002000.pt")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/propulsion_probe.json")
    args = ap.parse_args()

    sys.argv = [sys.argv[0]]
    import train as train_mod
    from models.jepa import JEPAModel
    from utils.config import load_yaml_with_family

    from scripts.probe_representation import (
        _embed_batch,
        group_split,
        paired_gain_bootstrap,
        ridge_r2,
    )

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or load_yaml_with_family(
        str(ROOT / "configs/base.yaml"), family="space_cpu")
    cfg["data"]["graph_path"] = str(ROOT / "artifacts/jepa-train-bundle/graph.json")

    print("loading dataset (once)...", flush=True)
    dataset = train_mod.build_dataloader(cfg, "graph").dataset

    trained = JEPAModel.from_config(cfg)
    trained.load_state_dict(payload["model"])
    trained.eval()
    torch.manual_seed(0)
    untrained = JEPAModel.from_config(cfg)
    untrained.eval()

    # Only records whose fields come from geometry: on a physics shard the
    # fields are the simulation, so predicting a quantity derived from that
    # simulation is integration of the input, not prediction.
    pool = [i for i in range(len(dataset))
            if getattr(dataset, "field_provenance", lambda _i: "geometry")(i) == "geometry"]
    rng = np.random.default_rng(0)
    picks = rng.choice(pool, size=min(args.samples, len(pool)), replace=False)
    print(f"{len(pool)} geometry-backed records; probing {len(picks)}", flush=True)

    results: dict = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        print(f"\n=== {target} ===", flush=True)
        xs_t, xs_r, ys, groups, pending = [], [], [], [], []

        def flush():
            if not pending:
                return
            pts = torch.stack([p for p, _f in pending])
            fld = torch.stack([f for _p, f in pending])
            xs_t.extend(_embed_batch(trained, pts, fld))
            xs_r.extend(_embed_batch(untrained, pts, fld))
            pending.clear()

        for i in picks:
            try:
                sample = dataset[int(i)]
            except Exception:  # noqa: BLE001
                continue
            value = dataset.physics_for(int(i)).get(target)
            if value is None or not np.isfinite(value) or value <= 0:
                continue
            pending.append((sample["points"], sample["fields"]))
            ys.append(np.log10(float(value)))
            rec = dataset.records[int(i)]
            groups.append(str(getattr(rec, "path", i)))
            if len(pending) >= 32:
                flush()
        flush()

        x_t = np.asarray(xs_t, dtype=np.float64)
        x_r = np.asarray(xs_r, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        groups_arr = np.asarray(groups, dtype=object)
        n_meshes = len(np.unique(groups_arr))
        print(f"  {len(y)} usable samples over {n_meshes} meshes", flush=True)
        if len(y) < 60:
            results[target] = {"usable": int(len(y)), "verdict": "too few samples"}
            args.out.write_text(json.dumps(results, indent=2))
            continue

        gains = []
        r2_t = r2_r = float("nan")
        for s in range(args.splits):
            tr = group_split(groups_arr, seed=s)
            te = ~tr
            if tr.sum() < 20 or te.sum() < 20:
                continue
            if s == 0:
                r2_t = ridge_r2(x_t[tr], y[tr], x_t[te], y[te])
                r2_r = ridge_r2(x_r[tr], y[tr], x_r[te], y[te])
            gain, _lo, _hi = paired_gain_bootstrap(
                x_r[tr], x_t[tr], y[tr], x_r[te], x_t[te], y[te], draws=200)
            gains.append(gain)

        sign = ("all +" if gains and all(g > 0 for g in gains)
                else "all -" if gains and all(g < 0 for g in gains) else "MIXED")
        results[target] = {
            "usable": int(len(y)),
            "meshes": int(n_meshes),
            "r2_trained": float(r2_t),
            "r2_random": float(r2_r),
            "mean_gain": float(np.mean(gains)) if gains else None,
            "min_gain": float(np.min(gains)) if gains else None,
            "max_gain": float(np.max(gains)) if gains else None,
            "sign": sign,
        }
        print(f"  R^2 trained {r2_t:.4f}  random {r2_r:.4f}", flush=True)
        if gains:
            print(f"  gain over {len(gains)} splits: mean {np.mean(gains):+.4f} "
                  f"[{np.min(gains):+.4f}, {np.max(gains):+.4f}]  {sign}", flush=True)
        args.out.write_text(json.dumps(results, indent=2))

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
