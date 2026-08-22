#!/usr/bin/env python3
"""What did the encoder actually learn?

A falling loss curve says the model is fitting its own objective. It does not
say the representation is useful, and for a JEPA it cannot: the objective is to
predict its own EMA target, which a collapsed encoder satisfies perfectly by
predicting a constant. embed_std guards against that particular failure and
nothing else.

The standard answer is a linear probe. Freeze the encoder, embed a held-out set,
fit a *linear* model from those embeddings to a physical quantity nobody trained
on, and see how much of the variance it explains. Linear is the point: if a
linear map recovers the physics, the encoder put it there, because a linear
probe cannot compute anything the representation does not already contain.

The comparison that matters is against the same architecture with random
weights. An untrained encoder is not a null of zero -- random projections of a
point cloud carry real information about its size and shape, and a probe on
them can score surprisingly well. Training has to beat that, not beat nothing.

Usage:
  python scripts/probe_representation.py --ckpt checkpoints/latest.pt \\
      --samples 400 --target max_stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def embed(model, batch: dict) -> torch.Tensor:
    """Mean-pooled context-encoder embedding for a batch."""
    with torch.no_grad():
        enc = model.context_encoder(batch["points"], batch["fields"])
        if isinstance(enc, dict):
            # The encoder returns both; the pooled vector is the one a probe
            # should see, since a linear map over per-token embeddings would be
            # doing the pooling itself and confusing the two questions.
            enc = enc.get("pooled_embedding")
            if enc is None:
                raise KeyError("encoder returned no pooled_embedding")
        if enc.dim() == 3:
            enc = enc.mean(dim=1)
    return enc


def clean_targets(y: np.ndarray, log_space: bool = True):
    """Drop placeholder values and put a heavy-tailed target in log space.

    Two things make max_stress unusable as it stands. It spans 8.4 decades --
    0.024 to 6.8 million -- so a squared error in linear space is decided
    entirely by a handful of extremes, and R^2 came out at -2.0 for every model
    including random weights. And 39% of the corpus carries max_stress exactly
    1.0, which is a placeholder rather than a measurement: 234 identical values
    out of 600, against 363 distinct values overall.

    A single value taking more than a tenth of a continuous target is not a
    coincidence, so it is treated as filler and removed. Returns the mask of
    rows to keep alongside the transformed target.
    """
    keep = np.ones(len(y), dtype=bool)
    values, counts = np.unique(np.round(y, 9), return_counts=True)
    for value, count in zip(values, counts):
        if count > 0.10 * len(y):
            keep &= np.abs(y - value) > 1e-9
    out = y[keep]
    if log_space:
        out = np.log10(np.clip(out, 1e-9, None))
    return keep, out


def collect(model, dataset, indices, target: str):
    """Embeddings and targets for a set of records.

    Only dataset read failures are tolerated, and only because 7% of this
    corpus is empty meshes. An encoder that will not run is a different thing
    entirely and is raised: swallowing it reported "0 usable samples", which
    said nothing about the actual fault -- a checkpoint whose weights expect
    three fields against a dataset serving six.
    """
    xs, ys = [], []
    for i in indices:
        try:
            sample = dataset[int(i)]
        except Exception:  # noqa: BLE001 - a bad file is not a bad probe
            continue
        if target not in sample:
            continue
        y = float(sample[target])
        if not np.isfinite(y):
            continue
        batch = {"points": sample["points"].unsqueeze(0),
                 "fields": sample["fields"].unsqueeze(0)}
        xs.append(embed(model, batch).squeeze(0).numpy())
        ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def ridge_r2_bootstrap(x_train, y_train, x_test, y_test, alpha: float = 1.0,
                       draws: int = 400, seed: int = 0):
    """Held-out R^2 with a bootstrap interval over the test set.

    Eighty-nine held-out points is not many, and a difference of 0.03 in R^2
    between two models is easy to read as a result when it is sampling noise.
    Resampling the test set with replacement gives the spread that difference
    has to clear before it means anything.
    """
    point = ridge_r2(x_train, y_train, x_test, y_test, alpha)
    rng = np.random.default_rng(seed)
    n = len(x_test)
    draws_r2 = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        draws_r2.append(ridge_r2(x_train, y_train, x_test[idx], y_test[idx], alpha))
    lo, hi = np.percentile(draws_r2, [2.5, 97.5])
    return point, float(lo), float(hi)


def ridge_r2(x_train, y_train, x_test, y_test, alpha: float = 1.0) -> float:
    """Held-out R^2 of a ridge fit. Closed form, no dependency on sklearn.

    Targets are standardised on the training split only -- standardising over
    everything would leak the test set's mean and variance into the fit, which
    on a few hundred samples is enough to matter.
    """
    mu, sigma = x_train.mean(0), x_train.std(0) + 1e-8
    xtr = (x_train - mu) / sigma
    xte = (x_test - mu) / sigma
    xtr = np.hstack([xtr, np.ones((len(xtr), 1))])
    xte = np.hstack([xte, np.ones((len(xte), 1))])

    ymu, ysig = y_train.mean(), y_train.std() + 1e-8
    ytr = (y_train - ymu) / ysig

    reg = alpha * np.eye(xtr.shape[1])
    reg[-1, -1] = 0.0                       # never penalise the intercept
    w = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ ytr)

    pred = (xte @ w) * ysig + ymu
    ss_res = float(((y_test - pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, nargs="+",
                    default=[ROOT / "checkpoints/latest.pt"],
                    help="one or more checkpoints; probing several shows the "
                         "trend, which a single number cannot -- 'learned "
                         "steadily' and 'learned then degraded' end up in the "
                         "same place")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--target", type=str, default="max_stress")
    ap.add_argument("--family", type=str, default="space_cpu")
    ap.add_argument("--graph", type=Path,
                    default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear-target", action="store_true",
                    help="probe the raw target instead of its logarithm; the "
                         "default is log because max_stress spans 8.4 decades")
    args = ap.parse_args()

    import train as train_mod
    from models.jepa import JEPAModel
    from utils.config import load_yaml_with_family

    ckpts = sorted(args.ckpt, key=lambda p: _step_of(p))
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = first.get("config") or load_yaml_with_family(
        str(ROOT / "configs/base.yaml"), family=args.family)
    cfg["data"]["graph_path"] = str(args.graph)

    # One dataset for every checkpoint: the graph is 430 MB and takes 39 s to
    # load, which would otherwise dominate a four-checkpoint sweep.
    dataset = train_mod.build_dataloader(cfg, "graph").dataset
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(dataset), size=min(args.samples, len(dataset)),
                       replace=False)

    expected_fields = int(cfg["data"]["num_fields"])

    # The random-init baseline is the same for every checkpoint, so it is
    # embedded once. It is also not zero: random projections of a point cloud
    # carry real information about its size and shape.
    torch.manual_seed(0)
    untrained = JEPAModel.from_config(cfg)
    untrained.eval()
    x0, y0_raw = collect(untrained, dataset, picks, args.target)
    keep, y0 = clean_targets(y0_raw, log_space=not args.linear_target)
    x0 = x0[keep]
    dropped = int((~keep).sum())
    if len(x0) < 40:
        print(f"only {len(x0)} usable samples; too few to probe")
        return 1
    n_tr = int(0.7 * len(x0))
    base_r2, base_lo, base_hi = ridge_r2_bootstrap(
        x0[:n_tr], y0[:n_tr], x0[n_tr:], y0[n_tr:])

    space = "raw" if args.linear_target else "log10"
    print(f"target {args.target!r} in {space}, {len(x0)} usable samples "
          f"({n_tr} train / {len(x0) - n_tr} held out); "
          f"{dropped} placeholder rows dropped\n")
    print(f"{'checkpoint':>14} {'step':>6} {'R^2':>8} {'95% interval':>18} "
          f"{'vs random':>10}")
    print(f"{'random init':>14} {0:>6} {base_r2:>8.4f} "
          f"{f'[{base_lo:+.3f}, {base_hi:+.3f}]':>18} {0.0:>+10.4f}")

    rows = []
    for path in ckpts:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weight = payload["model"].get("context_encoder.input_proj.weight")
        if weight is not None and int(weight.shape[1]) - 3 != expected_fields:
            print(f"{path.name:>14} {'--':>6}   trained on a different "
                  f"configuration; skipped")
            continue
        model = JEPAModel.from_config(cfg)
        model.load_state_dict(payload["model"], strict=False)
        model.eval()
        x, y_raw = collect(model, dataset, picks, args.target)
        x = x[keep]
        n = int(0.7 * len(x))
        r2, lo, hi = ridge_r2_bootstrap(x[:n], y0[:n], x[n:], y0[n:])
        step = int(payload.get("step") or _step_of(path))
        rows.append((step, r2, lo, hi))
        print(f"{path.name:>14} {step:>6} {r2:>8.4f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18} {r2 - base_r2:>+10.4f}")

    if len(rows) >= 2:
        first_r2, last_r2 = rows[0][1], rows[-1][1]
        width = base_hi - base_lo
        gain = max(r[1] for r in rows) - base_r2
        print(f"\n  the bootstrap interval on a single R^2 is {width:.3f} wide, "
              f"so a gain of {gain:+.3f} is {'inside' if abs(gain) < width/2 else 'outside'} "
              f"the noise on one model alone.")
        print(f"\n  across training: {first_r2:+.4f} -> {last_r2:+.4f} "
              f"({last_r2 - first_r2:+.4f})")
        best = max(rows, key=lambda r: r[1])
        print(f"  best at step {best[0]}: {best[1]:+.4f} "
              f"({best[1] - base_r2:+.4f} over random)")
        if best[1] < 0.0:
            print("  every probe is worse than predicting the mean; the target "
                  "or the split is at fault, not the encoder")
        elif best[1] - base_r2 > 0.02:
            print("  the representation carries physics random weights do not")
        elif best[1] - base_r2 > -0.02:
            print("  no measurable gain over random projections on this target")
        else:
            print("  training made the representation worse for this target")
    return 0


def _step_of(path: Path) -> int:
    """Step number from a checkpoint filename, for ordering."""
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 10**9


if __name__ == "__main__":
    raise SystemExit(main())
