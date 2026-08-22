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
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/latest.pt")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--target", type=str, default="max_stress")
    ap.add_argument("--family", type=str, default="space_cpu")
    ap.add_argument("--graph", type=Path,
                    default=ROOT / "artifacts/jepa-train-bundle/graph.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import train as train_mod
    from models.jepa import JEPAModel
    from utils.config import load_yaml_with_family

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or load_yaml_with_family(
        str(ROOT / "configs/base.yaml"), family=args.family)
    cfg["data"]["graph_path"] = str(args.graph)

    dataset = train_mod.build_dataloader(cfg, "graph").dataset
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(dataset), size=min(args.samples, len(dataset)),
                       replace=False)
    split = int(0.7 * len(picks))

    # Check the checkpoint against the data before embedding anything. A
    # mismatch here is the difference between "the model learned nothing" and
    # "the model was never run", and those deserve different reactions.
    expected_fields = int(cfg["data"]["num_fields"])
    weight = payload["model"].get("context_encoder.input_proj.weight")
    if weight is not None:
        ckpt_fields = int(weight.shape[1]) - 3          # xyz are concatenated
        if ckpt_fields != expected_fields:
            print(f"  checkpoint expects {ckpt_fields} fields, the dataset "
                  f"serves {expected_fields}. This checkpoint was trained on a "
                  f"different configuration and cannot be probed against this "
                  f"corpus.")
            return 2

    trained = JEPAModel.from_config(cfg)
    missing, unexpected = trained.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"  note: {len(missing)} parameters absent from the checkpoint "
              f"and left at initialisation")
    trained.eval()

    untrained = JEPAModel.from_config(cfg)
    untrained.eval()

    print(f"checkpoint step {payload.get('step')}, target {args.target!r}, "
          f"{len(picks)} samples")

    results = {}
    for name, model in (("trained", trained), ("random init", untrained)):
        x, y = collect(model, dataset, picks, args.target)
        if len(x) < 20:
            print(f"  {name}: only {len(x)} usable samples; cannot probe")
            return 1
        n_tr = int(0.7 * len(x))
        r2 = ridge_r2(x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:])
        results[name] = r2
        print(f"  {name:12s} held-out R^2 = {r2:+.4f}   "
              f"({len(x)} usable, {n_tr} train / {len(x)-n_tr} test)")

    gain = results["trained"] - results["random init"]
    print(f"\n  training gained {gain:+.4f} R^2 over random initialisation")
    if gain > 0.02:
        print("  the representation carries physics the random one does not")
    elif gain > -0.02:
        print("  no measurable gain: the encoder has not learned this target")
    else:
        print("  training made the representation WORSE for this target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
