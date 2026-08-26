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


def _embed_batch(model, points, fields):
    """Pooled embeddings for a batch, as a numpy array.

    Named export of what `embed` does, so other scripts fit and score on the
    same vector the probe reports accuracy for rather than a near-copy.
    """
    return embed(model, {"points": points, "fields": fields}).numpy()


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


def collect(model, dataset, indices, target: str, batch_size: int = 32,
            want_groups: bool = False):
    """Embeddings and targets for a set of records.

    Only dataset read failures are tolerated, and only because 7% of this
    corpus is empty meshes. An encoder that will not run is a different thing
    entirely and is raised: swallowing it reported "0 usable samples", which
    said nothing about the actual fault -- a checkpoint whose weights expect
    three fields against a dataset serving six.
    """
    # Embedded in batches. One sample at a time made the encoder pay full
    # per-call overhead for a batch of one, and the cost of that was not speed
    # but resolution: it kept the affordable sample size near 300, which put an
    # 0.247-wide bootstrap interval around every R^2 and made a 0.014 gain
    # unmeasurable. Batching buys the samples the interval needs.
    xs, ys, pending = [], [], []

    def flush():
        if not pending:
            return
        pts = torch.stack([p for p, _f in pending])
        fld = torch.stack([f for _p, f in pending])
        out = embed(model, {"points": pts, "fields": fld}).numpy()
        xs.extend(out)
        pending.clear()

    physics_for = getattr(dataset, "physics_for", None)
    groups = []
    for i in indices:
        try:
            sample = dataset[int(i)]
        except Exception:  # noqa: BLE001 - a bad file is not a bad probe
            continue
        if target in sample:
            y = float(sample[target])
        elif physics_for is not None:
            y = physics_for(int(i)).get(target)
            if y is None:
                continue
            y = float(y)
        else:
            continue
        if not np.isfinite(y):
            continue
        pending.append((sample["points"], sample["fields"]))
        ys.append(y)
        rec = dataset.records[int(i)] if hasattr(dataset, "records") else None
        groups.append(_group_key(getattr(rec, "path", None), i))
        if len(pending) >= batch_size:
            flush()
    flush()
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if want_groups:
        return x, y, np.asarray(groups, dtype=object)
    return x, y


_GROUP_KEY_CACHE: dict[str, str] = {}

#: How `_group_key` identifies a mesh. "content" hashes the file, "path" uses
#: the filename. Switchable only so the two can be measured against each other
#: on one pool -- claiming that content grouping changed a result requires
#: running both, not comparing against a remembered number from a run whose
#: configuration may have differed.
GROUP_MODE = "content"


def _group_key(path, index) -> str:
    """Group by mesh *content*, not by filename.

    Splitting on the path was already a fix for record-level leakage, and it is
    not enough. 17% of this corpus is byte-identical geometry stored under
    different names -- 71.5% within body_tube, where 1,000 files hold 285
    distinct shapes, and 64% within ring_frame. Those copies have different
    paths, so a path-grouped split cheerfully puts the same mesh in train and
    in test and the probe scores its own memory.

    Hashing the file closes that. Anything unreadable falls back to the path,
    and anything without a path to its record index, so a missing file cannot
    silently collapse many records into one group -- which would look like a
    much stricter split while actually being a broken one.
    """
    import hashlib

    if path is None:
        return f"idx:{index}"
    key = str(path)
    if GROUP_MODE == "path":
        return f"path:{key}"
    cached = _GROUP_KEY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        digest = hashlib.md5(Path(key).read_bytes()).hexdigest()
        out = f"sha:{digest}"
    except (OSError, ValueError):
        out = f"path:{key}"
    _GROUP_KEY_CACHE[key] = out
    return out


def group_split(groups, frac_train: float = 0.7, seed: int = 0):
    """Train/test split that never puts one mesh on both sides.

    87% of the geometry files in this corpus appear in more than one record,
    usually three. Splitting on record index therefore leaks: the same mesh is
    fitted and then scored, so the probe measures recall of a shape it has
    already seen. That inflates every absolute R^2 -- the 0.808 random-init
    baseline on stress is mostly this -- and it flatters random projections
    just as much as trained weights, which is why the *gains* moved far less
    than the levels.

    Grouping on the file's contents keeps every copy of a mesh on one side --
    including the copies stored under a different name, which grouping on the
    path alone does not catch.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_tr = int(frac_train * len(uniq))
    train_files = set(uniq[:n_tr].tolist())
    mask = np.array([g in train_files for g in groups], dtype=bool)
    return mask


def paired_gain_bootstrap(xa_tr, xb_tr, y_tr, xa_te, xb_te, y_te,
                          alpha: float = 1.0, draws: int = 400, seed: int = 0):
    """Bootstrap interval on the *difference* between two models' R^2.

    The two models are scored on the same held-out samples, so the comparison
    is paired and most of the spread in either R^2 is shared between them --
    a test set that happens to contain hard parts is hard for both. Comparing
    two independently-bootstrapped intervals throws that away: it put a
    0.247-wide band around each score and left a 0.014 gain unresolvable.

    Resampling the test set once per draw and taking the difference within the
    draw cancels the shared variation and measures what is actually being
    asked -- whether one model beats the other on the same data.
    """
    rng = np.random.default_rng(seed)
    n = len(y_te)
    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        ra = ridge_r2(xa_tr, y_tr, xa_te[idx], y_te[idx], alpha)
        rb = ridge_r2(xb_tr, y_tr, xb_te[idx], y_te[idx], alpha)
        diffs.append(rb - ra)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


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
    ap.add_argument("--fields-from", choices=("geometry", "solver", "any"),
                    default="geometry",
                    help="which records to probe, by where their `fields` came "
                         "from. Defaults to geometry, and that default is the "
                         "point: on a physics shard the fields ARE the "
                         "simulation result, so Cd is recoverable by "
                         "integrating the pressure field the encoder was "
                         "handed, and thrust follows from the flow solution. "
                         "Probing those measures integration of the input, not "
                         "prediction. 'geometry' restricts to records whose "
                         "fields were built from the mesh, where a solver "
                         "label is genuinely independent of what the model "
                         "sees. Pass 'solver' or 'any' only to demonstrate the "
                         "difference deliberately.")
    ap.add_argument("--real-labels-only", action="store_true",
                    help="restrict to records carrying a stored, solver-"
                         "computed max_stress. Without this the target for raw "
                         "geometry (.stl/.step, ~70%% of the corpus) is not a "
                         "measurement at all: graph_dataset falls back to "
                         "max() of an input field column, so the label is a "
                         "deterministic function of what the encoder is fed. "
                         "Dropping the exactly-1.0 rows removes the saturated "
                         "ones but leaves the rest, and a probe scored on them "
                         "measures whether the encoder can echo its own input.")
    ap.add_argument("--draws", type=int, default=1,
                    help="repeat the whole probe over N independent corpus "
                         "draws and report the spread. The bootstrap resamples "
                         "the held-out set but NOT which corpus rows were "
                         "drawn or how they were split, so it understates the "
                         "real uncertainty: on this corpus R^2 swings 0.640 to "
                         "0.684 across draws, four times the ~0.010 late-"
                         "training effect. A single draw once reported that "
                         "effect as significant when three others called it "
                         "noise. Use >1 before believing any small difference.")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="reference checkpoint to measure gains against, "
                         "instead of random initialisation. Asking whether "
                         "step 1000 beats step 750 by comparing their two "
                         "intervals against random is the same paired-vs-"
                         "independent error this script exists to avoid: "
                         "those intervals overlap, and overlap is not a test. "
                         "Point --baseline at 750 to measure it directly.")
    ap.add_argument("--group-by", choices=("content", "path"), default="content",
                    help="how a mesh is identified when splitting. 'path' was "
                         "the old behaviour and is not enough: 17%% of this "
                         "corpus is byte-identical geometry stored under "
                         "different names -- 71.5%% within body_tube -- so a "
                         "path-grouped split puts the same mesh in train and "
                         "test. Kept selectable so the two can be measured "
                         "against each other on one pool.")
    ap.add_argument("--out", type=Path, default=None,
                    help="write results as JSON. Without this a run exists "
                         "only in whatever captured its stdout, and a machine "
                         "restart erased four completed seed sweeps that way.")
    ap.add_argument("--linear-target", action="store_true",
                    help="probe the raw target instead of its logarithm; the "
                         "default is log because max_stress spans 8.4 decades")
    args = ap.parse_args()
    global GROUP_MODE
    GROUP_MODE = args.group_by
    print(f"grouping meshes by {GROUP_MODE}")
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
    if args.real_labels_only:
        import numpy as _np
        pool = [i for i, r in enumerate(dataset.records)
                if r.path is not None and r.path.suffix.lower() in (".npz", ".pt")]
        keep_pool = []
        for i in pool:
            try:
                with _np.load(dataset.records[i].path, allow_pickle=False) as d:
                    if "max_stress" in d:
                        keep_pool.append(i)
            except Exception:  # noqa: BLE001
                continue
            if len(keep_pool) >= args.samples * 3:
                break
        if len(keep_pool) < 60:
            print(f"only {len(keep_pool)} records carry a stored max_stress; "
                  f"too few to probe")
            return 1
        picks = rng.choice(keep_pool, size=min(args.samples, len(keep_pool)),
                           replace=False)
        print(f"restricted to {len(keep_pool)} records with solver-computed "
              f"max_stress; probing {len(picks)}")
    else:
        prov = getattr(dataset, "field_provenance", None)
        if args.fields_from != "any" and prov is not None:
            pool = [i for i in range(len(dataset))
                    if prov(i) == args.fields_from]
            if len(pool) < 60:
                print(f"only {len(pool)} records with fields from "
                      f"{args.fields_from}; too few to probe")
                return 1
            picks = rng.choice(pool, size=min(args.samples, len(pool)),
                               replace=False)
            print(f"{len(pool)} records with fields from {args.fields_from}; "
                  f"probing {len(picks)}")
        else:
            picks = rng.choice(len(dataset),
                               size=min(args.samples, len(dataset)),
                               replace=False)

    expected_fields = int(cfg["data"]["num_fields"])

    # The random-init baseline is the same for every checkpoint, so it is
    # embedded once. It is also not zero: random projections of a point cloud
    # carry real information about its size and shape.
    if args.baseline is not None:
        ref = JEPAModel.from_config(cfg)
        ref_payload = torch.load(args.baseline, map_location="cpu",
                                 weights_only=False)
        ref.load_state_dict(ref_payload["model"])
        ref_label = f"{args.baseline.name} (step {_step_of(args.baseline)})"
    else:
        torch.manual_seed(0)
        ref = JEPAModel.from_config(cfg)
        ref_label = "random init"
    ref.eval()
    untrained = ref
    x0, y0_raw, groups = collect(untrained, dataset, picks, args.target,
                                 want_groups=True)
    keep, y0 = clean_targets(y0_raw, log_space=not args.linear_target)
    x0 = x0[keep]
    groups = groups[keep]
    dropped = int((~keep).sum())
    if len(x0) < 40:
        print(f"only {len(x0)} usable samples; too few to probe")
        return 1
    tr_mask = group_split(groups, seed=args.seed)
    te_mask = ~tr_mask
    if tr_mask.sum() < 20 or te_mask.sum() < 20:
        print("group split left too few samples on one side")
        return 1
    base_r2, base_lo, base_hi = ridge_r2_bootstrap(
        x0[tr_mask], y0[tr_mask], x0[te_mask], y0[te_mask])

    space = "raw" if args.linear_target else "log10"
    print(f"target {args.target!r} in {space}, {len(x0)} usable samples "
          f"({int(tr_mask.sum())} train / {int(te_mask.sum())} held out, "
          f"split by file so no mesh is on both sides); "
          f"{dropped} placeholder rows dropped\n")
    print(f"{'checkpoint':>14} {'step':>6} {'R^2':>8} "
          f"{'95% CI on the gain':>20} {'gain':>10}  verdict")
    print(f"{ref_label:>14} {_step_of(args.baseline) if args.baseline else 0:>6} {base_r2:>8.4f} "
          f"{'(reference)':>20} {0.0:>+10.4f}")

    rows = []
    embedded = []
    spread: dict[int, dict] = {}
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
        embedded.append((int(payload.get("step") or _step_of(path)), x))
        r2 = ridge_r2(x[tr_mask], y0[tr_mask], x[te_mask], y0[te_mask])
        gain, glo, ghi = paired_gain_bootstrap(
            x0[tr_mask], x[tr_mask], y0[tr_mask],
            x0[te_mask], x[te_mask], y0[te_mask])
        step = int(payload.get("step") or _step_of(path))
        rows.append((step, r2, glo, ghi, gain))
        verdict = "significant" if glo > 0 else ("negative" if ghi < 0 else "noise")
        print(f"{path.name:>14} {step:>6} {r2:>8.4f} "
              f"{f'[{glo:+.4f}, {ghi:+.4f}]':>20} {gain:>+10.4f}  {verdict}")

    if args.draws > 1 and embedded:
        # Repeat the split, not the embedding. Which meshes land in the held-out
        # half is the dominant source of run-to-run spread here, and re-running
        # the whole probe per draw meant reloading a 430 MB graph and embedding
        # every sample again -- 9000 samples x 3 draws did not finish inside 90
        # minutes. The embeddings do not depend on the split, so they are
        # computed once and only the split is redrawn.
        print(f"\n  {args.draws} group splits over one embedded pool "
              f"({len(x0)} samples, {len(np.unique(groups))} distinct meshes):\n")
        print(f"{'step':>8} {'mean gain':>11} {'min':>9} {'max':>9}  sign")
        for step, xt in embedded:
            gains = []
            for d in range(args.draws):
                m = group_split(groups, seed=d)
                if m.sum() < 20 or (~m).sum() < 20:
                    continue
                g, _lo, _hi = paired_gain_bootstrap(
                    x0[m], xt[m], y0[m], x0[~m], xt[~m], y0[~m], draws=200)
                gains.append(g)
            if not gains:
                continue
            sign = ("all +" if all(v > 0 for v in gains)
                    else "all -" if all(v < 0 for v in gains) else "MIXED")
            spread[step] = {"mean": sum(gains) / len(gains), "min": min(gains),
                            "max": max(gains), "sign": sign, "n": len(gains)}
            print(f"{step:>8} {sum(gains)/len(gains):>+11.4f} {min(gains):>+9.4f} "
                  f"{max(gains):>+9.4f}  {sign}")

    if len(rows) >= 2:
        ref_name = "random projections" if args.baseline is None else ref_label
        print(f"\n  Intervals are on the paired difference against "
              f"{ref_name}, not on each R^2 separately -- both models see the "
              f"same held-out samples, so the shared variation cancels.")
        print(f"\n  across training: {rows[0][1]:+.4f} -> {rows[-1][1]:+.4f} "
              f"({rows[-1][1] - rows[0][1]:+.4f})")

        # The verdict follows the intervals, not a threshold on raw R^2. An
        # earlier version compared R^2 differences against a hardcoded 0.02 and
        # would announce a finding the intervals did not support -- the same
        # mistake, in the summary line, as reading two overlapping intervals as
        # a test.
        wins = [r for r in rows if r[2] > 0.0]      # lower bound above zero
        losses = [r for r in rows if r[3] < 0.0]    # upper bound below zero
        if wins:
            best = max(wins, key=lambda r: r[2])
            print(f"  beats {ref_name} at step {best[0]} and after: gain "
                  f"lower bound {best[2]:+.4f}")
        if losses:
            print(f"  worse than {ref_name} at steps "
                  f"{', '.join(str(r[0]) for r in losses)}")
        if not wins and not losses:
            print(f"  no checkpoint separates from {ref_name}: every interval "
                  f"straddles zero, so this measures a plateau, not a gain")

    if args.out is not None:
        import json as _json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_json.dumps({
            "target": args.target,
            "seed": args.seed,
            "samples_requested": args.samples,
            "usable": int(len(x0)),
            "train": int(tr_mask.sum()),
            "held_out": int(te_mask.sum()),
            "distinct_meshes": int(len(np.unique(groups))),
            "group_by": args.group_by,
            "draws": args.draws,
            "baseline_r2": float(base_r2),
            "checkpoints": [
                {"step": st, "r2": float(r2), "gain": float(g),
                 "ci": [float(lo), float(hi)],
                 "verdict": ("significant" if lo > 0
                             else "negative" if hi < 0 else "noise")}
                for st, r2, lo, hi, g in rows],
            "split_spread": {str(k): v for k, v in spread.items()},
        }, indent=1))
        print(f"\nwritten: {args.out}")

    return 0


def _step_of(path: Path) -> int:
    """Step number from a checkpoint filename, for ordering."""
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 10**9


if __name__ == "__main__":
    raise SystemExit(main())
