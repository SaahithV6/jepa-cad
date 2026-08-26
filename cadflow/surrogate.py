"""A learned screening surrogate, with its own error bar attached.

The plan says JEPA should be "not just a retrieval sidecar" but the core of the
modelling loop. It is currently a sidecar: `autodesign`, `design_loop`,
`planner`, `flywheel_loop` and `promotion` contain no reference to the trained
model between them. The flywheel promotes *checkpoints* by probe score; nothing
feeds the learned representation back into design.

This is the smallest honest step toward closing that: the encoder ranks
candidate geometries so the expensive solver runs on the promising ones. It is
deliberately not a replacement for CalculiX, and the numbers say why. Probed
under a group-aware split on consensus labels, the representation predicts
log10 max stress at R^2 = 0.77 against a random-projection baseline of 0.72 --
the learned part is about +0.057, reproducible across six splits, and the
absolute accuracy is nowhere near a solver's. Residual spread is reported in
decades because that is the unit the error actually lives in.

So the contract is:

* `screen` ranks candidates and says how much to trust the ranking;
* nothing here returns a number that may be reported as a stress result;
* a checkpoint whose gain over random projections does not clear zero is
  refused, because a surrogate built on it is a random projection with extra
  steps -- and this repository has already published one finding that was
  exactly that.

Every prediction is recorded with the solver outcome that follows it, when one
follows, so the flywheel can measure whether screening actually helped rather
than assuming it did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Ridge penalty, matching the probe so the reported accuracy is the accuracy
#: this surrogate actually has.
RIDGE_ALPHA = 1.0

#: A checkpoint must beat random projections by at least this much, measured as
#: the lower bound of the paired interval, before it may be used for screening.
MIN_GAIN_LOWER_BOUND = 0.0


@dataclass
class SurrogateAccuracy:
    """What the surrogate is worth, measured on held-out meshes."""
    r2: float
    gain_over_random: float
    gain_lower_bound: float
    residual_p50_decades: float
    residual_p90_decades: float
    n_train: int
    n_test: int
    n_meshes: int

    @property
    def trustworthy(self) -> bool:
        return self.gain_lower_bound > MIN_GAIN_LOWER_BOUND

    def summary(self) -> str:
        return (f"R^2 {self.r2:.3f} (random projections {self.r2 - self.gain_over_random:.3f}, "
                f"gain {self.gain_over_random:+.3f} [lower bound {self.gain_lower_bound:+.3f}]); "
                f"median error {self.residual_p50_decades:.2f} decades, "
                f"p90 {self.residual_p90_decades:.2f}; "
                f"{self.n_train}+{self.n_test} samples over {self.n_meshes} meshes")


@dataclass
class ScreeningRecord:
    """One prediction, and the solver result that later judged it."""
    candidate_id: str
    predicted_log10_mpa: float
    rank: int
    solver_log10_mpa: float | None = None
    promoted: bool = False
    context: dict[str, Any] = field(default_factory=dict)


class StressSurrogate:
    """Ranks candidate geometries by predicted peak stress.

    Fitted the same way the probe measures: ridge on pooled embeddings, split
    by mesh so no geometry appears on both sides. Fitting it any other way
    would make the accuracy it reports a different quantity from the accuracy
    it has -- a random-record split inflates R^2 here by roughly 0.09, which is
    most of the effect being claimed.
    """

    def __init__(self, model, weights: np.ndarray, bias: float,
                 accuracy: SurrogateAccuracy):
        self._model = model
        self._w = weights
        self._b = float(bias)
        self.accuracy = accuracy
        self.records: list[ScreeningRecord] = []

    # -- construction --------------------------------------------------------

    @classmethod
    def fit(cls, checkpoint: str | Path, dataset, *, n_samples: int = 1500,
            target: str = "max_stress", seed: int = 0) -> "StressSurrogate":
        import torch

        from models.jepa import JEPAModel

        payload = torch.load(Path(checkpoint), map_location="cpu",
                             weights_only=False)
        cfg = payload.get("config")
        if cfg is None:
            raise ValueError(f"{checkpoint} carries no config to rebuild from")
        model = JEPAModel.from_config(cfg)
        model.load_state_dict(payload["model"])
        model.eval()

        untrained = JEPAModel.from_config(cfg)
        untrained.eval()

        x, y, groups = _collect(model, dataset, n_samples, target, seed)
        if len(x) < 80:
            raise ValueError(f"only {len(x)} usable samples; refusing to fit "
                             f"a surrogate on that")
        x0, _y0, _g0 = _collect(untrained, dataset, n_samples, target, seed)

        train = _group_split(groups, seed=seed)
        test = ~train
        w, b = _ridge(x[train], y[train])
        pred = x[test] @ w + b
        resid = np.abs(pred - y[test])

        r2 = _r2(y[test], pred)
        w0, b0 = _ridge(x0[train], y[train])
        r2_rand = _r2(y[test], x0[test] @ w0 + b0)
        gain, lower = _paired_gain(x0[train], x[train], y[train],
                                   x0[test], x[test], y[test], seed=seed)

        acc = SurrogateAccuracy(
            r2=float(r2), gain_over_random=float(r2 - r2_rand),
            gain_lower_bound=float(lower),
            residual_p50_decades=float(np.percentile(resid, 50)),
            residual_p90_decades=float(np.percentile(resid, 90)),
            n_train=int(train.sum()), n_test=int(test.sum()),
            n_meshes=int(len(np.unique(groups))))
        return cls(model, w, b, acc)

    # -- use -----------------------------------------------------------------

    def predict_log10_mpa(self, points, fields) -> float:
        """Predicted log10 peak stress. Screening only -- see the class note."""
        import torch

        vec = _embed(self._model,
                     torch.as_tensor(points).unsqueeze(0),
                     torch.as_tensor(fields).unsqueeze(0))[0]
        return float(vec @ self._w + self._b)

    def screen(self, candidates: list[tuple[str, Any, Any]],
               keep: int = 3) -> list[ScreeningRecord]:
        """Rank candidates cheapest-stress-first and keep the best few.

        Refuses outright when the checkpoint did not beat random projections:
        ranking by a feature set that carries no signal is worse than not
        ranking, because it looks like a decision.
        """
        if not self.accuracy.trustworthy:
            raise RuntimeError(
                f"this checkpoint does not beat random projections "
                f"({self.accuracy.summary()}); screening with it would be a "
                f"random shuffle presented as a judgement")
        scored = []
        for cid, points, fields in candidates:
            scored.append((cid, self.predict_log10_mpa(points, fields)))
        scored.sort(key=lambda t: t[1])
        out = []
        for rank, (cid, value) in enumerate(scored[:max(1, keep)]):
            rec = ScreeningRecord(candidate_id=cid, predicted_log10_mpa=value,
                                  rank=rank)
            self.records.append(rec)
            out.append(rec)
        return out

    def screen_specs(self, candidates: list[tuple[str, dict]], *,
                     backend=None, workdir=None, num_points: int = 1024,
                     num_fields: int = 6, keep: int = 1) -> list[ScreeningRecord]:
        """Rank geometry specs without running a solver.

        A candidate costs a CAD build plus an STL export plus a parse -- call it
        seconds. The solver it replaces costs minutes. That ratio is the whole
        argument for screening, and it only pays if the ranking is better than
        chance, which is why `screen` refuses when the checkpoint is not.

        Specs that fail to build are dropped rather than ranked last: a
        candidate that could not be made is not a bad candidate, it is not a
        candidate, and scoring it would put a number on a failure.
        """
        import tempfile

        from data.graph_dataset import _load_sample_arrays

        from cadflow.backends import build_from_spec, get_backend

        b = backend or get_backend(prefer_real=True)
        root = Path(workdir or tempfile.mkdtemp(prefix="screen-"))
        root.mkdir(parents=True, exist_ok=True)

        built: list[tuple[str, Any, Any]] = []
        self.skipped: list[tuple[str, str]] = []
        for cid, spec in candidates:
            try:
                shape = build_from_spec(dict(spec), backend=b)
                stl = b.export_stl(shape, root / f"{cid}.stl")
                points, fields, _ = _load_sample_arrays(
                    Path(stl), num_points=num_points, num_fields=num_fields)
            except Exception as exc:  # noqa: BLE001
                self.skipped.append((cid, str(exc)[:120]))
                continue
            built.append((cid, points, fields))
        if not built:
            raise RuntimeError(
                f"no candidate could be built ({len(self.skipped)} failed); "
                f"nothing to screen")
        return self.screen(built, keep=keep)

    def record_solver_outcome(self, candidate_id: str,
                              solver_max_stress_mpa: float,
                              promoted: bool = False) -> None:
        """Attach the real result to the prediction that preceded it."""
        value = float(np.log10(max(float(solver_max_stress_mpa), 1e-9)))
        for rec in self.records:
            if rec.candidate_id == candidate_id and rec.solver_log10_mpa is None:
                rec.solver_log10_mpa = value
                rec.promoted = bool(promoted)
                return

    def screening_error(self) -> dict[str, float]:
        """How the predictions did against the solver, once it has spoken.

        The plan asks every JEPA cycle to record "the context window, target
        prediction, downstream test outcome, and whether the result was
        promoted". This is the part that closes the loop: without it the
        surrogate's usefulness is an assumption rather than a measurement.
        """
        judged = [r for r in self.records if r.solver_log10_mpa is not None]
        if not judged:
            return {"judged": 0}
        err = np.array([abs(r.predicted_log10_mpa - r.solver_log10_mpa)
                        for r in judged])
        return {
            "judged": float(len(judged)),
            "mean_abs_decades": float(err.mean()),
            "p90_abs_decades": float(np.percentile(err, 90)),
            "worse_than_fit_p90": float((err > self.accuracy.residual_p90_decades).mean()),
        }

    def dump_records(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "accuracy": self.accuracy.__dict__,
            "screening_error": self.screening_error(),
            "records": [r.__dict__ for r in self.records],
        }, indent=2))
        return out


# --- internals --------------------------------------------------------------

def _embed(model, points, fields) -> np.ndarray:
    """Pooled context-encoder embedding, the same vector the probe scores.

    Kept identical to `scripts/probe_representation.embed` on purpose: a
    surrogate fitted on a different vector from the one the reported accuracy
    was measured on would carry an error bar belonging to something else.
    """
    import torch

    with torch.no_grad():
        enc = model.context_encoder(points, fields)
        if isinstance(enc, dict):
            enc = enc.get("pooled_embedding")
            if enc is None:
                raise KeyError("encoder returned no pooled_embedding")
        if enc.dim() == 3:
            enc = enc.mean(dim=1)
    return enc.numpy()


def _collect(model, dataset, n_samples, target, seed):
    import torch

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=min(n_samples, len(dataset)),
                       replace=False)
    physics_for = getattr(dataset, "physics_for", None)
    xs, ys, groups, pending = [], [], [], []

    def flush():
        if not pending:
            return
        pts = torch.stack([p for p, _f in pending])
        fld = torch.stack([f for _p, f in pending])
        xs.extend(_embed(model, pts, fld))
        pending.clear()

    for i in picks:
        try:
            sample = dataset[int(i)]
        except Exception:  # noqa: BLE001
            continue
        if target in sample:
            value = float(sample[target])
        elif physics_for is not None:
            got = physics_for(int(i)).get("max_stress_mpa")
            if got is None:
                continue
            value = float(got)
        else:
            continue
        if not np.isfinite(value) or value <= 0.0:
            continue
        pending.append((sample["points"], sample["fields"]))
        ys.append(np.log10(value))
        rec = dataset.records[int(i)] if hasattr(dataset, "records") else None
        groups.append(str(getattr(rec, "path", i)))
        if len(pending) >= 32:
            flush()
    flush()
    return (np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(groups, dtype=object))


def _group_split(groups, frac_train: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    train_files = set(uniq[:int(frac_train * len(uniq))].tolist())
    return np.array([g in train_files for g in groups], dtype=bool)


def _ridge(x, y, alpha: float = RIDGE_ALPHA):
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc + alpha * np.eye(xc.shape[1])
    w = np.linalg.solve(gram, xc.T @ yc)
    return w, ym - float(xm @ w)


def _r2(y_true, y_pred) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _paired_gain(xa_tr, xb_tr, y_tr, xa_te, xb_te, y_te, draws: int = 200,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    wa, ba = _ridge(xa_tr, y_tr)
    wb, bb = _ridge(xb_tr, y_tr)
    diffs = []
    n = len(y_te)
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        diffs.append(_r2(y_te[idx], xb_te[idx] @ wb + bb)
                     - _r2(y_te[idx], xa_te[idx] @ wa + ba))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5))
