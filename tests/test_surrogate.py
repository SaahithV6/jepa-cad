"""The surrogate must know what it is worth, and refuse when it is worth nothing.

A screening surrogate is the first thing in this repository that lets the
learned model influence a design decision. That makes its error bar part of the
product rather than a footnote: a ranking presented with confidence is a
judgement, and a judgement made from features that carry no signal is worse
than no ranking at all, because it looks like one.

These tests pin the two properties that keep it honest -- it refuses to screen
on a checkpoint that does not beat random projections, and it scores itself on
meshes it did not fit on.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cadflow.surrogate import (  # noqa: E402
    ScreeningRecord,
    StressSurrogate,
    SurrogateAccuracy,
    _group_split,
    _r2,
    _ridge,
)


def _accuracy(gain_lower_bound: float) -> SurrogateAccuracy:
    return SurrogateAccuracy(
        r2=0.77, gain_over_random=0.057, gain_lower_bound=gain_lower_bound,
        residual_p50_decades=0.3, residual_p90_decades=0.8,
        n_train=1000, n_test=400, n_meshes=900)


def test_a_checkpoint_that_does_not_beat_random_is_refused():
    """The failure this repository already made once, in refusable form.

    A +0.08 gain was reported here from features that were partly echoing their
    own input. Screening on such a checkpoint reorders candidates by noise and
    hands the result to the design loop as a decision.
    """
    sur = StressSurrogate.__new__(StressSurrogate)
    sur.accuracy = _accuracy(gain_lower_bound=-0.01)
    sur.records = []
    assert not sur.accuracy.trustworthy
    with pytest.raises(RuntimeError, match="random projections"):
        sur.screen([("a", None, None)])


def test_a_checkpoint_that_clears_zero_is_allowed():
    sur = StressSurrogate.__new__(StressSurrogate)
    sur.accuracy = _accuracy(gain_lower_bound=+0.033)
    assert sur.accuracy.trustworthy


def test_the_split_never_puts_one_mesh_on_both_sides():
    """Fitting on a random record split inflates R^2 by about 0.09 here.

    87% of the meshes in this corpus appear in more than one record, so a split
    taken over records fits and scores the same geometry -- and the surrogate
    would then advertise an accuracy it does not have.
    """
    groups = np.array([f"mesh_{i // 3}.stl" for i in range(300)], dtype=object)
    train = _group_split(groups, seed=0)
    assert train.sum() > 0 and (~train).sum() > 0
    assert not (set(groups[train]) & set(groups[~train]))


def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 12))
    truth = rng.normal(size=12)
    y = x @ truth + 0.7
    w, b = _ridge(x, y, alpha=1e-6)
    assert _r2(y, x @ w + b) > 0.999
    assert b == pytest.approx(0.7, abs=1e-3)


def test_screening_error_is_zero_until_a_solver_has_spoken():
    """Predictions are not evidence until something judged them."""
    sur = StressSurrogate.__new__(StressSurrogate)
    sur.accuracy = _accuracy(gain_lower_bound=+0.033)
    sur.records = [ScreeningRecord("part-a", 2.3, 0)]
    assert sur.screening_error() == {"judged": 0}

    sur.record_solver_outcome("part-a", 10 ** 2.5)
    err = sur.screening_error()
    assert err["judged"] == 1
    assert err["mean_abs_decades"] == pytest.approx(0.2, abs=1e-6)


def test_a_solver_outcome_attaches_to_one_prediction_only():
    """Two candidates must not share a single verdict."""
    sur = StressSurrogate.__new__(StressSurrogate)
    sur.accuracy = _accuracy(gain_lower_bound=+0.033)
    sur.records = [ScreeningRecord("p", 1.0, 0), ScreeningRecord("p", 2.0, 1)]
    sur.record_solver_outcome("p", 100.0)
    attached = [r for r in sur.records if r.solver_log10_mpa is not None]
    assert len(attached) == 1


def test_accuracy_summary_states_the_baseline_not_just_the_score():
    """0.77 alone reads as good; 0.77 against 0.72 reads as what it is."""
    text = _accuracy(0.033).summary()
    assert "0.77" in text and "0.71" in text  # r2 minus gain
    assert "decades" in text
