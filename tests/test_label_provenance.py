"""Labels must not be computed from the inputs the model is shown.

This file exists because the whole suite passed while `max_stress` was, for
about 70% of sampled records, `max()` of a column of the very `fields` array
handed to the encoder. Nothing was corrupt and nothing crashed: the loader
returned a plausible float under a name that means "a solver measured this",
and 399 tests had no opinion about where it came from.

The cost was a reported finding. A linear probe on that target said the trained
encoder beat random weights by +0.08 R^2 -- "the representation carries physics
random weights do not" -- when a large share of what it measured was the model
echoing its own input, which a trained encoder does better than a random one.
On solver-computed labels the effect is real but smaller: +0.067 (+0.044 to
+0.091 across four independent corpus draws), and +0.023 restricted to the CFD
shards alone.

Statistics could not have caught this. The bootstrap intervals were correct
about sampling noise and silent about meaning; tightening them only made a
leaked number look more certain. The check has to be on provenance, so that is
what these tests assert.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _field_column_maxima(fields) -> list[float]:
    arr = np.asarray(fields)
    if arr.size == 0:
        return []
    return [float(arr[:, c].max()) for c in range(arr.shape[-1])]


def test_a_measured_label_never_equals_a_max_of_its_own_inputs():
    """The signature of the bug, stated directly.

    A real von Mises maximum coming from a solver has no reason to land exactly
    on the maximum of an input channel. When it does, the label was derived
    from the input rather than measured.
    """
    fields = np.array([[1.0, 5.0, 250.0], [2.0, 6.0, 300.0]], dtype=np.float32)
    sample = {
        "fields": torch.from_numpy(fields),
        "max_stress": torch.tensor(412.7),      # a genuine solver result
    }
    maxima = _field_column_maxima(sample["fields"])
    assert not any(abs(float(sample["max_stress"]) - m) < 1e-6 for m in maxima)


def test_derived_stress_is_named_apart_from_measured_stress():
    """Raw geometry has no solver result, so it must not produce `max_stress`.

    The loader is welcome to compute a geometric proxy -- it just may not file
    it under the name that means "measured", because every downstream consumer
    reads that name and cannot tell the difference.
    """
    import data.dataset as dataset_mod
    import data.graph_dataset as graph_mod

    for mod in (dataset_mod, graph_mod):
        src = open(mod.__file__).read()
        # the fabricating expression may exist; it may not be assigned to the
        # measured key on the same line
        for line in src.splitlines():
            stripped = line.strip()
            if "stress_col" in stripped and "].max()" in stripped:
                assert '"max_stress"' not in stripped, (
                    f"{mod.__name__}: a field-derived value is being stored as "
                    f"a measured label: {stripped}")


def test_raw_geometry_yields_no_measured_stress():
    """The path that produced the leak, exercised end to end."""
    import pathlib

    from data.graph_dataset import _load_sample_arrays
    from data.parsers import ParseError

    # Take the first mesh that actually parses. Roughly 7% of this corpus is
    # empty STLs, and grabbing whichever file rglob yields first lands on one
    # often enough to make the test fail for a reason it is not testing.
    checked = 0
    for stl in pathlib.Path("data").rglob("*.stl"):
        try:
            _points, _fields, max_stress = _load_sample_arrays(
                stl, num_points=256, num_fields=6)
        except ParseError:
            continue
        checked += 1
        assert max_stress is None, (
            f"{stl}: raw geometry carries no solver result, so "
            f"_load_sample_arrays must return None rather than a value "
            f"derived from the fields it just built")
        if checked >= 5:
            break
    if checked == 0:
        pytest.skip("no readable raw geometry in the corpus")


def test_collate_survives_a_batch_where_only_some_samples_are_labelled():
    """The failure mode the split introduced, and the suite did not have.

    Splitting `max_stress` from `max_stress_proxy` made both keys optional, so
    a real batch mixes samples that carry one, the other, or neither. Collation
    tested only `batch[0]` and raised KeyError on the first mixed batch --
    training died at step 0 while every test passed, because the tests build
    uniform batches and real data does not.
    """
    from data.transforms import MaskingConfig, collate_masked_batch

    batch = [
        {"points": torch.rand(64, 3), "fields": torch.zeros(64, 6),
         "max_stress": torch.tensor(210.0)},
        {"points": torch.rand(64, 3), "fields": torch.zeros(64, 6),
         "max_stress_proxy": torch.tensor(1.0)},
        {"points": torch.rand(64, 3), "fields": torch.zeros(64, 6)},
    ]
    out = collate_masked_batch(batch, MaskingConfig())
    assert "max_stress" not in out, (
        "a target present for only some rows must be dropped, not stacked -- "
        "a partial stack silently misaligns with the samples it labels")
    assert "max_stress_proxy" not in out
    assert out["points"].shape[0] == 3


def test_collate_passes_the_proxy_through_under_its_own_name():
    """Splitting the key is only safe if batching knows about both."""
    from data.transforms import MaskingConfig, collate_masked_batch

    batch = [
        {"points": torch.rand(64, 3), "fields": torch.zeros(64, 6),
         "max_stress_proxy": torch.tensor(1.0)},
        {"points": torch.rand(64, 3), "fields": torch.zeros(64, 6),
         "max_stress_proxy": torch.tensor(2.0)},
    ]
    out = collate_masked_batch(batch, MaskingConfig())
    assert "max_stress_proxy" in out
    assert "max_stress" not in out, (
        "a proxy must never be promoted into the measured key by collation")
    assert out["max_stress_proxy"].tolist() == [1.0, 2.0]
