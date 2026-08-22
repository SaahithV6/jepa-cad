"""The conditioning contract, checked end to end rather than assumed.

Adding a conditioning slot is three separate edits in three files: the slot goes
in CONDITIONING_QUANTITIES, the value gets computed in the corpus record, and
the value has to appear in the *shard metrics* under exactly the slot's name --
because the graph ingest seeds conditioning from a node's own properties by name
lookup. Miss the third and everything still runs, the corpus still contains the
value, and the model conditions on a zero.

That is precisely what nose_wave_factor did: computed correctly, validated
against theory, written into every corpus record, and invisible to the model.
These tests fail loudly on that rather than letting it pass silently.
"""

import json
from pathlib import Path

import pytest

from cadflow.physics_targets import CONDITIONING_QUANTITIES, clamp_conditioning

MANIFEST = Path("artifacts/physics_shards/traj_manifest.jsonl")

#: Slots the propulsion corpus is responsible for supplying. Not every slot --
#: most describe parts, not trajectories -- but each of these is a value this
#: corpus computes, so each must actually arrive.
TRAJECTORY_SLOTS = (
    "payload_kg",
    "delta_v_ms",
    "apogee_km",
    "downrange_km",
    "burn_time_s",
    "max_dynamic_pressure_kpa",
    "mass_fraction",
    "fineness_ratio",
    "nose_wave_factor",
    # chemistry and thermal, computed for the specific design rather than
    # sampled from a family range
    "mixture_ratio",
    "throat_heat_flux_MWm2",
    "wall_temp_max_K",
)


def test_conditioning_names_are_unique():
    """The tuple order is the contract, so a duplicate name silently shadows."""
    names = [n for n, _ in CONDITIONING_QUANTITIES]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]


def test_every_slot_has_a_positive_scale():
    for name, scale in CONDITIONING_QUANTITIES:
        assert scale > 0.0, name


def test_clamp_keeps_values_in_range():
    for value in (-1e9, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 1e9):
        assert -1.0 <= clamp_conditioning(value) <= 1.0


def test_nose_shape_has_a_conditioning_slot():
    """Shape must be representable, or the geometry work cannot be learned."""
    names = [n for n, _ in CONDITIONING_QUANTITIES]
    assert "nose_wave_factor" in names
    assert "fineness_ratio" in names


@pytest.mark.skipif(not MANIFEST.exists(), reason="no trajectory corpus generated")
def test_shard_metrics_carry_the_slots_the_corpus_claims_to_fill():
    """The step that was missing.

    A value present in the corpus record but absent from the shard metrics never
    reaches conditioning, because the ingest looks up slot names in the node's
    properties and finds nothing.
    """
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    assert rows, "manifest is empty"
    slot_names = {n for n, _ in CONDITIONING_QUANTITIES}

    metrics = rows[0]["metrics"]
    missing = [s for s in TRAJECTORY_SLOTS if s not in metrics]
    assert not missing, f"slots computed but never delivered to the model: {missing}"

    # and every metric that matches a slot name must be a usable number
    for name, value in metrics.items():
        if name in slot_names:
            assert isinstance(value, (int, float)), (name, value)


@pytest.mark.skipif(not MANIFEST.exists(), reason="no trajectory corpus generated")
def test_nose_wave_factor_actually_varies_across_the_corpus():
    """A constant conditioning signal carries no information.

    If every record came out at 1.0 the slot would be filled, well-formed, and
    still useless -- which is what would happen if shape sampling regressed to a
    single family.
    """
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    vals = [r["metrics"].get("nose_wave_factor") for r in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    assert len(vals) > 10, "not enough records to judge"
    assert min(vals) < 0.99, f"no non-ogive noses sampled: min {min(vals)}"
    assert max(vals) == pytest.approx(1.0, abs=1e-6), f"max {max(vals)}"
    # the validated range for tangent noses
    assert min(vals) > 0.80, f"outside the validated band: min {min(vals)}"


@pytest.mark.skipif(not MANIFEST.exists(), reason="no trajectory corpus generated")
def test_drag_responds_to_nose_shape_in_the_corpus():
    """The whole point: a better nose must show up as less drag."""
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    pairs = [
        (r["metrics"]["nose_wave_factor"], r["metrics"]["cd"])
        for r in rows
        if "nose_wave_factor" in r["metrics"] and "cd" in r["metrics"]
    ]
    assert len(pairs) > 50
    better = [cd for f, cd in pairs if f < 0.99]
    ogive = [cd for f, cd in pairs if f >= 0.99]
    assert better and ogive
    assert sum(better) / len(better) < sum(ogive) / len(ogive)


@pytest.mark.skipif(not MANIFEST.exists(), reason="no trajectory corpus generated")
def test_mixture_ratio_varies_and_drives_the_chamber():
    """The O/F axis the corpus did not have.

    One row per propellant meant the central trade in a liquid engine was
    absent from the training data entirely -- a constant carries no information
    however correctly it is computed.
    """
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    ratios = [r["metrics"].get("mixture_ratio") for r in rows]
    ratios = [x for x in ratios if isinstance(x, (int, float))]
    assert len(ratios) > 20
    assert max(ratios) > 1.4 * min(ratios), (min(ratios), max(ratios))


@pytest.mark.skipif(not MANIFEST.exists(), reason="no trajectory corpus generated")
def test_thermal_is_physical_across_the_corpus():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    fluxes = [r["metrics"].get("throat_heat_flux_MWm2") for r in rows]
    fluxes = [x for x in fluxes if isinstance(x, (int, float))]
    assert len(fluxes) > 20
    # real throats run from a few to a couple of hundred MW/m^2
    assert all(0.5 < f < 500.0 for f in fluxes), (min(fluxes), max(fluxes))
    assert max(fluxes) > 2.0 * min(fluxes), "heat flux carries no variation"


def test_the_configured_metadata_width_matches_the_dataset():
    """A hardcoded width that has to track a computed one.

    graph_metadata_dim is written out in the configs while the dataset derives
    the same number from len(CONDITIONING_QUANTITIES). Adding the 42nd
    conditioning slot moved the real width to 159 and left the configs at 158,
    and the way that surfaced was a matrix shape mismatch forty seconds into a
    training run -- "mat1 and mat2 shapes cannot be multiplied (8x159 and
    158x192)". This is the cheap version of that discovery.
    """
    from data.graph_dataset import GRAPH_METADATA_DIM
    from utils.config import load_yaml_with_family

    for family in (None, "space", "space_cpu", "space_24b"):
        cfg = (load_yaml_with_family("configs/base.yaml", family=family)
               if family else load_yaml_with_family("configs/base.yaml"))
        configured = cfg["data"].get("graph_metadata_dim")
        if configured is None:
            continue
        assert configured == GRAPH_METADATA_DIM, (
            f"family {family}: config says {configured}, dataset produces "
            f"{GRAPH_METADATA_DIM}")
