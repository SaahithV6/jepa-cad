"""A design must survive its own thermal environment.

Every packet this project has produced ran with the skin-temperature gate
switched off, and nothing noticed because the gate reported no violations --
which is what a disabled check and a satisfied check look like from outside.

The cause was two intentions sharing one sentinel. DEFAULT_LIMITS carried
``skin_temp_k = _NO_GATE_K``, which was meant as "impose no extra cap, the
material's own rating governs". The gate read the same value as "the caller has
switched this off". plan_and_verify calls autodesign with no limits, so the
default path took the disabled branch every time.

What it let through: aluminium 6061-T6, rated to 420 K, on a skin at 895 K,
returned as converged with an empty violation list. Aluminium is nearly molten
there. And because Inconel is three times its density, that selection then
decided whether the vehicle could afford its own tankage -- so a disabled
thermal gate was quietly setting a structural verdict.

The invariant below is the one that should always have been asserted, and it is
deliberately about the *result* rather than the mechanism: whatever path the
loop takes, the material it returns has to survive the temperature it reports.
"""

import pytest

from cadflow.autodesign import (
    DEFAULT_LIMITS, Knobs, _NO_GATE_K, _material, autodesign, evaluate)

PAYLOAD, APOGEE = 25.0, 4000.0


def test_the_default_imposes_no_cap_rather_than_disabling_the_gate():
    """The distinction the sentinel collapsed.

    None means "the material governs". _NO_GATE_K means "a caller is testing
    something unrelated to heating and wants this off". Those are different and
    the default must be the first.
    """
    assert DEFAULT_LIMITS["skin_temp_k"] is None


def test_a_returned_design_survives_its_own_skin_temperature():
    """The invariant. Whatever path the loop takes, the alloy must hold.

    This is asserted on the result rather than on the gate, so it stays true
    however the material comes to be chosen -- by the upgrade path, by
    right-sizing, or by never being challenged at all, which is how aluminium
    reached 895 K.
    """
    res = autodesign(PAYLOAD, APOGEE, max_iters=12)
    ev, knobs = res["evaluation"], res["knobs"]
    if not res["converged"] or getattr(ev, "tps", None):
        pytest.skip("unconverged, or protected by TPS rather than by the alloy")
    limit = _material(knobs.skin_material).max_service_temp_k
    assert ev.skin_temp_k <= limit, (
        f"{knobs.skin_material} is rated to {limit:.0f} K and this design "
        f"reaches {ev.skin_temp_k:.1f} K")


def test_a_hot_vehicle_on_aluminium_raises_a_violation():
    """Directly: hold the material and check the gate fires.

    Pinning the knob to aluminium on a mission that heats the skin well past
    420 K has to produce a violation. With the gate disabled this returned an
    empty list, which is why nothing downstream ever objected.
    """
    ev = evaluate(PAYLOAD, APOGEE, Knobs(skin_material="al-6061-t6"), None)
    if ev.skin_temp_k <= 420.0:
        pytest.skip("this mission does not heat the skin past aluminium")
    assert ev.violations, (
        f"skin at {ev.skin_temp_k:.0f} K on an alloy rated to 420 K raised "
        f"nothing")
    assert any("skin" in v.quantity.lower() for v in ev.violations)


def test_a_caller_can_still_switch_the_gate_off():
    """The capability the sentinel existed for, kept.

    Callers testing something unrelated to heating pass _NO_GATE_K, and that
    must still work -- the fix separates two meanings rather than removing one.
    """
    ev = evaluate(PAYLOAD, APOGEE, Knobs(skin_material="al-6061-t6"),
                  {"skin_temp_k": _NO_GATE_K})
    assert not any("skin" in v.quantity.lower() for v in ev.violations)


def test_a_caller_can_still_impose_something_stricter():
    """The other capability: a cap below what the alloy would allow.

    Inconel is rated to 920 K; a caller demanding 300 K should be obeyed.
    """
    ev = evaluate(PAYLOAD, APOGEE, Knobs(skin_material="inconel-718"),
                  {"skin_temp_k": 300.0})
    assert ev.skin_limit_k == pytest.approx(300.0)
    if ev.skin_temp_k > 300.0:
        assert any("skin" in v.quantity.lower() for v in ev.violations)
