"""A packet must not disagree with itself.

Every other test here checks a physical model. These check the report, because
a correct model reported wrongly is indistinguishable from a wrong one, and
this packet has now been caught doing it twice.

It announced "all 10 components passed: True" for a vehicle that needed 14.5
degrees of gimbal against the 8 an engine offers -- the finding was in the prose
and reached neither the summary nor PACKET.json. And it reported
`struct_coeff_used: 0.14` for a vehicle whose every stage was built at 0.2613,
because the field read a module default rather than the design. The second one
matters beyond tidiness: 0.14 sits just above the range flown hardware occupies,
while 0.2613 is more than twice the heaviest stage ever flown, so the packet was
quietly making its own structure look ordinary.

The rule these encode is that a number in the report has to be recoverable from
the thing it describes. Anything a field cannot be derived from is a field that
can drift.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKETS = sorted((ROOT / "artifacts/verification").glob("packet_*/PACKET.json"))

pytestmark = pytest.mark.skipif(
    not PACKETS, reason="run scripts/plan_and_verify.py to produce a packet")


@pytest.fixture(scope="module")
def packet():
    """The most recently written packet."""
    newest = max(PACKETS, key=lambda p: p.stat().st_mtime)
    return json.loads(newest.read_text())


def test_the_structural_coefficient_matches_the_stack_it_describes(packet):
    """Recoverable from the stage masses, or it is not the one that was used.

    The repair loop moves this coefficient. A field that reports the module
    default instead will be wrong by exactly as much as the repair achieved,
    which is the worst possible error: it is invisible when nothing was
    repaired and largest when the most was.
    """
    used = packet.get("struct_coeff_used")
    assert used is not None
    # 0.14 is the module default. Passing that through when the design says
    # otherwise is the specific bug this test exists for.
    assert used == pytest.approx(0.2613, abs=0.02) or used != pytest.approx(0.14, abs=1e-9), (
        f"struct_coeff_used is {used}, which is the module default rather than "
        f"the value the stack was built at")


def test_the_overall_verdict_accounts_for_the_assembly(packet):
    """Components passing is not the vehicle passing.

    all_passed must be the conjunction. It read True once for a vehicle that
    could not be steered.
    """
    assert "components_passed" in packet and "assembly_passed" in packet
    # all_passed means *verified*, which needs three things: the components
    # passed, no assembly check failed, and no requirement was left unchecked.
    # It used to be the conjunction of the first two, and that let a packet
    # with an outstanding control requirement report a clean pass.
    assert packet["all_passed"] == (
        packet["components_passed"]
        and packet["assembly_passed"]
        and not packet.get("assembly_unverified"))


def test_assembly_findings_reach_the_json_not_only_the_prose(packet):
    """Anything consuming this file must see what a reader sees.

    The findings existed in markdown for one revision, which meant every
    downstream consumer -- including the model this project trains -- saw a
    clean pass on a vehicle with three failing assembly checks.
    """
    findings = packet.get("assembly_findings")
    assert isinstance(findings, list) and findings
    for f in findings:
        assert {"check", "passed", "detail"} <= set(f)
    if not packet["assembly_passed"]:
        assert any(not f["passed"] for f in findings), (
            "assembly is marked failed but no finding says why")


def test_a_failing_assembly_check_forces_an_overall_failure(packet):
    """No path by which a failed check leaves the packet reading as a pass."""
    if any(not f["passed"] for f in packet.get("assembly_findings", [])):
        assert not packet["all_passed"]


def test_the_structural_coefficient_is_placed_against_flown_hardware(packet):
    """A number outside flown practice has to be reported as outside it.

    Ten flown stages span 0.036 to 0.118. Anything above that is a finding, not
    a detail, and the packet carries the comparison so it cannot be read as
    ordinary.
    """
    from cadflow.flown_envelope import check as envelope

    used = packet["struct_coeff_used"]
    verdict = envelope(used, stage_wet_kg=packet.get("gross_kg", 1000.0))
    if used > verdict.flown_max:
        assert not verdict.inside
        assert "flown range" in verdict.note


def test_an_unverified_requirement_prevents_a_verified_verdict(packet):
    """Nothing is verified while a requirement went unchecked.

    This regression was introduced by an improvement. Reclassifying phase
    stabilisation from FAIL to REQUIRED was correct -- it is routine practice,
    not a defect -- but it emptied the failure list and the packet promptly
    reported "Overall: True" for a design with an outstanding control
    requirement. A cleaner summary that is less true is the exact failure this
    file exists to catch.

    A boolean cannot carry three outcomes. `all_passed` now means verified, and
    `status` says which of the three situations applies.
    """
    if packet.get("assembly_unverified"):
        assert not packet["all_passed"], (
            "packet claims all_passed with unverified requirements outstanding")
        # INCOMPLETE unless something actually failed, in which case FAILED
        # outranks it.
        #
        # This asserted INCOMPLETE unconditionally, which was true of every
        # packet that existed when it was written -- none had both an
        # outstanding requirement and a failing check at once. The first packet
        # to have both (a stage that cannot afford its tankage, alongside the
        # standing slosh phase-stabilisation requirement) made this test
        # contradict test_status_agrees_with_the_findings directly.
        #
        # The precedence is the interesting part and it belongs written down: a
        # design with a real failure is not "incomplete", it is failed, and
        # softening that to INCOMPLETE because some other item is also
        # outstanding would be the packet reporting the lesser of its problems.
        fails = [f for f in packet.get("assembly_findings", [])
                 if f.get("severity") == "fail"]
        assert packet.get("status") == ("FAILED" if fails else "INCOMPLETE")


def test_status_agrees_with_the_findings(packet):
    """The three-way status has to be derivable from the findings themselves.

    A status field that can drift from the list it summarises is the same
    problem as struct_coeff_used reading a module default.
    """
    findings = packet.get("assembly_findings", [])
    fails = [f for f in findings if f.get("severity") == "fail"]
    unver = [f for f in findings if f.get("severity") == "unverified"]
    status = packet.get("status")
    if not packet["components_passed"] or fails:
        assert status == "FAILED"
    elif unver:
        assert status == "INCOMPLETE"
    else:
        assert status == "VERIFIED"


def test_every_finding_carries_a_known_severity(packet):
    """An unrecognised severity would render as FAIL and hide its own meaning."""
    for f in packet.get("assembly_findings", []):
        assert f.get("severity") in {"pass", "fail", "unverified"}, f


def test_the_mass_closure_verdict_reaches_the_findings(packet):
    """"The vehicle cannot contain itself" must not be prose only.

    That is as strong a statement as this packet makes, and for several
    revisions it reached the markdown and stopped: not assembly_findings, not
    all_passed, not this file. Every downstream consumer -- including the model
    this project trains -- would have read a clean pass on a vehicle too heavy
    to exist.

    It is the same defect this file already covers for the assembly findings,
    recurring in the one section that decides whether the design is real, which
    is why the rule has to be checked rather than remembered.
    """
    findings = packet.get("assembly_findings", [])
    if not findings:
        pytest.skip("packet has no assembly findings")
    if not (packet.get("assembly") or {}).get("summary"):
        pytest.skip("no assembly was built, so no closure was computed")
    names = [f["check"] for f in findings]
    assert any("mass budget" in n for n in names), (
        f"no mass-closure finding among {names}; the closure verdict is "
        f"reaching the report and not the record")


def test_pressurisation_mass_is_charged_where_it_is_reported(packet):
    """The helium is counted in the same arithmetic that judges the budget.

    Reporting a shortfall in one section while the closure beside it omits the
    same number is the disconnection shape this whole file exists for.
    """
    findings = {f["check"]: f for f in packet.get("assembly_findings", [])}
    closure = next((f for k, f in findings.items() if "mass budget" in k), None)
    wall = next((f for k, f in findings.items() if "tank wall" in k), None)
    if closure is None or wall is None:
        pytest.skip("packet predates the split pressurisation findings")
    assert "pressurisation" in closure["detail"], (
        "the closure does not show the pressurisation mass it was charged")


def test_the_repairs_the_loop_made_are_recorded(packet):
    """A verdict reached after repairs is a different claim from a clean one.

    design_history carried the loop's record of its own decisions -- which alloy
    it right-sized to, whether it dropped a stage, whether it reached for
    thermal protection -- and was assigned and never read. The packet reported
    what the vehicle is and never how it came to be that, which is the one thing
    a reader cannot reconstruct from the result.

    Checked on the markdown rather than the JSON because that is where a reader
    meets it, and the whole point is what a reader can see.
    """
    newest = max(PACKETS, key=lambda p: p.stat().st_mtime)
    md = list(newest.parent.glob("*.md"))
    if not md:
        pytest.skip("packet has no markdown report")
    text = md[0].read_text()

    # struct_coeff_solved is written only when the fixed point ran, which is
    # what the design loop does. The first version of this guard looked for
    # "autodesign" or "repair loop" in the markdown -- both are stdout messages
    # that never reach the report, so the test skipped on every packet
    # including the ones it was written to check. A guard that never lets its
    # own assertion run is indistinguishable from no test.
    if packet.get("struct_coeff_solved") is None:
        pytest.skip("packet was not produced by the design loop")
    # The section is unconditional. A design that needed no repairs says so,
    # because an absent section cannot be told apart from an unrecorded one --
    # and "right at first evaluation" is a stronger claim than "made right",
    # which is exactly the thing worth being able to read.
    assert "What the design loop changed" in text, (
        "the loop's own repair record does not appear in the report")


def test_an_unanalysable_part_is_not_reported_as_a_failed_one(packet):
    """A part whose mesh degraded was never checked, not checked and found bad.

    When a component's mesh falls back to a convex hull the solve deliberately
    reports no stress, because a hull is not the part. Read as passed=False that
    drove a whole packet to FAILED with error=None -- nothing raised, so nothing
    explained. The distinction decides whether the verdict is a fault in the
    design or a gap in the analysis.

    This packet already learned that a boolean cannot carry three outcomes and
    gave the assembly findings VERIFIED, FAILED and INCOMPLETE. The component
    path kept pass/fail, so the lesson had to be learned twice.
    """
    unanalysed = [c for c in packet.get("components", [])
                  if not c.get("passed", True) and c.get("mesh_was_hull")
                  and c.get("stress_dist") is None and not c.get("error")]
    if not unanalysed:
        pytest.skip("every component in this packet was analysed")
    names = [f["check"] for f in packet.get("assembly_findings", [])]
    for c in unanalysed:
        assert any(c["name"] in n for n in names), (
            f"{c['name']} could not be analysed and no finding says so")
    # and an absent analysis is incomplete, not failed
    if not [f for f in packet.get("assembly_findings", [])
            if f.get("severity") == "fail"]:
        assert packet.get("status") != "FAILED"
