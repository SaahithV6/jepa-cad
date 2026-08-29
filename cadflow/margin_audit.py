"""Is a passing margin larger than the uncertainty the packet quotes for it?

Packet v40 reports the tank wall as passing: von Mises 130.0 MPa against a 131
MPa allowable. That is a margin of 1.008 -- eight parts in a thousand. The same
document, forty lines earlier, states that swapping element order moves the p95
stress this loop sizes against by between -13.9% and +14.5%, and the pressure
driving that hoop stress rests on a net positive suction head the pressurisation
module labels ASSUMED from flown practice, not derived.

So the packet passes a structural check by 0.8% while quoting uncertainties an
order of magnitude larger on the quantities that check is built from. Both
statements are individually true and they cannot both be load-bearing. A margin
smaller than its own error bar is not a pass; it is a coin toss that happened to
land the right way, and reporting it as PASS is the most consequential kind of
overclaim this project can make, because it is the kind a reader has no way to
see.

This does not decide whether a design is good. It decides whether a *verdict is
supported*, which is a different question and the one the packet is for. A check
that clears its allowable by half a percent should read as unresolved, and the
honest response is a thicker wall, a tighter uncertainty, or an explicit
decision to accept the risk -- not a green tick.

Thresholds here are deliberately not clever. The packet already states its own
numbers; this compares against them rather than inventing a standard.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Numerical uncertainty this project has measured on the stresses it sizes
#: against, as a fraction.
#:
#: From artifacts/verification/element_order_ab.json: solving twelve real
#: components at identical meshes and loads under linear and quadratic elements
#: moved the p95 by a median of 1.1% but a range of -13.9% to +14.5%. The worst
#: case is the honest one to compare a margin against, because a reader cannot
#: know in advance which part they are looking at, and the double-digit ends
#: belong to fins and nose cones where the field is concentration-dominated.
MEASURED_STRESS_UNCERTAINTY = 0.145

#: Below this margin a verdict is reported as unresolved rather than passing.
#:
#: One plus the measured uncertainty. Not a safety factor and not an opinion:
#: it is the point at which this project's own A/B measurement can no longer
#: tell a pass from a failure.
RESOLVED_MARGIN = 1.0 + MEASURED_STRESS_UNCERTAINTY


@dataclass(frozen=True)
class MarginVerdict:
    check: str
    margin: float
    #: The uncertainty this margin is being judged against, as a fraction
    uncertainty: float
    resolved: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {"check": self.check, "margin": self.margin,
                "uncertainty": self.uncertainty, "resolved": self.resolved,
                "note": self.note}


def judge(check: str, margin: float,
          uncertainty: float = MEASURED_STRESS_UNCERTAINTY) -> MarginVerdict:
    """Is this margin bigger than the error bar on the number it came from?"""
    m = float(margin)
    u = abs(float(uncertainty))
    resolved = m >= 1.0 + u
    if m < 1.0:
        note = (f"{check} fails outright at margin {m:.3f}")
    elif resolved:
        note = (f"{check} clears its allowable by {100*(m-1):.0f}%, outside the "
                f"{100*u:.1f}% this project has measured on the stresses it "
                f"sizes against")
    else:
        note = (f"{check} clears its allowable by only {100*(m-1):.1f}%, inside "
                f"the {100*u:.1f}% this project has measured on the stresses it "
                f"sizes against. The verdict is not established either way: the "
                f"same analysis with quadratic elements could put it on the "
                f"other side")
    return MarginVerdict(check=check, margin=m, uncertainty=u,
                         resolved=resolved, note=note)


def unresolved(verdicts) -> list[MarginVerdict]:
    """Passing checks whose margin is inside the measured uncertainty."""
    return [v for v in verdicts if v.margin >= 1.0 and not v.resolved]


def audit(margins: dict,
          uncertainty: float = MEASURED_STRESS_UNCERTAINTY) -> list[MarginVerdict]:
    """Judge a mapping of check name to margin.

    Ordered worst-first, because the thinnest margin is the one that decides
    whether the packet's overall verdict means anything.
    """
    out = [judge(name, value, uncertainty) for name, value in margins.items()]
    out.sort(key=lambda v: v.margin)
    return out


def summary(verdicts) -> str:
    """One sentence a reader can act on, or silence when everything is clear."""
    thin = unresolved(verdicts)
    if not thin:
        return ""
    worst = min(thin, key=lambda v: v.margin)
    return (
        f"{len(thin)} passing check(s) clear their allowable by less than the "
        f"{100*worst.uncertainty:.1f}% numerical uncertainty this packet "
        f"reports for its own stresses -- the thinnest is {worst.check} at "
        f"{100*(worst.margin-1):.1f}%. Those verdicts are not established: the "
        f"same analysis at a different element order could move them to the "
        f"other side of the line. Read them as open questions rather than as "
        f"passes.")
