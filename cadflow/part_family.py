"""Classify existing TAO Parts into physics_targets families from labels/tags."""

from __future__ import annotations

import re
from typing import Any

from cadflow.physics_targets import FAMILY_ALIASES, FAMILY_PHYSICS, resolve_family

_SWEEP_RE = re.compile(r"^sweep-([a-z0-9_]+)-", re.I)

# tag / blob → family (first match wins)
_TAG_FAMILY: tuple[tuple[str, str], ...] = (
    (r"combustion|chamber|wall-stress|internal-flow", "combustion_chamber"),
    (r"\bnozzle\b|bell|throat", "nozzle"),
    (r"\binjector\b|turbopump|impeller|inducer", "injector"),
    (r"\btank\b|vessel|meop|pressure.?vessel", "tank"),
    (r"\bfin\b|canard|aero-load|root-stress", "fin"),
    (r"fairing|shroud", "fairing"),
    (r"nose_cone|\bnose\b|cone", "nose_cone"),
    (r"deployable|deployment|stowed-load", "deployable"),
    (r"antenna|panel", "antenna"),
    (r"spacecraft_bus|\bbus\b|cubesat", "spacecraft_bus"),
    (r"feed.?system|manifold|plumbing", "feed_system"),
    (r"valve", "valve"),
    (r"fastener|bolt|lug", "fastener"),
    (r"primary-structure|structure|bracket|frame", "structure"),
    (r"external-aero|aero-shape", "fin"),
)


def part_blob(part: dict[str, Any]) -> str:
    props = part.get("properties") or {}
    tags = props.get("tags")
    if isinstance(tags, list):
        tag_s = " ".join(str(t) for t in tags)
    else:
        tag_s = str(tags or "")
    return " ".join(
        str(x)
        for x in (
            part.get("label"),
            props.get("name"),
            props.get("part_class"),
            tag_s,
        )
        if x
    ).lower()


def sweep_family(part: dict[str, Any]) -> str | None:
    lab = str(part.get("label") or "")
    m = _SWEEP_RE.match(lab)
    if not m:
        return None
    raw = m.group(1).lower()
    return resolve_family(FAMILY_ALIASES.get(raw, raw))


def classify_part(part: dict[str, Any]) -> str:
    """Return a physics_targets family for this Part."""
    props = part.get("properties") or {}
    prop_fam = str(props.get("family") or "").lower()
    if prop_fam:
        return resolve_family(prop_fam)
    fam = sweep_family(part)
    if fam and fam in FAMILY_PHYSICS:
        return fam
    blob = part_blob(part)
    for pat, name in _TAG_FAMILY:
        if re.search(pat, blob, re.I):
            return resolve_family(name)
    pc = str(props.get("part_class") or "").lower()
    if pc in FAMILY_PHYSICS:
        return pc
    if pc in FAMILY_ALIASES:
        return resolve_family(pc)
    return "generic"


def preferred_modality(family: str) -> str:
    """Primary solver modality for alternates ('fea', 'cfd', or 'both')."""
    fam = resolve_family(family)
    spec = FAMILY_PHYSICS[fam]
    kind = spec["sim_kind"]
    if kind in {"external_aero", "internal_flow"}:
        return "both" if fam in {"fin", "nozzle", "fairing", "nose_cone"} else "cfd"
    if kind in {"thermo_structural", "pressure_vessel", "static_structural"}:
        return "fea"
    if kind in {"mechanism_flow", "kinematic", "deployment"}:
        return "fea"  # structural proxy until MBD lands
    return "fea"
