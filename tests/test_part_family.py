"""Tests for Part family classification used by physics alternates."""

from cadflow.part_family import classify_part, preferred_modality


def test_classify_sweep_combustion():
    p = {
        "id": "p1",
        "label": "sweep-combustion_chamber-home-best-foo",
        "properties": {"tags": ["combustion_chamber", "wall-stress"]},
    }
    assert classify_part(p) == "combustion_chamber"
    assert preferred_modality("combustion_chamber") == "fea"


def test_classify_fin_and_tank():
    fin = {
        "id": "p2",
        "label": "sweep-fin-oshrockets-root-stress-v01",
        "properties": {"tags": ["fin", "aero-load"]},
    }
    tank = {
        "id": "p3",
        "label": "abc tank",
        "properties": {"tags": ["tank"], "name": "propellant tank"},
    }
    assert classify_part(fin) == "fin"
    assert classify_part(tank) == "tank"
    assert preferred_modality("fin") == "both"


def test_classify_spacecraft_bus():
    p = {
        "id": "p4",
        "label": "sweep-spacecraft_bus-nasa3d-bus-structure-v01",
        "properties": {"tags": ["spacecraft_bus", "bus-structure"]},
    }
    assert classify_part(p) == "spacecraft_bus"
