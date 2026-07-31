"""Spaceflight materials catalog for LatticeZero / JEPA conditioning.

Covers structural metals, propulsion alloys, composites, polymers, and TPS
(tile / ablative / blanket) grades with engineering properties used to annotate
CAD parts so LatticeZero can reason about material-specific physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SpaceMaterial:
    material_id: str
    name: str
    category: str  # aluminum | titanium | steel | superalloy | copper | composite | polymer | tps | ceramic
    density_kg_m3: float
    youngs_modulus_gpa: float
    yield_mpa: float | None
    ultimate_mpa: float | None
    max_service_temp_k: float
    cte_1e6_k: float | None = None
    thermal_conductivity_w_mk: float | None = None
    notes: str = ""
    typical_uses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["typical_uses"] = list(self.typical_uses)
        return d


# Canonical catalog — keep IDs stable for graph node ids: material:<material_id>
SPACE_MATERIALS: tuple[SpaceMaterial, ...] = (
    # Aluminum family
    SpaceMaterial("al-6061-t6", "Al 6061-T6", "aluminum", 2700, 68.9, 276, 310, 420, 23.6, 167, "Workhorse airframe alloy", ("body_tube", "fin", "structure", "tank")),
    SpaceMaterial("al-7075-t6", "Al 7075-T6", "aluminum", 2810, 71.7, 503, 572, 400, 23.4, 130, "High-strength airframe", ("fin", "structure", "fairing")),
    SpaceMaterial("al-2219-t87", "Al 2219-T87", "aluminum", 2840, 73.1, 350, 455, 450, 22.3, 120, "Cryogenic tank alloy", ("tank", "feed_system")),
    SpaceMaterial("al-li-2195", "Al-Li 2195", "aluminum", 2710, 78.0, 480, 540, 400, 23.0, 95, "Lightweight cryo tank / Orion", ("tank", "structure")),
    SpaceMaterial("al-5052-h32", "Al 5052-H32", "aluminum", 2680, 70.3, 193, 228, 390, 23.8, 138, "Weldable sheet", ("fairing", "body_tube")),
    # Titanium
    SpaceMaterial("ti-6al-4v", "Ti-6Al-4V", "titanium", 4430, 113.8, 880, 950, 670, 8.6, 6.7, "Aerospace titanium", ("structure", "tank", "engine_mount", "fastener")),
    SpaceMaterial("ti-6al-4v-eli", "Ti-6Al-4V ELI", "titanium", 4430, 113.8, 795, 860, 650, 8.6, 6.7, "Cryo / fracture-critical", ("tank", "structure")),
    SpaceMaterial("cp-ti-grade2", "CP Titanium Grade 2", "titanium", 4510, 105.0, 275, 345, 590, 8.4, 16.4, "Corrosion-resistant tubing", ("feed_system", "tank")),
    # Steels
    SpaceMaterial("ss-304", "304 Stainless", "steel", 8000, 193, 215, 505, 870, 17.2, 16.2, "General stainless", ("feed_system", "structure", "fastener")),
    SpaceMaterial("ss-316", "316 Stainless", "steel", 8000, 193, 290, 580, 870, 16.0, 16.3, "Corrosion-resistant stainless", ("feed_system", "valve", "tank")),
    SpaceMaterial("ss-321", "321 Stainless", "steel", 8000, 193, 240, 620, 925, 16.6, 16.1, "High-temp stainless", ("nozzle", "combustion_chamber")),
    SpaceMaterial("a286", "A-286", "steel", 7920, 201, 655, 1030, 700, 16.5, 15.1, "High-temp fastener alloy", ("fastener", "engine_mount")),
    SpaceMaterial("maraging-250", "Maraging 250", "steel", 8000, 186, 1700, 1800, 480, 10.1, 19.7, "Ultra-high strength", ("structure", "engine_mount")),
    # Superalloys
    SpaceMaterial("inconel-625", "Inconel 625", "superalloy", 8440, 207, 460, 880, 1250, 12.8, 9.8, "Oxidizer / hot structures", ("nozzle", "combustion_chamber", "feed_system")),
    SpaceMaterial("inconel-718", "Inconel 718", "superalloy", 8190, 200, 1030, 1240, 920, 13.0, 11.4, "Turbomachinery / fasteners", ("turbopump", "fastener", "engine_mount")),
    SpaceMaterial("hastelloy-x", "Hastelloy X", "superalloy", 8220, 205, 360, 755, 1200, 13.9, 9.1, "Combustor liners", ("combustion_chamber", "nozzle")),
    SpaceMaterial("rene-41", "Rene 41", "superalloy", 8250, 218, 850, 1170, 1250, 12.5, 10.5, "Hot-section alloy", ("nozzle", "turbopump")),
    # Copper / chamber
    SpaceMaterial("cu-cr-zr", "Cu-Cr-Zr", "copper", 8900, 120, 300, 400, 750, 17.0, 320, "Regeneratively cooled chambers", ("combustion_chamber", "nozzle", "injector")),
    SpaceMaterial("naarloy-z", "NARloy-Z", "copper", 9000, 117, 280, 380, 800, 16.5, 340, "SSME-class chamber alloy", ("combustion_chamber", "nozzle")),
    SpaceMaterial("ofhc-copper", "OFHC Copper", "copper", 8960, 115, 70, 220, 500, 17.0, 390, "High conductivity", ("injector", "combustion_chamber")),
    # Composites
    SpaceMaterial("cfrp-epoxy", "CFRP/Epoxy", "composite", 1550, 70.0, None, 600, 400, 0.5, 5.0, "Carbon fiber epoxy laminate", ("fairing", "body_tube", "structure", "deployable")),
    SpaceMaterial("cfrp-bmi", "CFRP/BMI", "composite", 1580, 75.0, None, 650, 450, 0.4, 4.5, "Higher-temp BMI matrix", ("fairing", "structure")),
    SpaceMaterial("gfrp-epoxy", "GFRP/Epoxy", "composite", 1850, 25.0, None, 350, 380, 8.0, 0.4, "Glass fiber composite", ("fairing", "body_tube", "fin")),
    SpaceMaterial("cmc-c-sic", "C/SiC CMC", "composite", 2100, 60.0, None, 250, 1600, 2.0, 15.0, "Ceramic matrix hot structures", ("nozzle", "tps_tile", "fairing")),
    SpaceMaterial("phenolic-impregnated", "Phenolic Impregnated Carbon", "composite", 1400, 20.0, None, 80, 2500, 2.5, 1.0, "Ablative composite", ("tps_tile", "nozzle", "fairing")),
    # Polymers
    SpaceMaterial("peek", "PEEK", "polymer", 1320, 3.6, 100, 110, 520, 47.0, 0.25, "High-performance thermoplastic", ("structure", "fastener", "sensor")),
    SpaceMaterial("vespel-sp1", "Vespel SP-1", "polymer", 1430, 3.1, 86, 90, 560, 50.0, 0.35, "High-temp polyimide", ("fastener", "valve", "sensor")),
    SpaceMaterial("ptfe", "PTFE", "polymer", 2200, 0.5, None, 25, 530, 100.0, 0.25, "Seals / soft goods", ("valve", "feed_system")),
    # TPS / ceramic / ablators
    SpaceMaterial("li-900", "LI-900 Silica Tile", "tps", 144, 0.05, None, 0.5, 1530, 0.6, 0.05, "Shuttle-class low-density tile", ("tps_tile", "fairing")),
    SpaceMaterial("li-2200", "LI-2200 Silica Tile", "tps", 352, 0.12, None, 1.5, 1640, 0.7, 0.08, "Higher-density silica tile", ("tps_tile",)),
    SpaceMaterial("fri-12", "FRCI-12", "tps", 192, 0.08, None, 1.0, 1600, 0.6, 0.06, "Fibrous refractory composite insulation", ("tps_tile",)),
    SpaceMaterial("toughened-uni-piece", "TUFI / AETB", "tps", 288, 0.15, None, 2.0, 1650, 0.8, 0.1, "Toughened tile / AETB family", ("tps_tile",)),
    SpaceMaterial("rcc", "Reinforced Carbon-Carbon", "tps", 1650, 40.0, None, 50, 1920, 1.0, 8.0, "Leading-edge / nosecap RCC", ("tps_tile", "nose_cone", "fairing")),
    SpaceMaterial("avion-ablator", "Avcoat Ablator", "tps", 512, 0.3, None, 5.0, 3000, 15.0, 0.2, "Capsule heatshield ablator", ("tps_tile", "fairing")),
    SpaceMaterial("pica", "PICA", "tps", 280, 0.2, None, 3.0, 3200, 2.0, 0.15, "Phenolic Impregnated Carbon Ablator", ("tps_tile", "fairing")),
    SpaceMaterial("tufroc", "TUFROC", "tps", 400, 0.25, None, 4.0, 1900, 1.5, 0.2, "Toughened uni-piece fibrous refractory", ("tps_tile",)),
    SpaceMaterial("mlti-blanket", "MLI Blanket", "tps", 40, 0.01, None, 0.1, 400, 10.0, 0.01, "Multi-layer insulation", ("blanket", "spacecraft_bus")),
    SpaceMaterial("afrsi-blanket", "AFRSI Blanket", "tps", 96, 0.05, None, 0.5, 920, 5.0, 0.04, "Advanced flexible reusable surface insulation", ("blanket", "tps_tile")),
    SpaceMaterial("Nextel-afb", "Nextel AFB", "ceramic", 2700, 70.0, None, 100, 1400, 5.0, 2.0, "Ceramic fiber / flexible TPS", ("blanket", "tps_tile")),
)


MATERIALS_BY_ID: dict[str, SpaceMaterial] = {m.material_id: m for m in SPACE_MATERIALS}


# Preferred materials per geometric family (ordered = preference weight)
FAMILY_MATERIAL_PRESETS: dict[str, tuple[str, ...]] = {
    "nose_cone": ("al-6061-t6", "cfrp-epoxy", "rcc", "al-7075-t6"),
    "fin": ("al-6061-t6", "al-7075-t6", "gfrp-epoxy", "cfrp-epoxy"),
    "body_tube": ("al-6061-t6", "al-5052-h32", "cfrp-epoxy", "gfrp-epoxy"),
    "tank": ("al-2219-t87", "al-li-2195", "ti-6al-4v", "ss-316"),
    "nozzle": ("inconel-625", "cu-cr-zr", "naarloy-z", "hastelloy-x", "cmc-c-sic"),
    "transition": ("al-6061-t6", "ss-304", "cfrp-epoxy"),
    "engine_mount": ("al-7075-t6", "ti-6al-4v", "inconel-718", "maraging-250"),
    "fairing": ("cfrp-epoxy", "cfrp-bmi", "al-7075-t6", "gfrp-epoxy"),
    "tps_tile": ("li-900", "li-2200", "fri-12", "toughened-uni-piece", "rcc", "pica", "avion-ablator", "tufroc"),
    "blanket": ("mlti-blanket", "afrsi-blanket", "Nextel-afb"),
    "solar_panel": ("cfrp-epoxy", "al-6061-t6"),
    "antenna": ("al-6061-t6", "cfrp-epoxy", "ss-304"),
    "ring_frame": ("al-7075-t6", "ti-6al-4v", "cfrp-epoxy"),
    "bulkhead": ("al-2219-t87", "ti-6al-4v", "ss-316"),
    "strut": ("ti-6al-4v", "al-7075-t6", "cfrp-epoxy"),
}


def assign_material_for_family(family: str, index: int = 0) -> SpaceMaterial:
    """Deterministic material assignment cycling presets for a part family."""
    preset = FAMILY_MATERIAL_PRESETS.get(family)
    if not preset:
        # fallback structural aluminum
        return MATERIALS_BY_ID["al-6061-t6"]
    mid = preset[index % len(preset)]
    return MATERIALS_BY_ID[mid]


def catalog_as_dicts() -> list[dict[str, Any]]:
    return [m.to_dict() for m in SPACE_MATERIALS]


def iter_materials() -> Iterable[SpaceMaterial]:
    return iter(SPACE_MATERIALS)
