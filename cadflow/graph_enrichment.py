"""Graph enrichment layer for the JEPA/TAO corpus.

This module adds richness that the baseline corpus sweep does not produce:

  1. RealPart nodes with human-readable names extracted from source filenames
  2. Material nodes (316 SS, Al 6061, Inconel, etc.) linked to parts
  3. Document→Part edges linking PDFs and drawings to the parts they describe
  4. Assembly COMPOSED_OF hierarchy inferred from sldasm/sldprt co-location
  5. ExperimentalResult nodes from CSV test data (ME411 pintle flow tests)
  6. Registered Source nodes for the liquid engine / test-stand repos
  7. Reclassification of generic Part nodes by part-name heuristics
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph_schema import GraphDocument, GraphEdge, GraphNode


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s or "node"


def _uid(*parts: str) -> str:
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Material extraction from part/file names
# ---------------------------------------------------------------------------

_MATERIAL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b316\s*ss\b|\b316\s*stainless\b", re.I), "316 SS", "steel"),
    (re.compile(r"\b304\s*ss\b|\b304\s*stainless\b", re.I), "304 SS", "steel"),
    (re.compile(r"\b321\s*ss\b|\b321\s*stainless\b", re.I), "321 Stainless", "steel"),
    (re.compile(r"\ba-?286\b", re.I), "A-286", "steel"),
    (re.compile(r"\bmaraging\b", re.I), "Maraging 250", "steel"),
    (re.compile(r"\binconel\s*718\b", re.I), "Inconel 718", "superalloy"),
    (re.compile(r"\binconel\b", re.I), "Inconel 625", "superalloy"),
    (re.compile(r"\bhastelloy\b", re.I), "Hastelloy X", "superalloy"),
    (re.compile(r"\brene\s*41\b", re.I), "Rene 41", "superalloy"),
    (re.compile(r"\bcu-?cr-?zr\b|\bcopper.?chrome\b", re.I), "Cu-Cr-Zr", "copper"),
    (re.compile(r"\bnarloy\b", re.I), "NARloy-Z", "copper"),
    (re.compile(r"\bofhc\b", re.I), "OFHC Copper", "copper"),
    (re.compile(r"\bcopper\b|\bcu\b", re.I), "Copper", "copper"),
    (re.compile(r"\bal-?li\b|\b2195\b", re.I), "Al-Li 2195", "aluminum"),
    (re.compile(r"\bal\s*2219\b|\b2219\b", re.I), "Al 2219-T87", "aluminum"),
    (re.compile(r"\bal\s*7075\b|\b7075\b", re.I), "Al 7075-T6", "aluminum"),
    (re.compile(r"\bal\s*6061\b|\b6061\b", re.I), "Al 6061-T6", "aluminum"),
    (re.compile(r"\bal\s*5052\b|\b5052\b", re.I), "Al 5052-H32", "aluminum"),
    (re.compile(r"\baluminum\b|\baluminium\b|\bal\b", re.I), "Aluminum", "aluminum"),
    (re.compile(r"\bti-?6al-?4v\s*eli\b", re.I), "Ti-6Al-4V ELI", "titanium"),
    (re.compile(r"\bti-?6al-?4v\b|\bti[-\s]?6\b|\btitanium\b", re.I), "Ti-6Al-4V", "titanium"),
    (re.compile(r"\bc\/?\s*sic\b|\bcmc\b", re.I), "C/SiC CMC", "composite"),
    (re.compile(r"\bcfrp\b|\bcarbon\s*fiber\b", re.I), "CFRP", "composite"),
    (re.compile(r"\bgfrp\b|\bglass\s*fiber\b", re.I), "GFRP/Epoxy", "composite"),
    (re.compile(r"\bpica\b", re.I), "PICA", "tps"),
    (re.compile(r"\bavcoat\b", re.I), "Avcoat Ablator", "tps"),
    (re.compile(r"\bli-?900\b|\bsilica\s*tile\b", re.I), "LI-900 Silica Tile", "tps"),
    (re.compile(r"\bli-?2200\b", re.I), "LI-2200 Silica Tile", "tps"),
    (re.compile(r"\brcc\b|reinforced\s*carbon", re.I), "Reinforced Carbon-Carbon", "tps"),
    (re.compile(r"\bmlti\b|\bmli\b", re.I), "MLI Blanket", "tps"),
    (re.compile(r"\bafrsi\b", re.I), "AFRSI Blanket", "tps"),
    (re.compile(r"\bpeek\b", re.I), "PEEK", "polymer"),
    (re.compile(r"\bvespel\b", re.I), "Vespel SP-1", "polymer"),
    (re.compile(r"\bptfe\b|\bteflon\b", re.I), "PTFE", "polymer"),
    (re.compile(r"\babs\b|\bpla\b|\bnylon\b|\bpetg\b", re.I), "Polymer", "polymer"),
    (re.compile(r"\bstainless\b|\bss\b", re.I), "Stainless Steel", "steel"),
    (re.compile(r"\bsteel\b", re.I), "Steel", "steel"),
]


_FAMILY_KEYWORDS: list[tuple[list[str], str]] = [
    (["nozzle", "throat", "diverge", "converge", "bell"], "nozzle"),
    (["chamber", "combustion", "cc"], "combustion_chamber"),
    (["injector", "pintle", "impinge", "orifice"], "injector"),
    (["turbopump", "impeller", "turbine", "pump"], "turbopump"),
    (["tank", "propellant", "lox", "fuel", "oxidizer"], "tank"),
    (["valve", "poppet", "regulator", "solenoid"], "valve"),
    (["feed", "manifold", "tube", "tubing", "pipe", "fitting"], "feed_system"),
    (["fairing", "shroud", "aeroshell", "ogive"], "fairing"),
    (["fin", "airfoil", "wing", "control surface"], "fin"),
    (["tile", "tps", "heatshield", "heat-shield", "li-900", "pica", "avcoat", "rcc"], "tps_tile"),
    (["blanket", "mlti", "mli", "afrsi", "insulation"], "blanket"),
    (["solar", "panel", "array", "boom"], "deployable"),
    (["antenna", "dish", "horn"], "antenna"),
    (["bulkhead", "ring_frame", "ring-frame"], "structure"),
    (["bracket", "mount", "adapter", "coupler", "plate", "stand", "frame", "strut"], "structure"),
    (["washer", "bolt", "screw", "nut", "fastener", "bushing", "bearing", "dowel", "seal", "gasket", "o-ring", "retainer"], "fastener"),
    (["load cell", "sensor", "gauge", "transducer"], "sensor"),
    (["spacecraft", "satellite", "bus", "probe"], "spacecraft_bus"),
    (["cubesat", "smallsat", "nanosatellite"], "cubesat"),
]


def _extract_material(text: str) -> tuple[str, str] | None:
    """Return (material_label, category) if text hints at a material."""
    for pattern, label, category in _MATERIAL_PATTERNS:
        if pattern.search(text):
            return label, category
    return None


def _reclassify_family(name: str) -> str | None:
    """Return a specific family or None if no match."""
    low = name.lower()
    for keywords, family in _FAMILY_KEYWORDS:
        if any(kw in low for kw in keywords):
            return family
    return None


# ---------------------------------------------------------------------------
# Assembly hierarchy from co-located sldprt + sldasm files
# ---------------------------------------------------------------------------

def _discover_assemblies(raw_dir: Path) -> list[tuple[Path, list[Path]]]:
    """For each .sldasm find sibling .sldprt files in the same directory."""
    results = []
    for asm_file in raw_dir.rglob("*.SLDASM"):
        parent = asm_file.parent
        parts = [
            p for p in parent.iterdir()
            if p.is_file() and p.suffix.upper() in {".SLDPRT", ".SLDDRW"}
        ]
        if parts:
            results.append((asm_file, parts))
    for asm_file in raw_dir.rglob("*.sldasm"):
        parent = asm_file.parent
        parts = [
            p for p in parent.iterdir()
            if p.is_file() and p.suffix.lower() in {".sldprt", ".slddrw"}
        ]
        if parts:
            results.append((asm_file, parts))
    return results


# ---------------------------------------------------------------------------
# CSV experimental result parsing
# ---------------------------------------------------------------------------

def _parse_me411_csv(csv_path: Path) -> dict[str, Any] | None:
    """Parse an ME411 flow-test CSV, returning numeric channel stats or None."""
    try:
        with open(csv_path, newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Find the numeric data rows (skip comment/header rows starting with #!)
        data_rows: list[list[str]] = []
        header_row: list[str] = []
        for row in rows:
            if not row or row[0].startswith("#!"):
                continue
            # Rows that look like timestamps or numeric
            if all(
                cell.strip() == "" or _is_numeric(cell.strip()) or ":" in cell
                for cell in row[:4]
            ):
                if not header_row and not _is_numeric(row[0].strip()):
                    header_row = [c.strip() for c in row]
                else:
                    data_rows.append(row)
        if not data_rows:
            return None
        # Pick up to 4 numeric channels
        channels: dict[str, list[float]] = {}
        col_start = 0
        for ci in range(col_start, min(col_start + 8, len(data_rows[0]))):
            vals = []
            for row in data_rows:
                if ci < len(row):
                    v = _try_float(row[ci])
                    if v is not None:
                        vals.append(v)
            if len(vals) >= 10:
                col_name = header_row[ci] if ci < len(header_row) else f"ch{ci}"
                channels[col_name or f"ch{ci}"] = vals
        if not channels:
            return None
        stats: dict[str, Any] = {"n_samples": len(data_rows), "channels": {}}
        for ch, vals in channels.items():
            stats["channels"][ch] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        return stats
    except Exception:
        return None


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _try_float(s: str) -> float | None:
    try:
        return float(s.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Document→Part linking
# ---------------------------------------------------------------------------

_PART_NAME_SUFFIXES = {".sldprt", ".step", ".stp", ".stl", ".x_t", ".iges", ".igs", ".obj", ".ply"}


def _stem_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity between two stems."""
    a_toks = set(re.split(r"[\W_]+", a.lower())) - {"v1","v2","v3","v4","v5","v6","v7","v8",""}
    b_toks = set(re.split(r"[\W_]+", b.lower())) - {"v1","v2","v3","v4","v5","v6","v7","v8",""}
    if not a_toks or not b_toks:
        return 0.0
    return len(a_toks & b_toks) / max(len(a_toks), len(b_toks))


# ---------------------------------------------------------------------------
# Main enrichment builder
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentReport:
    realpart_nodes: int = 0
    material_nodes: int = 0
    material_edges: int = 0
    document_nodes: int = 0
    document_edges: int = 0
    assembly_nodes: int = 0
    assembly_edges: int = 0
    experimental_nodes: int = 0
    experimental_edges: int = 0
    source_nodes: int = 0
    reclassified_parts: int = 0
    total_new_nodes: int = 0
    total_new_edges: int = 0


def build_enrichment_graph(
    raw_dirs: list[Path | str],
    existing_graph: GraphDocument | None = None,
) -> tuple[GraphDocument, EnrichmentReport]:
    """Build an enrichment graph from raw source directories.

    Returns a GraphDocument containing only the *new* nodes/edges (enrichment
    layer), plus an EnrichmentReport with counts. Caller should merge this into
    the existing master graph.
    """
    report = EnrichmentReport()
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # Collect existing node IDs so we don't duplicate
    existing_ids: set[str] = set()
    if existing_graph:
        existing_ids = {n.id for n in existing_graph.nodes}

    def _add_node(n: GraphNode) -> bool:
        if n.id not in existing_ids:
            nodes.append(n)
            existing_ids.add(n.id)
            return True
        return False

    def _add_edge(e: GraphEdge) -> None:
        edges.append(e)

    # ------------------------------------------------------------------
    # 1. Registered Source nodes for the liquid engine repos
    # ------------------------------------------------------------------
    liquid_sources = [
        {
            "key": "liquid-propellant-engine",
            "title": "Open Source Liquid Propellant Rocket Engine (LogRocket)",
            "domain": "propulsion",
            "url": "https://github.com/[local]/liquid-propellant-engine",
            "license": "open-hardware",
            "tier": "A",
            "notes": "Full SolidWorks assembly (V1–V7), pintle injector evolution, 6kN heat-sink engine, CFD analysis, ME411 pintle flow test CSVs, PDF drawings.",
            "geometry_count": 318,
            "document_count": 829,
        },
        {
            "key": "liquid-engine-test-stand",
            "title": "Liquid Fuel Engine Test Stand (LFETS)",
            "domain": "propulsion/test-infrastructure",
            "url": "https://github.com/[local]/liquid-engine-test-stand",
            "license": "open-hardware",
            "tier": "A",
            "notes": "Full test stand SolidWorks CAD, P&ID diagrams, BOM, procedure docs, TSAR safety review.",
            "geometry_count": 1083,
            "document_count": 149,
        },
    ]
    for src in liquid_sources:
        nid = f"source:{src['key']}"
        if _add_node(GraphNode(
            id=nid,
            type="Source",
            label=src["title"],
            properties={k: v for k, v in src.items() if k != "key"},
        )):
            report.source_nodes += 1

    # ------------------------------------------------------------------
    # 2. Walk raw dirs for geometry + documents
    # ------------------------------------------------------------------
    geom_exts = {".stl", ".step", ".stp", ".x_t", ".obj", ".ply", ".iges", ".igs"}
    doc_exts = {".pdf", ".md", ".txt", ".rst", ".csv"}
    skip_dirs = {".git", "__pycache__", "node_modules"}

    material_node_ids: dict[str, str] = {}  # material_label -> node_id

    for raw_dir_arg in raw_dirs:
        raw_dir = Path(raw_dir_arg)
        if not raw_dir.exists():
            continue

        # Determine which registered source this dir belongs to
        dir_key = raw_dir.name.lower().replace(" ", "-").replace("_", "-")
        src_node_id = f"source:{dir_key}" if f"source:{dir_key}" in existing_ids or any(s["key"] in dir_key for s in liquid_sources) else None
        if src_node_id and src_node_id not in existing_ids:
            src_node_id = None  # not registered

        # Walk all geometry files (includes SolidWorks files for naming/material)
        sldprt_exts = {".sldprt", ".sldasm", ".slddrw"}
        all_geom_exts = geom_exts | sldprt_exts
        geom_files: list[Path] = []
        doc_files: list[Path] = []
        for p in raw_dir.rglob("*"):
            if p.is_file():
                if any(part in skip_dirs for part in p.parts):
                    continue
                suf = p.suffix.lower()
                if suf in all_geom_exts and p.stat().st_size > 500:
                    geom_files.append(p)
                elif suf in doc_exts and p.stat().st_size > 100:
                    doc_files.append(p)

        # ---- RealPart nodes from geometry files ----
        for gp in geom_files:
            part_name = gp.stem
            family = _reclassify_family(part_name) or _reclassify_family(str(gp.relative_to(raw_dir))) or "generic"
            mat_result = _extract_material(str(gp)) or _extract_material(part_name)
            version_match = re.search(r"[Vv](\d+)", part_name)
            version = version_match.group(0) if version_match else None
            nid = f"realpart:{_slug(str(gp.relative_to(raw_dir.parent)))}"

            new = _add_node(GraphNode(
                id=nid,
                type="RealPart",
                label=part_name,
                properties={
                    "source_path": str(gp),
                    "file_format": gp.suffix.lower().lstrip("."),
                    "size_bytes": gp.stat().st_size,
                    "family": family,
                    "version": version,
                    "source_dir": raw_dir.name,
                    "relative_path": str(gp.relative_to(raw_dir)),
                },
            ))
            if new:
                report.realpart_nodes += 1
                if family != "generic":
                    report.reclassified_parts += 1

            # Link to source
            if src_node_id:
                _add_edge(GraphEdge(
                    id=f"edge:{src_node_id}:contains:{nid}",
                    type="CONTAINS",
                    source=src_node_id,
                    target=nid,
                    properties={"role": "geometry"},
                ))

            # Material node + edge
            if mat_result:
                mat_label, mat_category = mat_result
                mat_nid = f"material:{_slug(mat_label)}"
                if mat_nid not in material_node_ids:
                    if _add_node(GraphNode(
                        id=mat_nid,
                        type="Material",
                        label=mat_label,
                        properties={"name": mat_label, "category": mat_category},
                    )):
                        material_node_ids[mat_label] = mat_nid
                        report.material_nodes += 1
                else:
                    mat_nid = material_node_ids.get(mat_label, mat_nid)
                _add_edge(GraphEdge(
                    id=f"edge:{nid}:made-of:{mat_nid}",
                    type="MADE_OF",
                    source=nid,
                    target=mat_nid,
                    properties={"confidence": "name_heuristic"},
                ))
                report.material_edges += 1

        # ---- Document nodes + similarity-based Part→Doc edges ----
        geom_stems = {gp.stem.lower(): f"realpart:{_slug(str(gp.relative_to(raw_dir.parent)))}" for gp in geom_files}

        for dp in doc_files:
            doc_name = dp.stem
            doc_nid = f"document:{_slug(str(dp.relative_to(raw_dir.parent)))}"
            doc_type = {
                ".pdf": "drawing",
                ".md": "readme",
                ".txt": "notes",
                ".rst": "documentation",
                ".csv": "test_data",
            }.get(dp.suffix.lower(), "document")

            new = _add_node(GraphNode(
                id=doc_nid,
                type="Document",
                label=doc_name,
                properties={
                    "source_path": str(dp),
                    "doc_type": doc_type,
                    "size_bytes": dp.stat().st_size,
                    "source_dir": raw_dir.name,
                    "relative_path": str(dp.relative_to(raw_dir)),
                },
            ))
            if new:
                report.document_nodes += 1

            if src_node_id:
                _add_edge(GraphEdge(
                    id=f"edge:{src_node_id}:documents:{doc_nid}",
                    type="DOCUMENTS",
                    source=src_node_id,
                    target=doc_nid,
                    properties={"role": "documentation"},
                ))

            # Link document → best-matching part by stem similarity
            best_part_id: str | None = None
            best_score = 0.3  # minimum threshold
            doc_low = doc_name.lower()
            for gstem, gid in geom_stems.items():
                score = _stem_similarity(doc_low, gstem)
                if score > best_score:
                    best_score = score
                    best_part_id = gid
            if best_part_id:
                _add_edge(GraphEdge(
                    id=f"edge:{doc_nid}:describes:{best_part_id}",
                    type="DESCRIBES",
                    source=doc_nid,
                    target=best_part_id,
                    properties={"similarity": round(best_score, 3), "method": "stem_overlap"},
                ))
                report.document_edges += 1

        # ---- Assembly hierarchy ----
        for asm_path, part_paths in _discover_assemblies(raw_dir):
            asm_nid = f"realpart:{_slug(str(asm_path.relative_to(raw_dir.parent)))}"
            asm_name = asm_path.stem
            asm_family = _reclassify_family(asm_name) or "assembly"
            _add_node(GraphNode(
                id=asm_nid,
                type="Assembly",
                label=asm_name,
                properties={
                    "source_path": str(asm_path),
                    "family": asm_family,
                    "child_count": len(part_paths),
                    "source_dir": raw_dir.name,
                },
            ))
            for pp in part_paths:
                child_nid = f"realpart:{_slug(str(pp.relative_to(raw_dir.parent)))}"
                _add_node(GraphNode(
                    id=child_nid,
                    type="RealPart",
                    label=pp.stem,
                    properties={
                        "source_path": str(pp),
                        "file_format": pp.suffix.lower().lstrip("."),
                        "size_bytes": pp.stat().st_size,
                        "family": _reclassify_family(pp.stem) or "generic",
                        "source_dir": raw_dir.name,
                    },
                ))
                _add_edge(GraphEdge(
                    id=f"edge:{asm_nid}:composed:{child_nid}",
                    type="COMPOSED_OF",
                    source=asm_nid,
                    target=child_nid,
                    properties={"role": "component"},
                ))
                report.assembly_edges += 1
            report.assembly_nodes += 1

    # ------------------------------------------------------------------
    # 3. ME411 / CSV experimental result nodes
    # ------------------------------------------------------------------
    for raw_dir_arg in raw_dirs:
        raw_dir = Path(raw_dir_arg)
        if not raw_dir.exists():
            continue
        for csv_path in raw_dir.rglob("*.csv"):
            if any(part in {".git", "__pycache__"} for part in csv_path.parts):
                continue
            stats = _parse_me411_csv(csv_path)
            if not stats:
                continue
            exp_nid = f"experimentalresult:{_slug(str(csv_path.relative_to(raw_dir.parent)))}"
            # Infer test kind from path/name
            test_kind = "flow_test"
            low = csv_path.name.lower()
            if "pressure" in low:
                test_kind = "pressure_test"
            elif "thrust" in low or "force" in low:
                test_kind = "thrust_test"
            elif "temperature" in low or "temp" in low:
                test_kind = "thermal_test"

            _add_node(GraphNode(
                id=exp_nid,
                type="ExperimentalResult",
                label=csv_path.stem,
                properties={
                    "source_path": str(csv_path),
                    "test_kind": test_kind,
                    "n_samples": stats["n_samples"],
                    "channels": list(stats["channels"].keys()),
                    "channel_stats": stats["channels"],
                    "source_dir": raw_dir.name,
                },
            ))
            report.experimental_nodes += 1

            # Try to link to a sibling geometry
            parent = csv_path.parent
            for neighbor in parent.rglob("*"):
                if neighbor.is_file() and neighbor.suffix.lower() in {".stl", ".step", ".stp", ".x_t"}:
                    geom_nid = f"realpart:{_slug(str(neighbor.relative_to(raw_dir.parent)))}"
                    if geom_nid in existing_ids:
                        _add_edge(GraphEdge(
                            id=f"edge:{exp_nid}:validates:{geom_nid}",
                            type="VALIDATES",
                            source=exp_nid,
                            target=geom_nid,
                            properties={"method": "co_location"},
                        ))
                        report.experimental_edges += 1

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    report.total_new_nodes = len(nodes)
    report.total_new_edges = len(edges)

    doc = GraphDocument(
        name="spaceflight-enrichment",
        generated_at=_utc_now(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "kind": "enrichment",
            "raw_dirs": [str(d) for d in raw_dirs],
            "new_nodes": len(nodes),
            "new_edges": len(edges),
        },
    )
    return doc, report


def merge_graphs(*docs: GraphDocument, name: str = "merged-spaceflight-graph") -> GraphDocument:
    """Merge multiple GraphDocuments, deduplicating by node/edge id."""
    seen_nodes: dict[str, GraphNode] = {}
    seen_edges: set[str] = set()
    all_edges: list[GraphEdge] = []

    for doc in docs:
        for n in doc.nodes:
            if n.id not in seen_nodes:
                seen_nodes[n.id] = n
        for e in doc.edges:
            if e.id not in seen_edges:
                seen_edges.add(e.id)
                all_edges.append(e)

    return GraphDocument(
        name=name,
        generated_at=_utc_now(),
        nodes=tuple(seen_nodes.values()),
        edges=tuple(all_edges),
        metadata={
            "kind": "merged",
            "source_graph_count": len(docs),
            "node_count": len(seen_nodes),
            "edge_count": len(all_edges),
        },
    )


def render_enrichment_report(report: EnrichmentReport, *, as_json: bool = False) -> str:
    d = {
        "realpart_nodes": report.realpart_nodes,
        "material_nodes": report.material_nodes,
        "material_edges": report.material_edges,
        "document_nodes": report.document_nodes,
        "document_edges": report.document_edges,
        "assembly_nodes": report.assembly_nodes,
        "assembly_edges": report.assembly_edges,
        "experimental_nodes": report.experimental_nodes,
        "experimental_edges": report.experimental_edges,
        "source_nodes": report.source_nodes,
        "reclassified_parts": report.reclassified_parts,
        "total_new_nodes": report.total_new_nodes,
        "total_new_edges": report.total_new_edges,
    }
    if as_json:
        return json.dumps(d, indent=2)
    lines = ["=== Graph Enrichment Report ==="]
    for k, v in d.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
