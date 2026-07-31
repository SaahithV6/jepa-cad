"""Adaptable graph schema and source-graph export for spaceflight data.

This module provides two layers:

1. A schema catalog describing how spaceflight entities can vary by subtype.
   The catalog is open-world by design: each node type has a shared base and
   subtype-specific extension fields, so rocket nozzles, tanks, valves, and
   assemblies can all carry the attributes they actually need.

2. A concrete graph export of the current dataset registry, so the corpus can
   be materialized immediately as nodes and edges for downstream retrieval,
   embedding, and graph-database ingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .datasets import DATASET_REGISTRY, DatasetSource, infer_information_mode
from .source_validation import validate_source_registry


@dataclass(frozen=True, slots=True)
class GraphFieldSchema:
    """A property that may appear on a graph node or edge type."""

    name: str
    data_type: str
    required: bool = False
    description: str = ""
    nullable: bool = True
    repeated: bool = False
    examples: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GraphNodeTypeSchema:
    """Schema contract for a node type in the graph."""

    name: str
    description: str
    extends: str | None = None
    open_world: bool = True
    properties: tuple[GraphFieldSchema, ...] = ()
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["properties"] = [prop.to_dict() for prop in self.properties]
        return payload


@dataclass(frozen=True, slots=True)
class GraphEdgeTypeSchema:
    """Schema contract for an edge type in the graph."""

    name: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    description: str
    properties: tuple[GraphFieldSchema, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["properties"] = [prop.to_dict() for prop in self.properties]
        return payload


@dataclass(frozen=True, slots=True)
class GraphSchemaCatalog:
    """Complete graph-schema catalog."""

    name: str
    version: str
    node_types: tuple[GraphNodeTypeSchema, ...]
    edge_types: tuple[GraphEdgeTypeSchema, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "node_types": [node_type.to_dict() for node_type in self.node_types],
            "edge_types": [edge_type.to_dict() for edge_type in self.edge_types],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "properties": dict(self.properties),
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "properties": dict(self.properties),
        }


@dataclass(frozen=True, slots=True)
class GraphDocument:
    """A graph materialization ready for JSON, Neo4j, ArangoDB, or RDF adapters."""

    name: str
    generated_at: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _field(name: str, data_type: str, *, required: bool = False, description: str = "", nullable: bool = True, repeated: bool = False, examples: Sequence[Any] = ()) -> GraphFieldSchema:
    return GraphFieldSchema(
        name=name,
        data_type=data_type,
        required=required,
        description=description,
        nullable=nullable,
        repeated=repeated,
        examples=tuple(examples),
    )


def build_spaceflight_graph_schema() -> GraphSchemaCatalog:
    """Return the adaptable graph schema for the current spaceflight corpus."""

    base_node_fields = (
        _field("id", "string", required=True, nullable=False, description="Stable graph identifier"),
        _field("type", "string", required=True, nullable=False, description="Node type name"),
        _field("source_ref", "string", description="Provenance pointer back to a source record"),
        _field("provenance", "object", description="Opaque provenance / traceability payload"),
        _field("domain", "string", description="High-level domain such as space, propulsion, structure"),
        _field("confidence", "number", description="Extraction confidence in [0, 1]"),
        _field("tags", "array[string]", repeated=True, description="Free-form semantic tags"),
        _field("version", "string", description="Schema or content version"),
    )

    node_types = (
        GraphNodeTypeSchema(
            name="Entity",
            description="Universal graph entity with provenance and open-world extension fields.",
            open_world=True,
            properties=base_node_fields,
            notes="All specializations inherit from this node type.",
        ),
        GraphNodeTypeSchema(
            name="Source",
            description="A retrievable document, dataset, web page, patent, or model source.",
            extends="Entity",
            properties=(
                _field("url", "string", required=True, nullable=False, description="Source URL"),
                _field("license", "string", description="Declared or inferred rights/terms"),
                _field("source_kind", "string", description="Heuristic classification such as pdf, patent, cad-asset"),
                _field("status", "string", description="Validator status such as usable or reference-only"),
            ),
            notes="This is the primary ingress node for the registry-driven corpus.",
        ),
        GraphNodeTypeSchema(
            name="Document",
            description="A document or report extracted from a source.",
            extends="Entity",
            properties=(
                _field("title", "string", required=True, nullable=False),
                _field("doc_type", "string", description="Report, patent, brochure, paper, manual, drawing"),
                _field("pages", "integer", description="Page count when available"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Part",
            description="A physical part or component that can exist in a product tree.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("part_class", "string", description="Specialization such as nozzle, tank, valve, bracket"),
                _field("material_ref", "string", description="Pointer to a material node or material name"),
                _field("geometry_ref", "string", description="Pointer to mesh/CAD geometry or shape record"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Assembly",
            description="A composed part hierarchy with mating and interface structure.",
            extends="Part",
            properties=(
                _field("children_count", "integer", description="Number of immediate child parts"),
                _field("interface_count", "integer", description="Interface or mate count"),
                _field("bom_ref", "string", description="Bill of materials reference"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Feature",
            description="A localized geometric or functional feature of a part.",
            extends="Entity",
            properties=(
                _field("feature_kind", "string", required=True, nullable=False, description="Fillet, rib, hole, cooling channel, flange, etc."),
                _field("geometry_ref", "string", description="Associated geometric region or parametrization"),
                _field("impact", "string", description="Functional impact such as cooling, stiffness, sealing, flow"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Dimension",
            description="A numeric parameter that may drive variant behavior.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("value", "number", required=True, nullable=False),
                _field("unit", "string", required=True, nullable=False),
                _field("min_value", "number", description="Optional admissible lower bound"),
                _field("max_value", "number", description="Optional admissible upper bound"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Material",
            description="A material definition or material family.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("family", "string", description="Alloy family, ceramic family, composite family, etc."),
                _field("properties", "object", description="Mechanical, thermal, and chemical properties"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Process",
            description="A manufacturing or treatment process.",
            extends="Entity",
            properties=(
                _field("process_kind", "string", required=True, nullable=False),
                _field("parameters", "object", description="Process parameters and limits"),
            ),
        ),
        GraphNodeTypeSchema(
            name="TuningGuidance",
            description="A structured design or solver-tuning hint derived from targets, outcomes, or family-specific heuristics.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("guidance_kind", "string", required=True, nullable=False, description="target_window, failure_recovery, or family_heuristic"),
                _field("solver", "string", description="Solver or analysis family the guidance applies to"),
                _field("family", "string", description="Part or source family that produced the guidance"),
                _field("targets", "object", description="Concrete parameter targets or ranges"),
                _field("rationale", "string", description="Short explanation of why the guidance exists"),
            ),
        ),
        GraphNodeTypeSchema(
            name="SimulationCase",
            description="A solver run, CFD/FEA/MBD case, or digital twin state.",
            extends="Entity",
            properties=(
                _field("solver", "string", required=True, nullable=False),
                _field("inputs", "object", description="Simulation inputs and boundary conditions"),
                _field("outputs", "object", description="Simulation outputs"),
                _field("status", "string", description="queued, completed, failed, verified"),
            ),
        ),
        GraphNodeTypeSchema(
            name="TestCase",
            description="An observed test, inspection, or qualification result.",
            extends="Entity",
            properties=(
                _field("test_kind", "string", required=True, nullable=False),
                _field("conditions", "object", description="Test conditions and setup"),
                _field("results", "object", description="Measured outputs"),
            ),
        ),
        GraphNodeTypeSchema(
            name="FailureMode",
            description="A defect, issue, or failure classification.",
            extends="Entity",
            properties=(
                _field("failure_kind", "string", required=True, nullable=False),
                _field("severity", "string", description="Low, medium, high, critical"),
                _field("symptoms", "array[string]", repeated=True, description="Observed symptoms or signatures"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Subsystem",
            description="A subsystem such as propulsion, tanks and feed, structures, or thermal.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("subsystem_kind", "string", description="Propulsion, tanks_and_feed, structures, thermal, mechanisms, etc."),
            ),
        ),
        GraphNodeTypeSchema(
            name="Vehicle",
            description="A rocket, spacecraft, launch vehicle, probe, or station.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("vehicle_kind", "string", description="Launch vehicle, spacecraft, lander, probe, station"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Mission",
            description="A mission program or flight campaign.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("agency", "string", description="Lead organization or program owner"),
                _field("year", "integer", description="Mission or publication year"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Dataset",
            description="A curated dataset or processed corpus manifest.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("manifest_path", "string", description="Path to the dataset manifest"),
                _field("processed_dir", "string", description="Path to the processed shard directory"),
                _field("source_count", "integer", description="Number of source rows in the manifest"),
                _field("shard_count", "integer", description="Number of shards in the processed set"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Statistic",
            description="A queryable summary value for a corpus, dataset, or source collection.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("statistic_kind", "string", required=True, nullable=False),
                _field("bucket", "string", description="Named bucket or category for the statistic"),
                _field("value", "number", required=True, nullable=False),
                _field("unit", "string", description="Unit or count label"),
                _field("scope", "string", description="Graph scope that the statistic summarizes"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Shard",
            description="A processed training shard or exported corpus slice.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("shard_path", "string", description="Path to the shard file"),
                _field("source_path", "string", description="Original raw file path"),
                _field("format", "string", description="Shard format such as npz or pt"),
                _field("index", "integer", description="Shard order in the manifest"),
            ),
        ),
        GraphNodeTypeSchema(
            name="RawAsset",
            description="A raw source asset such as a STEP, STL, GLB, PDF, or image file.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("exists", "boolean", description="Whether the asset exists on disk"),
                _field("size_bytes", "integer", description="File size in bytes"),
                _field("extension", "string", description="Lower-case file extension"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Sample",
            description="A single processed sample with summarized tensor or point-cloud statistics.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("num_points", "integer", description="Number of points in the sample"),
                _field("num_fields", "integer", description="Number of per-point fields"),
                _field("max_stress", "number", description="Optional stress or scalar label"),
                _field("point_bounds", "object", description="Summary bounds for the point cloud"),
                _field("field_stats", "object", description="Summary statistics for fields"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Analogue",
            description="A detailed analogue entry that summarizes one downloaded asset or training item.",
            extends="Entity",
            properties=(
                _field("name", "string", required=True, nullable=False),
                _field("analogue_kind", "string", required=True, nullable=False, description="geometry, field, document, tensor, or mixed"),
                _field("source_path", "string", required=True, nullable=False),
                _field("source_kind", "string", description="Underlying source classification"),
                _field("summary", "object", description="Detailed summary of the asset or sample"),
                _field("feature_summary", "object", description="Extracted geometric / semantic features"),
                _field("parametric_summary", "object", description="Dimension and effect statistics"),
                _field("physical_summary", "object", description="Physics-related breakdown such as stress, area, or bounds"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Corpus",
            description="A top-level local data collection or corpus root.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("corpus_kind", "string", description="repo_root, raw_downloads, processed, dataset_collection"),
                _field("file_count", "integer", description="Number of files under this corpus"),
                _field("directory_count", "integer", description="Number of directories under this corpus"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Directory",
            description="A directory within the local data tree.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("depth", "integer", description="Directory depth relative to the data root"),
                _field("file_count", "integer", description="Number of immediate files"),
                _field("directory_count", "integer", description="Number of immediate child directories"),
            ),
        ),
        GraphNodeTypeSchema(
            name="CodeArtifact",
            description="A source code or script artifact in the local data tree.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("language", "string", description="Language such as python"),
                _field("line_count", "integer", description="Total line count"),
                _field("size_bytes", "integer", description="File size in bytes"),
            ),
        ),
        GraphNodeTypeSchema(
            name="DocumentAsset",
            description="A manifest, metadata file, report, or other text document.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("document_kind", "string", description="manifest, metadata, readme, report, note, etc."),
                _field("line_count", "integer", description="Total line count when text is available"),
                _field("size_bytes", "integer", description="File size in bytes"),
            ),
        ),
        GraphNodeTypeSchema(
            name="TensorShard",
            description="A processed NPZ/PT tensor shard used for JEPA training.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("format", "string", description="npz or pt"),
                _field("shard_index", "integer", description="Order within the processed dataset"),
                _field("source_path", "string", description="Original raw source path"),
                _field("num_points", "integer", description="Number of points in the shard"),
                _field("num_fields", "integer", description="Number of fields in the shard"),
                _field("max_stress", "number", description="Max stress or scalar label when present"),
            ),
        ),
        GraphNodeTypeSchema(
            name="ImageAsset",
            description="A raster image or texture asset.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("extension", "string", description="Lower-case file extension"),
                _field("size_bytes", "integer", description="File size in bytes"),
            ),
        ),
        GraphNodeTypeSchema(
            name="ArchiveAsset",
            description="A compressed or packaged asset archive.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("extension", "string", description="Lower-case file extension"),
                _field("size_bytes", "integer", description="File size in bytes"),
            ),
        ),
        GraphNodeTypeSchema(
            name="OtherArtifact",
            description="A data artifact that does not fit the primary categories.",
            extends="Entity",
            properties=(
                _field("path", "string", required=True, nullable=False),
                _field("extension", "string", description="Lower-case file extension"),
                _field("size_bytes", "integer", description="File size in bytes"),
                _field("artifact_kind", "string", description="Fallback classification"),
            ),
        ),
        GraphNodeTypeSchema(
            name="RocketNozzle",
            description="A propulsion nozzle with geometry and operating-point properties.",
            extends="Part",
            properties=(
                _field("chamber_pressure", "number", description="Chamber pressure, typically in Pa, bar, or MPa depending on unit field"),
                _field("expansion_ratio", "number", description="Nozzle expansion ratio"),
                _field("contraction_ratio", "number", description="Chamber-to-throat or related contraction ratio"),
                _field("throat_diameter", "number", description="Throat diameter"),
                _field("exit_diameter", "number", description="Exit diameter"),
                _field("area_ratio", "number", description="Nozzle area ratio"),
                _field("construction_ratio", "number", description="Construction or structural ratio used by a specific source family"),
                _field("wall_thickness", "number", description="Wall thickness"),
                _field("cooling_method", "string", description="Regenerative, film, ablative, transpiration, etc."),
                _field("regen_channel_count", "integer", description="Regenerative channel count"),
                _field("film_cooling", "boolean", description="Whether film cooling is used"),
                _field("mixture_ratio", "number", description="Engine mixture ratio"),
                _field("mass_flow", "number", description="Mass flow through the nozzle or engine"),
                _field("heat_flux", "number", description="Estimated or measured wall heat flux"),
            ),
            notes="This node type is intentionally specialized because nozzles often need operating-state data, not just geometry.",
        ),
        GraphNodeTypeSchema(
            name="Tank",
            description="A propellant, oxidizer, or pressurant tank.",
            extends="Part",
            properties=(
                _field("volume", "number", description="Internal volume"),
                _field("design_pressure", "number", description="Design pressure"),
                _field("proof_pressure", "number", description="Proof pressure"),
                _field("wall_thickness", "number", description="Nominal wall thickness"),
                _field("pressurant_type", "string", description="Helium, nitrogen, autogenous, etc."),
                _field("domed_endcaps", "boolean", description="Whether the tank has domed endcaps"),
                _field("baffle_count", "integer", description="Number of internal baffles"),
            ),
        ),
        GraphNodeTypeSchema(
            name="Valve",
            description="A fluid-control valve or actuation element.",
            extends="Part",
            properties=(
                _field("cv", "number", description="Valve flow coefficient"),
                _field("actuation_type", "string", description="Solenoid, pneumatic, pyrotechnic, motorized, etc."),
                _field("response_time", "number", description="Opening or closing response time"),
                _field("max_delta_p", "number", description="Maximum differential pressure"),
                _field("seal_type", "string", description="Seat or seal family"),
            ),
        ),
        GraphNodeTypeSchema(
            name="StructurePart",
            description="A load-bearing structural component.",
            extends="Part",
            properties=(
                _field("load_cases", "array[string]", repeated=True, description="Named structural load cases"),
                _field("buckling_margin", "number", description="Buckling margin"),
                _field("stiffness", "number", description="Representative stiffness metric"),
                _field("modal_freqs", "array[number]", repeated=True, description="Modal frequencies"),
            ),
        ),
    )

    edge_types = (
        GraphEdgeTypeSchema(
            name="PART_OF",
            source_types=("Part", "Assembly", "Feature", "Sample"),
            target_types=("Part", "Assembly", "Subsystem", "Vehicle", "Mission"),
            description="Hierarchy / containment relation.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_FEATURE",
            source_types=("Part", "Assembly", "Sample"),
            target_types=("Feature",),
            description="Links a part or assembly to a geometric/function feature.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_DIMENSION",
            source_types=("Part", "Feature", "Assembly"),
            target_types=("Dimension",),
            description="Attaches a numeric parameter to a part or feature.",
        ),
        GraphEdgeTypeSchema(
            name="MADE_OF",
            source_types=("Part", "Assembly", "Feature", "Sample"),
            target_types=("Material",),
            description="Material assignment edge.",
        ),
        GraphEdgeTypeSchema(
            name="MANUFACTURED_BY",
            source_types=("Part", "Assembly", "Feature", "Sample"),
            target_types=("Process",),
            description="Process or treatment relation.",
        ),
        GraphEdgeTypeSchema(
            name="SIMULATED_IN",
            source_types=("Part", "Assembly", "Feature", "Vehicle", "Subsystem"),
            target_types=("SimulationCase",),
            description="Connects an entity to a simulation case.",
        ),
        GraphEdgeTypeSchema(
            name="TESTED_IN",
            source_types=("Part", "Assembly", "Feature", "Vehicle", "Subsystem"),
            target_types=("TestCase",),
            description="Connects an entity to a physical test or inspection record.",
        ),
        GraphEdgeTypeSchema(
            name="VERIFIED_BY",
            source_types=("SimulationCase", "TestCase"),
            target_types=("TestCase", "SimulationCase"),
            description="Evidence relation between simulation and physical observation.",
        ),
        GraphEdgeTypeSchema(
            name="AFFECTS",
            source_types=("Feature", "Dimension", "Process", "Material"),
            target_types=("Feature", "Dimension", "SimulationCase", "TestCase", "FailureMode"),
            description="Causal or correlational influence edge.",
        ),
        GraphEdgeTypeSchema(
            name="GUIDES",
            source_types=("TuningGuidance",),
            target_types=("Part", "Assembly", "Feature", "Dimension", "SimulationCase", "TestCase", "FailureMode"),
            description="Structured guidance that should be applied to or evaluated against an entity.",
        ),
        GraphEdgeTypeSchema(
            name="VARIANT_OF",
            source_types=("Part", "Assembly", "Feature", "SimulationCase"),
            target_types=("Part", "Assembly", "Feature", "SimulationCase"),
            description="Variant / family relation with parameter deltas.",
        ),
        GraphEdgeTypeSchema(
            name="SOURCE_OF_TRUTH_FOR",
            source_types=("Source", "Document", "TestCase", "SimulationCase"),
            target_types=("Part", "Assembly", "Feature", "Dimension", "Material", "Process", "Subsystem", "Vehicle", "Mission"),
            description="Traceability from evidence to semantic entity.",
        ),
        GraphEdgeTypeSchema(
            name="CONTAINS",
            source_types=("Corpus", "Directory", "Source"),
            target_types=("Corpus", "Directory", "Source", "Statistic", "RawAsset", "TensorShard", "DocumentAsset", "CodeArtifact", "ImageAsset", "ArchiveAsset", "OtherArtifact", "Sample"),
            description="Filesystem or corpus containment.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_STATISTIC",
            source_types=("Corpus", "Dataset", "Source"),
            target_types=("Statistic",),
            description="Attaches an aggregate or summary statistic to a graph scope.",
        ),
        GraphEdgeTypeSchema(
            name="DESCRIBES",
            source_types=("DocumentAsset",),
            target_types=("Corpus", "Directory", "RawAsset", "TensorShard", "Sample"),
            description="Documentary metadata or manifest relation.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_SHARD",
            source_types=("Dataset",),
            target_types=("Shard",),
            description="A dataset manifest contains a processed shard.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_SAMPLE",
            source_types=("Dataset", "Shard"),
            target_types=("Sample",),
            description="A dataset or shard contains a sample summary.",
        ),
        GraphEdgeTypeSchema(
            name="HAS_ANALOGUE",
            source_types=("Corpus", "Dataset", "Shard", "Source", "RawAsset", "DocumentAsset", "TensorShard", "Sample"),
            target_types=("Analogue",),
            description="A graph entity has a detailed analogue entry.",
        ),
        GraphEdgeTypeSchema(
            name="ANALOGUE_OF",
            source_types=("Analogue",),
            target_types=("Source", "Document", "RawAsset", "TensorShard", "Shard", "Sample", "Corpus", "Dataset"),
            description="A detailed analogue corresponds to a source item or training item.",
        ),
        GraphEdgeTypeSchema(
            name="DERIVED_FROM",
            source_types=("Shard", "Sample", "Dataset"),
            target_types=("RawAsset", "Source", "Document"),
            description="A processed item derived from a raw asset or source.",
        ),
        GraphEdgeTypeSchema(
            name="STORED_AT",
            source_types=("RawAsset", "Shard", "Sample", "Dataset"),
            target_types=("Source", "Document", "RawAsset", "Shard", "Dataset"),
            description="Physical or logical storage relation.",
        ),
        GraphEdgeTypeSchema(
            name="MENTIONS",
            source_types=("Source", "Document"),
            target_types=("Part", "Assembly", "Feature", "Dimension", "Material", "Process", "Subsystem", "Vehicle", "Mission"),
            description="Loose mention or reference relation from a document or source.",
        ),
        GraphEdgeTypeSchema(
            name="RECOMMENDED_FOR",
            source_types=("Source",),
            target_types=("Subsystem", "Part", "Feature"),
            description="Source-to-domain recommendation relation used by the registry.",
        ),
    )

    notes = (
        "Open-world schemas allow part families to define new fields without breaking the catalog.",
        "Rocket nozzles intentionally carry operating-state fields (pressure, mixture ratio, heat flux) in addition to geometry.",
        "Source registry edges are included so the current corpus can be materialized immediately as a graph document.",
    )
    return GraphSchemaCatalog(
        name="spaceflight-graph",
        version="1.0",
        node_types=node_types,
        edge_types=edge_types,
        notes=notes,
    )


def build_source_registry_graph(sources: Iterable[DatasetSource] | None = None) -> GraphDocument:
    """Materialize the current source registry as a graph document."""

    selected = tuple(sources or DATASET_REGISTRY.values())
    validation = validate_source_registry(keys=[source.key for source in selected], live_check=False) if selected else None
    validation_by_key = {result.source.key: result for result in validation.results} if validation is not None else {}

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_domain_nodes: set[str] = set()
    seen_use_case_nodes: set[str] = set()
    seen_recommendation_nodes: set[str] = set()
    domain_node_index: dict[str, int] = {}

    information_mode_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    analogue_kind_counts: dict[str, int] = {}
    analogue_count = 0

    corpus_node_id = "corpus:source-registry"
    nodes.append(
        GraphNode(
            id=corpus_node_id,
            type="Corpus",
            label="source-registry",
            properties={
                "path": "source-registry",
                "corpus_kind": "dataset_collection",
                "file_count": len(selected),
                "directory_count": len({source.domain for source in selected}),
            },
        )
    )

    for source in selected:
        information_mode = source.information_mode or infer_information_mode(source)
        information_mode_counts[information_mode] = information_mode_counts.get(information_mode, 0) + 1
        validation_result = validation_by_key.get(source.key)
        source_kind = validation_result.source_kind if validation_result is not None else "unknown"
        source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1
        status = validation_result.status if validation_result is not None else "manual-review"
        status_counts[status] = status_counts.get(status, 0) + 1
        domain_counts[source.domain] = domain_counts.get(source.domain, 0) + 1

        source_node_id = f"source:{source.key}"
        nodes.append(
            GraphNode(
                id=source_node_id,
                type="Source",
                label=source.title,
                properties={
                    "key": source.key,
                    "domain": source.domain,
                    "url": source.url,
                    "license": source.license,
                    "use_cases": list(source.use_cases),
                    "notes": source.notes,
                    "size_hint": source.size_hint,
                    "recommended_for": list(source.recommended_for),
                    "information_mode": information_mode,
                    "source_kind": source_kind,
                    "status": status,
                    "training_eligible": validation_result.training_eligible if validation_result is not None else False,
                    "reference_only": validation_result.reference_only if validation_result is not None else False,
                    "manual_review": validation_result.manual_review if validation_result is not None else True,
                    "blocked": validation_result.blocked if validation_result is not None else False,
                },
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{corpus_node_id}:contains:{source.key}",
                type="CONTAINS",
                source=corpus_node_id,
                target=source_node_id,
                properties={"role": "source_registry_entry"},
            )
        )
        source_analogue_id = f"analogue:source:{source.key}"
        source_analogue_kind = "registry_source"
        analogue_kind_counts[source_analogue_kind] = analogue_kind_counts.get(source_analogue_kind, 0) + 1
        analogue_count += 1
        nodes.append(
            GraphNode(
                id=source_analogue_id,
                type="Analogue",
                label=f"{source.title} analogue",
                properties={
                    "name": source.title,
                    "analogue_kind": source_analogue_kind,
                    "source_path": source.url,
                    "source_kind": "Source",
                    "summary": {
                        "key": source.key,
                        "domain": source.domain,
                        "status": status,
                        "information_mode": information_mode,
                        "url": source.url,
                    },
                    "feature_summary": {
                        "use_case_count": len(source.use_cases),
                        "recommended_for_count": len(source.recommended_for),
                    },
                    "parametric_summary": {
                        "url_length": len(source.url),
                        "title_length": len(source.title),
                    },
                    "physical_summary": {
                        "size_hint": source.size_hint,
                        "license": source.license,
                    },
                },
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{source_node_id}:analogue",
                type="HAS_ANALOGUE",
                source=source_node_id,
                target=source_analogue_id,
                properties={"role": "registry_source_analogue"},
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{source_analogue_id}:of",
                type="ANALOGUE_OF",
                source=source_analogue_id,
                target=source_node_id,
                properties={"role": "registry_source_analogue"},
            )
        )

        domain_id = f"domain:{source.domain}"
        if domain_id not in seen_domain_nodes:
            seen_domain_nodes.add(domain_id)
            domain_node_index[domain_id] = len(nodes)
            nodes.append(
                GraphNode(
                    id=domain_id,
                    type="Subsystem",
                    label=source.domain,
                    properties={"subsystem_kind": source.domain, "source_count": domain_counts[source.domain]},
                )
            )
        else:
            domain_node = nodes[domain_node_index[domain_id]]
            nodes[domain_node_index[domain_id]] = GraphNode(
                id=domain_node.id,
                type=domain_node.type,
                label=domain_node.label,
                properties={**domain_node.properties, "source_count": domain_counts[source.domain]},
            )
        edges.append(
            GraphEdge(
                id=f"edge:{source.key}:domain",
                type="PART_OF",
                source=source_node_id,
                target=domain_id,
                properties={"relation": "registry_domain"},
            )
        )

        for idx, use_case in enumerate(source.use_cases):
            use_case_id = f"use-case:{source.key}:{idx}:{_slug(use_case)}"
            if use_case_id not in seen_use_case_nodes:
                seen_use_case_nodes.add(use_case_id)
                nodes.append(
                    GraphNode(
                        id=use_case_id,
                        type="Feature",
                        label=use_case,
                        properties={"feature_kind": "use_case", "impact": source.domain},
                    )
                )
            edges.append(
                GraphEdge(
                    id=f"edge:{source.key}:use-case:{idx}",
                    type="MENTIONS",
                    source=source_node_id,
                    target=use_case_id,
                    properties={"role": "use_case"},
                )
            )

        for idx, recommendation in enumerate(source.recommended_for):
            rec_id = f"recommendation:{source.key}:{idx}:{_slug(recommendation)}"
            if rec_id not in seen_recommendation_nodes:
                seen_recommendation_nodes.add(rec_id)
                nodes.append(
                    GraphNode(
                        id=rec_id,
                        type="Subsystem",
                        label=recommendation,
                        properties={"subsystem_kind": recommendation},
                    )
                )
            edges.append(
                GraphEdge(
                    id=f"edge:{source.key}:recommended_for:{idx}",
                    type="RECOMMENDED_FOR",
                    source=source_node_id,
                    target=rec_id,
                    properties={},
                )
            )

    def add_statistic(statistic_kind: str, bucket: str, value: int, *, unit: str = "count", scope: str = "source-registry") -> None:
        stat_id = f"statistic:{_slug(statistic_kind)}:{_slug(bucket)}"
        nodes.append(
            GraphNode(
                id=stat_id,
                type="Statistic",
                label=bucket,
                properties={
                    "name": bucket,
                    "statistic_kind": statistic_kind,
                    "bucket": bucket,
                    "value": value,
                    "unit": unit,
                    "scope": scope,
                },
            )
        )
        edges.append(
            GraphEdge(
                id=f"edge:{corpus_node_id}:stat:{_slug(statistic_kind)}:{_slug(bucket)}",
                type="HAS_STATISTIC",
                source=corpus_node_id,
                target=stat_id,
                properties={"statistic_kind": statistic_kind, "bucket": bucket},
            )
        )

    add_statistic("source-count", "all-sources", len(selected))
    add_statistic("usable-count", "usable", status_counts.get("usable", 0))
    add_statistic("reference-only-count", "reference-only", status_counts.get("reference-only", 0))
    add_statistic("manual-review-count", "manual-review", status_counts.get("manual-review", 0))
    add_statistic("blocked-count", "blocked", status_counts.get("blocked", 0))

    for information_mode, count in sorted(information_mode_counts.items()):
        add_statistic("information-mode-count", information_mode, count)
    for source_kind, count in sorted(source_kind_counts.items()):
        add_statistic("source-kind-count", source_kind, count)
    for domain, count in sorted(domain_counts.items()):
        add_statistic("domain-count", domain, count)

    return GraphDocument(
        name="source-registry-graph",
        generated_at=_utc_now(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "source_count": len(selected),
            "domain_count": len(seen_domain_nodes),
            "use_case_count": len(seen_use_case_nodes),
            "recommendation_count": len(seen_recommendation_nodes),
            "information_mode_counts": information_mode_counts,
            "source_kind_counts": source_kind_counts,
            "status_counts": status_counts,
            "domain_counts": domain_counts,
            "analogue_count": analogue_count,
            "analogue_kind_counts": analogue_kind_counts,
            "training_eligible_count": status_counts.get("usable", 0),
        },
    )


def _slug(text: str) -> str:
    slug = [ch.lower() if ch.isalnum() else "-" for ch in text.strip()]
    collapsed = "".join(slug)
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed.strip("-") or "node"


def render_graph_schema(catalog: GraphSchemaCatalog, *, as_json: bool = False) -> str:
    if as_json:
        import json

        return json.dumps(catalog.to_dict(), indent=2)

    lines = [f"name={catalog.name}", f"version={catalog.version}", "", "node_types:"]
    for node_type in catalog.node_types:
        lines.append(f"- {node_type.name}{' < ' + node_type.extends if node_type.extends else ''}: {node_type.description}")
        for prop in node_type.properties:
            req = " required" if prop.required else ""
            lines.append(f"  - {prop.name} ({prop.data_type}{req})")
    lines.append("\nedge_types:")
    for edge_type in catalog.edge_types:
        lines.append(f"- {edge_type.name}: {edge_type.description}")
        lines.append(f"  source_types={', '.join(edge_type.source_types)}")
        lines.append(f"  target_types={', '.join(edge_type.target_types)}")
    lines.append("\nnotes:")
    for note in catalog.notes:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def render_graph_document(graph: GraphDocument, *, as_json: bool = False) -> str:
    if as_json:
        import json

        return json.dumps(graph.to_dict(), indent=2)

    lines = [
        f"name={graph.name}",
        f"generated_at={graph.generated_at}",
        f"nodes={len(graph.nodes)}",
        f"edges={len(graph.edges)}",
        f"metadata={graph.metadata}",
        "",
        "sample_nodes:",
    ]
    for node in graph.nodes[:15]:
        lines.append(f"- {node.id} [{node.type}] {node.label}")
    if len(graph.nodes) > 15:
        lines.append(f"- ... {len(graph.nodes) - 15} more nodes")
    lines.append("\nsample_edges:")
    for edge in graph.edges[:15]:
        lines.append(f"- {edge.source} -[{edge.type}]-> {edge.target}")
    if len(graph.edges) > 15:
        lines.append(f"- ... {len(graph.edges) - 15} more edges")
    return "\n".join(lines).rstrip() + "\n"
