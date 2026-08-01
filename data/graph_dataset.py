"""Graph-backed dataset adapters for JEPA-CAD.

This module turns a materialized graph export into a PyTorch dataset. It is
meant to consume the JSON payload emitted by the existing Neo4j and corpus
export helpers, so training can be driven by graph metadata as well as the raw
geometry/shard files the graph points to.

Conditioning vector layout (``GRAPH_METADATA_DIM``):
  9  legacy file-node stats
  25 family one-hot (``FAMILY_VOCAB``)
  N  physics targets / measured FEA-CFD / airflow overlays (``CONDITIONING_QUANTITIES``)
  11 geometry metrics (+ mesh structural slots folded into geom/text)
  8  material engineering props
  P  generation params (``PARAM_QUANTITIES``)
  32 hashed text bag (spec_prompt + Document excerpts + categorical params)
  = computed (see GRAPH_METADATA_DIM)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data.parsers import ParseError, parse_raw_file

_GRAPH_FILE_CANDIDATES = (
    "spaceflight-graph.json",
    "corpus-graph.json",
    "local-data-inventory.json",
    "graph.json",
)

_FILE_NODE_TYPES = {"RawAsset", "TensorShard", "Shard", "Sample", "Analogue"}
_PATH_KEYS = ("shard_path", "source_shard_path", "path", "source_path", "file_path", "local_path", "geometry_ref")

# Family vocabulary for one-hot conditioning (order is the contract).
FAMILY_VOCAB: tuple[str, ...] = (
    "generic",
    "nose_cone",
    "combustion_chamber",
    "injector",
    "nozzle",
    "tank",
    "valve",
    "feed_system",
    "fin",
    "fairing",
    "structure",
    "fastener",
    "mechanism",
    "deployable",
    "antenna",
    "spacecraft_bus",
    # Rocket / TPS generation families (kept distinct for label fidelity)
    "body_tube",
    "transition",
    "engine_mount",
    "tps_tile",
    "blanket",
    "solar_panel",
    "ring_frame",
    "bulkhead",
    "strut",
)
_FAMILY_INDEX = {name: i for i, name in enumerate(FAMILY_VOCAB)}

# Node types that carry association payloads worth folding into conditioning.
_ASSOC_NODE_TYPES = {
    "Part",
    "GeometricMetric",
    "TuningGuidance",
    "DesignTarget",
    "PhysicsTarget",
    "RealPart",
    "Material",
    "Dimension",
    "Document",
    "SimulationCase",
    "Source",
}

# Leaf association nodes: ingest payloads but do not expand through them
# (e.g. Material hub would otherwise pull every MADE_OF neighbor's params).
_ASSOC_LEAF_TYPES = {
    "Material",
    "Dimension",
    "GeometricMetric",
    "PhysicsTarget",
    "DesignTarget",
    "TuningGuidance",
    "Document",
    "Source",
    "SimulationCase",
}

# Prefer typed training edges; avoid wandering through COMPOSED_OF assemblies.
_ASSOC_EDGE_TYPES = {
    "REPRESENTS",
    "PART_OF",
    "HAS_SAMPLE",
    "DERIVED_FROM",
    "HAS_PHYSICS_TARGET",
    "HAS_GEOMETRIC_METRIC",
    "HAS_DESIGN_TARGET",
    "HAS_DIMENSION",
    "MADE_OF",
    "GUIDES",
    "HAS_FEATURE",
    "VARIANT_OF",
    "SIMULATED_IN",
    "HAS_ANALOGUE",
    "ANALOGUE_OF",
    "HAS_SOLVER_SETUP",
    "DESCRIBES",
    "MENTIONS",
    "DOCUMENTS",
    "SOURCE_OF_TRUTH_FOR",
    "HAS_SHARD",
    "VERIFIED_BY",
}

# Fixed hashed bag-of-tokens from linked Document text + Part spec_prompt
# for hybrid text→CAD conditioning (appended after structured metadata).
TEXT_META_DIM = 32

# Canonical generation-param slots (mm-scale dims + a few dimensionless).
PARAM_QUANTITIES: tuple[tuple[str, float], ...] = (
    ("diameter_mm", 1.0 / 500.0),
    ("length_mm", 1.0 / 1000.0),
    ("thickness_mm", 1.0 / 50.0),
    ("wall_mm", 1.0 / 20.0),
    ("height_mm", 1.0 / 500.0),
    ("root_chord_mm", 1.0 / 500.0),
    ("tip_chord_mm", 1.0 / 500.0),
    ("sweep_mm", 1.0 / 500.0),
    ("throat_diameter_mm", 1.0 / 200.0),
    ("expansion_ratio", 1.0 / 100.0),
    ("fore_diameter_mm", 1.0 / 500.0),
    ("aft_diameter_mm", 1.0 / 500.0),
    ("motor_diameter_mm", 1.0 / 200.0),
    ("side_mm", 1.0 / 500.0),
    ("width_mm", 1.0 / 1000.0),
    ("power", 1.0),
    ("depth_mm", 1.0 / 500.0),
    ("radial_width_mm", 1.0 / 500.0),
    ("densified_cap_mm", 1.0 / 50.0),
    ("fineness_ratio", 1.0 / 10.0),
)
# String params folded into the text bag (not numeric slots).
_CATEGORICAL_PARAM_KEYS = ("shape", "nose_shape", "profile", "cross_section", "finish")

_MATERIAL_CATEGORIES = (
    "aluminum",
    "titanium",
    "steel",
    "superalloy",
    "copper",
    "composite",
    "polymer",
    "tps",
    "ceramic",
)
_MATERIAL_CAT_INDEX = {name: i for i, name in enumerate(_MATERIAL_CATEGORIES)}

# graph_metadata layout — keep in lockstep with ``_metadata_vector`` (9 legacy)
# and ``_association_vector`` (family + physics + geometry + material + params).
# Physics/geometry widths are derived so the constant can't drift from the
# actual emitted vector (which previously desynced: 82 declared vs 89 emitted).
_LEGACY_META_DIM = 9
_GEOMETRY_META_DIM = 11
_MATERIAL_META_DIM = 8
try:  # physics width is defined by the conditioning contract
    from cadflow.physics_targets import CONDITIONING_QUANTITIES as _COND_Q

    _PHYSICS_META_DIM = len(_COND_Q)
except Exception:  # noqa: BLE001 — fall back to historical width
    _PHYSICS_META_DIM = 20
GRAPH_METADATA_DIM = (
    _LEGACY_META_DIM
    + len(FAMILY_VOCAB)
    + _PHYSICS_META_DIM
    + _GEOMETRY_META_DIM
    + _MATERIAL_META_DIM
    + len(PARAM_QUANTITIES)
    + TEXT_META_DIM
)


@dataclass(frozen=True, slots=True)
class GraphDatasetRecord:
    node_id: str
    node_type: str
    label: str
    path: Path
    properties: dict[str, Any]


def _load_graph_payload(graph_path: Path) -> dict[str, Any]:
    if graph_path.is_dir():
        for candidate in _GRAPH_FILE_CANDIDATES:
            nested = graph_path / candidate
            if nested.exists():
                graph_path = nested
                break
        else:
            raise FileNotFoundError(
                f"No graph export found in {graph_path}; expected one of: {', '.join(_GRAPH_FILE_CANDIDATES)}"
            )
    if not graph_path.exists():
        raise FileNotFoundError(f"graph export not found: {graph_path}")
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _candidate_paths(
    raw_value: str,
    *,
    graph_path: Path,
    graph_root: Path | None,
    extra_roots: list[Path] | None = None,
) -> list[Path]:
    raw = Path(raw_value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if graph_root is not None:
            candidates.append((graph_root / raw).resolve())
        candidates.append((graph_path.parent / raw).resolve())
        # Walk upward from the graph export to cover repo-root-relative paths.
        for parent in list(graph_path.resolve().parents)[:6]:
            candidates.append((parent / raw).resolve())
        candidates.append((Path.cwd() / raw).resolve())
        # Common corpus roots (physics shards, processed NASA3D, raw CAD).
        for root in extra_roots or []:
            candidates.append((Path(root) / raw).resolve())
        # nasa3d/*.npz historically stored without data/processed/ prefix
        if raw_value.startswith("nasa3d/"):
            for parent in list(graph_path.resolve().parents)[:6]:
                candidates.append((parent / "data" / "processed" / raw_value).resolve())
            candidates.append((Path.cwd() / "data" / "processed" / raw_value).resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_node_path(
    node: dict[str, Any],
    *,
    graph_path: Path,
    graph_root: Path | None,
    extra_roots: list[Path] | None = None,
) -> Path | None:
    props = node.get("properties", {}) or {}
    for key in _PATH_KEYS:
        value = props.get(key)
        if not value:
            continue
        candidates = _candidate_paths(
            str(value),
            graph_path=graph_path,
            graph_root=graph_root,
            extra_roots=extra_roots,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _sanitize_fields(fields: np.ndarray, num_fields: int) -> np.ndarray:
    if fields.ndim == 1:
        fields = fields.reshape(-1, 1)
    if fields.shape[1] < num_fields:
        pad = np.zeros((fields.shape[0], num_fields - fields.shape[1]), dtype=np.float32)
        fields = np.concatenate([fields, pad], axis=1)
    elif fields.shape[1] > num_fields:
        fields = fields[:, :num_fields]
    return fields.astype(np.float32)


def _resample_points_fields(
    points: np.ndarray,
    fields: np.ndarray,
    *,
    num_points: int,
    seed_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == num_points:
        return points, fields
    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    if points.shape[0] > num_points:
        idx = rng.choice(points.shape[0], num_points, replace=False)
    else:
        idx = rng.choice(points.shape[0], num_points, replace=True)
    return points[idx], fields[idx]


def _load_npz_or_pt(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        if "points" not in data or "fields" not in data:
            raise ParseError(f"npz shard missing points/fields: {path}")
        payload: dict[str, np.ndarray] = {
            "points": np.asarray(data["points"], dtype=np.float32),
            "fields": np.asarray(data["fields"], dtype=np.float32),
        }
        if "max_stress" in data:
            payload["max_stress"] = np.asarray(data["max_stress"], dtype=np.float32)
        return payload

    if path.suffix.lower() == ".pt":
        obj = torch.load(path, weights_only=True)
        if isinstance(obj, dict) and "points" in obj and "fields" in obj:
            payload = {
                "points": np.asarray(obj["points"], dtype=np.float32),
                "fields": np.asarray(obj["fields"], dtype=np.float32),
            }
            if "max_stress" in obj:
                payload["max_stress"] = np.asarray(obj["max_stress"], dtype=np.float32)
            return payload
        raise ParseError(f"pt shard missing points/fields: {path}")

    raise ParseError(f"unsupported tensor shard format: {path.suffix}")


def _load_sample_arrays(path: Path, *, num_points: int, num_fields: int) -> tuple[np.ndarray, np.ndarray, float | None]:
    if path.suffix.lower() in {".npz", ".pt"}:
        payload = _load_npz_or_pt(path)
        points = payload["points"]
        fields = _sanitize_fields(payload["fields"], num_fields)
        points, fields = _resample_points_fields(points, fields, num_points=num_points, seed_text=str(path))
        max_stress = float(np.asarray(payload["max_stress"]).reshape(())) if "max_stress" in payload else None
        return points.astype(np.float32), fields.astype(np.float32), max_stress

    sample = parse_raw_file(path, num_points=num_points, num_fields=num_fields, allow_synthetic_fallback=False)
    points = sample.points.astype(np.float32)
    fields = _sanitize_fields(sample.fields.astype(np.float32), num_fields)
    points, fields = _resample_points_fields(points, fields, num_points=num_points, seed_text=str(path))
    max_stress = float(fields[:, min(2, fields.shape[1] - 1)].max()) if fields.size else None
    return points, fields, max_stress


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _family_from_props(props: dict[str, Any], fallback: str | None = None) -> str | None:
    for key in ("family", "part_class"):
        raw = props.get(key)
        if raw:
            return str(raw).lower()
    return fallback


class GraphBackedCADDataset(Dataset):
    """Dataset that resolves training shards from a graph export.

    The graph can be the spaceflight registry graph, the processed-corpus graph,
    or the local data inventory graph. It extracts file-backed nodes, resolves
    their paths, and loads the actual geometry/shard arrays while exposing a
    compact metadata vector for graph conditioning.
    """

    def __init__(
        self,
        graph_path: str | Path,
        *,
        data_root: str | Path | None = None,
        num_points: int = 1024,
        num_fields: int = 3,
        node_types: tuple[str, ...] | None = None,
        limit: int | None = None,
        prefer_physics_shards: bool = True,
        physics_shards_only: bool = False,
        extra_search_roots: list[str | Path] | None = None,
    ):
        self.graph_path = Path(graph_path)
        self.data_root = Path(data_root) if data_root is not None else None
        self.num_points = num_points
        self.num_fields = num_fields
        self.node_types = set(node_types or tuple(_FILE_NODE_TYPES))
        self.prefer_physics_shards = prefer_physics_shards
        self.physics_shards_only = physics_shards_only
        # Always search repo-ish roots so artifacts/physics_shards and data/processed resolve.
        defaults: list[Path] = []
        for parent in list(self.graph_path.resolve().parents)[:6]:
            defaults.extend(
                [
                    parent,
                    parent / "artifacts",
                    parent / "data",
                    parent / "data" / "processed",
                ]
            )
        defaults.append(Path.cwd())
        extras = [Path(p).expanduser().resolve() for p in (extra_search_roots or [])]
        seen: set[str] = set()
        self.extra_search_roots: list[Path] = []
        for p in extras + defaults:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            self.extra_search_roots.append(p)
        payload = _load_graph_payload(self.graph_path)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        graph_root = self.data_root
        if graph_root is None:
            for key in ("data_root", "processed_dir"):
                if metadata.get(key):
                    graph_root = Path(str(metadata[key])).expanduser().resolve()
                    break
        self.graph_root = graph_root

        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        outgoing: dict[str, list[dict[str, Any]]] = {}
        incoming: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            outgoing.setdefault(edge.get("source", ""), []).append(edge)
            incoming.setdefault(edge.get("target", ""), []).append(edge)

        records: list[GraphDatasetRecord] = []
        for node in nodes:
            if node.get("type") not in self.node_types:
                continue
            path = _resolve_node_path(
                node,
                graph_path=self.graph_path,
                graph_root=self.graph_root,
                extra_roots=self.extra_search_roots,
            )
            if path is None:
                continue
            props = dict(node.get("properties", {}))
            # Prefer real solver-field shards (FEA FRD / CFD volume extracts) over
            # geometry-only placeholders when training for oneshot physics CAD.
            is_physics = (
                str(props.get("source") or "") == "solver_field_extract"
                or str(props.get("kind") or "") in {"fea", "cfd"}
                or "physics_shards" in str(path)
            )
            if physics_shards_only and not is_physics:
                continue
            records.append(
                GraphDatasetRecord(
                    node_id=str(node.get("id", "")),
                    node_type=str(node.get("type", "")),
                    label=str(node.get("label", path.name)),
                    path=path,
                    properties=props,
                )
            )
        # Physics-field TensorShards first so batch sampling hits real FEA/CFD
        # signal before geometry placeholders (critical for oneshot rocket specs).
        def _rank(rec: GraphDatasetRecord) -> tuple:
            props = rec.properties
            is_phys = int(
                str(props.get("source") or "") == "solver_field_extract"
                or str(props.get("kind") or "") in {"fea", "cfd"}
                or "physics_shards" in str(rec.path)
            )
            type_rank = {"TensorShard": 0, "Shard": 1, "Sample": 2, "Analogue": 3, "RawAsset": 4}.get(rec.node_type, 5)
            return (-is_phys, type_rank, str(rec.path), rec.node_id)

        records.sort(key=_rank)
        if limit is not None:
            records = records[: max(0, limit)]
        if not records:
            raise FileNotFoundError(f"No graph-backed training nodes found in {self.graph_path}")

        self.records = records
        self._outgoing = outgoing
        self._incoming = incoming
        self._type_code = {"TensorShard": 0.0, "Shard": 0.2, "Sample": 0.4, "Analogue": 0.6, "RawAsset": 1.0}
        self._node_by_id = {str(n.get("id", "")): n for n in nodes}
        self._assoc_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _ingest_part_payload(self, payload: dict[str, Any], neighbor: dict[str, Any]) -> None:
        """Fold Part/RealPart params, labels, materials, and FEA into assoc payload."""
        props = neighbor.get("properties", {}) or {}
        if payload["family"] is None:
            fam = _family_from_props(props)
            if fam:
                payload["family"] = fam

        params = props.get("params")
        if isinstance(params, dict) and not payload["params"]:
            for key, value in params.items():
                val = _as_float(value)
                if val is not None:
                    payload["params"][str(key)] = val
                elif key in _CATEGORICAL_PARAM_KEYS and value is not None:
                    payload.setdefault("text_parts", []).append(f"{key}={value}")
        elif isinstance(params, dict):
            for key in _CATEGORICAL_PARAM_KEYS:
                if params.get(key) is not None:
                    payload.setdefault("text_parts", []).append(f"{key}={params[key]}")

        # Mesh / structural stats on Part props (faces, watertight, extents).
        faces = _as_float(props.get("faces"))
        if faces is not None and "face_count" not in payload["geometry"]:
            payload["geometry"]["face_count"] = faces
        extents = props.get("extents_mm")
        if isinstance(extents, (list, tuple)) and len(extents) >= 3:
            for i, axis in enumerate(("bbox_x", "bbox_y", "bbox_z")):
                val = _as_float(extents[i])
                if val is not None and axis not in payload["geometry"]:
                    # extents are mm; geometry vector expects ~meters-ish scale via /10
                    payload["geometry"][axis] = float(val) / 1000.0
        if props.get("watertight") is not None:
            payload.setdefault("text_parts", []).append(
                f"watertight={1 if props.get('watertight') else 0}"
            )
        tags = props.get("tags")
        if isinstance(tags, list) and tags:
            payload.setdefault("text_parts", []).append(
                "tags=" + ",".join(str(t) for t in tags[:12])
            )

        # Material may live on Part props even without walking MADE_OF yet.
        if not payload["material"]:
            for key in ("density_kg_m3", "youngs_modulus_gpa", "yield_mpa", "ultimate_mpa",
                        "max_service_temp_k", "cte_1e6_k", "thermal_conductivity_w_mk"):
                val = _as_float(props.get(key))
                if val is not None:
                    payload["material"][key] = val
            if props.get("material_category") or props.get("category"):
                payload["material"]["category"] = str(props.get("material_category") or props.get("category")).lower()
            if props.get("material_id") and "material_id" not in payload["material"]:
                payload["material"]["material_id"] = str(props["material_id"])

        # Mass / mass-distribution (STL × density on Part props or top-level).
        mp = props.get("mass_properties") if isinstance(props.get("mass_properties"), dict) else {}
        if not isinstance(mp, dict):
            mp = {}
        top_mp = neighbor.get("mass_properties") if isinstance(neighbor.get("mass_properties"), dict) else {}
        if top_mp:
            mp = {**mp, **top_mp}
        mass = _as_float(props.get("mass_kg") if props.get("mass_kg") is not None else neighbor.get("mass_kg"))
        if mass is None:
            mass = _as_float(mp.get("mass_kg"))
        if mass is not None and "mass_kg" not in payload["physics"]:
            payload["physics"]["mass_kg"] = mass
        inertia = mp.get("inertia_kg_m2") if isinstance(mp.get("inertia_kg_m2"), dict) else props.get("inertia_kg_m2")
        if isinstance(inertia, dict):
            for src, dst in (("Ixx", "Ixx_kg_m2"), ("Iyy", "Iyy_kg_m2"), ("Izz", "Izz_kg_m2")):
                val = _as_float(inertia.get(src))
                if val is not None and dst not in payload["physics"]:
                    payload["physics"][dst] = val
        com = props.get("center_of_mass_m") or mp.get("center_of_mass_m")
        if isinstance(com, (list, tuple)) and len(com) >= 3:
            for i, axis in enumerate(("com_x_m", "com_y_m", "com_z_m")):
                val = _as_float(com[i])
                if val is not None and axis not in payload["geometry"]:
                    payload["geometry"][axis] = val
        vol = _as_float(props.get("volume_m3") if props.get("volume_m3") is not None else mp.get("volume_m3"))
        if vol is not None and vol > 0 and "log_volume" not in payload["geometry"]:
            import math
            payload["geometry"]["log_volume"] = math.log10(vol)

        fea = neighbor.get("simulation_results_fea")
        if isinstance(fea, dict):
            for key in ("max_stress_mpa", "mean_stress_mpa", "max_displacement_mm", "safety_factor"):
                val = _as_float(fea.get(key))
                if val is not None and key not in payload["physics"]:
                    payload["physics"][key] = val
            parse = fea.get("parse")
            if isinstance(parse, dict):
                for key in ("max_stress_mpa", "mean_stress_mpa", "max_displacement_mm"):
                    val = _as_float(parse.get(key))
                    if val is not None and key not in payload["physics"]:
                        payload["physics"][key] = val
        cfd = neighbor.get("simulation_results_cfd")
        if isinstance(cfd, dict):
            for src, dst in (
                ("Cd", "Cd"),
                ("cd", "Cd"),
                ("CL_alpha_per_rad", "CL_alpha_per_rad"),
                ("U_mag_max", "U_mag_max"),
                ("U_mag_mean", "U_mag_max"),  # fallback into same slot if max missing
                ("p_mean", "p_mean"),
                ("p_delta", "p_delta"),
            ):
                val = _as_float(cfd.get(src))
                if val is not None and dst not in payload["physics"]:
                    payload["physics"][dst] = val
            if "p_delta" not in payload["physics"]:
                pmax = _as_float(cfd.get("p_max"))
                pmin = _as_float(cfd.get("p_min"))
                if pmax is not None and pmin is not None:
                    payload["physics"]["p_delta"] = abs(pmax - pmin)
            # pressure_drop_bar slot from kinematic Δp when bar-scale missing
            if "pressure_drop_bar" not in payload["physics"] and payload["physics"].get("p_delta") is not None:
                payload["physics"]["pressure_drop_bar"] = float(payload["physics"]["p_delta"])
        # Hybrid text: Part.spec_prompt + linked doc excerpts live in payload["text"]
        prompt = props.get("spec_prompt")
        if isinstance(prompt, str) and prompt.strip():
            payload.setdefault("text_parts", []).append(prompt.strip())
        for key in ("Cd", "U_mag_max", "p_mean", "p_delta"):
            val = _as_float(props.get(key))
            if val is not None and key not in payload["physics"]:
                payload["physics"][key] = val
    def _walk_associations(self, node_id: str, depth: int = 3) -> dict[str, Any]:
        """Gather association payloads (family, physics, geometry, material, params)
        by walking typed TAO edges around a file node up to ``depth`` hops."""
        if node_id in self._assoc_cache:
            return self._assoc_cache[node_id]

        payload: dict[str, Any] = {
            "family": None,
            "physics": {},
            "geometry": {},
            "material": {},
            "params": {},
            "text_parts": [],
        }
        # Seed from the file node itself (associated Samples carry params/family).
        seed = self._node_by_id.get(node_id)
        if seed is not None:
            seed_props = seed.get("properties", {}) or {}
            payload["family"] = _family_from_props(seed_props)
            params = seed_props.get("params")
            if isinstance(params, dict):
                for key, value in params.items():
                    val = _as_float(value)
                    if val is not None:
                        payload["params"][str(key)] = val
            elif isinstance(seed_props.get("parametric_summary"), dict):
                for key, value in seed_props["parametric_summary"].items():
                    val = _as_float(value)
                    if val is not None:
                        payload["params"][str(key)] = val

        seen: set[str] = set()
        frontier = [node_id]
        for _ in range(depth):
            next_frontier: list[str] = []
            for nid in frontier:
                if nid in seen:
                    continue
                seen.add(nid)
                for edge in self._outgoing.get(nid, []) + self._incoming.get(nid, []):
                    etype = str(edge.get("type", ""))
                    if etype and etype not in _ASSOC_EDGE_TYPES:
                        continue
                    for neighbor_id in (str(edge.get("source", "")), str(edge.get("target", ""))):
                        if neighbor_id in seen or not neighbor_id:
                            continue
                        neighbor = self._node_by_id.get(neighbor_id)
                        if neighbor is None:
                            continue
                        ntype = str(neighbor.get("type", ""))
                        if ntype not in _ASSOC_NODE_TYPES:
                            # Bridge through non-assoc nodes only if they are file/sample hubs.
                            if ntype in _FILE_NODE_TYPES or ntype in {"Part", "RealPart"}:
                                next_frontier.append(neighbor_id)
                            continue
                        props = neighbor.get("properties", {}) or {}
                        if payload["family"] is None:
                            fam = _family_from_props(props)
                            if fam:
                                payload["family"] = fam
                        if ntype in ("Part", "RealPart"):
                            self._ingest_part_payload(payload, neighbor)
                        if ntype == "Document":
                            dprops = neighbor.get("properties", {}) or {}
                            excerpt = dprops.get("text") or dprops.get("content") or dprops.get("title")
                            if isinstance(excerpt, str) and excerpt.strip():
                                payload.setdefault("text_parts", []).append(excerpt.strip()[:2000])
                        if ntype == "SimulationCase":
                            sprops = neighbor.get("properties", {}) or {}
                            for key in ("kind", "solver", "status"):
                                val = sprops.get(key)
                                if isinstance(val, str) and val:
                                    payload.setdefault("text_parts", []).append(f"{key}={val}")
                        if ntype in ("TuningGuidance", "DesignTarget", "PhysicsTarget"):
                            targets = props.get("targets")
                            if isinstance(targets, dict):
                                for key, value in targets.items():
                                    val = _as_float(value)
                                    if val is not None and key not in payload["physics"]:
                                        payload["physics"][str(key)] = val
                            for key in ("max_stress_mpa", "observed_objective"):
                                val = _as_float(props.get(key))
                                if val is not None and key not in payload["physics"]:
                                    payload["physics"][key] = val
                        if ntype == "GeometricMetric":
                            for key in ("volume", "log_volume", "aspect_ratio_xy", "aspect_ratio_xz",
                                        "compactness", "bbox_x", "bbox_y", "bbox_z", "face_count"):
                                val = _as_float(props.get(key))
                                if val is not None and key not in payload["geometry"]:
                                    payload["geometry"][key] = val
                        if ntype == "Material":
                            for key in ("density_kg_m3", "youngs_modulus_gpa", "yield_mpa", "ultimate_mpa",
                                        "max_service_temp_k", "cte_1e6_k", "thermal_conductivity_w_mk"):
                                val = _as_float(props.get(key))
                                if val is not None and key not in payload["material"]:
                                    payload["material"][key] = val
                            cat = props.get("category") or props.get("material_category")
                            if cat and "category" not in payload["material"]:
                                payload["material"]["category"] = str(cat).lower()
                        if ntype == "Dimension":
                            # Only accept dimensions while params are still empty, or matching existing keys.
                            name = str(props.get("name") or "")
                            val = _as_float(props.get("value"))
                            if name and val is not None:
                                if not payload["params"] or name in payload["params"]:
                                    if name not in payload["params"]:
                                        payload["params"][name] = val
                        # Expand further only through hub nodes (Parts / file bridges).
                        if ntype not in _ASSOC_LEAF_TYPES:
                            next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        # If family still missing, classify from nearest Part blob via part_family.
        if payload["family"] is None:
            for nid in seen:
                node = self._node_by_id.get(nid)
                if node is None or node.get("type") not in {"Part", "RealPart"}:
                    continue
                try:
                    from cadflow.part_family import classify_part
                    payload["family"] = classify_part(node)
                except Exception:
                    pass
                break

        self._assoc_cache[node_id] = payload
        return payload

    def _association_vector(self, record: GraphDatasetRecord) -> torch.Tensor:
        """Family + physics + geometry + material + generation params."""
        from cadflow.physics_targets import CONDITIONING_QUANTITIES

        assoc = self._walk_associations(record.node_id)

        family_vec = [0.0] * len(FAMILY_VOCAB)
        fam = assoc.get("family")
        if fam in _FAMILY_INDEX:
            family_vec[_FAMILY_INDEX[fam]] = 1.0
        else:
            # Map aliased physics families into vocab when rocket-specific missing.
            try:
                from cadflow.physics_targets import resolve_family
                resolved = resolve_family(fam)
            except Exception:
                resolved = "generic"
            family_vec[_FAMILY_INDEX.get(resolved if resolved in _FAMILY_INDEX else "generic", 0)] = 1.0

        physics = assoc.get("physics", {})
        physics_vec = []
        for name, scale in CONDITIONING_QUANTITIES:
            value = physics.get(name)
            physics_vec.append(float(value) * scale if isinstance(value, (int, float)) else 0.0)

        geom = assoc.get("geometry", {})
        geometry_vec = [
            min(float(geom.get("log_volume", 0.0)) / 6.0 + 0.5, 1.0),
            min(float(geom.get("aspect_ratio_xy", 1.0)) / 10.0, 1.0),
            min(float(geom.get("aspect_ratio_xz", 1.0)) / 10.0, 1.0),
            min(float(geom.get("compactness", 0.5)), 1.0),
            min(float(geom.get("bbox_x", 0.0)) / 10.0, 1.0),
            min(float(geom.get("bbox_y", 0.0)) / 10.0, 1.0),
            min(float(geom.get("bbox_z", 0.0)) / 10.0, 1.0),
            min(float(geom.get("face_count", 0.0)) / 200.0, 1.0),
            # COM in meters, scaled for part-scale (~0–1 m)
            min(abs(float(geom.get("com_x_m", 0.0))) / 1.0, 1.0),
            min(abs(float(geom.get("com_y_m", 0.0))) / 1.0, 1.0),
            min(abs(float(geom.get("com_z_m", 0.0))) / 2.0, 1.0),
        ]

        mat = assoc.get("material", {})
        cat = str(mat.get("category") or "")
        cat_code = (_MATERIAL_CAT_INDEX.get(cat, 0) + 1) / max(len(_MATERIAL_CATEGORIES), 1)
        material_vec = [
            min(float(mat.get("density_kg_m3", 0.0)) / 10000.0, 1.0),
            min(float(mat.get("youngs_modulus_gpa", 0.0)) / 250.0, 1.0),
            min(float(mat.get("yield_mpa", 0.0)) / 2000.0, 1.0),
            min(float(mat.get("ultimate_mpa", 0.0)) / 2000.0, 1.0),
            min(float(mat.get("max_service_temp_k", 0.0)) / 2000.0, 1.0),
            min(float(mat.get("cte_1e6_k", 0.0)) / 30.0, 1.0),
            min(float(mat.get("thermal_conductivity_w_mk", 0.0)) / 400.0, 1.0),
            float(cat_code),
        ]

        params = assoc.get("params", {})
        param_vec = []
        for name, scale in PARAM_QUANTITIES:
            value = params.get(name)
            param_vec.append(min(float(value) * scale, 1.5) if isinstance(value, (int, float)) else 0.0)

        # Hashed bag-of-tokens from spec_prompt + linked Document text
        text_vec = [0.0] * TEXT_META_DIM
        blobs = assoc.get("text_parts") or []
        if isinstance(blobs, list) and blobs:
            joined = " ".join(str(b) for b in blobs[:8]).lower()
            tokens = re.findall(r"[a-z0-9_./-]{2,}", joined)[:400]
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                text_vec[h % TEXT_META_DIM] += 1.0
            s = sum(text_vec) or 1.0
            text_vec = [min(v / s * 4.0, 1.0) for v in text_vec]

        return torch.tensor(
            family_vec + physics_vec + geometry_vec + material_vec + param_vec + text_vec,
            dtype=torch.float32,
        )

    def _metadata_vector(self, record: GraphDatasetRecord) -> torch.Tensor:
        props = record.properties
        size_bytes = props.get("size_bytes")
        if size_bytes is None:
            try:
                size_bytes = record.path.stat().st_size
            except OSError:
                size_bytes = 0
        path_depth = len(record.path.parts)
        index_val = props.get("index")
        if index_val is None:
            index_val = 0
        source_flag = 1.0 if any(key in props for key in ("source_path", "source_shard_path", "manifest_path", "source_key")) else 0.0
        summary = props.get("summary") if isinstance(props.get("summary"), dict) else {}
        feature_summary = props.get("feature_summary") if isinstance(props.get("feature_summary"), dict) else {}
        parametric_summary = props.get("parametric_summary") if isinstance(props.get("parametric_summary"), dict) else {}
        physical_summary = props.get("physical_summary") if isinstance(props.get("physical_summary"), dict) else {}
        analogue_flag = 1.0 if record.node_type == "Analogue" else 0.0
        return torch.tensor(
            [
                float(np.log1p(float(size_bytes)) / 20.0),
                float(min(path_depth, 20) / 20.0),
                float(self._type_code.get(record.node_type, 0.5)),
                float(min(float(index_val), 4096.0) / 4096.0),
                source_flag,
                analogue_flag,
                float(min(len(summary), 20) / 20.0),
                float(min(len(feature_summary), 20) / 20.0),
                float(min(len(parametric_summary) + len(physical_summary), 40) / 40.0),
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        points, fields, max_stress = _load_sample_arrays(record.path, num_points=self.num_points, num_fields=self.num_fields)
        legacy = self._metadata_vector(record)
        assoc = self._association_vector(record)
        assoc_payload = self._walk_associations(record.node_id)
        sample: dict[str, torch.Tensor] = {
            "points": torch.from_numpy(points),
            "fields": torch.from_numpy(fields),
            "is_synthetic": torch.tensor(0, dtype=torch.long),
            "graph_metadata": torch.cat([legacy, assoc]),
        }
        fea_stress = assoc_payload.get("physics", {}).get("max_stress_mpa")
        if isinstance(fea_stress, (int, float)):
            sample["max_stress"] = torch.tensor(float(fea_stress), dtype=torch.float32)
        elif max_stress is not None:
            sample["max_stress"] = torch.tensor(float(max_stress), dtype=torch.float32)
        else:
            stress_col = min(2, fields.shape[-1] - 1)
            sample["max_stress"] = torch.tensor(float(fields[:, stress_col].max()), dtype=torch.float32)
        return sample
