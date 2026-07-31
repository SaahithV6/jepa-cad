#!/usr/bin/env python3
"""Densify the TAO train graph for hybrid text + CAD + physics training.

Why this exists
---------------
Disk holds ~140GB of FRD/OpenFOAM + ~865MB of PDFs/docs, but the train graph
(~277MB) mostly stored *scalar* FEA/CFD summaries and Document nodes with
titles only (no text). Cd was never written. Modal would have trained on a
thin slice of the real corpus.

This script (idempotent, graph-locked):
  1. Extracts document text (pdftotext / utf-8) onto Document nodes
  2. Links Documents ↔ Parts (DESCRIBES / MENTIONS / DOCUMENTS)
  3. Derives Cd_proxy + airflow fields from bodyfit/internal CFD metas
  4. Writes spec_prompt strings on Parts (params + material + aero + mass)
  5. Registers missing CFD/FEA physics shards into the graph
  6. Re-associates Samples / Dimensions / PhysicsTargets
  7. Writes artifacts/tao_densify_report.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from cadflow.graph_lock import graph_lock, read_graph, write_graph_atomic  # noqa: E402
from cadflow.build_physics_shards import (  # noqa: E402
    CFD_MANIFEST,
    FEA_MANIFEST,
    register_manifest_to_graph,
)

GRAPH = ROOT / "artifacts/jepa-train-bundle/graph.json"
REPORT = ROOT / "artifacts/tao_densify_report.json"

DOC_EXTS = {".pdf", ".md", ".txt", ".rst", ".csv", ".html"}
TEXT_MAX = 12_000


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s[:120] or "x"


def _stem_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    ta, tb = set(re.findall(r"[a-z0-9]+", a)), set(re.findall(r"[a-z0-9]+", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_text(path: Path) -> str:
    if not path.is_file():
        return ""
    suf = path.suffix.lower()
    try:
        if suf == ".pdf":
            r = subprocess.run(
                ["pdftotext", "-layout", "-q", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            text = r.stdout or ""
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:TEXT_MAX]


def cd_proxy_from_metrics(m: dict[str, Any]) -> float | None:
    """Nondimensional aero signature from kinematic OpenFOAM fields.

    Bodyfit cases rarely write forceCoeffs; use pressure span / U^2 as a
    Cd-like conditioning signal so the model sees airflow physics.
    """
    try:
        u = float(m.get("U_inlet") or m.get("U_mag_mean") or 0.0)
        pmax = float(m.get("p_max") or 0.0)
        pmin = float(m.get("p_min") or 0.0)
    except (TypeError, ValueError):
        return None
    if u < 1e-6:
        return None
    # OpenFOAM simpleFoam often stores p/rho → (Δp)/U² ≈ Cd-scale O(0.1–2)
    val = abs(pmax - pmin) / (u * u)
    return float(max(0.01, min(val, 3.0)))


def load_cfd_metas() -> dict[str, dict[str, Any]]:
    """part_id / case_id → metrics from rocket bodyfit + legacy roots."""
    out: dict[str, dict[str, Any]] = {}
    roots = [
        ROOT / "artifacts/rocket_cfd_bodyfit",
        ROOT / "artifacts/cfd_bodyfit",
        ROOT / "artifacts/cfd_internal",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for meta in root.glob("*/meta.json"):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            metrics = dict(m.get("metrics") or {})
            if m.get("U_inlet") is not None:
                metrics.setdefault("U_inlet", m["U_inlet"])
            if float(metrics.get("U_mag_max") or 0) <= 1e-6:
                continue
            pid = str(m.get("part_id") or meta.parent.name)
            out[pid] = metrics
            out[meta.parent.name] = metrics
    return out


def densify_documents(graph: dict[str, Any], stats: dict[str, Any]) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    by_id = {n["id"]: n for n in nodes}
    edge_ids = {e["id"] for e in edges}

    # Index parts for linking
    parts = [n for n in nodes if n.get("type") in ("Part", "RealPart")]
    part_stems: list[tuple[str, str, str]] = []  # stem, id, family
    for p in parts:
        props = p.get("properties") or {}
        fam = str(props.get("family") or props.get("part_class") or "")
        label = str(p.get("label") or p["id"])
        stem = Path(str(props.get("stl") or props.get("geometry_ref") or label)).stem
        part_stems.append((stem.lower(), p["id"], fam.lower()))

    text_filled = 0
    linked = 0
    new_docs = 0

    # Fill text on existing Document nodes
    for n in nodes:
        if n.get("type") != "Document":
            continue
        props = n.setdefault("properties", {})
        existing = str(props.get("text") or props.get("content") or "")
        if len(existing) > 50:
            continue
        path = props.get("source_path") or props.get("path") or props.get("manifest_path")
        if not path:
            continue
        p = Path(str(path))
        if not p.is_file():
            # try relative to repo
            cand = ROOT / str(path)
            if cand.is_file():
                p = cand
            else:
                continue
        text = extract_text(p)
        if len(text) < 40:
            continue
        props["text"] = text
        props["text_chars"] = len(text)
        props["text_sha1"] = hashlib.sha1(text.encode()).hexdigest()[:16]
        props["title"] = props.get("title") or p.stem
        text_filled += 1

        # Link to best-matching parts
        doc_low = p.stem.lower()
        scored: list[tuple[float, str]] = []
        for stem, pid, fam in part_stems:
            score = _stem_similarity(doc_low, stem)
            # family keyword boost from text head
            head = text[:800].lower()
            if fam and fam in head:
                score = max(score, 0.45)
            if score >= 0.35:
                scored.append((score, pid))
        scored.sort(reverse=True)
        for score, pid in scored[:5]:
            eid = f"edge:{n['id']}:describes:{pid}"
            if eid in edge_ids:
                continue
            edges.append(
                {
                    "id": eid,
                    "type": "DESCRIBES",
                    "source": n["id"],
                    "target": pid,
                    "properties": {"similarity": round(score, 3), "method": "densify_text"},
                }
            )
            edge_ids.add(eid)
            # reverse MENTIONS for walkability
            mid = f"edge:{pid}:mentions:{n['id']}"
            if mid not in edge_ids:
                edges.append(
                    {
                        "id": mid,
                        "type": "MENTIONS",
                        "source": pid,
                        "target": n["id"],
                        "properties": {"similarity": round(score, 3)},
                    }
                )
                edge_ids.add(mid)
            linked += 1

    # Discover docs on disk not yet in graph
    raw_roots = [
        ROOT / "data/raw_downloads/external",
        ROOT / "data/raw_downloads/nasa3d",
        ROOT / "data/spaceflight_components",
    ]
    for raw in raw_roots:
        if not raw.is_dir():
            continue
        for dp in raw.rglob("*"):
            if not dp.is_file() or dp.suffix.lower() not in DOC_EXTS:
                continue
            if dp.stat().st_size < 200:
                continue
            try:
                rel = str(dp.relative_to(ROOT))
            except ValueError:
                rel = str(dp)
            nid = f"document:{_slug(rel)}"
            if nid in by_id:
                continue
            text = extract_text(dp)
            props = {
                "source_path": str(dp),
                "relative_path": rel,
                "doc_type": {
                    ".pdf": "drawing",
                    ".md": "readme",
                    ".txt": "notes",
                    ".rst": "documentation",
                    ".csv": "test_data",
                    ".html": "webpage",
                }.get(dp.suffix.lower(), "document"),
                "size_bytes": dp.stat().st_size,
                "title": dp.stem,
            }
            if len(text) >= 40:
                props["text"] = text
                props["text_chars"] = len(text)
                props["text_sha1"] = hashlib.sha1(text.encode()).hexdigest()[:16]
                text_filled += 1
            node = {"id": nid, "type": "Document", "label": dp.stem, "properties": props}
            nodes.append(node)
            by_id[nid] = node
            new_docs += 1

    stats["documents_text_filled"] = text_filled
    stats["document_part_links_added"] = linked
    stats["documents_created"] = new_docs


def densify_cfd_physics(graph: dict[str, Any], stats: dict[str, Any]) -> None:
    metas = load_cfd_metas()
    cd_n = 0
    airflow_n = 0
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        pid = str(n.get("id") or "")
        bare = pid.replace("part:rocket:", "").replace("part:", "")
        metrics = metas.get(pid) or metas.get(bare) or metas.get(n.get("label") or "")
        # also try case_id on properties
        props = n.setdefault("properties", {})
        case_id = props.get("cfd_case_id") or (n.get("simulation_results_cfd") or {}).get("case_id")
        if not metrics and case_id:
            metrics = metas.get(str(case_id))
        if not metrics:
            continue

        cfd = n.get("simulation_results_cfd")
        if not isinstance(cfd, dict):
            cfd = {}
        # copy airflow fields
        for k in (
            "U_mag_max",
            "U_mag_mean",
            "U_x_mean",
            "p_min",
            "p_max",
            "p_mean",
            "U_inlet",
            "nu",
            "mesh",
            "solver",
            "n_cells_sampled",
        ):
            if metrics.get(k) is not None and cfd.get(k) is None:
                cfd[k] = metrics[k]
        u = float(metrics.get("U_mag_max") or 0)
        if u > 1e-6:
            airflow_n += 1
        cd = cd_proxy_from_metrics(metrics)
        if cd is not None:
            cfd["Cd"] = round(cd, 6)
            cfd["Cd_source"] = "pressure_span_over_U2"
            cfd["p_delta"] = round(abs(float(metrics.get("p_max") or 0) - float(metrics.get("p_min") or 0)), 6)
            cd_n += 1
        cfd["has_flow_field"] = True
        n["simulation_results_cfd"] = cfd
        n["has_cfd"] = True

        # Mirror into PhysicsTarget-friendly props for association walk
        props["Cd"] = cfd.get("Cd")
        props["U_mag_max"] = cfd.get("U_mag_max")
        props["p_mean"] = cfd.get("p_mean")
        props["p_delta"] = cfd.get("p_delta")

    stats["parts_with_Cd"] = cd_n
    stats["parts_with_airflow"] = airflow_n


def write_spec_prompts(graph: dict[str, Any], stats: dict[str, Any]) -> None:
    """Human-readable generation prompts for hybrid text→CAD conditioning."""
    n_written = 0
    for n in graph["nodes"]:
        if n.get("type") != "Part":
            continue
        props = n.setdefault("properties", {})
        params = props.get("params") if isinstance(props.get("params"), dict) else {}
        fam = props.get("family") or props.get("part_class") or "part"
        mat = props.get("material_name") or props.get("material_id") or props.get("material_category") or "unspecified"
        mass = n.get("mass_kg") or props.get("mass_kg")
        cfd = n.get("simulation_results_cfd") if isinstance(n.get("simulation_results_cfd"), dict) else {}
        fea = n.get("simulation_results_fea") if isinstance(n.get("simulation_results_fea"), dict) else {}
        bits = [f"Generate a spaceflight {fam}"]
        for k in (
            "diameter_mm",
            "length_mm",
            "height_mm",
            "thickness_mm",
            "wall_mm",
            "root_chord_mm",
            "tip_chord_mm",
            "sweep_mm",
            "throat_diameter_mm",
            "expansion_ratio",
            "shape",
            "nose_shape",
        ):
            if k in params and params[k] is not None:
                bits.append(f"{k}={params[k]}")
        bits.append(f"material={mat}")
        if mass is not None:
            bits.append(f"mass_kg={mass}")
        if cfd.get("Cd") is not None:
            bits.append(f"Cd={cfd['Cd']}")
        if cfd.get("U_mag_max") is not None:
            bits.append(f"U_mag_max={round(float(cfd['U_mag_max']), 4)}")
        if cfd.get("p_delta") is not None:
            bits.append(f"p_delta={cfd['p_delta']}")
        if fea.get("max_stress_mpa") is not None:
            bits.append(f"max_stress_mpa={fea['max_stress_mpa']}")
        elif isinstance(fea.get("parse"), dict) and fea["parse"].get("max_stress_mpa") is not None:
            bits.append(f"max_stress_mpa={fea['parse']['max_stress_mpa']}")
        prompt = "; ".join(str(b) for b in bits)
        props["spec_prompt"] = prompt
        props["spec_prompt_sha1"] = hashlib.sha1(prompt.encode()).hexdigest()[:16]
        n_written += 1
    stats["spec_prompts_written"] = n_written


def link_simulation_cases(graph: dict[str, Any], stats: dict[str, Any]) -> None:
    """Ensure rocket Parts with FEA/CFD have SIMULATED_IN edges to case hubs."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    edge_ids = {e["id"] for e in edges}
    by_id = {n["id"]: n for n in nodes}
    added = 0
    for n in nodes:
        if n.get("type") != "Part":
            continue
        pid = str(n["id"])
        props = n.get("properties") or {}
        for kind, key in (("fea", "fea_case_id"), ("cfd", "cfd_case_id")):
            case = props.get(key)
            if not case:
                sim = n.get(f"simulation_results_{kind}")
                if isinstance(sim, dict):
                    case = sim.get("case_id") or sim.get("part_id")
            if not case:
                if pid.startswith("part:rocket:"):
                    case = pid.replace("part:rocket:", "")
                else:
                    continue
            sid = f"simulationcase:rocket-{kind}-{_slug(str(case))}"
            if sid not in by_id:
                node = {
                    "id": sid,
                    "type": "SimulationCase",
                    "label": f"{kind}:{case}",
                    "properties": {
                        "solver": "calculix" if kind == "fea" else "openfoam",
                        "kind": kind,
                        "case_id": str(case),
                        "status": "completed",
                        "source": "densify_tao_hybrid",
                    },
                }
                nodes.append(node)
                by_id[sid] = node
            eid = f"edge:{pid}:simulated_in:{sid}"
            if eid not in edge_ids:
                edges.append(
                    {
                        "id": eid,
                        "type": "SIMULATED_IN",
                        "source": pid,
                        "target": sid,
                        "properties": {"kind": kind},
                    }
                )
                edge_ids.add(eid)
                added += 1
    stats["simulation_case_links_added"] = added


def register_shards(stats: dict[str, Any]) -> None:
    try:
        reg = register_manifest_to_graph(GRAPH, manifest_paths=(FEA_MANIFEST, CFD_MANIFEST))
        stats["shard_register"] = reg
    except Exception as exc:  # noqa: BLE001
        stats["shard_register"] = {"error": str(exc)}


def validate(graph: dict[str, Any]) -> dict[str, Any]:
    types = Counter(n.get("type") for n in graph["nodes"])
    edges = Counter(e.get("type") for e in graph["edges"])
    docs_text = sum(
        1
        for n in graph["nodes"]
        if n.get("type") == "Document"
        and len(str((n.get("properties") or {}).get("text") or "")) > 50
    )
    cd = sum(
        1
        for n in graph["nodes"]
        if n.get("type") == "Part"
        and isinstance(n.get("simulation_results_cfd"), dict)
        and n["simulation_results_cfd"].get("Cd") is not None
    )
    prompts = sum(
        1
        for n in graph["nodes"]
        if n.get("type") == "Part" and (n.get("properties") or {}).get("spec_prompt")
    )
    return {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "node_types": dict(types.most_common(25)),
        "edge_types": dict(edges.most_common(25)),
        "documents_with_text": docs_text,
        "parts_with_Cd": cd,
        "parts_with_spec_prompt": prompts,
        "TensorShard": types.get("TensorShard", 0),
        "Document": types.get("Document", 0),
        "SimulationCase": types.get("SimulationCase", 0),
        "DESCRIBES": edges.get("DESCRIBES", 0),
        "MENTIONS": edges.get("MENTIONS", 0),
        "ready_for_hybrid_train": docs_text >= 200 and cd >= 2000 and prompts >= 10000,
    }


def main() -> int:
    t0 = time.time()
    stats: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print("densify: loading graph", flush=True)
    with graph_lock(GRAPH):
        graph = read_graph(GRAPH)
        densify_documents(graph, stats)
        print(f"  docs text={stats.get('documents_text_filled')} links={stats.get('document_part_links_added')} new={stats.get('documents_created')}", flush=True)
        densify_cfd_physics(graph, stats)
        print(f"  Cd={stats.get('parts_with_Cd')} airflow={stats.get('parts_with_airflow')}", flush=True)
        write_spec_prompts(graph, stats)
        link_simulation_cases(graph, stats)
        print(f"  prompts={stats.get('spec_prompts_written')} sim_links={stats.get('simulation_case_links_added')}", flush=True)
        meta = graph.setdefault("metadata", {})
        meta["tao_densify"] = {**stats, "finished_partial": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_graph_atomic(GRAPH, graph)
        print("densify: graph written", flush=True)

    # Shard register uses its own lock
    print("densify: registering physics shards", flush=True)
    register_shards(stats)

    # Associate training wiring
    try:
        from cadflow.associate_training_data import associate_graph_file

        assoc = associate_graph_file(GRAPH)
        stats["associate"] = assoc
        print(f"densify: associate {assoc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        stats["associate_error"] = str(exc)
        print(f"densify: associate error {exc}", flush=True)

    with graph_lock(GRAPH):
        graph = read_graph(GRAPH)
        stats["validate"] = validate(graph)
        stats["elapsed_s"] = round(time.time() - t0, 1)
        graph.setdefault("metadata", {})["tao_densify"] = stats
        write_graph_atomic(GRAPH, graph)

    REPORT.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    print(json.dumps(stats["validate"], indent=2), flush=True)
    print(f"wrote {REPORT}", flush=True)
    return 0 if stats["validate"].get("ready_for_hybrid_train") else 1


if __name__ == "__main__":
    raise SystemExit(main())
