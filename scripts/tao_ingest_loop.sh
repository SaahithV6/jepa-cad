#!/usr/bin/env bash
# Periodically ingest rocket FEA FRDs + bodyfit CFD metas into the live TAO
# graph, then re-apply the mass sidecar so mass_kg never gets wiped.
set -u
ROOT=/home/best/jepa-cad
cd "$ROOT" || exit 1
LOG=artifacts/logs/tao_ingest_loop.log
mkdir -p artifacts/logs
# Append, don't truncate: the loop gets restarted whenever this script changes
# (bash caches an already-parsed loop body) and history is worth keeping.
echo "$(date -Is) ==== loop start (pid $$) ====" >>"$LOG"
while true; do
  echo "$(date -Is) ingest begin" >>"$LOG"
  .venv/bin/python -u - <<'PY' >>"$LOG" 2>&1
from pathlib import Path
import json
from cadflow.rocket_physics_suite import ingest_fea_to_graph, DEFAULT_GRAPH, DEFAULT_FEA_ROOT, DEFAULT_CFD_ROOT
from cadflow.enrich_tao_mass_properties import apply_sidecar_to_graph

# FEA ingest (parses meta.json / FRDs under rocket_fea_8k)
linked_fea = ingest_fea_to_graph(DEFAULT_GRAPH, DEFAULT_FEA_ROOT)
print("fea_linked", linked_fea)

# Bodyfit CFD: read metas with success metrics and ingest
results = []
cfd_root = Path("artifacts/rocket_cfd_bodyfit")
if cfd_root.is_dir():
    for meta in cfd_root.glob("*/meta.json"):
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        if not m.get("success") and not (m.get("metrics") or {}).get("U_mag_max"):
            # accept either success flag or live velocity field
            metrics = m.get("metrics") or {}
            if not metrics:
                continue
        metrics = m.get("metrics") or {}
        if float(metrics.get("U_mag_max") or 0) <= 1e-6:
            continue
        results.append({
            "part_id": m.get("part_id") or meta.parent.name,
            "success": True,
            "metrics": metrics,
        })
if results:
    from cadflow.rocket_physics_suite import ingest_cfd_to_graph
    linked_cfd = ingest_cfd_to_graph(DEFAULT_GRAPH, results)
    print("cfd_linked", linked_cfd, "of", len(results))
else:
    print("cfd_linked 0")

n = apply_sidecar_to_graph()
print("mass_reapplied", n)

# Fold any newly-built real physics-field shards into the graph (single writer).
# The 253MB graph is rewritten non-atomically by the FEA/CFD ingest steps above, so a
# read here can land mid-write and raise JSONDecodeError. Retry instead of skipping a
# whole cycle's shards.
import time as _time
for _attempt in range(4):
    try:
        from cadflow.build_physics_shards import register_manifest_to_graph
        reg = register_manifest_to_graph(Path(DEFAULT_GRAPH))
        print("shard_register", reg)
        break
    except json.JSONDecodeError as exc:
        print("shard_register_torn_read", _attempt, exc)
        _time.sleep(20)
    except Exception as exc:  # noqa: BLE001
        print("shard_register_error", exc)
        break
else:
    print("shard_register_gave_up")
PY
  echo "$(date -Is) ingest end; sleeping 600s" >>"$LOG"
  sleep 600
done
