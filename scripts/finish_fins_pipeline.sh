#!/usr/bin/env bash
# Finish realistic fins: FEA → shards → mass sidecar → TAO ingest picks up.
set -euo pipefail
cd /home/best/jepa-cad
mkdir -p artifacts/logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBER_OF_CPUS=1

echo "$(date -Is) finish_fins_pipeline begin" | tee -a artifacts/logs/finish_fins_pipeline.log

# Stop any existing FEA parent (leave CFD/ingest alone).
if [[ -f artifacts/rocket_fea.pid ]]; then
  old=$(cat artifacts/rocket_fea.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "stopping old FEA pid=${old}" | tee -a artifacts/logs/finish_fins_pipeline.log
    kill -TERM "${old}" 2>/dev/null || true
    sleep 2
    pkill -P "${old}" 2>/dev/null || true
    sleep 1
    kill -KILL "${old}" 2>/dev/null || true
  fi
fi
pkill -f 'run_rocket_physics_8k.py --fea-only' 2>/dev/null || true
sleep 2

# Purge stale fin shards + drop fin lines from FEA manifest so new FRDs re-extract.
.venv/bin/python -u <<'PY' | tee -a artifacts/logs/finish_fins_pipeline.log
from pathlib import Path
import json

fea = Path("artifacts/rocket_fea_8k")
shards = Path("artifacts/physics_shards/fea")
man = Path("artifacts/physics_shards/fea_manifest.jsonl")
good = {
    p.parent.name
    for p in fea.glob("fin_*/case.frd")
    if p.is_file() and p.stat().st_size >= 50_000
}
removed = forced = 0
for p in shards.glob("fin_*.npz"):
    if p.stem not in good:
        p.unlink(missing_ok=True)
        removed += 1
    else:
        p.unlink(missing_ok=True)
        forced += 1
dropped = 0
kept: list[str] = []
if man.is_file():
    for line in man.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = str(rec.get("part_id") or "")
        case = str(rec.get("case_id") or "")
        if pid.startswith("part:rocket:fin_") or case.startswith("fin_"):
            dropped += 1
            continue
        kept.append(line)
    man.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
print(
    {
        "orphan_shards_removed": removed,
        "good_shards_cleared": forced,
        "manifest_fin_lines_dropped": dropped,
        "good_fin_frds": len(good),
    },
    flush=True,
)
PY

# Start fins-only FEA.
echo "$(date -Is) starting fins-only FEA" | tee -a artifacts/logs/finish_fins_pipeline.log artifacts/logs/fea_fins.log
nohup .venv/bin/python -u run_rocket_physics_8k.py \
  --fea-only --workers 3 --skip-register --no-ingest \
  --families fin \
  --target-tets 12000 --mesh-timeout 180 --timeout 300 \
  >> artifacts/logs/fea_fins.log 2>&1 &
echo $! > artifacts/rocket_fea.pid
echo "fea_pid=$(cat artifacts/rocket_fea.pid)" | tee -a artifacts/logs/finish_fins_pipeline.log

# Extract shards for current good FRDs (non-fin pending + cleared fins).
nohup .venv/bin/python -u run_physics_shards.py --source rocket --workers 2 --num-points 2048 \
  >> artifacts/logs/physics_shards_rocket.log 2>&1 &
echo "shard_pid=$!" | tee -a artifacts/logs/finish_fins_pipeline.log

# Faster shard watcher while fins FEA runs.
if [[ -f artifacts/fin_shard_watch.pid ]] && kill -0 "$(cat artifacts/fin_shard_watch.pid)" 2>/dev/null; then
  echo "shard watch already up" | tee -a artifacts/logs/finish_fins_pipeline.log
else
  nohup bash -c '
    while true; do
      sleep 90
      cd /home/best/jepa-cad
      .venv/bin/python -u run_physics_shards.py --source rocket --workers 2 --num-points 2048 \
        >> artifacts/logs/physics_shards_rocket.log 2>&1
    done
  ' >/dev/null 2>&1 &
  echo $! > artifacts/fin_shard_watch.pid
  echo "fin_shard_watch=$(cat artifacts/fin_shard_watch.pid)" | tee -a artifacts/logs/finish_fins_pipeline.log
fi

# Fin mass → sidecar merge (ingest loop applies to graph).
nohup .venv/bin/python -u <<'PY' >> artifacts/logs/fin_mass_enrich.log 2>&1 &
from pathlib import Path
import json
from cadflow.rocket_physics_suite import load_manifest, DEFAULT_CORPUS
from cadflow.mass_properties import mass_properties_from_stl
from cadflow.enrich_tao_mass_properties import SIDECAR, DEFAULT_DENSITY

man = load_manifest(Path("data/openrocket_hardware_8k"))
fins = [e for e in man if e.get("family") == "fin"]
payload = {"by_part_id": {}}
if SIDECAR.is_file():
    try:
        payload = json.loads(SIDECAR.read_text(encoding="utf-8"))
        if not isinstance(payload.get("by_part_id"), dict):
            payload["by_part_id"] = {}
    except Exception:
        payload = {"by_part_id": {}}
by = payload["by_part_id"]
updated = failed = 0
for i, e in enumerate(fins):
    pid = f"part:rocket:{e['part_id']}"
    stl = DEFAULT_CORPUS / e["stl"]
    mat = e.get("material") if isinstance(e.get("material"), dict) else {}
    dens = float(mat.get("density_kg_m3") or DEFAULT_DENSITY)
    prev = by.get(pid) if isinstance(by.get(pid), dict) else {}
    if prev.get("density_kg_m3"):
        dens = float(prev["density_kg_m3"])
    mp_obj = mass_properties_from_stl(stl, dens, extents_mm=e.get("extents_mm"))
    if mp_obj is None:
        failed += 1
        continue
    mp = mp_obj.as_dict()
    mp["material_id"] = prev.get("material_id") or e.get("material_id") or mat.get("id")
    by[pid] = mp
    updated += 1
    if (i + 1) % 200 == 0:
        print(f"mass {i+1}/{len(fins)} updated={updated} failed={failed}", flush=True)
payload["by_part_id"] = by
tmp = SIDECAR.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
tmp.replace(SIDECAR)
print({"fin_mass_updated": updated, "failed": failed, "sidecar": str(SIDECAR)}, flush=True)
PY
echo "mass_pid=$!" | tee -a artifacts/logs/finish_fins_pipeline.log

sleep 8
ps -p "$(cat artifacts/rocket_fea.pid)" -o pid,etime,cmd | tee -a artifacts/logs/finish_fins_pipeline.log
tail -15 artifacts/logs/fea_fins.log | tee -a artifacts/logs/finish_fins_pipeline.log
echo "$(date -Is) finish_fins_pipeline launched" | tee -a artifacts/logs/finish_fins_pipeline.log
