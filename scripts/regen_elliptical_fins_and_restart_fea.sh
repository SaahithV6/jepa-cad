#!/usr/bin/env bash
# Regen meshable elliptical fins, restart fins FEA with hull-fallback fix.
set -euo pipefail
cd /home/best/jepa-cad
mkdir -p artifacts/logs

echo "$(date -Is) regen elliptical fins + restart FEA" | tee -a artifacts/logs/finish_fins_pipeline.log

.venv/bin/python -u <<'PY' | tee -a artifacts/logs/finish_fins_pipeline.log
from pathlib import Path
from cadflow.rocket_physics_suite import load_manifest, DEFAULT_CORPUS, run_fea_for_entry, DEFAULT_FEA_ROOT, DEFAULT_CCX
from cadflow.rocket_hardware_generator import mesh_fin, write_stl

man = load_manifest(Path("data/openrocket_hardware_8k"))
n = ok = 0
for e in man:
    if e.get("family") != "fin":
        continue
    p = e.get("params") or {}
    if p.get("shape") != "elliptical":
        continue
    m = mesh_fin(
        height_mm=p["height_mm"],
        root_chord_mm=p["root_chord_mm"],
        tip_chord_mm=p["tip_chord_mm"],
        thickness_mm=p["thickness_mm"],
        sweep_mm=p.get("sweep_mm", 0.0),
        shape="elliptical",
    )
    write_stl(m, DEFAULT_CORPUS / e["stl"])
    n += 1
    if m.is_watertight and float(m.volume) > 0:
        ok += 1
    if n % 100 == 0:
        print(f"elliptical regen {n} ok={ok}", flush=True)
print({"elliptical_regen": n, "watertight_posvol": ok}, flush=True)

# smoke previously-failing case
e = next(x for x in man if x["part_id"] == "fin_01242")
r = run_fea_for_entry(
    e, DEFAULT_CORPUS, DEFAULT_FEA_ROOT, force=True, timeout=240,
    target_tets=12000, mesh_timeout_s=120, ccx=DEFAULT_CCX,
)
print({
    "smoke_fin_01242": r.success,
    "err": r.error,
    "tets": (r.metrics or {}).get("mesh_tets"),
    "stress": (r.metrics or {}).get("max_stress_mpa"),
}, flush=True)
PY

# Restart FEA so workers import new hull-fallback / mesh code
if [[ -f artifacts/rocket_fea.pid ]]; then
  old=$(cat artifacts/rocket_fea.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    kill -TERM "${old}" 2>/dev/null || true
    sleep 2
    pkill -P "${old}" 2>/dev/null || true
    sleep 1
    kill -KILL "${old}" 2>/dev/null || true
  fi
fi
pkill -f 'run_rocket_physics_8k.py --fea-only' 2>/dev/null || true
sleep 2

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBER_OF_CPUS=1
echo "$(date -Is) restart fins-only FEA after mesh fix" >> artifacts/logs/fea_fins.log
nohup .venv/bin/python -u run_rocket_physics_8k.py \
  --fea-only --workers 3 --skip-register --no-ingest \
  --families fin \
  --target-tets 12000 --mesh-timeout 180 --timeout 300 \
  >> artifacts/logs/fea_fins.log 2>&1 &
echo $! > artifacts/rocket_fea.pid
echo "fea_pid=$(cat artifacts/rocket_fea.pid)" | tee -a artifacts/logs/finish_fins_pipeline.log
sleep 6
ps -p "$(cat artifacts/rocket_fea.pid)" -o pid,etime,cmd | tee -a artifacts/logs/finish_fins_pipeline.log
tail -12 artifacts/logs/fea_fins.log | tee -a artifacts/logs/finish_fins_pipeline.log
