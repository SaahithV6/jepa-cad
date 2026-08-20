"""One mission specification -> two-stage vehicle -> both disciplines verified."""
import json, math, subprocess, sys
sys.path.insert(0, "/home/lain_iwakura/Documents/jepa-cad-vm")
from pathlib import Path
from generate_propulsion_trajectory_corpus import load_coupling
from solve_multistage_corpus import fly, build_stack
from scripts.params_to_physics_confirmed import run_confirmed

ROOT = "/home/lain_iwakura/Documents/jepa-cad-vm"
OUT = f"{ROOT}/artifacts/final_demo"
PROMPT = ("two-stage vehicle delivering 20 kg payload to 400 km apogee "
          "using lox/rp1 at 55 bar chamber pressure")

load_coupling()
print("SPECIFICATION")
print(f"  {PROMPT}")
print()

subprocess.run(["python", "scripts/infer_text_to_assembly.py",
    "--prompt", PROMPT, "--ckpt", "artifacts/mission_train_v14/latest.pt",
    "--out", OUT], cwd=ROOT, capture_output=True, text=True)
p = json.load(open(f"{OUT}/INFER_REPORT.json"))["decoded_params_mm"]

m_dry = p["struct_mass_kg"] + p["payload_kg"]
mr = math.exp(p["log_mass_ratio"])
gross = m_dry * mr
total_prop = gross - m_dry
stages, g, split = build_stack(total_prop, p["payload_kg"],
                               p["chamber_pressure_bar"] * 1e5, "lox_rp1")
p1, s1, p2, s2 = split

print("DESIGNED TWO-STAGE VEHICLE")
print(f"  payload                {p['payload_kg']:9.2f} kg")
print(f"  stage 1  propellant    {p1:9.2f} kg    structure {s1:8.2f} kg")
print(f"  stage 2  propellant    {p2:9.2f} kg    structure {s2:8.2f} kg")
print(f"  gross liftoff          {g:9.2f} kg    (mass ratio {mr:.2f})")
print(f"  chamber pressure       {p['chamber_pressure_bar']:9.2f} bar")
print(f"  stage 1 / 2 expansion  {stages[0].expansion_ratio:9.1f} / {stages[1].expansion_ratio:.1f}")
print(f"  airframe               {p['body_radius_mm']:9.2f} mm radius, "
      f"{p['body_height_mm']:.1f} mm long")

a, _, _, r = fly(total_prop, p["payload_kg"], p["chamber_pressure_bar"] * 1e5, "lox_rp1")
print()
print("DISCIPLINE 1 - FLOWN (staged gravity turn, exponential atmosphere)")
print(f"  apogee                 {a:9.1f} km   (asked 400)")
print(f"  downrange              {r['downrange_m']/1000:9.1f} km")
print(f"  max-Q                  {r['max_q_pa']/1000:9.1f} kPa")
print(f"  stage separation at    {r['separations'][0]:9.1f} s")

geom = {k: p[k] for k in ("body_radius_mm", "body_height_mm", "nose_radius_mm",
                          "nose_height_mm", "fin_span_mm", "fin_thickness_mm",
                          "fin_chord_mm", "fillet_radius_mm")}
geom.update({"cl_max_mm": 8.0, "cl_min_mm": 2.0})
thrust = g * 9.80665 * 4.5
rep = run_confirmed(params_mm=geom, out=Path(f"{OUT}/verify"),
    max_stress_mpa=200.0, max_disp_mm=3.0, max_iters=1,
    load_n=thrust, prompt=PROMPT)
acc = rep.get("accepted") or {}

print()
print("DISCIPLINE 2 - STRUCTURE (gmsh + CalculiX, own liftoff thrust)")
print(f"  applied load           {thrust:9.0f} N")
print(f"  max von Mises          {acc.get('max_von_mises_mpa', 0):9.2f} MPa  (limit 200)")
print(f"  max displacement       {acc.get('max_displacement_mm', 0):9.4f} mm   (limit 3.0)")
print(f"  solver                 {acc.get('solver_mode'):>9}")
print(f"  targets met            {str(acc.get('targets_met')):>9}")
print(f"  FRD bytes              {acc.get('frd_bytes'):9d}")
