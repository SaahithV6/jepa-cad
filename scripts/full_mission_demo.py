"""One specification, all the way through both disciplines."""
import json, math, subprocess, sys
sys.path.insert(0, "/home/lain_iwakura/Documents/jepa-cad-vm")
from pathlib import Path
from generate_propulsion_trajectory_corpus import (
    PROPELLANTS, integrate_trajectory, load_coupling)
from solve_mission_corpus import CD
from scripts.params_to_physics_confirmed import run_confirmed

ROOT = "/home/lain_iwakura/Documents/jepa-cad-vm"
OUT = f"{ROOT}/artifacts/full_demo"
PROMPT = ("deliver 8 kg payload to 650 km apogee "
          "using lox/rp1 at 55 bar chamber pressure")

load_coupling()
print("SPEC:", PROMPT)
print()

subprocess.run(["python", "scripts/infer_text_to_assembly.py",
    "--prompt", PROMPT, "--ckpt", "artifacts/mission_train_v13/latest.pt",
    "--out", OUT], cwd=ROOT, capture_output=True, text=True)
p = json.load(open(f"{OUT}/INFER_REPORT.json"))["decoded_params_mm"]

m_dry = p["struct_mass_kg"] + p["payload_kg"]
mr = math.exp(p["log_mass_ratio"])
m0 = m_dry * mr
m_prop = m0 - m_dry

print("DESIGNED VEHICLE")
print(f"  payload            {p['payload_kg']:8.2f} kg")
print(f"  structure          {p['struct_mass_kg']:8.2f} kg")
print(f"  propellant         {m_prop:8.2f} kg")
print(f"  gross liftoff      {m0:8.2f} kg   (mass ratio {mr:.2f})")
print(f"  chamber pressure   {p['chamber_pressure_bar']:8.2f} bar")
print(f"  expansion ratio    {p['expansion_ratio']:8.2f}")
print(f"  throat area        {p['throat_area_mm2']:8.1f} mm2")
print(f"  airframe radius    {p['body_radius_mm']:8.2f} mm")
print(f"  airframe length    {p['body_height_mm']:8.2f} mm")

gamma, tc, mol = PROPELLANTS["lox_rp1"]
dia = max(0.08, (m0 / 1000.0) ** (1.0 / 3.0) * 0.5)
t = integrate_trajectory(m0=m0, m_prop=m_prop,
    throat_area=p["throat_area_mm2"] / 1e6,
    chamber_pressure=p["chamber_pressure_bar"] * 1e5, chamber_temp=tc,
    expansion_ratio=p["expansion_ratio"], gamma=gamma, mol_mass=mol,
    cd=CD, ref_area=math.pi * (dia / 2) ** 2,
    pitchover_time=8.0, pitchover_angle=math.radians(3.0), dt=0.2)

print()
print("DISCIPLINE 1 - FLOWN (gravity turn, exponential atmosphere)")
print(f"  apogee             {t['apogee_m']/1000:8.1f} km   (asked 650)")
print(f"  downrange          {t['downrange_m']/1000:8.1f} km")
print(f"  max-Q              {t['max_q_pa']/1000:8.1f} kPa")
print(f"  burnout            {t['burnout_s']:8.1f} s")

geom = {k: p[k] for k in ("body_radius_mm", "body_height_mm", "nose_radius_mm",
                          "nose_height_mm", "fin_span_mm", "fin_thickness_mm",
                          "fin_chord_mm", "fillet_radius_mm")}
geom.update({"cl_max_mm": 8.0, "cl_min_mm": 2.0})
thrust = m0 * 9.80665 * 5.0          # T/W 5, the corpus convention
rep = run_confirmed(params_mm=geom, out=Path(f"{OUT}/verify"),
    max_stress_mpa=200.0, max_disp_mm=3.0, max_iters=1,
    load_n=thrust, prompt=PROMPT)
acc = rep.get("accepted") or {}

print()
print("DISCIPLINE 2 - STRUCTURE (gmsh + CalculiX, under its own liftoff thrust)")
print(f"  applied load       {thrust:8.0f} N")
print(f"  max von Mises      {acc.get('max_von_mises_mpa', 0):8.2f} MPa   (limit 200)")
print(f"  max displacement   {acc.get('max_displacement_mm', 0):8.4f} mm   (limit 3.0)")
print(f"  solver mode        {acc.get('solver_mode')}")
print(f"  targets met        {acc.get('targets_met')}")
print(f"  FRD bytes          {acc.get('frd_bytes')}")
