"""Sweep requested apogee for two-stage vehicles, decode, and fly the stack."""
import json, math, subprocess, sys
sys.path.insert(0, "/home/lain_iwakura/Documents/jepa-cad-vm")
from generate_propulsion_trajectory_corpus import load_coupling
from solve_multistage_corpus import fly

CKPT = sys.argv[1]
BASE = "/tmp/claude-1000/-home-lain-iwakura/6a071b7a-7311-4dbb-85f8-8d63c135e9bd/scratchpad"
load_coupling()

print(f"{'asked km':>9} {'flown km':>10} {'ratio':>8} {'gross kg':>10} {'sep s':>7}")
errs = []
for A in (50, 100, 200, 400, 800, 1500, 3000, 6000):
    out = f"{BASE}/v14_{A}"
    subprocess.run([
        "python", "scripts/infer_text_to_assembly.py",
        "--prompt", f"two-stage vehicle delivering 8 kg payload to {A} km apogee "
                    f"using lox/rp1 at 55 bar chamber pressure",
        "--ckpt", CKPT, "--out", out],
        cwd="/home/lain_iwakura/Documents/jepa-cad-vm",
        capture_output=True, text=True)
    p = json.load(open(f"{out}/INFER_REPORT.json"))["decoded_params_mm"]

    # gross follows from payload, structure and mass ratio; total propellant
    # is what the stack burns. The stage split is fixed by convention, so the
    # solver's build_stack reconstructs the whole vehicle from it.
    m_dry = p["struct_mass_kg"] + p["payload_kg"]
    mr = math.exp(p["log_mass_ratio"])
    gross = m_dry * mr
    total_prop = gross - m_dry

    a, g, split, r = fly(total_prop, p["payload_kg"],
                         p["chamber_pressure_bar"] * 1e5, "lox_rp1")
    errs.append(abs(math.log10(max(a, 0.1) / A)))
    seps = ",".join(f"{s:.0f}" for s in r["separations"]) or "-"
    print(f"{A:9d} {a:10.1f} {a/A:7.2f}x {g:10.1f} {seps:>7}")

print()
print(f"mean |log10 error| : {sum(errs)/len(errs):.3f} decades")
print(f"within 2x of target: {sum(1 for e in errs if e < 0.301)}/{len(errs)}")
print(f"within 20% of target: {sum(1 for e in errs if e < 0.0792)}/{len(errs)}")
