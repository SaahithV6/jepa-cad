"""Harvest NASA TPSX thermal-protection material properties.

The design loop selects a skin material by max service temperature. Its
catalogue holds about forty materials, of which ten are TPS, and the loop
already hits the end of it: for 25 kg to 4,000 km the skin reaches 854 K and
the search returns Inconel 718 because nothing lighter survives. Above roughly
1,250 K it returns nothing at all and the honest answer becomes "needs ablative
or reusable TPS", which is a real answer the loop cannot currently act on.

TPSX is NASA Ames' thermal protection materials database: public, no login, no
API, ~1,500 materials across 32 categories, each record carrying density,
conductivity, specific heat, emissivity and -- the field that matters here --
Multiple Use Temperature Limit and Single Use Temperature Limit. Those two are
precisely the selection variable, and the distinction between them is one the
catalogue does not currently make: a tile that survives 1,590 K every flight is
a different design decision from an ablator that survives 1,760 K once.

Values are at standard conditions only; TPSX has no temperature-dependent
curves and says so. That limitation is carried into the records rather than
papered over.

Polite by construction: one request at a time, a delay between them, and a stop
after a run of empty ids rather than walking the whole space.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://tpsx.arc.nasa.gov/Material?id={}"

#: TPSX lays each property out as
#:     Property | Value | Units | Uncertainty | Source | STP
#: so the units are stated and there is no reason to guess them. An earlier
#: version of this parser read only the value and filed it under a name ending
#: in _mpa or _gpa, which turned LI-900's 469 kPa tensile strength into
#: "469000 MPa" -- a number under a label asserting something it is not, which
#: is the same defect that produced a fabricated stress label in this corpus.
#: Field names here carry no units; the unit TPSX reported travels with the
#: value, along with whether it was measured or predicted.
PROPERTY_KEYS = (
    "density",
    "multiple use temperature limit",
    "single use temperature limit",
    "thermal conductivity",
    "specific heat",
    "emissivity",
    "tensile strength",
    "compressive strength",
    "tensile modulus",
    "coefficient of thermal expansion",
)


def fetch(url: str, timeout: int = 45) -> str | None:
    try:
        proc = subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                              capture_output=True, text=True, timeout=timeout + 15)
        return proc.stdout if proc.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def parse_record(html: str) -> dict | None:
    """Pull the name and property rows out of one TPSX material page."""
    if not html or "Material" not in html:
        return None
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)

    name = None
    m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", text, flags=re.S | re.I)
    if m:
        name = re.sub(r"<[^>]+>", " ", m.group(1))
        name = " ".join(name.split())
    if not name:
        return None

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I)
    props: dict = {}
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        cells = [" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in cells]
        if len(cells) < 3:
            continue
        # property names carry a trailing footnote index: "Density 1"
        label = re.sub(r"\s+\d+$", "", cells[0]).strip().rstrip(":")
        low = label.lower()
        if not any(low.startswith(k) for k in PROPERTY_KEYS):
            continue
        num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cells[1])
        if not num:
            continue
        entry = {
            "value": float(num.group()),
            "unit": cells[2].strip(),
            "source": cells[4].strip() if len(cells) > 4 else "",
        }
        if len(cells) > 3:
            unc = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cells[3])
            if unc:
                entry["uncertainty"] = float(unc.group())
        # "Thermal Conductivity (Thru-the-Thickness)" and "(In-Plane)" are two
        # different quantities; keying on the bare property name kept only
        # whichever came last.
        props[label] = entry
    return {"name": name, "properties": props} if props else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--max-id", type=int, default=400)
    ap.add_argument("--delay", type=float, default=0.7,
                    help="seconds between requests; this is a courtesy to a "
                         "public government server, not a tuning knob")
    ap.add_argument("--stop-after-empty", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/materials/tpsx_materials.json")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    found: list[dict] = []
    empty_run = 0

    for mat_id in range(args.start, args.max_id + 1):
        html = fetch(BASE.format(mat_id))
        rec = parse_record(html) if html else None
        if rec is None:
            empty_run += 1
            if empty_run >= args.stop_after_empty:
                print(f"stopping: {empty_run} consecutive ids with no record")
                break
        else:
            empty_run = 0
            rec["tpsx_id"] = mat_id
            rec["source"] = BASE.format(mat_id)
            found.append(rec)
            if len(found) % 25 == 0:
                print(f"  {len(found)} materials so far (id {mat_id})", flush=True)
                args.out.write_text(json.dumps(found, indent=1))
        time.sleep(args.delay)

    args.out.write_text(json.dumps(found, indent=1))
    print(f"\n{len(found)} TPSX materials harvested -> {args.out}")

    def use_limit(rec):
        best = 0.0
        for label, entry in rec["properties"].items():
            if "temperature limit" in label.lower() and entry["unit"].upper() == "K":
                best = max(best, entry["value"])
        return best

    with_limit = [r for r in found if use_limit(r) > 0]
    print(f"{len(with_limit)} carry a use-temperature limit in kelvin "
          f"(the field skin selection keys on)")
    if with_limit:
        temps = sorted(use_limit(r) for r in with_limit)
        print(f"  limits span {temps[0]:.0f} K to {temps[-1]:.0f} K")
        print(f"  above the 1250 K ceiling of the current catalogue: "
              f"{sum(1 for t in temps if t > 1250)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
