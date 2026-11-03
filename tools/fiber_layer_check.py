"""Check the layer model against every overlapping pair (ply-aware same-sheet test)."""
import csv
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
L = json.loads((d / "layers.json").read_text())
lay, p, S = L["fibre_layer"], L["ply_offset_vx"], L["spacing_vx"]
rows = [r for r in csv.DictReader(open(d / "crossings.csv")) if r["verdict"] != "no overlap"]


def z_of(r):
    return float(r["at"].strip("[]").split()[2])


def fits(r, dl):
    """Offset within half a layer of the canonical position for layer difference dl."""
    return abs(float(r["normal_offset_vx"]) - (dl * S + p)) < S / 2


same = [r for r in rows if lay[r["vertical"]] == lay[r["horizontal"]]]
ok = [r for r in same if fits(r, 0)]
print(f"same-layer pairs: {len(same)}; at the canonical position ({p - S / 2:+.1f} .. {p + S / 2:+.1f} vx): {len(ok)} "
      f"({len(ok) / max(len(same), 1) * 100:.0f}%)")
diff = [r for r in rows if lay[r["vertical"]] != lay[r["horizontal"]]]
okd = [r for r in diff if fits(r, lay[r["horizontal"]] - lay[r["vertical"]])]
print(f"different-layer pairs: {len(diff)}; offset in the dL window: {len(okd)}")
for v in sorted({r["vertical"] for r in rows}):
    rs = sorted([r for r in rows if r["vertical"] == v], key=z_of)
    good = sum(fits(r, lay[r["horizontal"]] - lay[v]) for r in rs)
    print(f"  {v:20s} layer {lay[v]}  fits {good}/{len(rs)}  offsets by z: " +
          " ".join(f"z{z_of(r):.0f}:{float(r['normal_offset_vx']):+.0f}" for r in rs))
