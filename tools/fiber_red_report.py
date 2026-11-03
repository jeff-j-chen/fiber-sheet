"""Which fibres account for the off-model ('red') crossings, under the ply-aware layer model."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

d = Path(sys.argv[1])
L = json.loads((d / "layers.json").read_text())
lay, p, S = L["fibre_layer"], L["ply_offset_vx"], L["spacing_vx"]
seeds = {}
for line in (Path(sys.argv[2]).read_text().splitlines() if len(sys.argv) > 2 else []):
    parts = line.split()
    if len(parts) >= 4 and parts[2] == "seed":
        seeds[parts[0]] = float(parts[3])

bad, total, worst = Counter(), Counter(), {}
for r in csv.DictReader(open(d / "crossings.csv")):
    if r["verdict"] == "no overlap":
        continue
    v, h, o = r["vertical"], r["horizontal"], float(r["normal_offset_vx"])
    dl = lay[h] - lay[v]
    res = o - (dl * S + p)
    for n in (v, h):
        total[n] += 1
        if abs(res) >= S / 2:
            bad[n] += 1
            worst.setdefault(n, []).append((round(res, 1), v if n == h else h))
print(f"model: spacing {S} vx, ply offset {p:+} vx (red = more than half a layer from the canonical position)\n")
print(f"{'fibre':22s} layer  red/total  seed->point  residuals at red crossings")
for n in sorted(total, key=lambda n: (-bad[n] / total[n], n)):
    res = ", ".join(f"{r:+.0f} ({o.split('_')[0]}{o.split('_')[1]})" for r, o in sorted(worst.get(n, []))[:6])
    sd = f"{seeds[n]:5.1f} vx" if n in seeds else "   -"
    print(f"{n:22s} {lay[n]:5d}  {bad[n]:3d}/{total[n]:<3d}    {sd}    {res}")
