"""Assign traced fibres to papyrus layers from their signed crossing offsets.

Model: for a vertical/horizontal pair that overlaps, the signed offset of the horizontal fibre along
the sheet normal is  offset = (L_h - L_v) * S + p, where S = layer spacing and p = ply offset (the two
fibre families lie on opposite faces of the same sheet). S, p and the integer layers L are fitted.

Usage: python fiber_layers.py <fiber_dir> [--max-gap 60]
Reads crossings.csv (from fiber_crossings.py). Writes layers.json.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--max-gap", type=float, default=60.0)
ap.add_argument("--min-spacing", type=float, default=12.0, help="smallest layer spacing tried (vx)")
ap.add_argument("--max-spacing", type=float, default=40.0)
args = ap.parse_args()
fdir = Path(args.fibers)
rows = [r for r in csv.DictReader(open(fdir / "crossings.csv"))
        if r["verdict"] != "no overlap" and float(r["distance_vx"]) <= args.max_gap]
names = sorted({r["vertical"] for r in rows} | {r["horizontal"] for r in rows})
edges = [(r["vertical"], r["horizontal"], float(r["normal_offset_vx"])) for r in rows]
print(f"{len(edges)} overlapping vertical/horizontal pairs among {len(names)} fibres")

offs = np.array([e[2] for e in edges])
print("signed offsets (+ = horizontal on the +normal side), sorted:")
print("  " + " ".join(f"{o:+.0f}" for o in np.sort(offs)))


def assign(S, p):
    """Layer per fibre by robust iterative averaging (start: all 0)."""
    L = {n: 0.0 for n in names}
    for _ in range(50):
        acc = defaultdict(list)
        for v, h, o in edges:
            d = (o - p) / S                              # = L_h - L_v
            acc[h].append(L[v] + d)
            acc[v].append(L[h] - d)
        L = {n: float(np.median(acc[n])) if acc[n] else 0.0 for n in names}
    Li = {n: int(round(L[n] - L[names[0]])) for n in names}
    res = np.array([o - ((Li[h] - Li[v]) * S + p) for v, h, o in edges])
    return Li, res


best = None
for S in np.arange(args.min_spacing, args.max_spacing + 0.01, 0.5):
    # canonical side: the horizontal ply sits on the +normal side of its own vertical ply (0 < p < S),
    # so the touching sheet's vertical (at p - S) can never be read as the same sheet
    for p in np.arange(0.5, S, 0.5):
        Li, res = assign(S, p)
        cost = np.sum(np.minimum(np.abs(res), S / 2) ** 2)    # robust: outliers count at most (S/2)^2
        if best is None or cost < best[0]:
            best = (cost, S, p, Li, res)
cost, S, p, Li, res = best
base = min(Li.values())
Li = {n: v - base for n, v in Li.items()}
print(f"\nbest fit: layer spacing S = {S:.1f} vx ({S * 9.362:.0f} um), ply offset p = {p:+.1f} vx; "
      f"pairs explained within 3 vx: {(np.abs(res) <= 3).mean() * 100:.0f}%, "
      f"median |residual| {np.median(np.abs(res)):.1f} vx")

layers = defaultdict(list)
for n, l in Li.items():
    layers[l].append(n)
print("\nlayers (0 = innermost along -normal):")
for l in sorted(layers):
    print(f"  layer {l}: {', '.join(sorted(layers[l]))}")
print("\npairs the model explains worst (possible tracing errors or fibre jumps):")
order = np.argsort(-np.abs(res))
for i in order[:8]:
    v, h, o = edges[i]
    print(f"  {v:20s} x {h:20s} offset {o:+6.1f}  model {(Li[h] - Li[v]) * S + p:+6.1f}  residual {res[i]:+6.1f}")
(fdir / "layers.json").write_text(json.dumps(
    {"spacing_vx": S, "ply_offset_vx": p, "layers": {str(k): sorted(v) for k, v in sorted(layers.items())},
     "fibre_layer": Li}, indent=1))
print("\nwrote", fdir / "layers.json")
