"""Score uv guide lines (blends of bracketing kept horizontals) against the horizontal-probability grid for a target z.

Usage: python guide_scan.py <sheet> <fibre dir with kept h_*> <target z> [--tol 100] [--top 12]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_trace import Sheet, neighbour_lines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sheet")
ap.add_argument("nbr")
ap.add_argument("z", type=float)
ap.add_argument("--tol", type=float, default=100)
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--grid", required=True, help="npz from build_hz_grid.py")
args = ap.parse_args()
G = np.load(args.grid)
P = np.nan_to_num(G["P"], nan=-1)
ROW0 = float(G["rows"][0])
COL0 = float(G["cols"][0])
sh = Sheet(args.sheet)
lines = neighbour_lines(sh, [args.nbr], "h")
cols = np.array(G["cols"], float)
MID = float(cols[len(cols) // 2])
# rows at column 215 of every neighbour, to find the bracketing pair for the target
ref = sorted((float(L(MID)), n, L) for n, L in lines)
zr = [(sh.xyz(np.array([[r, MID]]))[0, 2], r, n, L) for r, n, L in ref]
print("neighbours (z at col 215, row):", [(round(z), round(r, 1), n[:12]) for z, r, n, L in zr])
below = [t for t in zr if t[0] <= args.z]
above = [t for t in zr if t[0] > args.z]
zb, rb, nb, Lb = max(below, key=lambda t: t[0]) if below else (None,) * 4
za, ra, na, La = min(above, key=lambda t: t[0]) if above else (None,) * 4
print("bracketing:", nb, na)
res = []
for w in np.linspace(-0.1, 1.1, 241):
    g = (1 - w) * Lb(cols) + w * La(cols) if (Lb is not None and La is not None) else None
    ri = (g - ROW0) / 0.25
    if ri.min() < 0 or ri.max() > P.shape[0] - 1:
        continue
    v = map_coordinates(P, [ri, cols - COL0], order=1, mode="nearest")
    ok = v >= 0
    if ok.mean() < 0.85:
        continue
    ii = (len(cols) // 8, len(cols) // 2, len(cols) * 7 // 8)
    zs = sh.xyz(np.array([[g[i], cols[i]] for i in ii]))[:, 2]
    res.append((float((v[ok] > 0.5).mean()), float(v[ok].mean()), w, zs))
sel = [r for r in res if abs(r[3][1] - args.z) <= args.tol]
sel.sort(key=lambda t: (-t[0], -t[1]))
kept = []
for r in sel:
    if all(abs(r[2] - k[2]) > 0.02 for k in kept):
        kept.append(r)
print("frac(p>.5)  mean p  w   z at col 125 / 215 / 305")
for f, m, w, zs in kept[: args.top]:
    print(f"  {f:5.2f}  {m:5.2f}  {w:5.2f}   {zs[0]:.0f} / {zs[1]:.0f} / {zs[2]:.0f}")
