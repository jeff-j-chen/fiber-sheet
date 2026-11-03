"""Find straight strand lines in the sheet's uv grid along which the horizontal-fibre probability stays high.

Usage: python line_search.py <hz_uv_grid.npy> [--row0 48 --col0 95] [--top 12]
Grid: rows 48..84 step 0.25, columns 95..334 step 1 (see the sampling script). For each (row at column 215, slope)
the mean probability and the fraction of columns with p > 0.5 along the line are computed; the best distinct lines are listed
with the sheet position (x, y, z) of the line at column 215.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_trace import Sheet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("grid")
ap.add_argument("--sheet", default="fiber_only_20261102b_flatboi")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--zlo", type=float, default=11400)
ap.add_argument("--zhi", type=float, default=12200)
args = ap.parse_args()
P = np.load(args.grid)
rows = np.arange(48, 84, 0.25)
cols = np.arange(95, 335, 1.0)
sh = Sheet(args.sheet)
c = np.arange(100, 331, 1.0)
out = []
for r0 in np.arange(50, 82, 0.25):
    for s in np.arange(-0.05, 0.0501, 0.005):
        r = r0 + s * (c - 215)
        ri = (r - 48) / 0.25
        ci = c - 95
        v = map_coordinates(np.nan_to_num(P, nan=-1), [ri, ci], order=1, mode="nearest")
        valid = v >= 0
        if valid.mean() < 0.9:
            continue
        vv = v[valid]
        out.append((float((vv > 0.5).mean()), float(vv.mean()), r0, s))
out.sort(key=lambda t: (-t[0], -t[1]))
kept = []
for f, m, r0, s in out:
    if all(abs(r0 - k[2]) > 0.75 for k in kept):
        kept.append((f, m, r0, s))
    if len(kept) >= args.top * 3:
        break
print("frac(p>0.5)  mean p  row@col215  slope  -> sheet xyz at col 215, z at col 120 / 330")
n = 0
for f, m, r0, s in kept:
    uv = np.array([[r0, 215.0], [r0 + s * (120 - 215), 120.0], [r0 + s * (330 - 215), 330.0]])
    xyz = sh.xyz(uv)
    if not (args.zlo <= xyz[0, 2] <= args.zhi):
        continue
    print(f"  {f:5.2f}  {m:5.2f}  {r0:6.2f}  {s:+.3f}  -> ({xyz[0, 0]:.0f}, {xyz[0, 1]:.0f}, {xyz[0, 2]:.0f}); z {xyz[1, 2]:.0f} / {xyz[2, 2]:.0f}")
    n += 1
    if n >= args.top:
        break
