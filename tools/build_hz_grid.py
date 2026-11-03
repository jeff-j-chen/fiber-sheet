"""Sample the horizontal-fibre probability on the sheet (uv grid) for the rows whose z at the middle column lies in [zlo, zhi].

Usage: python build_hz_grid.py <sheet> <zlo> <zhi> <out.npz>
P[i, j] = max of the probability over normal offsets -2, 0, +2 at (rows[i], cols[j]); NaN off the sheet.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402
from surf_trace import Sheet  # noqa: E402

sheet, zlo, zhi, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
sh = Sheet(sheet)
allr = np.arange(0, sh.X.shape[0] - 1, 0.25)
z215 = sh.xyz(np.column_stack([allr, np.full(len(allr), 115.0)]))[:, 2]
rows = allr[(z215 >= zlo) & (z215 <= zhi)]
cols = np.arange(5, sh.X.shape[1] - 4, 1.0)
print("rows %.2f..%.2f (%d), cols %d..%d" % (rows[0], rows[-1], len(rows), cols[0], cols[-1]), flush=True)
RR, CC = np.meshgrid(rows, cols, indexing="ij")
uv = np.column_stack([RR.ravel(), CC.ravel()])
ok = sh.on_sheet(uv)
xyz = sh.xyz(uv)
n = sh.normal(uv)
P = np.full(len(uv), np.nan)
idx = np.where(ok)[0]
best = np.zeros(len(idx))
for off in (-2.0, 0.0, 2.0):
    pts = xyz[idx] + off * n[idx]
    v = np.concatenate([core.sample("h", pts[i:i + 20000]) for i in range(0, len(pts), 20000)])
    best = np.maximum(best, v)
    print("offset", off, "done", flush=True)
P[idx] = best
np.savez(out, rows=rows, cols=cols, P=P.reshape(RR.shape))
print("saved", out)
