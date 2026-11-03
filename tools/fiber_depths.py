import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# fibre depth along the reference surface normal, tabulated: fiber_depths.py <ref_tifxyz> <fiber_dir> [h|v] [step]
ref, fdir = sys.argv[1], sys.argv[2]
fam = sys.argv[3] if len(sys.argv) > 3 else "h"
step = float(sys.argv[4]) if len(sys.argv) > 4 else 250
X, Y, Z = (cv2.imread(os.path.join(ref, f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1)
gs = 1 / json.load(open(os.path.join(ref, "meta.json")))["scale"][0]
xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
f = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False)


def depth(p):
    x, y, z = p.T
    fx = (f(np.c_[z, x + 2]) - f(np.c_[z, x - 2])) / 4
    fz = (f(np.c_[z + 2, x]) - f(np.c_[z - 2, x])) / 4
    n = np.c_[fx, -np.ones_like(fx), fz]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return (y - f(np.c_[z, x])) * n[:, 1]


pa, sa = (0, 2) if fam == "h" else (2, 0)
grid = np.arange(2000, 6601, step) if fam == "h" else np.arange(10900, 13801, step * 0.6)
rows = []
for p in glob.glob(os.path.join(fdir, f"{fam}_*.path.npy")):
    a = np.load(p).astype(np.float64)
    a = a[np.argsort(a[:, pa])]
    rows.append((np.median(a[:, sa]), os.path.basename(p)[:-9], a))
print(f"{'':22s}" + "".join(f"{g:6.0f}" for g in grid))
for _, n, a in sorted(rows):
    cells = []
    for g in grid:
        if a[0, pa] <= g <= a[-1, pa]:
            q = np.stack([np.interp(g, a[:, pa], a[:, c]) for c in range(3)])[None]
            cells.append(f"{float(depth(q)[0]):+6.0f}")
        else:
            cells.append("     .")
    print(f"{n:22s}" + "".join(cells))
