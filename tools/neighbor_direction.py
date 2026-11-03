"""Which way did a neighbour surface go? Signed offset along the source normal and change in distance
from a scroll-centre estimate (circle fitted to the source surface in the xy plane).

Usage: python neighbor_direction.py <source_tifxyz> <neighbour_tifxyz>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load(d):
    g = np.stack([cv2.imread(str(Path(d) / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float64)
    return g, (g != -1).all(-1)


S, sok = load(sys.argv[1])
N, nok = load(sys.argv[2])
n = np.cross(np.gradient(S, axis=1), np.gradient(S, axis=0))
n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
good = sok & np.isfinite(n).all(-1)
sp, sn = S[good], n[good]
d, k = cKDTree(sp).query(N[nok])
off = ((N[nok] - sp[k]) * sn[k]).sum(1)
m = d < 300
# circle fit (algebraic) to the source xy
xy = sp[:, :2]
A = np.column_stack([2 * xy, np.ones(len(xy))])
cx, cy, c = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
if len(sys.argv) > 4:
    cx, cy, c = float(sys.argv[3]), float(sys.argv[4]), 0.0
r_src = np.hypot(sp[k[m], 0] - cx, sp[k[m], 1] - cy)
r_nb = np.hypot(N[nok][m, 0] - cx, N[nok][m, 1] - cy)
print(f"neighbour vs source: median offset along source normal {np.median(off[m]):+.1f} vx "
      f"({(off[m] > 0).mean() * 100:.0f}% positive); fitted centre ({cx:.0f}, {cy:.0f}), "
      f"radius {np.sqrt(c + cx * cx + cy * cy):.0f}; neighbour is {np.median(r_nb - r_src):+.1f} vx "
      f"{'farther from' if np.median(r_nb - r_src) > 0 else 'closer to'} the centre")
