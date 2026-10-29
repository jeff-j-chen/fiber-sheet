"""3D version of band_check: label hz > thr in a box around the fibre and report the component under each point.

Usage: python band_check3d.py <fiber.path.npy> <xlo:xhi> [--thr 0.5] [--pad 6] [--every 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import CORE, Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fiber")
ap.add_argument("xr")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--pad", type=int, default=6)
ap.add_argument("--every", type=float, default=10.0)
ap.add_argument("--fam", default="hz")
args = ap.parse_args()
x0, x1 = map(int, args.xr.split(":"))
a = np.load(args.fiber)[:, :3]
ax = 0 if args.fam == "hz" else 2
a = a[(a[:, ax] >= x0) & (a[:, ax] < x1)]
lo = np.floor(a.min(0)).astype(int) - [40, 40, args.pad]
hi = np.ceil(a.max(0)).astype(int) + [40, 40, args.pad]
if ax == 2:
    lo, hi = np.floor(a.min(0)).astype(int) - [args.pad, 40, 40], np.ceil(a.max(0)).astype(int) + [args.pad, 40, 40]
pr = Predictor()
for kz in range(lo[2] // CORE, hi[2] // CORE + 1):
    for ky in range(lo[1] // CORE, hi[1] // CORE + 1):
        for kx in range(lo[0] // CORE, hi[0] // CORE + 1):
            pr.ensure((kz, ky, kx))
vol = pr.root[args.fam][lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32) / 255
lab, n = ndimage.label(vol > args.thr, structure=np.ones((3, 3, 3)))
sizes = np.bincount(lab.ravel())
arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
prev = None
for s in np.arange(0, arc[-1], args.every):
    q = a[int(np.searchsorted(arc, s))]
    i = np.round(q[::-1] - lo[::-1]).astype(int)
    w = lab[max(i[0] - 1, 0):i[0] + 2, max(i[1] - 1, 0):i[1] + 2, max(i[2] - 1, 0):i[2] + 2]
    ids = w[w > 0]
    c = int(np.bincount(ids).argmax()) if len(ids) else 0
    runs = ""
    if c and ax == 0:
        col = lab[i[0], :, i[2]] == c
        starts = np.flatnonzero(np.diff(np.concatenate([[0], col.astype(int)])) == 1) + lo[1]
        runs = f", same component at y {starts.tolist()}"
    if c != prev or (s % 50 < args.every):
        print(f"{'xyz'[ax]} {q[ax]:6.0f} ({q.round().astype(int).tolist()}): component {c} (size {sizes[c] if c else 0}){runs}")
        prev = c
print(f"{n} components in box {lo.tolist()}..{hi.tolist()}")
