"""CT xy slice at fixed z with the intersection curves of several tifxyz surfaces overlaid.

Usage: python xy_compare.py <z0> <xlo:xhi> <ylo:yhi> <out.png> --surf name1,name2 [--paths DIR] [--scale 1]
       [--fibers dir]   (horizontal fibre points within 2 vx of z0 drawn as dots)
Colours in order: orange, blue, green, magenta, cyan.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("z0", type=int)
ap.add_argument("xr")
ap.add_argument("yr")
ap.add_argument("out")
ap.add_argument("--surf", required=True)
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fibers")
args = ap.parse_args()
x0, x1 = map(int, args.xr.split(":"))
y0, y1 = map(int, args.yr.split(":"))
pr = Predictor()
ct = pr.ct[args.z0 - 1:args.z0 + 2, y0:y1, x0:x1].astype(np.float32).mean(0)
lo, hi = np.percentile(ct, [1, 99])
img = cv2.cvtColor(np.clip((ct - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
s = args.scale
img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
cols = [(0, 140, 255), (255, 80, 40), (60, 220, 60), (220, 60, 220), (230, 230, 0)]


def section(seg, z0):
    """Polyline segments where the surface grid crosses z = z0 (marching over quad edges)."""
    X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    ok = (X != -1) & (Z > 0)
    pts = []
    for A, B in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        za, zb = Z[A], Z[B]
        m = ok[A] & ok[B] & ((za - z0) * (zb - z0) <= 0) & (za != zb)
        t = np.where(m, (z0 - za) / np.where(za == zb, 1, zb - za), 0)
        px = X[A] + t * (X[B] - X[A])
        py = Y[A] + t * (Y[B] - Y[A])
        pts.append(np.column_stack([px[m], py[m]]))
    return np.vstack(pts)


for k, name in enumerate(n for n in args.surf.split(",") if n):
    P = section(Path(args.paths) / name, args.z0)
    c = cols[k % len(cols)]
    for q in P:
        if x0 <= q[0] < x1 and y0 <= q[1] < y1:
            cv2.circle(img, (int((q[0] - x0) * s), int((q[1] - y0) * s)), 2, c, -1)
    cv2.putText(img, name[:40], (5, 15 + 15 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
if args.fibers:
    for p in Path(args.fibers).glob("*.path.npy"):
        a = np.load(p)[:, :3]
        m = (np.abs(a[:, 2] - args.z0) <= 3) & (a[:, 0] >= x0) & (a[:, 0] < x1) & (a[:, 1] >= y0) & (a[:, 1] < y1)
        if not m.any():
            continue
        h = p.name[0] == "h"
        for q in a[m][::3]:
            cv2.circle(img, (int((q[0] - x0) * s), int((q[1] - y0) * s)), 1 if h else 4, (0, 0, 255) if h else (0, 255, 0), -1)
        if not h:
            q = a[m][0]
            cv2.putText(img, p.name[2:7], (int((q[0] - x0) * s) + 5, int((q[1] - y0) * s) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
for x in range((x0 // 100 + 1) * 100, x1, 100):
    cv2.putText(img, str(x), (int((x - x0) * s) - 14, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
for y in range((y0 // 50 + 1) * 50, y1, 50):
    cv2.putText(img, str(y), (2, int((y - y0) * s) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(args.out, img)
print(f"wrote {args.out}")
