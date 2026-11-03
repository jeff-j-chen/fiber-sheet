"""CT yz slice at fixed x with the intersection curves of several tifxyz surfaces and the fibres crossing it.

Usage: python yz_compare.py <x0> <ylo:yhi> <zlo:zhi> <out.png> --surf name1,name2 [--fibers dir] [--scale 1]
Image: horizontal axis y, vertical axis z (z grows downward). Colours: orange, blue, green, magenta, cyan.
Fibre points within 3 vx of x0: horizontals red dots, verticals green dots.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("x0", type=int)
ap.add_argument("yr")
ap.add_argument("zr")
ap.add_argument("out")
ap.add_argument("--surf", required=True)
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fibers")
args = ap.parse_args()
y0, y1 = map(int, args.yr.split(":"))
z0, z1 = map(int, args.zr.split(":"))
pr = Predictor()
ct = pr.ct[z0:z1, y0:y1, args.x0 - 1:args.x0 + 2].astype(np.float32).mean(2)
lo, hi = np.percentile(ct, [1, 99])
s = args.scale
img = cv2.cvtColor(np.clip((ct - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
cols = [(0, 140, 255), (255, 80, 40), (60, 220, 60), (220, 60, 220), (230, 230, 0)]


def section(seg, x0):
    X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    ok = (X != -1) & (Z > 0)
    pts = []
    for A, B in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        xa, xb = X[A], X[B]
        m = ok[A] & ok[B] & ((xa - x0) * (xb - x0) <= 0) & (xa != xb)
        t = np.where(m, (x0 - xa) / np.where(xa == xb, 1, xb - xa), 0)
        py = Y[A] + t * (Y[B] - Y[A])
        pz = Z[A] + t * (Z[B] - Z[A])
        pts.append(np.column_stack([py[m], pz[m]]))
    return np.vstack(pts)


for k, name in enumerate(n for n in args.surf.split(",") if n):
    P = section(Path(args.paths) / name, args.x0)
    c = cols[k % len(cols)]
    for q in P:
        if y0 <= q[0] < y1 and z0 <= q[1] < z1:
            cv2.circle(img, (int((q[0] - y0) * s), int((q[1] - z0) * s)), 2, c, -1)
    cv2.putText(img, name[:40], (5, 15 + 15 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
if args.fibers:
    for p in Path(args.fibers).glob("*.path.npy"):
        a = np.load(p)[:, :3]
        m = (np.abs(a[:, 0] - args.x0) <= 3) & (a[:, 1] >= y0) & (a[:, 1] < y1) & (a[:, 2] >= z0) & (a[:, 2] < z1)
        if not m.any():
            continue
        h = p.name[0] == "h"
        for q in a[m][::2]:
            cv2.circle(img, (int((q[1] - y0) * s), int((q[2] - z0) * s)), 4 if h else 1, (0, 0, 255) if h else (0, 255, 0), -1)
        if h:
            q = a[m][0]
            cv2.putText(img, p.name[2:6], (int((q[1] - y0) * s) + 5, int((q[2] - z0) * s) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
for z in range((z0 // 100 + 1) * 100, z1, 100):
    cv2.putText(img, str(z), (2, int((z - z0) * s) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
for y in range((y0 // 50 + 1) * 50, y1, 50):
    cv2.putText(img, str(y), (int((y - y0) * s) - 12, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
cv2.imwrite(args.out, img)
print(f"wrote {args.out}")
