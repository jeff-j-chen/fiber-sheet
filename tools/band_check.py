"""Which connected hz-probability band (in an xy slice) each part of a horizontal lies on.

For a slice at the fibre's local z, hz > --thr is labelled into connected components; points of the fibre within
--dz of the slice are assigned the component under them. A fibre staying on one component never hopped bands
in this slice (unless bands merge; merged components are reported with their y-extent at each x).
Usage: python band_check.py <fiber.path.npy> <z0> <xlo:xhi> [--thr 0.5] [--half 2] [--dz 4]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import CORE, Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fiber")
ap.add_argument("z0", type=int)
ap.add_argument("xr")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--half", type=int, default=2)
ap.add_argument("--dz", type=float, default=4)
args = ap.parse_args()
x0, x1 = map(int, args.xr.split(":"))
a = np.load(args.fiber)[:, :3]
m = (np.abs(a[:, 2] - args.z0) <= args.dz) & (a[:, 0] >= x0) & (a[:, 0] < x1)
pts = a[m]
y0, y1 = int(pts[:, 1].min()) - 80, int(pts[:, 1].max()) + 80
pr = Predictor()
za, zb = args.z0 - args.half, args.z0 + args.half + 1
for kz in range(za // CORE, (zb - 1) // CORE + 1):
    for ky in range(y0 // CORE, (y1 - 1) // CORE + 1):
        for kx in range(x0 // CORE, (x1 - 1) // CORE + 1):
            pr.ensure((kz, ky, kx))
hz = pr.root["hz"][za:zb, y0:y1, x0:x1].astype(np.float32).max(0) / 255
n, lab = cv2.connectedComponents((hz > args.thr).astype(np.uint8), connectivity=8)
prev = None
for q in pts[np.argsort(pts[:, 0])][::3]:
    xi, yi = int(round(q[0])) - x0, int(round(q[1])) - y0
    win = lab[max(yi - 2, 0):yi + 3, max(xi - 2, 0):xi + 3]
    ids = win[win > 0]
    c = int(np.bincount(ids).argmax()) if len(ids) else 0
    if c != prev:
        col = np.flatnonzero(lab[:, xi] == c) + y0 if c else []
        span = f"y {col.min()}..{col.max()}" if len(col) else "-"
        print(f"x {q[0]:6.0f} y {q[1]:6.0f} z {q[2]:6.0f}: band {c:4d} ({span} in this column)")
        prev = c
