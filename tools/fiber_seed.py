"""Seed new fibres on the sheet: find the strongest strand of a family near (x, z), at the depth where that
family's existing fibres sit relative to a surface, and write a short straight starter path for fiber_march.py.

Usage: python fiber_seed.py <surface_tifxyz> <fiber_dir> <out_dir> --family h --at "x,z;x,z" [--search 25]
Names: h_<x>_<y>_<z> / v_<x>_<y>_<z> of the found point.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("surface")
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--family", choices="hv", required=True)
ap.add_argument("--at", required=True, help='"x,z;..." or "x,y,z;..." (the latter fixes the depth)')
ap.add_argument("--search", type=float, default=25.0, help="+/- vx across the fibre direction within the sheet")
ap.add_argument("--dwin", type=float, default=6.0, help="+/- vx around the family's typical depth")
args = ap.parse_args()

seg = Path(args.surface)
X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1) & (Y != -1)
_li = LinearNDInterpolator(np.c_[Z[ok], X[ok]], Y[ok])   # scattered, so SLIM-flattened grids work too


def f(zx):
    return _li(zx)


def frame(x, z):
    x, z = np.atleast_1d(x).astype(float), np.atleast_1d(z).astype(float)
    y = f(np.c_[z, x])
    fx = (f(np.c_[z, x + 3]) - f(np.c_[z, x - 3])) / 6
    fz = (f(np.c_[z + 3, x]) - f(np.c_[z - 3, x])) / 6
    n = np.c_[fx, -np.ones_like(fx), fz]
    return np.c_[x, y, z], n / np.linalg.norm(n, axis=1, keepdims=True)


fam = args.family
pa, sa = (0, 2) if fam == "h" else (2, 0)
pts = []
for p in glob.glob(os.path.join(args.fibers, f"{fam}_*.path.npy")):
    pts.append(np.load(p)[::10, :3].astype(np.float64))
pts = np.concatenate(pts)
S, n = frame(pts[:, 0], pts[:, 2])
dep = ((pts - S) * n).sum(1)

pred = Predictor()
arr = pred.root["hz" if fam == "h" else "vt"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for tok in args.at.split(";"):
    c3 = list(map(float, tok.split(",")))
    x0, z0 = (c3[0], c3[2]) if len(c3) == 3 else tuple(c3)
    near = np.hypot(pts[:, 0] - x0, pts[:, 2] - z0) < 400
    d0 = float(np.median(dep[near])) if near.sum() > 20 else float(np.median(dep))
    if len(c3) == 3:       # x,y,z given: take the depth of that point on the surface
        S0, n0 = frame(x0, z0)
        d0 = float(((np.array(c3) - S0[0]) * n0[0]).sum())
    along = np.arange(-10, 10.1, 2.0)
    across = np.arange(-args.search, args.search + 0.1, 1.0)
    depths = np.arange(d0 - args.dwin, d0 + args.dwin + 0.1, 1.0)
    best = (-1, None)
    grid = []
    for c in across:
        px = x0 + (along if fam == "h" else c)
        pz = z0 + (c if fam == "h" else along)
        px, pz = np.broadcast_arrays(px, pz)
        Sx, nx = frame(px, pz)
        P = Sx[:, None, :] + nx[:, None, :] * depths[None, :, None]           # along, depth, 3
        grid.append(P)
    G = np.stack(grid)                                                        # across, along, depth, 3
    flat = G.reshape(-1, 3)[:, ::-1]
    lo = np.maximum(np.floor(flat.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(flat.max(0)).astype(int) + 2, pred.shape)
    for tz in range(lo[0] // 96, (hi[0] - 1) // 96 + 1):
        for ty in range(lo[1] // 96, (hi[1] - 1) // 96 + 1):
            for tx in range(lo[2] // 96, (hi[2] - 1) // 96 + 1):
                pred.ensure((tz, ty, tx))
    blk = arr[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32) / 255
    v = map_coordinates(blk, (flat - lo).T, order=1).reshape(G.shape[:3]).mean(1)    # across, depth
    i, j = np.unravel_index(int(np.argmax(v)), v.shape)
    P0 = G[i, len(along) // 2, j]
    # starter path: 41 vx straight along the family direction, following the surface
    t = np.arange(-20, 20.1, 1.0)
    px = P0[0] + (t if fam == "h" else 0 * t)
    pz = P0[2] + (0 * t if fam == "h" else t)
    Sx, nx = frame(px, pz)
    path = Sx + nx * depths[j]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    name = f"{fam}_{P0[0]:.0f}_{P0[1]:.0f}_{P0[2]:.0f}"
    np.save(out / f"{name}.path.npy", np.column_stack([path, np.full(len(t), v[i, j]), arc]).astype(np.float32))
    print(f"{tok}: family depth {d0:+.1f}; best p {v[i, j]:.2f} at offset {across[i]:+.0f}, depth {depths[j]:+.0f} -> {name}")
