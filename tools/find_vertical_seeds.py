"""Find vertical-fibre seeds on the same sheet as a horizontal fibre at given x positions.

Samples the vertical-fibre probability along the sheet normal at the horizontal's point for each x and
lists peaks; same-sheet verticals sit at about -ply (the horizontal is on the +normal side of its vertical).
Usage: python find_vertical_seeds.py <fiber_dir> <segment_dir> <h_name> <x1,x2,...> [--ply 9.5]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from afv_core import STORE

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("name")
ap.add_argument("xs")
ap.add_argument("--ply", type=float, default=9.5)
args = ap.parse_args()

seg = Path(args.segment)
sx, sy = json.loads((seg / "meta.json").read_text())["scale"]
g = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (g != -1).all(-1)
uu, vv = np.meshgrid(np.arange(0, (g.shape[1] - 1) / sx, 2.0) * sx, np.arange(0, (g.shape[0] - 1) / sy, 2.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(g[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
tree, nvec = cKDTree(sp[ok]), nrm[ok]
vt = zarr.open_group(str(STORE), mode="r")["vt"]

P = np.load(Path(args.fibers) / f"{args.name}.path.npy")[:, :3].astype(np.float64)
T = np.arange(-25, 26, 1.0)
if ":" in args.xs:
    a, b, s = map(float, args.xs.split(":"))
    xlist = np.arange(a, b, s)
else:
    xlist = list(map(float, args.xs.split(",")))
for x in xlist:
    i = int(np.argmin(np.abs(P[:, 0] - x)))
    p = P[i]
    t = P[min(i + 5, len(P) - 1)] - P[max(i - 5, 0)]
    t /= np.linalg.norm(t)
    n = nvec[tree.query(p, k=8)[1]].mean(0)
    n -= (n @ t) * t
    n /= np.linalg.norm(n)
    lo = np.floor(p - 30).astype(int)
    blk = vt[lo[2]:lo[2] + 61, lo[1]:lo[1] + 61, lo[0]:lo[0] + 61].astype(np.float32) / 255
    if blk.max() == 0:
        print(f"x {x:.0f}: no vertical predictions here (tiles not predicted)")
        continue
    best = np.zeros(len(T))
    for dz in (-3, 0, 3):                                  # tolerate the vertical's position along z
        q = p + T[:, None] * n + np.array([0, 0, dz]) - lo
        best = np.maximum(best, map_coordinates(blk, [q[:, 2], q[:, 1], q[:, 0]], order=1))
    pk, _ = find_peaks(best, height=0.5, distance=4)
    txt = "  ".join(f"{T[k]:+.0f}:{best[k]:.2f}" for k in pk)
    cand = [k for k in pk if abs(T[k] + args.ply) <= 5]
    seed = f"  seed {(p + T[cand[0]] * n).round().astype(int).tolist()}" if cand else ""
    print(f"x {x:.0f} at {p.round().astype(int).tolist()}: vt peaks {txt}{seed}")
