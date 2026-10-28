"""Horizontal-fibre sanity check along x: signed offset to the sheet (median per x bin), and agreement
with a reference trace set.

A whole column shifting together = the sheet moved; one fibre jumping ~15 vx alone = that fibre changed layer.
Usage: python fiber_hcheck.py <fiber_dir> <segment_dir> [--ref <other_fiber_dir>] [--bin 250]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--ref")
ap.add_argument("--bin", type=float, default=250.0)
args = ap.parse_args()
fdir = Path(args.fibers)

seg = Path(args.segment)
sx, sy = json.loads((seg / "meta.json").read_text())["scale"]
g = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (g != -1).all(-1)
uu, vv = np.meshgrid(np.arange(0, (g.shape[1] - 1) / sx, 1.0) * sx, np.arange(0, (g.shape[0] - 1) / sy, 1.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(g[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
tree, nvec, spts = cKDTree(sp[ok]), nrm[ok], sp[ok]

names = sorted((p.name[:-9] for p in fdir.glob("h_*.path.npy")), key=lambda n: -int(n.split("_")[3]))
paths = {n: np.load(fdir / f"{n}.path.npy")[:, :4].astype(np.float64) for n in names}
xs = np.arange(1500, 6750, args.bin)
print(f"median signed offset to the sheet (vx, + = +normal side) per x bin of {args.bin:.0f}; "
      "'.' = no fibre, '*' = median sheet distance > 45 vx")
print(f"{'fibre':20s}" + "".join(f"{x:>7.0f}" for x in xs))
for n in names:
    P = paths[n]
    d, k = tree.query(P[:, :3])
    off = ((P[:, :3] - spts[k]) * nvec[k]).sum(1)
    cells = []
    for x0 in xs:
        m = (P[:, 0] >= x0) & (P[:, 0] < x0 + args.bin)
        if m.sum() < 10:
            cells.append("      .")
        elif np.median(d[m]) > 45:
            cells.append("      *")
        else:
            cells.append(f"{np.median(off[m]):+7.1f}")
    print(f"{n:20s}" + "".join(cells))

if args.ref:
    rdir = Path(args.ref)
    print(f"\nagreement with {rdir.name} (distance from each reference point to this trace, vx):")
    for n in names:
        rp = rdir / f"{n}.path.npy"
        if not rp.exists():
            continue
        R = np.load(rp)[:, :3].astype(np.float64)
        d, _ = cKDTree(paths[n][:, :3]).query(R)
        cells = []
        for x0 in xs:
            m = (R[:, 0] >= x0) & (R[:, 0] < x0 + args.bin)
            cells.append(f"{np.median(d[m]):7.1f}" if m.sum() >= 10 else "      .")
        print(f"{n:20s}" + "".join(cells))
