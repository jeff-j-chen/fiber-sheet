"""Seed a new fibre one ply away from an anchor fibre of the other family, then write a starter path.

Crossing convention (fiber_crossings_fn.py): n = t_h x t_v (h along +x, v along +z); a same-sheet vertical
sits at (v - h).n = +ply. So a vertical seed is anchor_h + ply*n and a horizontal seed is anchor_v - ply*n.
The strongest point of the wanted family within +/-dwin of that depth and +/-search sideways is used.

Usage: python fiber_seed2.py <fiber_dir> <out_dir> --seed "anchor_name@pos;anchor_name@pos" [--ply 9]
  pos = x for a horizontal anchor (new vertical), z for a vertical anchor (new horizontal).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--seed", required=True)
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--dwin", type=float, default=4.0)
ap.add_argument("--search", type=float, default=20.0)
args = ap.parse_args()
fdir, out = Path(args.fibers), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pred = Predictor()

for tok in args.seed.split(";"):
    anchor, pos = tok.split("@")
    pos = float(pos)
    a = np.load(next(fdir.glob(anchor + "*.path.npy"))).astype(np.float64)[:, :3]
    afam = Path(anchor).name[0]
    pa = 0 if afam == "h" else 2
    a = a[np.argsort(a[:, pa])]
    i = int(np.argmin(np.abs(a[:, pa] - pos)))
    A = a[i]
    t = a[min(i + 15, len(a) - 1)] - a[max(i - 15, 0)]
    t /= np.linalg.norm(t)
    if afam == "h":
        th, tv = t, np.array([0.0, 0.0, 1.0])
    else:
        tv, th = t, np.array([1.0, 0.0, 0.0])
    tv = tv - (tv @ th) * th if afam == "h" else tv
    th = th - (th @ tv) * tv if afam == "v" else th
    n = np.cross(th, tv)
    n /= np.linalg.norm(n)
    fam = "v" if afam == "h" else "h"
    arr = pred.root["vt" if fam == "v" else "hz"]
    side = th if fam == "v" else tv          # new fibre may sit anywhere along the anchor
    along = tv if fam == "v" else th         # new fibre direction
    along = along - (along @ n) * n
    along /= np.linalg.norm(along)
    side = side - (side @ n) * n - (side @ along) * along
    side /= np.linalg.norm(side)
    base = A + (args.ply if fam == "v" else -args.ply) * n
    dd = np.arange(-args.dwin, args.dwin + 0.1, 1.0)
    ss = np.arange(-args.search, args.search + 0.1, 1.0)
    ll = np.arange(-10, 10.1, 2.0)
    P = (base[None, None, None] + dd[:, None, None, None] * n + ss[None, :, None, None] * side
         + ll[None, None, :, None] * along)
    flat = P.reshape(-1, 3)[:, ::-1]
    lo = np.maximum(np.floor(flat.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(flat.max(0)).astype(int) + 2, pred.shape)
    for tz in range(lo[0] // 96, (hi[0] - 1) // 96 + 1):
        for ty in range(lo[1] // 96, (hi[1] - 1) // 96 + 1):
            for tx in range(lo[2] // 96, (hi[2] - 1) // 96 + 1):
                pred.ensure((tz, ty, tx))
    blk = arr[tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))].astype(np.float32) / 255
    v = map_coordinates(blk, (flat - lo).T, order=1).reshape(P.shape[:3]).mean(2)     # depth, side
    j, k = np.unravel_index(int(np.argmax(v)), v.shape)
    P0 = base + dd[j] * n + ss[k] * side
    tt = np.arange(-20, 20.1, 1.0)
    path = P0[None] + tt[:, None] * along[None]
    arc = tt - tt[0]
    name = f"{fam}_{P0[0]:.0f}_{P0[1]:.0f}_{P0[2]:.0f}"
    np.save(out / f"{name}.path.npy", np.column_stack([path, np.full(len(tt), v[j, k]), arc]).astype(np.float32))
    print(f"{tok}: p {v[j, k]:.2f} at depth {dd[j]:+.0f}, side {ss[k]:+.0f} -> {name}")
