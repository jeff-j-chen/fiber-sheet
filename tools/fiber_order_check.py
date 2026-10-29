"""Check that parallel fibres keep their order on the sheet (they must not cross in the sheet's uv frame).

Projects every fibre onto the surface grid and compares, for each pair of same-family fibres, their
across-coordinate (v for horizontals, u for verticals) along the shared range. Pairs that come closer than
--min-gap grid cells or swap order are reported.

Usage: python fiber_order_check.py <surface_tifxyz> <fiber_dir> [--min-gap 1.0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("surface")
ap.add_argument("fibers")
ap.add_argument("--min-gap", type=float, default=1.0, help="grid cells (one cell = 1/scale vx)")
args = ap.parse_args()
seg = Path(args.surface)
G = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
ok = (G != -1).all(-1)
H, W = ok.shape
uu, vv = np.meshgrid(np.arange(0, W - 1, 0.25, dtype=np.float32), np.arange(0, H - 1, 0.25, dtype=np.float32))
D = np.stack([cv2.remap(G[..., c], uu, vv, cv2.INTER_LINEAR) for c in range(3)], -1)
dok = cv2.remap(ok.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
tree, UV = cKDTree(D[dok]), np.stack([uu[dok], vv[dok]], 1)

proj = {}
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    a = np.load(p)[:, :3]
    d, k = tree.query(a)
    uv = UV[k[d < 40]]
    if len(uv) > 20:
        proj[p.name[:-9]] = uv
issues = 0
for fam, along, across in (("h", 0, 1), ("v", 1, 0)):
    names = [n for n in proj if n[0] == fam]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = proj[a], proj[b]
            lo, hi = max(A[:, along].min(), B[:, along].min()), min(A[:, along].max(), B[:, along].max())
            if hi - lo < 2:
                continue
            t = np.linspace(lo, hi, 100)
            oa, ob = np.argsort(A[:, along]), np.argsort(B[:, along])
            ca = np.interp(t, A[oa, along], A[oa, across])
            cb = np.interp(t, B[ob, along], B[ob, across])
            diff = cb - ca
            if (np.sign(diff) != np.sign(np.median(diff))).any() or np.abs(diff).min() < args.min_gap:
                issues += 1
                k = int(np.argmin(np.abs(diff)))
                print(f"{a} / {b}: closest {diff[k]:+.2f} cells at {'uv'[along]}={t[k]:.0f}, "
                      f"median gap {np.median(diff):+.1f}, order {'SWAPS' if (np.sign(diff) != np.sign(np.median(diff))).any() else 'kept'}")
print(f"{issues} pairs too close or crossing")
