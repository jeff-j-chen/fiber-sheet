import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

# same-family fibres that run on top of each other: fiber_dupes.py <dir> [near_vx] [min_overlap_vx]
d = sys.argv[1]
near = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
minov = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
F = {}
for p in glob.glob(os.path.join(d, "*.path.npy")):
    a = np.load(p)[:, :3].astype(np.float64)
    F[os.path.basename(p)[:-9]] = a[::5]
names = sorted(F)
trees = {n: cKDTree(F[n]) for n in names}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        if a[0] != b[0]:
            continue
        dist, _ = trees[b].query(F[a])
        close = dist < near
        if close.sum() * 5 < minov:
            continue
        pa = 0 if a[0] == "h" else 2
        xs = F[a][close, pa]
        print(f"{a:22s} {b:22s} overlap {close.sum() * 5:5d} vx  {'xz'[pa == 2]} {xs.min():.0f}..{xs.max():.0f}  "
              f"median dist {np.median(dist[close]):.1f}")
