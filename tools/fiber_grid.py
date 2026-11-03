import glob
import os
import sys

import numpy as np

# grid overview: z of each horizontal at regular x, x of each vertical at regular z: fiber_grid.py <dir>
d = sys.argv[1]
xs = np.arange(2000, 6601, 250)
zs = np.arange(10900, 13801, 150)
H, V = [], []
for p in sorted(glob.glob(os.path.join(d, "*.path.npy"))):
    n = os.path.basename(p)[:-9]
    a = np.load(p).astype(np.float64)
    ln = float(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).sum())
    if n[0] == "h":
        a = a[np.argsort(a[:, 0])]
        H.append((np.median(a[:, 2]), n, a, ln))
    else:
        a = a[np.argsort(a[:, 2])]
        V.append((np.median(a[:, 0]), n, a, ln))
print("H: z at x=          len " + " ".join(f"{x:5d}" for x in xs))
for _, n, a, ln in sorted(H):
    row = [f"{np.interp(x, a[:, 0], a[:, 2]):5.0f}" if a[0, 0] <= x <= a[-1, 0] else "    ." for x in xs]
    print(f"{n[:19]:19s} {ln:5.0f} " + " ".join(row))
print("V: x at z=          len " + " ".join(f"{z:5d}" for z in zs))
for _, n, a, ln in sorted(V):
    row = [f"{np.interp(z, a[:, 2], a[:, 0]):5.0f}" if a[0, 2] <= z <= a[-1, 2] else "    ." for z in zs]
    print(f"{n[:19]:19s} {ln:5.0f} " + " ".join(row))
