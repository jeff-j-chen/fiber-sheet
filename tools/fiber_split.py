"""Split a fibre into two named pieces along its primary axis (x for h, z for v), dropping a gap between them.

Usage: python fiber_split.py <fiber_dir> <name> <cut_lo> <cut_hi>
Writes <name>_a (below cut_lo) and <name>_b (above cut_hi) style names: the second keeps the original name
with 'b' appended to its first number (e.g. h_3849_3228_11573 -> h_3849b_3228_11573), the original file keeps
the lower piece.
"""
import sys
from pathlib import Path

import numpy as np

d, name = Path(sys.argv[1]), sys.argv[2]
lo, hi = float(sys.argv[3]), float(sys.argv[4])
a = np.load(d / f"{name}.path.npy").astype(np.float64)
ax = 0 if name[0] == "h" else 2
parts = name.split("_")
name_b = "_".join([parts[0], parts[1] + "b"] + parts[2:])
for nm, m in ((name, a[:, ax] <= lo), (name_b, a[:, ax] >= hi)):
    p = a[m].copy()
    p[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1))])
    np.save(d / f"{nm}.path.npy", p.astype(np.float32))
    print(f"{nm}: {len(p)} pts, {'xz'[ax == 2]} {p[:, ax].min():.0f}..{p[:, ax].max():.0f}")
(d / f"{name}.json").unlink(missing_ok=True)
