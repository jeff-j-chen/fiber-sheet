"""Edit a fibre dir: crop fibres to a z/x range or merge two fibres of the same strand.

Usage: python fiber_edit.py <fiber_dir> --crop name:axis:lo:hi [...] --merge a:b:cut:newname [...]
  --crop keeps lo <= axis <= hi (axis x|z); original saved as <name>.path.npy.bak
  --merge takes a for primary-axis < cut and b for >= cut (axis z for v, x for h); a and b are removed, newname written
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--crop", action="append", default=[])
ap.add_argument("--merge", action="append", default=[])
args = ap.parse_args()
d = Path(args.dir)


def find(prefix):
    m = sorted(d.glob(prefix + "*.path.npy"))
    if len(m) != 1:
        raise SystemExit(f"{prefix}: {len(m)} matches")
    return m[0]


for spec in args.crop:
    name, ax, lo, hi = spec.split(":")
    p = find(name)
    A = np.load(p)
    c = A[:, 0 if ax == "x" else 2]
    keep = (c >= float(lo)) & (c <= float(hi))
    shutil.copy(p, str(p) + ".bak")
    np.save(p, A[keep])
    print(f"{p.name[:-9]}: {len(A)} -> {keep.sum()} points")

for spec in args.merge:
    a, b, cut, new = spec.split(":")
    pa, pb = find(a), find(b)
    ax = 0 if pa.name[0] == "h" else 2
    A, B = np.load(pa), np.load(pb)
    A, B = A[np.argsort(A[:, ax])], B[np.argsort(B[:, ax])]
    M = np.vstack([A[A[:, ax] < float(cut)], B[B[:, ax] >= float(cut)]])
    gap = np.linalg.norm(A[A[:, ax] < float(cut)][-1, :3] - B[B[:, ax] >= float(cut)][0, :3])
    for p in (pa, pb):
        shutil.move(str(p), str(p) + ".bak")
        j = p.with_suffix("").with_suffix(".json")
        if j.exists():
            j.unlink()
    np.save(d / f"{new}.path.npy", M)
    print(f"{new}: {len(M)} points, joint gap {gap:.1f} vx")
