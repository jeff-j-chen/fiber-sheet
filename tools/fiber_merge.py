"""Join two overlapping horizontals: A left of --cut, B right of it (blended over --blend vx).

Usage: python fiber_merge.py <A.path.npy> <B.path.npy> <out_dir> --cut X --name N [--blend 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("out")
ap.add_argument("--cut", type=float, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--blend", type=float, default=40.0)
args = ap.parse_args()
A, B = (np.load(p).astype(np.float64) for p in (args.a, args.b))
A, B = A[np.argsort(A[:, 0])], B[np.argsort(B[:, 0])]
c, h = args.cut, args.blend / 2
for F, s in ((A, "A"), (B, "B")):
    if not (F[0, 0] <= c - h and F[-1, 0] >= c + h):
        sys.exit(f"{s} does not cover {c - h:.0f}..{c + h:.0f}")
xb = np.arange(c - h, c + h + 0.1, 1.0)
w = (xb - xb[0]) / (xb[-1] - xb[0])
mid = np.column_stack([xb] + [(1 - w) * np.interp(xb, A[:, 0], A[:, k]) + w * np.interp(xb, B[:, 0], B[:, k])
                              for k in (1, 2, 3)])
gap = np.linalg.norm([np.interp(c, A[:, 0], A[:, k]) - np.interp(c, B[:, 0], B[:, k]) for k in range(3)])
L, R = A[A[:, 0] < c - h, :4], B[B[:, 0] > c + h, :4]
C = np.vstack([L, mid, R])
core.save(args.out, args.name, C[:, :3], C[:, 3])
print(f"{args.name}: A gap at cut {gap:.1f} vx, x {C[0, 0]:.0f}..{C[-1, 0]:.0f}")
