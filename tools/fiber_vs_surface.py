"""Distance of every fibre to a reference tifxyz surface (e.g. a hand-made sheet), segment by segment.

Usage: python fiber_vs_surface.py <fiber_dir> <surface> [--paths DIR] [--thr 10] [--box x0:x1:z0:z1]
Surface grid is bilinearly upsampled to ~2 vx; distance = nearest surface point (unsigned).
Prints per fibre: median distance and the stretches (axis ranges) where it exceeds --thr.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def surface_grid(seg):
    seg = Path(seg)
    X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    return X, Y, Z, (X != -1) & (Z > 0)


def surface_points(seg, step=2.0, return_uv=False):
    """Bilinear upsampling of the grid over quads whose 4 corners are valid; uv = fractional (row, col)."""
    X, Y, Z, ok = surface_grid(seg)
    q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    d = np.median(np.hypot(X[:, 1:] - X[:, :-1], Z[:, 1:] - Z[:, :-1])[ok[:, 1:] & ok[:, :-1]])
    k = max(int(round(d / step)), 1)
    t = np.arange(k) / k
    qi, qj = np.nonzero(q)
    ii = (qi[:, None, None] + t[None, :, None] + 0 * t[None, None, :]).ravel()
    jj = (qj[:, None, None] + 0 * t[None, :, None] + t[None, None, :]).ravel()
    P = np.column_stack([uv_to_xyz(A, ii, jj) for A in (X, Y, Z)])
    return (P, np.column_stack([ii, jj])) if return_uv else P


def uv_to_xyz(A, ii, jj):
    i0, j0 = np.clip(np.floor(ii).astype(int), 0, A.shape[0] - 2), np.clip(np.floor(jj).astype(int), 0, A.shape[1] - 2)
    a, b = ii - i0, jj - j0
    return ((1 - a) * (1 - b) * A[i0, j0] + a * (1 - b) * A[i0 + 1, j0] + (1 - a) * b * A[i0, j0 + 1]
            + a * b * A[i0 + 1, j0 + 1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fibers")
    ap.add_argument("surface")
    ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
    ap.add_argument("--thr", type=float, default=10.0)
    ap.add_argument("--box")
    args = ap.parse_args()
    S = surface_points(Path(args.paths) / args.surface)
    tree = cKDTree(S)
    box = list(map(float, args.box.split(":"))) if args.box else None
    for p in sorted(Path(args.fibers).glob("*.path.npy")):
        a = np.load(p)[:, :3].astype(np.float64)
        if box:
            m = (a[:, 0] >= box[0]) & (a[:, 0] <= box[1]) & (a[:, 2] >= box[2]) & (a[:, 2] <= box[3])
            if m.sum() < 20:
                continue
            a = a[m]
        d, _ = tree.query(a)
        ax = 0 if p.name[0] == "h" else 2
        bad = d > args.thr
        segs = []
        i = 0
        while i < len(a):
            if bad[i]:
                j = i
                while j + 1 < len(a) and bad[j + 1]:
                    j += 1
                if j - i >= 10:
                    segs.append(f"{a[i, ax]:.0f}..{a[j, ax]:.0f}(max {d[i:j + 1].max():.0f})")
                i = j + 1
            else:
                i += 1
        print(f"{p.name[:-9]:22s} median {np.median(d):5.1f}  p90 {np.percentile(d, 90):5.1f}  " + " ".join(segs))
