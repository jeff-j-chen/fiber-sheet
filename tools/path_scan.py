"""Free Viterbi paths through the horizontal-probability grid on the sheet (no neighbour guide), best K distinct strands.

Usage: python path_scan.py <sheet> <grid.npz> <target z> [--tol 150] [--k 12] [--lam 0.35]
A path is row(col) over every column of the grid; a step of one row-quarter costs lam, so it follows one strand instead of
hopping. After each path its corridor (+-1 row) is masked, so successive paths are distinct strands.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_trace import Sheet  # noqa: E402


def viterbi(C, lam, maxstep=2):
    R, W = C.shape
    D = np.full((R, W), np.inf)
    B = np.zeros((R, W), int)
    D[:, 0] = C[:, 0]
    for j in range(1, W):
        best = np.full(R, np.inf)
        arg = np.zeros(R, int)
        for d in range(-maxstep, maxstep + 1):
            src = np.arange(R) + d
            ok = (src >= 0) & (src < R)
            cand = np.full(R, np.inf)
            cand[ok] = D[src[ok], j - 1] + lam * abs(d)
            m = cand < best
            best[m], arg[m] = cand[m], src[m]
        D[:, j] = best + C[:, j]
        B[:, j] = arg
    i = int(np.argmin(D[:, -1]))
    path = [i]
    for j in range(W - 1, 0, -1):
        i = B[i, j]
        path.append(i)
    return np.array(path[::-1]), float(D[:, -1].min())


def find_paths(P, lam, k, corridor=4):
    C = -np.log(np.clip(np.nan_to_num(P, nan=0.3), 0.02, 1.0))
    out = []
    for _ in range(k):
        path, cost = viterbi(C, lam)
        out.append(path)
        for j, i in enumerate(path):
            C[max(0, i - corridor): i + corridor + 1, j] = 8.0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("grid")
    ap.add_argument("z", type=float)
    ap.add_argument("--tol", type=float, default=150)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.35)
    args = ap.parse_args()
    G = np.load(args.grid)
    rows, cols, P = G["rows"], G["cols"], G["P"]
    sh = Sheet(args.sheet)
    paths = find_paths(P, args.lam, args.k)
    print("strand  z@25%/50%/75% of cols   mean p  frac>.5  valid  maxjump(rows) ")
    for n, path in enumerate(paths):
        r = rows[path]
        uv = np.column_stack([r, cols])
        v = P[path, np.arange(len(cols))]
        ok = np.isfinite(v)
        xyz = sh.xyz(uv)
        z = xyz[:, 2]
        W = len(cols)
        zz = [z[W // 4], z[W // 2], z[3 * W // 4]]
        mark = "  <== near target" if abs(zz[1] - args.z) <= args.tol else ""
        print(f"{n:3d}   {zz[0]:.0f}/{zz[1]:.0f}/{zz[2]:.0f}   {np.nanmean(v):.2f}   {np.nanmean(v[ok] > 0.5):.2f}  {ok.mean():.2f}  {np.abs(np.diff(r)).max():.2f}{mark}")
    np.savez(args.grid.replace(".npz", "_paths.npz"), rows=rows, cols=cols, paths=np.array(paths))
