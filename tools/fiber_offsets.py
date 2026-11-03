"""Signed offset (along the reference sheet normal, +ve = -y side) of each fibre near a box, in bins along the fibre.

Usage: python fiber_offsets.py <fiber_dir> <ref_surface> --box x0:x1:z0:z1 [--pad 300] [--bin 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiber_vs_surface import surface_grid, surface_points, uv_to_xyz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("ref")
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--box", required=True)
ap.add_argument("--pad", type=float, default=300)
ap.add_argument("--bin", type=float, default=100)
args = ap.parse_args()
b = list(map(float, args.box.split(":")))
ref = Path(args.paths) / args.ref
X, Y, Z, ok = surface_grid(ref)
S, UV = surface_points(ref, step=1.5, return_uv=True)
tree = cKDTree(S)
gi = [np.gradient(np.where(ok, A, np.nan), axis=0) for A in (X, Y, Z)]
gj = [np.gradient(np.where(ok, A, np.nan), axis=1) for A in (X, Y, Z)]
NG = np.cross(np.stack(gi, -1), np.stack(gj, -1))
NG /= np.linalg.norm(NG, axis=-1, keepdims=True)
NG = np.nan_to_num(NG * np.where(NG[..., 1:2] > 0, -1.0, 1.0))
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    a = np.load(p)[:, :3].astype(np.float64)
    m = (a[:, 0] >= b[0] - args.pad) & (a[:, 0] <= b[1] + args.pad) & (a[:, 2] >= b[2] - args.pad) & (a[:, 2] <= b[3] + args.pad)
    if m.sum() < 30:
        continue
    a = a[m]
    d, i = tree.query(a)
    uv = UV[i]
    n = np.column_stack([uv_to_xyz(NG[..., c], uv[:, 0], uv[:, 1]) for c in range(3)])
    s = ((a - S[i]) * n).sum(1)
    s[d > 40] = np.nan
    ax = 0 if p.name[0] == "h" else 2
    edges = np.arange(np.floor(a[:, ax].min() / args.bin) * args.bin, a[:, ax].max() + args.bin, args.bin)
    row = []
    for e in edges[:-1]:
        k = (a[:, ax] >= e) & (a[:, ax] < e + args.bin)
        v = s[k]
        v = v[np.isfinite(v)]
        row.append(f"{e:.0f}:{np.median(v):+.0f}" if len(v) > 5 else f"{e:.0f}:  .")
    print(f"{p.name[:-9]:22s} " + " ".join(row))
