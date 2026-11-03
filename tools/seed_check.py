"""Where do the user's hand-picked points sit relative to two surfaces and the fibres in a set?

Usage: python seed_check.py <fiber_dir> <surfA> <surfB> --h "x,y,z;..." --v "x,y,z;..."
Reports signed distance along the surface normal (n ~ -y) to each surface and the nearest fibre of each family.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiber_vs_surface import surface_points  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--h", default="")
ap.add_argument("--v", default="")
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
args = ap.parse_args()

surf = {}
for nm in (args.a, args.b):
    S = surface_points(Path(args.paths) / nm, step=3.0)
    surf[nm] = (cKDTree(S[:, [0, 2]]), S)

fib = {"h": [], "v": []}
for p in glob.glob(os.path.join(args.fibers, "*.path.npy")):
    a = np.load(p)[::3, :3]
    nm = os.path.basename(p)[:-9]
    fib[nm[0]].append((nm, a))


def near(fam, q):
    best = (1e9, "")
    for nm, a in fib[fam]:
        d = np.linalg.norm(a - q, axis=1).min()
        if d < best[0]:
            best = (d, nm)
    return best


for fam, spec in (("h", args.h), ("v", args.v)):
    for tok in [t for t in spec.split(";") if t.strip()]:
        q = np.array(list(map(float, tok.split(","))))
        row = []
        for nm in (args.a, args.b):
            tr, S = surf[nm]
            d, i = tr.query(q[[0, 2]])
            row.append(f"dy {q[1] - S[i, 1]:+6.1f} (xz gap {d:3.0f})")
        dh, nh = near("h", q)
        dv, nv = near("v", q)
        print(f"{fam} {tok.strip():20s} {args.a[-6:]}: {row[0]} | {args.b[-12:]}: {row[1]} | nearest h {nh} {dh:4.0f}  v {nv} {dv:4.0f}")
