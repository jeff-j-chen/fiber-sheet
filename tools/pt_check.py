"""For points h:x:y:z / v:x:y:z report the nearest same-family fibres and the point's distance to reference surfaces.

Usage: python pt_check.py <fiber_dir> --surf a,b --pts h:x:y:z,v:x:y:z,...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiber_vs_surface import surface_points  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--surf", required=True)
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--pts", required=True)
args = ap.parse_args()
F = {p.name[:-9]: np.load(p)[:, :3] for p in Path(args.fibers).glob("*.path.npy")}
S = []
for n in args.surf.split(","):
    P = surface_points(Path(args.paths) / n, step=1.5)
    S.append((n, P, cKDTree(P)))
for spec in args.pts.split(","):
    fam, x, y, z = spec.split(":")
    q = np.array([float(x), float(y), float(z)])
    r = sorted((float(np.min(np.linalg.norm(a - q, axis=1))), n) for n, a in F.items() if n[0] == fam)[:3]
    s = []
    for n, P, t in S:
        d, i = t.query(q)
        s.append(f"{n[-14:]} {d:5.1f} (dy {q[1] - P[i, 1]:+.0f})")
    print(f"{spec:22s} " + " | ".join(f"{n} {d:.0f}" for d, n in r) + "   surf: " + "  ".join(s))
