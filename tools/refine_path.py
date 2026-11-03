"""Turn a path from path_scan.py into a refined fibre (strand lock on the model probability, narrow depth band).

Usage: python refine_path.py <sheet> <paths.npz> <grid.npz> <index> <name> <out_dir> [--depth 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402
from surf_trace import Sheet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sheet")
ap.add_argument("paths")
ap.add_argument("grid")
ap.add_argument("index", type=int)
ap.add_argument("name")
ap.add_argument("out")
ap.add_argument("--depth", type=int, default=3)
args = ap.parse_args()
Pz = np.load(args.paths)
G = np.load(args.grid)
rows, cols = Pz["rows"], Pz["cols"]
path = Pz["paths"][args.index]
v = G["P"][path, np.arange(len(cols))]
ok = np.isfinite(v)
idx = np.where(ok)[0]
sh = Sheet(args.sheet)
uv = np.column_stack([rows[path][idx], cols[idx]])
ok2 = sh.on_sheet(uv)
uv = uv[ok2]
C0 = core.resample(sh.xyz(uv), 2.0)
refn = lambda P: sh.normal(sh.uv_of(P)[0])
C, P = core.lock_strand(C0, "h", sigma=150.0, depth=args.depth, side=24, scost=5.0, refn=refn)
d = np.linalg.norm(C - core.resample(C0, 2.0)[: len(C)], axis=1) if len(C) == len(C0) else np.array([0.0])
so = sh.signed_offset(C)
core.save(args.out, args.name, C, P)
print(f"{args.name}: x {C[:, 0].min():.0f}..{C[:, 0].max():.0f}, z {C[:, 2].min():.0f}..{C[:, 2].max():.0f}, {len(C)} pts, mean p {P.mean():.2f}, "
      f"weak {np.mean(P < 0.3) * 100:.0f}%, sheet offset median {np.median(so):+.1f} max {np.abs(so).max():.1f}, move from path {d.mean():.1f} vx")
