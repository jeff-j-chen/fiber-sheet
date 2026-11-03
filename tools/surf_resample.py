"""Resample a hand-edited tifxyz onto the xz grid of a TPS surface (same layout as fiber_surface_tps output).

Usage: python surf_resample.py <template_surface> <edited_surface> <out_name> [--paths DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiber_vs_surface import surface_points  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("template")
ap.add_argument("edited")
ap.add_argument("out")
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
args = ap.parse_args()
P = Path(args.paths)
X, Y, Z = (cv2.imread(str(P / args.template / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32) for c in "xyz")
ok = (X != -1) & (Y != -1)
S = surface_points(P / args.edited, step=1.5)
near = cKDTree(S[:, [0, 2]]).query(np.column_stack([X[ok], Z[ok]]))[0] <= 25
Yn = griddata(S[:, [0, 2]], S[:, 1], np.column_stack([X[ok], Z[ok]]), method="linear")
good = near & np.isfinite(Yn)
Y2 = np.full_like(Y, -1)
idx = np.nonzero(ok)
Y2[idx[0][good], idx[1][good]] = Yn[good]
Xo, Zo = np.where(Y2 != -1, X, -1), np.where(Y2 != -1, Z, -1)
out = P / args.out
if out.exists():
    sys.exit(f"{out} exists")
out.mkdir(parents=True)
meta = json.loads((P / args.template / "meta.json").read_text())
meta["uuid"] = meta["seed_surface_id"] = args.out
(out / "meta.json").write_text(json.dumps(meta, indent=4))
for c, A in zip("xyz", (Xo, Y2, Zo)):
    cv2.imwrite(str(out / f"{c}.tif"), A)
d = Y2[ok & (Y2 != -1)] - Y[ok & (Y2 != -1)]
print(f"{good.sum()}/{ok.sum()} cells; y change vs template: median {np.median(d):+.1f}, 95% |d| {np.percentile(np.abs(d), 95):.1f}, max {np.abs(d).max():.1f}")
