"""Trace the horizontal-fibre ply through a hard stretch in fixed-z slices (a "slice zipper").

Horizontals lie at nearly constant z and the horizontal ply is a continuous band of side-by-side strands, so
in a fixed-z slice the sheet is a curve in x-y. For each z, the curve of a reference surface (passing
through trusted horizontals) between x=lo and x=hi is used as guide, and a Viterbi pass with a wide depth
band but z held (side band +/-2) finds the ply. Ends can be pinned to the surface (where it is trusted):
--pin both | left | right. With one free end, the reported landing depth at the free end against the
surface tells whether both sides of the stretch are on the same layer (about 0) or a layer apart.

Usage: python fiber_slice_zip.py <surface> <out_dir> --x lo:hi --z z0:z1:step [--pin both] [--depth 30]
Writes h_slice<z>_<pin>.path.npy pseudo-fibres.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("surface")
ap.add_argument("out")
ap.add_argument("--x", required=True)
ap.add_argument("--z", required=True)
ap.add_argument("--pin", default="both", choices=["both", "left", "right"])
ap.add_argument("--depth", type=int, default=30)
ap.add_argument("--dcost", type=float, default=1.0)
ap.add_argument("--ct", type=float, default=0.0, help="weight of CT brightness in the emission (0..1)")
ap.add_argument("--zwin", type=int, default=6)
ap.add_argument("--ztrack", help="fibre whose z(x) the slice follows; --z then gives offsets from it")
args = ap.parse_args()
zt = None
if args.ztrack:
    _a = np.load(args.ztrack).astype(np.float64)
    _a = _a[np.argsort(_a[:, 0])]
    zt = (_a[:, 0], gaussian_filter1d(_a[:, 2], 30))


def sampler(pts):
    v = core.sample("h", pts)
    return (1 - args.ct) * v + args.ct * core.sample_ct(pts) if args.ct > 0 else v


seg = Path(args.surface)
X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1) & (Y != -1)
gs = 1.0 / json.loads((seg / "meta.json").read_text())["scale"][0]
xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
f = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False, fill_value=None)
lo, hi = map(float, args.x.split(":"))
z0, z1, zst = map(float, args.z.split(":"))
for z in np.arange(z0, z1 + 0.1, zst):
    x = np.arange(lo, hi + 0.1, 1.0)
    zz = np.full_like(x, z) if zt is None else np.interp(x, *zt) + z
    G = core.resample(np.column_stack([x, f(np.c_[zz, x]), zz]), 2.0)
    zg = G[:, 2].copy()
    C, P = core.dp(G, "h", depth=args.depth, side=args.zwin, dcost=args.dcost, scost=0.2, sampler=sampler,
                   pin_start=args.pin in ("both", "left"), pin_end=args.pin in ("both", "right"))
    C[:, 2] = zg
    N, _ = core.frame(G, "h")
    d = ((C - G) * N).sum(1)
    name = f"h_slice{z:.0f}_{args.pin}"
    core.save(args.out, name, C, P)
    w = P < 0.3
    print(f"z {z:6.0f}: mean p {P.mean():.2f}, weak {w.mean() * 100:3.0f}%, depth vs surface: "
          f"left {d[:15].mean():+5.1f}  mid {d[len(d) // 2]:+5.1f}  right {d[-15:].mean():+5.1f}  "
          f"max |d| {np.abs(d).max():4.1f}")
