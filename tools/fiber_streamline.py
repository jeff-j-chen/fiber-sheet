"""Follow a sheet layer through a fixed-z slice by streamlining the CT layer orientation (for tight bends).

In a fixed-z slice, horizontals and the layers run as curves in x-y. The local layer direction comes from
the structure tensor of the smoothed CT; a streamline is stepped along it from a start point, and after each
step the point is nudged (at most --snap vx) across the layer to the nearest maximum of CT brightness plus
horizontal-fibre probability. Neighbouring layers are parallel, so the streamline cannot drift onto another
layer except by the small snaps, which keep it centred on its own one.

Usage: python fiber_streamline.py <surface> <out_dir> --z Z[,Z..] --x0 X --x1 X [--y0 Y] [--half 2]
       [--compare fibre_dir]
Start point: (x0, surface(x0, z)) unless --y0 is given. Writes hl<z>.path.npy and prints the depth between
the streamline and the surface every 100 vx of x.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("surface")
ap.add_argument("out")
ap.add_argument("--z", required=True)
ap.add_argument("--x0", type=float, required=True)
ap.add_argument("--x1", type=float, required=True)
ap.add_argument("--y0", type=float)
ap.add_argument("--half", type=int, default=2)
ap.add_argument("--margin", type=int, default=400)
ap.add_argument("--step", type=float, default=1.5)
ap.add_argument("--snap", type=float, default=0.6)
ap.add_argument("--rho", type=float, default=8.0)
args = ap.parse_args()
seg = Path(args.surface)
X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1) & (Y != -1)
gs = 1.0 / json.loads((seg / "meta.json").read_text())["scale"][0]
xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
f = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False, fill_value=None)
p = core.pred()
for z in [float(v) for v in args.z.split(",")]:
    xa, xb = int(min(args.x0, args.x1)) - 50, int(max(args.x0, args.x1)) + 50
    xx = np.arange(xa, xb, 10.0)
    ys = f(np.c_[np.full_like(xx, z), xx])
    ya, yb = int(np.nanmin(ys)) - args.margin, int(np.nanmax(ys)) + args.margin
    ya = max(ya, 0)
    za, zb = int(z) - args.half, int(z) + args.half + 1
    for tz in range(za // 96, (zb - 1) // 96 + 1):
        for ty in range(ya // 96, (yb - 1) // 96 + 1):
            for tx in range(xa // 96, (xb - 1) // 96 + 1):
                p.ensure((tz, ty, tx))
    ct = p.ct[za:zb, ya:yb, xa:xb].astype(np.float32).mean(0)
    hz = p.root["hz"][za - 4:zb + 4, ya:yb, xa:xb].astype(np.float32).max(0) / 255
    lo_, hi_ = np.percentile(ct, [5, 99])
    ctn = np.clip((ct - lo_) / (hi_ - lo_), 0, 1)
    score = gaussian_filter(ctn, 1.5) + 0.7 * gaussian_filter(hz, 1.5)
    g = gaussian_filter(ctn, 1.5)
    gy, gx = np.gradient(g)
    Jxx, Jxy, Jyy = (gaussian_filter(a, args.rho) for a in (gx * gx, gx * gy, gy * gy))
    # layer direction = eigenvector of the smaller eigenvalue (along the layer)
    ang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
    dx_f, dy_f = np.cos(ang), np.sin(ang)

    def at(arr, P):
        return map_coordinates(arr, [[P[1] - ya], [P[0] - xa]], order=1, mode="nearest")[0]

    y0 = args.y0 if args.y0 is not None else float(f([[z, args.x0]])[0])
    P = np.array([args.x0, y0])
    sgn = 1.0 if args.x1 > args.x0 else -1.0
    prev = np.array([sgn, 0.0])
    pts = [P.copy()]
    for _ in range(int(6 * abs(args.x1 - args.x0) / args.step)):
        d = np.array([at(dx_f, P), at(dy_f, P)])
        d /= np.linalg.norm(d) + 1e-9
        if d @ prev < 0:
            d = -d
        if d[0] * sgn < 0.15:
            # never turn back in x: at a junction with flat layers keep heading on along the bend
            d = prev if prev[0] * sgn >= 0.15 else np.array([sgn * 0.15, np.sign(d[1] or 1) * 0.99])
            d /= np.linalg.norm(d)
        P = P + args.step * d
        nrm = np.array([-d[1], d[0]])
        offs = np.arange(-3, 3.01, 0.5)
        vals = [at(score, P + o * nrm) for o in offs]
        o = offs[int(np.argmax(vals))]
        P = P + np.clip(o, -args.snap, args.snap) * nrm
        prev = d
        pts.append(P.copy())
        if (P[0] - args.x1) * sgn >= 0 or not (xa < P[0] < xb and ya < P[1] < yb):
            break
    pts = np.array(pts)
    C = np.column_stack([pts, np.full(len(pts), z)])
    core.save(args.out, f"h_hl{z:.0f}_{args.x0:.0f}", C, np.ones(len(C)))
    msg = f"z {z:.0f}: {len(pts)} steps, ends at x {pts[-1, 0]:.0f}; y - surface:"
    for xv in np.arange(min(args.x0, args.x1), max(args.x0, args.x1) + 1, 100):
        k = np.flatnonzero(np.abs(pts[:, 0] - xv) < 2)
        if len(k):
            msg += f" {xv:.0f}:{pts[k[0], 1] - float(f([[z, xv]])[0]):+.0f}"
    print(msg)
