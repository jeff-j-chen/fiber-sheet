"""Fibre-only surface: a thin-plate spline through the fibre points, bounded by the fibres.

No reference sheet is used. In this region horizontals run along x and verticals along z, and the
z-locked tracer makes horizontals monotonic in x, so the papyrus sheet is a height field y = f(x, z) over
the fibres' footprint. The spline is fitted to the sheet's mid-plane: horizontals and verticals lie on
opposite faces of the sheet (ply offset p), so after a first fit every sample is moved p/2 along the local
surface normal towards the other ply and the spline is refitted (repeated --iterations times).

Output: a tifxyz on a regular (x, z) grid (columns = x, rows = z, spacing 1/scale vx) with y from the
spline; cells outside the fibres' convex hull or farther than --reach vx from any fibre sample are -1.

Usage: python fiber_surface_tps.py <fiber_dir> <out_tifxyz> [--scale 0.05] [--smooth 50] [--reach 400]
       [--verified-margin 200] [--exclude a,b] [--compare <old tifxyz>]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=0.05, help="grid cells per vx (0.05 = one cell every 20 vx)")
ap.add_argument("--ply", type=float, help="ply offset (vx); default from layers.json, else 9")
ap.add_argument("--step", type=float, default=15.0, help="fibre sample spacing (vx)")
ap.add_argument("--trim", type=float, default=50.0, help="vx dropped at each fibre end")
ap.add_argument("--min-prob", type=float, default=0.5)
ap.add_argument("--smooth", type=float, default=50.0, help="thin-plate-spline smoothing")
ap.add_argument("--iterations", type=int, default=3, help="mid-plane refits")
ap.add_argument("--reach", type=float, default=400.0, help="max distance (vx) from a fibre sample to keep a cell")
ap.add_argument("--no-hull", action="store_true",
                help="don't clip to the fibres' convex hull; extrapolate up to --reach beyond them on every side")
ap.add_argument("--verified-margin", type=float,
                help="use each fibre only between its first and last same-sheet crossing, plus this many vx")
ap.add_argument("--exclude", default="", help="comma-separated fibre names to leave out")
ap.add_argument("--compare", help="old tifxyz: report how far it is from the new surface (not used in the fit)")
ap.add_argument("--plane", choices=["mid", "h"], default="mid",
                help="fit the sheet mid-plane, or the horizontal-fibre face (verticals moved a full ply onto it)")
ap.add_argument("--hweight", type=float, default=1.0,
                help="horizontal samples get --smooth / hweight (they are followed more closely than verticals)")
ap.add_argument("--hstep", type=float, help="sample spacing for horizontals (default --step)")
ap.add_argument("--fweight", default="", help="prefix:w,... these fibres' samples get --smooth / w (followed more closely)")
args = ap.parse_args()
fdir, out = Path(args.fibers), Path(args.out)
L = json.loads((fdir / "layers.json").read_text()) if (fdir / "layers.json").exists() else {}
ply = args.ply if args.ply is not None else L.get("ply_offset_vx", 9.0)
S = L.get("spacing_vx", 14.0)

span = {}
if args.verified_margin is not None:
    for r in csv.DictReader(open(fdir / "crossings.csv")):
        if r["verdict"] == "no overlap" or abs(float(r["normal_offset_vx"]) - ply) >= S / 2:
            continue
        at = [float(v) for v in r["at"].strip("[]").split()]
        for nm, ax in ((r["horizontal"], 0), (r["vertical"], 2)):
            lo, hi = span.get(nm, (np.inf, -np.inf))
            span[nm] = (min(lo, at[ax]), max(hi, at[ax]))

pts, fam, names = [], [], []
for p in sorted(fdir.glob("*.path.npy")):
    name = p.name[:-9]
    if name in args.exclude.split(","):
        continue
    a = np.load(p).astype(np.float64)
    xyz, prob, arc = a[:, :3], a[:, 3], a[:, 4]
    s = np.arange(args.trim, arc[-1] - args.trim, (args.hstep or args.step) if name[0] == "h" else args.step)
    P = np.stack([np.interp(s, arc, xyz[:, c]) for c in range(3)], 1)
    m = np.interp(s, arc, prob) >= args.min_prob
    if args.verified_margin is not None:
        if name not in span:
            continue
        ax = 0 if name[0] == "h" else 2
        lo, hi = span[name]
        m &= (P[:, ax] >= lo - args.verified_margin) & (P[:, ax] <= hi + args.verified_margin)
    pts.append(P[m])
    fam += [name[0]] * int(m.sum())
    names += [name] * int(m.sum())
pts, fam, names = np.concatenate(pts), np.array(fam), np.array(names)
print(f"{len(pts)} samples from {len(set(names))} fibres; ply offset {ply:g} vx")


def fit(P):
    # coordinates in grid cells, so --smooth has the same meaning as in fiber_sheet_tps.py
    sm = np.where(fam == "h", args.smooth / args.hweight, args.smooth)
    for spec in [s for s in args.fweight.split(",") if s]:
        pre, w = spec.rsplit(":", 1)
        sm = np.where(np.char.startswith(names, pre), sm / float(w), sm)
    rbf = RBFInterpolator(P[:, [0, 2]] * args.scale, P[:, 1], kernel="thin_plate_spline", smoothing=sm)
    return lambda xz: rbf(np.asarray(xz) * args.scale)


def normal(f, xz, h=2.0):
    fx = (f(xz + [h, 0]) - f(xz - [h, 0])) / (2 * h)
    fz = (f(xz + [0, h]) - f(xz - [0, h])) / (2 * h)
    n = np.stack([fx, -np.ones_like(fx), fz], 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


t0 = time.time()
f = fit(pts)
target = pts.copy()
for it in range(args.iterations):
    xz = target[:, [0, 2]]
    n = normal(f, xz)
    d = ((pts - np.column_stack([xz[:, 0], f(xz), xz[:, 1]])) * n).sum(1)
    # which side the horizontal ply is on, along this surface's normal
    side = np.sign(np.median(d[fam == "h"]) - np.median(d[fam == "v"])) or 1.0
    if args.plane == "h":
        shift = np.where(fam == "h", 0.0, side * ply)
    else:
        shift = np.where(fam == "h", -side, side) * ply / 2
    target = pts + shift[:, None] * normal(f, pts[:, [0, 2]])
    f = fit(target)
    print(f"  iteration {it + 1}: horizontals {np.median(d[fam == 'h']):+.1f} vx, verticals "
          f"{np.median(d[fam == 'v']):+.1f} vx from the surface (want +/-{ply / 2:g})")
res = target[:, 1] - f(target[:, [0, 2]])
print(f"fitted in {time.time() - t0:.1f} s; mid-plane residual (in y): median {np.median(np.abs(res)):.2f}, "
      f"95% {np.percentile(np.abs(res), 95):.2f}, max {np.abs(res).max():.2f} vx")

# grid over the fibres' footprint
step = 1.0 / args.scale
pad = args.reach if args.no_hull else 0.0
x0, x1 = np.floor((pts[:, 0].min() - pad) / step) * step, np.ceil((pts[:, 0].max() + pad) / step) * step
z0, z1 = np.floor((pts[:, 2].min() - pad) / step) * step, np.ceil((pts[:, 2].max() + pad) / step) * step
gx, gz = np.meshgrid(np.arange(x0, x1 + 1e-6, step), np.arange(z0, z1 + 1e-6, step))
cells = np.column_stack([gx.ravel(), gz.ravel()])
inside = True if args.no_hull else Delaunay(pts[:, [0, 2]]).find_simplex(cells) >= 0
near = cKDTree(pts[:, [0, 2]]).query(cells)[0] <= args.reach
mask = (inside & near).reshape(gx.shape)
gy = np.full(gx.shape, -1.0)
gy[mask] = f(cells[mask.ravel()])
X, Y, Z = (np.where(mask, g, -1.0).astype(np.float32) for g in (gx, gy, gz))

out.mkdir(parents=True, exist_ok=True)
for name, g in (("x", X), ("y", Y), ("z", Z)):
    cv2.imwrite(str(out / f"{name}.tif"), g)
cv2.imwrite(str(out / "generations.tif"), mask.astype(np.uint16))
G = np.stack([X, Y, Z], -1).astype(np.float64)
q = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
area = float(np.linalg.norm(np.cross(G[:-1, 1:] - G[:-1, :-1], G[1:, :-1] - G[:-1, :-1]), axis=-1)[q].sum())
P = G[mask]
meta = {"uuid": out.name, "seed_surface_id": out.name, "type": "seg", "format": "tifxyz",
        "scale": [args.scale, args.scale], "bbox": [P.min(0).tolist(), P.max(0).tolist()],
        "area_vx2": area, "area_cm2": area * (9.362e-4) ** 2,
        "date_last_modified": time.strftime("%Y%m%d%H%M%S000"),
        "source": f"fibre-only thin-plate spline y=f(x,z) from {fdir.name}"}
(out / "meta.json").write_text(json.dumps(meta, indent=4))
print(f"grid {gx.shape[1]} x {gx.shape[0]}, {mask.sum()} valid cells, area {meta['area_cm2']:.2f} cm2 -> {out}")

for nm in sorted(set(names)):
    r = np.abs(res[names == nm])
    print(f"  {nm:22s} {len(r):4d} samples, residual median {np.median(r):4.1f} / 95% {np.percentile(r, 95):4.1f} vx")

if args.compare:
    old = Path(args.compare)
    O = np.stack([cv2.imread(str(old / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).reshape(-1, 3)
    O = O[(O != -1).all(1)].astype(np.float64)
    inside_o = (Delaunay(pts[:, [0, 2]]).find_simplex(O[:, [0, 2]]) >= 0)
    O = O[inside_o]
    dy = O[:, 1] - f(O[:, [0, 2]])
    print(f"old sheet {old.name} vs new surface (in y, inside the fibre footprint): median {np.median(dy):+.1f}, "
          f"|dy| > 7 vx on {(np.abs(dy) > 7).mean() * 100:.0f}% of its points, 5-95% "
          f"{np.percentile(dy, 5):+.1f}..{np.percentile(dy, 95):+.1f} vx")
