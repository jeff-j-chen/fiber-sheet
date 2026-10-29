"""Turn a verified fibre set into a tifxyz surface (thin-plate spline pinned to the fibres).

The fibres define where the papyrus sheet really is; an existing tifxyz sheet only provides the 2D grid
(u, v) and the normal direction. Each fibre sample is moved half a ply towards the sheet's mid-plane
(horizontals -ply/2, verticals +ply/2 along the normal), projected onto the reference sheet to get its grid
position and its signed normal offset, and a smoothing thin-plate spline over (u, v) interpolates that
offset between the fibres. Every grid vertex is moved along its normal by the interpolated offset. Only
the area spanned by the fibres (their convex hull plus --margin cells) is kept; the rest is -1 (invalid).
The output has the same grid size and scale as the reference, so renders of both line up pixel for pixel.

Example
  python fibers_to_tifxyz.py data/grid1_final7 <reference tifxyz> <paths>/fiber_tps_mysheet \
      --keep v_3253_3439_12976:z:12200:12850
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fibers", help="fibre-set folder (<name>.path.npy files; layers.json is used for the ply offset)")
ap.add_argument("reference", help="reference tifxyz folder: grid and normal orientation")
ap.add_argument("out", help="output tifxyz folder (created); its name becomes the segment uuid")
ap.add_argument("--ply", type=float, help="ply offset (vx); default from <fibers>/layers.json, else 9")
ap.add_argument("--step", type=float, default=15.0, help="fibre sample spacing (vx)")
ap.add_argument("--trim", type=float, default=50.0, help="vx dropped at each fibre end")
ap.add_argument("--min-prob", type=float, default=0.5)
ap.add_argument("--max-dist", type=float, default=60.0, help="ignore samples farther than this from the sheet")
ap.add_argument("--smooth", type=float, default=50.0, help="thin-plate-spline smoothing")
ap.add_argument("--margin", type=int, default=5, help="grid cells kept beyond the fibres' convex hull")
ap.add_argument("--keep", default="", help="name:axis:lo:hi -> only use that fibre where lo <= axis <= hi")
ap.add_argument("--exclude", default="", help="comma-separated fibre names to leave out")
args = ap.parse_args()
ref, out = Path(args.reference), Path(args.out)
if args.ply is None:
    lay = Path(args.fibers) / "layers.json"
    args.ply = json.loads(lay.read_text())["ply_offset_vx"] if lay.exists() else 9.0
print(f"ply offset {args.ply:g} vx")

meta = json.loads((ref / "meta.json").read_text())
G = np.stack([cv2.imread(str(ref / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float64)
valid = (G != -1).all(-1)
H, W = valid.shape


def normals(grid, ok):
    gu = np.gradient(grid, axis=1)
    gv = np.gradient(grid, axis=0)
    n = np.cross(gu, gv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n[~ok] = np.nan
    return n


# densify 4x for projecting fibre points onto the reference sheet
D = 4
uu, vv = np.meshgrid(np.arange(0, W - 1 + 1e-9, 1 / D), np.arange(0, H - 1 + 1e-9, 1 / D))
dense = np.stack([cv2.remap(G[..., c].astype(np.float32), uu.astype(np.float32), vv.astype(np.float32),
                            cv2.INTER_LINEAR) for c in range(3)], -1).astype(np.float64)
dok = cv2.remap(valid.astype(np.float32), uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR) > 0.999
dn = normals(dense, dok)
dok &= np.isfinite(dn).all(-1)
tree = cKDTree(dense[dok])
d_uv = np.stack([uu[dok], vv[dok]], 1)
d_xyz, d_n = dense[dok], dn[dok]

keep = {}
for k in filter(None, args.keep.split(",")):
    name, axis, lo, hi = k.split(":")
    keep[name] = ("xyz".index(axis), float(lo), float(hi))

uv_pts, offs, info = [], [], []
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    name = p.name[:-9]
    if name in args.exclude.split(","):
        continue
    a = np.load(p).astype(np.float64)
    xyz, prob, arc = a[:, :3], a[:, 3], a[:, 4]
    s = np.arange(args.trim, arc[-1] - args.trim, args.step)
    P = np.stack([np.interp(s, arc, xyz[:, c]) for c in range(3)], 1)
    pr = np.interp(s, arc, prob)
    m = pr >= args.min_prob
    if name in keep:
        ax, lo, hi = keep[name]
        m &= (P[:, ax] >= lo) & (P[:, ax] <= hi)
    P = P[m]
    dist, k = tree.query(P)
    n = d_n[k]
    sign = -1.0 if name.startswith("h") else 1.0
    target = P + sign * args.ply / 2 * n
    o = ((target - d_xyz[k]) * n).sum(1)
    ok = dist <= args.max_dist
    uv_pts.append(d_uv[k][ok])
    offs.append(o[ok])
    info.append((name, ok.sum(), np.median(o[ok]) if ok.any() else np.nan))
bounds = np.concatenate([[0], np.cumsum([len(o) for o in offs])])
uv_pts, offs = np.concatenate(uv_pts), np.concatenate(offs)
print(f"{len(offs)} fibre samples from {len(info)} fibres; offsets to the reference sheet: "
      f"median {np.median(offs):+.1f}, 5-95% {np.percentile(offs, 5):+.1f}..{np.percentile(offs, 95):+.1f} vx")

t0 = time.time()
tps = RBFInterpolator(uv_pts, offs, kernel="thin_plate_spline", smoothing=args.smooth)
res = offs - tps(uv_pts)
print(f"thin-plate spline fitted in {time.time() - t0:.1f} s; residual at fibre samples: median |r| "
      f"{np.median(np.abs(res)):.2f}, 95% {np.percentile(np.abs(res), 95):.2f}, max {np.abs(res).max():.2f} vx")

# area spanned by the fibres
gu, gv = np.meshgrid(np.arange(W), np.arange(H))
cells = np.stack([gu.ravel(), gv.ravel()], 1).astype(np.float64)
inside = Delaunay(uv_pts).find_simplex(cells) >= 0
near = cKDTree(uv_pts).query(cells)[0] <= args.margin
region = (inside.reshape(H, W) | near.reshape(H, W)) & valid
region = cv2.dilate(region.astype(np.uint8), np.ones((2 * args.margin + 1,) * 2, np.uint8)).astype(bool) & valid

N = normals(G, valid)
region &= np.isfinite(N).all(-1)
o_grid = np.zeros((H, W))
o_grid[region] = tps(cells[region.ravel()])
new = G + o_grid[..., None] * np.nan_to_num(N)
new[~region] = -1

out.mkdir(parents=True, exist_ok=True)
for c, name in enumerate("xyz"):
    cv2.imwrite(str(out / f"{name}.tif"), new[..., c].astype(np.float32))
gen = cv2.imread(str(ref / "generations.tif"), cv2.IMREAD_UNCHANGED)
if gen is not None:
    cv2.imwrite(str(out / "generations.tif"), np.where(region, np.maximum(gen, 1), 0).astype(gen.dtype))

# area from the quads that are fully valid
q = region[:-1, :-1] & region[1:, :-1] & region[:-1, 1:] & region[1:, 1:]
e1 = new[:-1, 1:] - new[:-1, :-1]
e2 = new[1:, :-1] - new[:-1, :-1]
area_vx2 = float(np.linalg.norm(np.cross(e1, e2), axis=-1)[q].sum())
pts = new[region]
meta.update({
    "uuid": out.name, "seed_surface_id": out.name, "type": "seg", "format": "tifxyz",
    "bbox": [pts.min(0).tolist(), pts.max(0).tolist()],
    "area_vx2": area_vx2, "area_cm2": area_vx2 * (9.362e-4) ** 2,
    "date_last_modified": time.strftime("%Y%m%d%H%M%S000"),
    "source": f"fiber thin-plate spline from {Path(args.fibers).name} over {ref.name}",
})
(out / "meta.json").write_text(json.dumps(meta, indent=4))
print(f"region: {region.sum()} of {valid.sum()} valid vertices; applied offset median {np.median(o_grid[region]):+.1f}, "
      f"5-95% {np.percentile(o_grid[region], 5):+.1f}..{np.percentile(o_grid[region], 95):+.1f} vx")
print(f"wrote {out} (area {meta['area_cm2']:.2f} cm2)")
for i, (name, n_s, med) in enumerate(info):
    r = np.abs(res[bounds[i]:bounds[i + 1]])
    print(f"  {name:22s} {n_s:4d} samples, median offset {med:+6.1f} vx, fit residual median {np.median(r):4.1f} "
          f"/ 95% {np.percentile(r, 95):4.1f} vx")
