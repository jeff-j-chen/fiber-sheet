"""Follow one predicted fibre on the sheet from a 3D seed point and write a vc3d_fiber JSON.

Usage: python trace_predicted_fiber.py <pred_dir> <x> <y> <z> <h|v> <out.json>
         [--threshold 0.35] [--max-gap 40] [--cp-spacing 60]
pred_dir comes from afv_infer_surface.py (vt_prob.tif, hz_prob.tif, dir_uv.npy, info.json).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from fiber_meta import FIBER_COORDS
from scipy.ndimage import gaussian_filter, map_coordinates

ap = argparse.ArgumentParser()
ap.add_argument("pred")
ap.add_argument("x", type=float)
ap.add_argument("y", type=float)
ap.add_argument("z", type=float)
ap.add_argument("family", choices=["h", "v"])
ap.add_argument("out")
ap.add_argument("--threshold", type=float, default=0.35)
ap.add_argument("--max-gap", type=int, default=40, help="pixels allowed below threshold before stopping")
ap.add_argument("--cp-spacing", type=float, default=60.0, help="control point spacing in voxels")
args = ap.parse_args()

pred = Path(args.pred)
info = json.loads((pred / "info.json").read_text())
seg = Path(info["segment"])
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], 0).astype(np.float64)
gvalid = (grid != -1).all(0).astype(np.float64)

prob = cv2.imread(str(pred / ("hz_prob.tif" if args.family == "h" else "vt_prob.tif")),
                  cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
prob_s = gaussian_filter(prob, 1.0)
duv = np.load(pred / "dir_uv.npy").astype(np.float32)
H, W = prob.shape
target = np.array([args.x, args.y, args.z])


def sheet_xyz(u, v):
    rows, cols = np.asarray(v, float) * sy, np.asarray(u, float) * sx
    p = np.stack([map_coordinates(grid[i], [rows, cols], order=1) for i in range(3)], -1)
    ok = map_coordinates(gvalid, [rows, cols], order=1) > 0.999
    return p, ok


# seed: sheet pixel closest to the 3D point (coarse search on the grid, then refine)
gv = gvalid > 0
d = np.linalg.norm(np.moveaxis(grid, 0, -1) - target, axis=-1)
d[~gv] = np.inf
r0, c0 = np.unravel_index(np.argmin(d), d.shape)
cu, cv_ = c0 / sx, r0 / sy
uu, vv = np.meshgrid(np.arange(cu - 25, cu + 26), np.arange(cv_ - 25, cv_ + 26))
p, ok = sheet_xyz(uu.ravel(), vv.ravel())
dd = np.linalg.norm(p - target, axis=1)
dd[~ok] = np.inf
k = int(np.argmin(dd))
seed = np.array([uu.ravel()[k], vv.ravel()[k]], float)
print(f"seed sheet pixel (u={seed[0]:.0f}, v={seed[1]:.0f}), {dd[k]:.2f} vx from the point; "
      f"{args.family}-prob there {prob[int(seed[1]), int(seed[0])]:.2f}")


def local_dir(pt, prev):
    u, v = int(round(pt[0])), int(round(pt[1]))
    win = duv[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].reshape(-1, 2)
    win = win[np.linalg.norm(win, axis=1) > 0.2]
    if len(win) == 0:
        return prev
    win = win * np.sign(win @ prev)[:, None]          # orientation is sign-free
    dvec = win.mean(0)
    n = np.linalg.norm(dvec)
    return prev if n < 1e-6 else 0.7 * prev + 0.3 * dvec / n


def snap(pt, dvec):
    """Move pt across the fibre to the ridge maximum within +/-3 px."""
    perp = np.array([-dvec[1], dvec[0]])
    offs = np.arange(-3, 3.01, 0.5)
    cand = pt[None] + offs[:, None] * perp[None]
    vals = map_coordinates(prob_s, [cand[:, 1], cand[:, 0]], order=1, mode="constant")
    j = int(np.argmax(vals))
    return cand[j], float(vals[j])


seed, _ = snap(seed, np.array([1.0, 0.0]) if args.family == "h" else np.array([0.0, 1.0]))
path = {0: seed}
for sign in (1, -1):
    pt = seed.copy()
    dvec = (np.array([1.0, 0.0]) if args.family == "h" else np.array([0.0, 1.0])) * sign
    dvec = local_dir(pt, dvec)
    gap, i = 0, 0
    while True:
        i += 1
        nxt = pt + dvec / np.linalg.norm(dvec)
        if not (2 <= nxt[0] < W - 3 and 2 <= nxt[1] < H - 3):
            break
        nxt, val = snap(nxt, dvec)
        nxt = pt + 0.6 * (nxt - pt) + 0.4 * (pt + dvec / np.linalg.norm(dvec) - pt)  # damp lateral jumps
        gap = gap + 1 if val < args.threshold else 0
        if gap > args.max_gap:
            break
        dvec = local_dir(nxt, dvec)
        pt = nxt
        path[sign * i] = pt
    # trim the trailing low-probability stretch
    for _ in range(gap):
        path.pop(sign * i, None)
        i -= 1

keys = sorted(path)
uv = np.array([path[k] for k in keys])
xyz, ok = sheet_xyz(uv[:, 0], uv[:, 1])
uv, xyz = uv[ok], xyz[ok]
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
seed_i = int(np.argmin(np.linalg.norm(xyz - target, axis=1)))

ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], args.cp_spacing)),
                                 [len(arc) - 1, seed_i]]))
line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
cps = xyz[ctrl].round(3).tolist()
lps = xyz[line].round(3).tolist()
cps[int(np.where(ctrl == seed_i)[0][0])] = target.tolist()           # pass exactly through the point
lps[int(np.where(line == seed_i)[0][0])] = target.tolist()
Path(args.out).write_text(json.dumps(
    {"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": cps, "line_points": lps}, indent=1))

mean_p = map_coordinates(prob, [uv[:, 1], uv[:, 0]], order=1).mean()
print(f"traced {arc[-1]:.0f} voxels ({arc[-1] * 9.362 / 1000:.2f} mm) along the sheet, "
      f"{len(cps)} control points, {len(lps)} line points, mean prob on path {mean_p:.2f}")
print(f"u range {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f}, v range {uv[:, 1].min():.0f}..{uv[:, 1].max():.0f}")
print("wrote", args.out)
np.save(Path(args.out).with_suffix(".uv.npy"), uv.astype(np.float32))
