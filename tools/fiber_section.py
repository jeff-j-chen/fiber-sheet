"""Cross-section along a traced fibre: CT and fibre probability vs offset along the sheet normal.

Usage: python fiber_section.py <fiber.path.npy> <segment_dir> <out.png> [--half 40] [--family h]
Columns = arc length along the fibre (2 vx), rows = offset along the local sheet normal.
Green line = the fibre (row 0); red curve = where the sheet is, measured along that normal.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("segment")
ap.add_argument("out")
ap.add_argument("--half", type=int, default=40)
ap.add_argument("--family", default="h")
args = ap.parse_args()

path = np.load(args.path).astype(np.float64)
xyz, arc = path[:, :3], path[:, 4]
s = np.arange(0, arc[-1], 2.0)
F = np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1)            # fibre samples, every 2 vx

# sheet at 1 px/vx with normals
seg = Path(args.segment)
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (grid != -1).all(-1)
gh, gw = gv.shape
uu, vv = np.meshgrid(np.arange(0, (gw - 1) / sx, 1.0) * sx, np.arange(0, (gh - 1) / sy, 1.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(grid[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
vi, ui = np.nonzero(ok)
tree = cKDTree(sp[vi, ui])

# normal direction at each fibre sample: from the nearest sheet points, smoothed, made perpendicular to the fibre
_, nn = tree.query(F, k=25)
n = nrm[vi[nn], ui[nn]]
n = n * np.sign((n * n[:, :1, :]).sum(-1, keepdims=True))
n = n.mean(1)
ref = n[len(n) // 2]
n *= np.sign(n @ ref)[:, None]
n = gaussian_filter1d(n, 10, axis=0)
tan = np.gradient(F, axis=0)
tan /= np.linalg.norm(tan, axis=1, keepdims=True)
n -= (n * tan).sum(1, keepdims=True) * tan
n /= np.linalg.norm(n, axis=1, keepdims=True)

# where the sheet crosses each normal line (signed offset), if it's within reach
offs = np.arange(-args.half, args.half + 1, 1.0)
line = F[:, None, :] + n[:, None, :] * offs[None, :, None]                   # cols, rows, 3
d, _ = tree.query(line.reshape(-1, 3))
d = d.reshape(len(F), len(offs))
best = d.argmin(1)
sheet_row = np.where(d[np.arange(len(F)), best] <= 2.0, best, -1)

# sample CT and fibre probability along the normal lines, window by window
pred = Predictor()
ct, prob_arr = pred.ct, pred.root["hz" if args.family == "h" else "vt"]
img_ct = np.zeros((len(offs), len(F)), np.float32)
img_p = np.zeros_like(img_ct)
for a in range(0, len(F), 50):
    b = min(a + 50, len(F))
    pts = line[a:b].reshape(-1, 3)[:, ::-1]                                    # z,y,x
    lo = np.floor(pts.min(0)).astype(int) - 1
    hi = np.ceil(pts.max(0)).astype(int) + 2
    lo, hi = np.maximum(lo, 0), np.minimum(hi, pred.shape)
    for t in {tuple(k) for k in (np.floor(pts / 96)).astype(int)}:
        pred.ensure(t)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    block_ct = ct[sl].astype(np.float32)
    block_p = prob_arr[sl].astype(np.float32) / 255
    loc = (pts - lo).T
    img_ct[:, a:b] = map_coordinates(block_ct, loc, order=1).reshape(b - a, len(offs)).T
    img_p[:, a:b] = map_coordinates(block_p, loc, order=1).reshape(b - a, len(offs)).T

ct8 = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.normalize(img_ct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
top = cv2.cvtColor(ct8, cv2.COLOR_GRAY2BGR)
bot = cv2.applyColorMap((img_p * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
centre = args.half
for img in (top, bot):
    img[centre, :] = (0, 255, 0)
    for c in np.nonzero(sheet_row >= 0)[0]:
        img[sheet_row[c], c] = (0, 0, 255)
scale = 3
out = np.vstack([cv2.resize(top, None, fx=1, fy=scale, interpolation=cv2.INTER_NEAREST),
                 np.full((6, top.shape[1], 3), 255, np.uint8),
                 cv2.resize(bot, None, fx=1, fy=scale, interpolation=cv2.INTER_NEAREST)])
cv2.imwrite(args.out, out)
on = (sheet_row >= 0) & (np.abs(sheet_row - centre) <= 4)
print(f"{len(F)} columns ({arc[-1]:.0f} vx); sheet within 4 vx of the fibre in {on.mean() * 100:.0f}% of columns, "
      f"outside +/-{args.half} vx (or absent) in {(sheet_row < 0).mean() * 100:.0f}%")
print("wrote", args.out)
