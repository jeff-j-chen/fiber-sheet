"""Horizontal and vertical fibre maps: fibre-depth CT strips laid out in the sheet's (u, v) frame.

Usage: python fiber_maps.py <fiber_dir> <segment_dir> [--half 15] [--xmax 5100] [--trim 100] [--min-prob 0.5]
Writes <fiber_dir>/maps/{horizontal,vertical}_map.png (+ .tif, 16-bit counts-weighted mean) and
{horizontal,vertical}_labeled.png. Pixel (u, v) = sheet grid (row v*scale_y, col u*scale_x), i.e. the
same frame as vc_render_tifxyz --scale 1 for this sheet. Each strip is sampled at the fibre's own depth
(a 3-voxel slab along the local layer normal), +/-half voxels across the fibre within the layer.
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
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--half", type=int, default=15)
ap.add_argument("--xmax", type=float, default=5100.0)
ap.add_argument("--trim", type=float, default=100.0, help="vx dropped from each end of every fibre")
ap.add_argument("--min-prob", type=float, default=0.5, help="drop columns where the fibre's own probability is lower")
args = ap.parse_args()
fdir = Path(args.fibers)
out_dir = fdir / "maps"
out_dir.mkdir(exist_ok=True)

# sheet at 1 px/vx: positions, normals and a KD-tree to map 3D points to (u, v)
seg = Path(args.segment)
sx, sy = json.loads((seg / "meta.json").read_text())["scale"]
g = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (g != -1).all(-1)
Wm, Hm = int(round((g.shape[1] - 1) / sx)) + 1, int(round((g.shape[0] - 1) / sy)) + 1
uu, vv = np.meshgrid(np.arange(Wm, dtype=np.float32) * sx, np.arange(Hm, dtype=np.float32) * sy)
sp = np.stack([cv2.remap(g[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
vi, ui = np.nonzero(ok)
tree = cKDTree(sp[vi, ui])
nvec = nrm[vi, ui]
print(f"sheet frame {Wm} x {Hm} px")

pred = Predictor()
ct, hz_arr, vt_arr = pred.ct, pred.root["hz"], pred.root["vt"]
across = np.arange(-args.half, args.half + 1, 1.0)


def sample_block(arr, pts_zyx, scale=1.0):
    lo = np.maximum(np.floor(pts_zyx.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(pts_zyx.max(0)).astype(int) + 2, pred.shape)
    blk = arr[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32) * scale
    return map_coordinates(blk, (pts_zyx - lo).T, order=1, mode="nearest")


maps = {f: (np.zeros((Hm, Wm), np.float64), np.zeros((Hm, Wm), np.float64)) for f in "hv"}
labels = {f: [] for f in "hv"}
names = sorted(p.name[: -len(".path.npy")] for p in fdir.glob("*.path.npy"))
for name in names:
    fam = name[0]
    path = np.load(fdir / f"{name}.path.npy").astype(np.float64)
    xyz, arc = path[:, :3], path[:, 4]
    s = np.arange(args.trim, arc[-1] - args.trim, 1.0)
    if len(s) < 50:
        continue
    F = np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1)
    t = gaussian_filter1d(np.gradient(F, axis=0), 5, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    _, k = tree.query(F, k=12)
    n = gaussian_filter1d(nvec[k].mean(1), 15, axis=0)
    n -= (n * t).sum(1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    side = np.cross(t, n)

    W, Hh = len(F), len(across)
    strip = np.zeros((Hh, W), np.float32)
    own_p = np.zeros(W, np.float32)
    own_arr = hz_arr if fam == "h" else vt_arr
    for a0 in range(0, W, 150):
        a1 = min(a0 + 150, W)
        base = F[a0:a1, None, :] + side[a0:a1, None, :] * across[None, :, None]
        for t_ in {tuple(v) for v in (np.floor(base.reshape(-1, 3)[:, ::-1] / 96)).astype(int)}:
            pred.ensure(t_)
        acc = 0
        for dd in (-1.0, 0.0, 1.0):
            acc = acc + sample_block(ct, (base + n[a0:a1, None, :] * dd).reshape(-1, 3)[:, ::-1])
        strip[:, a0:a1] = (acc / 3).reshape(a1 - a0, Hh).T
        own_p[a0:a1] = sample_block(own_arr, F[a0:a1, ::-1], 1 / 255)

    keep = (F[:, 0] <= args.xmax) & (gaussian_filter1d(own_p, 10) >= args.min_prob)
    # place every strip sample at the sheet pixel nearest to its own 3D position
    pts = (F[:, None, :] + side[:, None, :] * across[None, :, None])[keep]          # cols, rows, 3
    dist, idx = tree.query(pts.reshape(-1, 3))
    near = dist <= 45.0                                   # beyond the sheet's edge -> no place in its frame
    pu, pv = ui[idx][near], vi[idx][near]
    val = strip.T[keep].reshape(-1)[near]
    acc_map, cnt_map = maps[fam]
    np.add.at(acc_map, (pv, pu), val)
    np.add.at(cnt_map, (pv, pu), 1)
    centre = pts[:, Hh // 2]
    cd, ci = tree.query(centre)
    ci = ci[cd <= 45.0]
    if len(ci) == 0:
        continue
    labels[fam].append((name, ui[ci], vi[ci]))
    print(f"  {name:22s} kept {len(ci):5d}/{W} columns "
          f"(u {ui[ci].min()}-{ui[ci].max()}, v {vi[ci].min()}-{vi[ci].max()})")

vals = np.concatenate([(maps[f][0][maps[f][1] > 0] / maps[f][1][maps[f][1] > 0]) for f in "hv" if (maps[f][1] > 0).any()])
lo, hi = np.percentile(vals, [1, 99.5])
clahe = cv2.createCLAHE(2.5, (16, 16))
for fam, word in (("h", "horizontal"), ("v", "vertical")):
    acc_map, cnt_map = maps[fam]
    has = cnt_map > 0
    mean = np.zeros_like(acc_map)
    mean[has] = acc_map[has] / cnt_map[has]
    img8 = np.clip((mean - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    img8 = np.where(has, clahe.apply(img8), 0).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{word}_map.png"), img8)
    cv2.imwrite(str(out_dir / f"{word}_map.tif"), np.where(has, mean * 256, 0).astype(np.uint16))
    lab = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    lab[~has] = (40, 22, 18)
    for name, cu, cv_ in labels[fam]:
        pts2 = np.stack([cu, cv_], 1).astype(np.int32)
        j = len(pts2) // 2
        cv2.putText(lab, name, (int(pts2[j, 0]) + 6, int(pts2[j, 1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{word}_labeled.png"), lab)
    print(f"{word}: covered {has.sum():,} px -> {out_dir / (word + '_map.png')}")

both_acc = maps["h"][0] + maps["v"][0]
both_cnt = maps["h"][1] + maps["v"][1]
has = both_cnt > 0
mean = np.zeros_like(both_acc)
mean[has] = both_acc[has] / both_cnt[has]
img8 = np.where(has, clahe.apply(np.clip((mean - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)), 0).astype(np.uint8)
cv2.imwrite(str(out_dir / "combined_map.png"), img8)
print("combined ->", out_dir / "combined_map.png")
