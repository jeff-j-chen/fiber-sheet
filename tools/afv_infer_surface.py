"""Run the AFV fibre model only where a tifxyz sheet passes, and keep 2D on-sheet maps.

Usage:
  python afv_infer_surface.py <segment_dir> <out_dir> [--around X Y Z --radius R] [--dry-run]

Outputs (in out_dir): vt_prob.tif / hz_prob.tif (uint8, 0..255), dir_uv.npy (float16 fibre
direction projected into sheet u/v), valid.png, preview.png. Pixel (u, v) maps to tifxyz grid
(row = v * scale_y, col = u * scale_x), the same convention as vc_render_tifxyz --scale 1.
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import fsspec

MODEL_DIR = Path(os.environ.get("AFV_MODEL_DIR", Path.home() / ".VC3D" / "afv_model"))
sys.path.insert(0, str(MODEL_DIR))
from model import load_model, normalize_ct  # noqa: E402

CT_URL = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes/"
          "20250728152254-9.362um-1.2m-113keV-masked.zarr")
PATCH, STRIDE, MARGIN = 128, 96, 16          # tile core = [MARGIN, PATCH-MARGIN) along each axis
NORMAL_OFFSETS = np.arange(-4, 5, dtype=np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("segment")
ap.add_argument("out")
ap.add_argument("--around", nargs=3, type=float)
ap.add_argument("--radius", type=float, default=600.0)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
seg, out = Path(args.segment), Path(args.out)

# ---- sheet at 1 voxel / pixel ------------------------------------------------------------
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gvalid = (grid != -1).all(-1)
gh, gw = gvalid.shape
W, H = int(round((gw - 1) / sx)) + 1, int(round((gh - 1) / sy)) + 1
uu, vv = np.meshgrid(np.arange(W, dtype=np.float32) * sx, np.arange(H, dtype=np.float32) * sy)
pts = np.stack([cv2.remap(grid[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
valid = cv2.remap(gvalid.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
du = np.gradient(pts, axis=1)
dv = np.gradient(pts, axis=0)
nrm = np.cross(du, dv)
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
valid &= np.isfinite(nrm).all(-1)

if args.around:
    centre = np.array(args.around, np.float32)
    valid &= np.linalg.norm(pts - centre, axis=-1) <= args.radius
ys, xs = np.nonzero(valid)
print(f"sheet {W}x{H} px at 1 vx/px, {len(ys):,} sheet pixels selected")
if len(ys) == 0:
    sys.exit("no sheet pixels selected")

# ---- tiles that contain sheet points (incl. the normal band) ----------------------------
P = pts[ys, xs]                                   # (N,3) xyz
N = nrm[ys, xs]
band = np.concatenate([P + N * NORMAL_OFFSETS.min(), P, P + N * NORMAL_OFFSETS.max()])
zyx = band[:, ::-1]
origin = np.floor(zyx.min(0)) - MARGIN
core_idx = np.floor((zyx - origin - MARGIN) / STRIDE).astype(np.int64)
tiles = np.unique(core_idx, axis=0)
ct = zarr.open(zarr.LRUStoreCache(fsspec.get_mapper(CT_URL), max_size=6 * 2**30), mode="r")["0"]
chunk = np.array(ct.chunks)
starts = origin + tiles * STRIDE
n_chunks = len({tuple(((s // chunk) + np.array(o)).astype(int))
                for s in starts for o in np.ndindex(2, 2, 2)})
print(f"{len(tiles)} tiles of {PATCH}^3 (stride {STRIDE}); about {n_chunks} CT chunks of "
      f"{tuple(chunk)} (~{n_chunks * np.prod(chunk) / 2**30:.1f} GiB raw, streamed, not saved)")
if args.dry_run:
    sys.exit(0)

# ---- inference ----------------------------------------------------------------------------
out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
model = load_model(MODEL_DIR, device=dev)
vt = np.zeros(len(ys), np.float32)
hz = np.zeros(len(ys), np.float32)
dir3 = np.zeros((len(ys), 3), np.float32)
order = np.lexsort(core_idx[len(ys):2 * len(ys)].T[::-1])   # points grouped by their own tile
own = core_idx[len(ys):2 * len(ys)]
groups = OrderedDict()
for k, key in enumerate(map(tuple, own[order])):
    groups.setdefault(key, []).append(order[k])

shape = np.array(ct.shape)
t0 = time.time()
for i, key in enumerate(groups):
    s = (origin + np.array(key) * STRIDE).astype(np.int64)
    lo, hi = np.maximum(s, 0), np.minimum(s + PATCH, shape)
    vol = np.zeros((PATCH,) * 3, np.uint8)
    if (hi > lo).all():
        vol[tuple(slice(a - b, c - b) for a, b, c in zip(lo, s, hi))] = ct[tuple(slice(a, c) for a, c in zip(lo, hi))]
    if vol.max() == 0:
        continue
    x = torch.from_numpy(vol)[None, None].to(dev)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        o = model(normalize_ct(x))
    prob = torch.softmax(o["seg"].float(), 1)[0]               # 4,Z,Y,X
    d = F.normalize(o["direction"].float()[0], dim=0)          # 3,Z,Y,X (x,y,z)
    fields = torch.cat([prob[1:3], d], 0)[None]                # 1,5,Z,Y,X

    idx = np.array(groups[key])
    p = P[idx][:, None, :] + N[idx][:, None, :] * NORMAL_OFFSETS[None, :, None]   # n,9,3 xyz
    local = p - s[::-1].astype(np.float32)                                         # xyz in tile
    g = torch.from_numpy(local / (PATCH - 1) * 2 - 1).to(dev)                     # grid_sample wants x,y,z
    samp = F.grid_sample(fields, g[None, :, :, None, :], align_corners=True)[0, :, :, :, 0]  # 5,n,9
    v_, h_ = samp[0], samp[1]
    best = torch.maximum(v_, h_).argmax(1)
    vt[idx] = v_.max(1).values.cpu().numpy()
    hz[idx] = h_.max(1).values.cpu().numpy()
    dir3[idx] = samp[2:5].permute(1, 2, 0)[torch.arange(len(idx)), best].cpu().numpy()
    if (i + 1) % 20 == 0 or i + 1 == len(groups):
        el = time.time() - t0
        print(f"  tile {i + 1}/{len(groups)}  {el:.0f}s elapsed, ~{el / (i + 1) * (len(groups) - i - 1):.0f}s left",
              flush=True)

# ---- 2D outputs ---------------------------------------------------------------------------
def to_img(vals):
    img = np.zeros((H, W), np.uint8)
    img[ys, xs] = np.clip(vals * 255 + 0.5, 0, 255).astype(np.uint8)
    return img

vt_img, hz_img = to_img(vt), to_img(hz)
cv2.imwrite(str(out / "vt_prob.tif"), vt_img)
cv2.imwrite(str(out / "hz_prob.tif"), hz_img)
cv2.imwrite(str(out / "valid.png"), valid.astype(np.uint8) * 255)
# fibre direction expressed in sheet axes (u = along columns, v = along rows)
eu = du[ys, xs] / (np.linalg.norm(du[ys, xs], axis=1, keepdims=True) + 1e-9)
ev = dv[ys, xs] / (np.linalg.norm(dv[ys, xs], axis=1, keepdims=True) + 1e-9)
duv = np.zeros((H, W, 2), np.float16)
duv[ys, xs] = np.stack([(dir3 * eu).sum(1), (dir3 * ev).sum(1)], 1)
np.save(out / "dir_uv.npy", duv)
preview = np.zeros((H, W, 3), np.uint8)
preview[..., 2] = hz_img            # red   = horizontal
preview[..., 1] = vt_img            # green = vertical
cv2.imwrite(str(out / "preview.png"), preview)
(out / "info.json").write_text(json.dumps({
    "segment": str(seg), "ct_url": CT_URL, "model": "Qualzz20/afv_fiber_9um_v2",
    "pixel_to_grid": {"row": f"v * {sy}", "col": f"u * {sx}"}, "size_wh": [W, H],
    "normal_band_voxels": NORMAL_OFFSETS.tolist(), "tiles": len(groups),
    "around": args.around, "radius": args.radius if args.around else None}, indent=1))
print(f"done in {time.time() - t0:.0f}s -> {out}")
