"""AFV fibre predictions in a 3D band around a tifxyz sheet, stored as a sparse zarr.

Usage:
  python afv_infer_band.py <segment_dir> [--band 40] [--around X Y Z --radius R] [--dry-run]

Store: .VC3D/afv_predictions/<volume>.zarr with arrays (full CT shape, z,y,x):
  vt, hz : uint8 probability * 255         chunks (96,96,96)
  dir    : int8 unit fibre axis (x,y,z)*127 chunks (3,96,96,96)
Each 96^3 chunk is one prediction tile (128^3 input, 16-voxel context margin); tiles already in
the store are skipped, so runs on other regions or sheets extend it. CT chunks are cached on disk.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numcodecs
import numpy as np
import requests
import torch
import zarr

MODEL_DIR = Path(os.environ.get("AFV_MODEL_DIR", Path.home() / ".VC3D" / "afv_model"))
sys.path.insert(0, str(MODEL_DIR))
from model import load_model, normalize_ct  # noqa: E402

ROOT = Path.home() / ".VC3D"
VOLUME_ID = "20250728152254"
CT_URL = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes/"
          f"{VOLUME_ID}-9.362um-1.2m-113keV-masked.zarr")
STORE = ROOT / "afv_predictions" / f"PHerc0846A_{VOLUME_ID}.zarr"
CT_CACHE = ROOT / "afv_ct_cache" / VOLUME_ID
CORE, MARGIN = 96, 16
PATCH = CORE + 2 * MARGIN

ap = argparse.ArgumentParser()
ap.add_argument("segment")
ap.add_argument("--band", type=float, default=40.0, help="voxels either side of the sheet along its normal")
ap.add_argument("--around", nargs=3, type=float)
ap.add_argument("--radius", type=float, default=600.0)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
seg = Path(args.segment)

# ---- sheet points (every 4 px is enough to find 96^3 tiles) -------------------------------
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gvalid = (grid != -1).all(-1)
gh, gw = gvalid.shape
step = 4.0
uu, vv = np.meshgrid(np.arange(0, (gw - 1) / sx, step) * sx, np.arange(0, (gh - 1) / sy, step) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
pts = np.stack([cv2.remap(grid[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gvalid.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
if args.around:
    ok &= np.linalg.norm(pts - np.array(args.around, np.float32), axis=-1) <= args.radius
P, N = pts[ok], nrm[ok]
offsets = np.arange(-args.band, args.band + 0.1, 8.0, dtype=np.float32)
band = (P[:, None, :] + N[:, None, :] * offsets[None, :, None]).reshape(-1, 3)
tiles = np.unique(np.floor(band[:, ::-1] / CORE).astype(np.int64), axis=0)       # z,y,x tile index

# ---- stores --------------------------------------------------------------------------------
class CachedHttpStore(MutableMapping):
    """Read-only zarr store over HTTP with an atomic, thread-safe on-disk chunk cache."""

    def __init__(self, url, cache_dir):
        self.url, self.cache = url.rstrip("/"), Path(cache_dir)
        self.local = threading.local()
        self.key_locks, self.guard = {}, threading.Lock()

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def __getitem__(self, key):
        path = self.cache / key
        if path.exists():
            return path.read_bytes()
        with self.guard:
            lock = self.key_locks.setdefault(key, threading.Lock())
        with lock:
            if path.exists():
                return path.read_bytes()
            return self._download(key, path)

    def _download(self, key, path):
        for attempt in range(4):
            r = self._session().get(f"{self.url}/{key}", timeout=60)
            if r.status_code in (403, 404):
                raise KeyError(key)
            r.raise_for_status()
            want = r.headers.get("Content-Length")
            if want is None or int(want) == len(r.content):
                break
            time.sleep(1 + attempt)
        else:
            raise IOError(f"incomplete download for {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        tmp.write_bytes(r.content)
        os.replace(tmp, path)
        return r.content

    def __iter__(self):
        raise NotImplementedError("listing is not supported")

    def __len__(self):
        return 0

    def __setitem__(self, key, value):
        raise PermissionError("read-only store")

    def __delitem__(self, key):
        raise PermissionError("read-only store")


CT_CACHE.mkdir(parents=True, exist_ok=True)
ct = zarr.open(zarr.storage.KVStore(CachedHttpStore(CT_URL, CT_CACHE)), mode="r")["0"]
shape = tuple(ct.shape)
STORE.parent.mkdir(parents=True, exist_ok=True)
root = zarr.open_group(str(STORE), mode="a")
comp = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
for name in ("vt", "hz"):
    if name not in root:
        root.create_dataset(name, shape=shape, chunks=(CORE,) * 3, dtype="u1", fill_value=0,
                            compressor=comp, write_empty_chunks=False)
if "dir" not in root:
    root.create_dataset("dir", shape=(3,) + shape, chunks=(3,) + (CORE,) * 3, dtype="i1", fill_value=0,
                        compressor=comp, write_empty_chunks=False)
root.attrs.update({"model": "Qualzz20/afv_fiber_9um_v2", "ct_url": CT_URL, "core": CORE, "margin": MARGIN,
                   "vt_hz": "softmax probability * 255", "dir": "unit fibre axis (x,y,z) * 127, sign-free"})
done_file = STORE / "tiles_done.txt"
done = set()
if done_file.exists():
    done = {tuple(map(int, line.split(","))) for line in done_file.read_text().split()}
todo = [tuple(t) for t in tiles if tuple(t) not in done]
in_vol = [t for t in todo if all(0 <= t[i] * CORE < shape[i] for i in range(3))]
print(f"{len(P):,} sheet samples, band +/-{args.band:.0f} vx -> {len(tiles)} tiles; "
      f"{len(tiles) - len(todo)} already in store, {len(in_vol)} to predict")
if args.dry_run or not in_vol:
    sys.exit(0)

# ---- inference with CT prefetch --------------------------------------------------------------
def read_patch(t):
    s = np.array(t) * CORE - MARGIN
    lo, hi = np.maximum(s, 0), np.minimum(s + PATCH, shape)
    vol = np.zeros((PATCH,) * 3, np.uint8)
    vol[tuple(slice(a - b, c - b) for a, b, c in zip(lo, s, hi))] = ct[tuple(slice(a, c) for a, c in zip(lo, hi))]
    return t, vol


model = load_model(MODEL_DIR, device="cuda")
t0 = time.time()
skipped_empty = 0
with ThreadPoolExecutor(max_workers=6) as pool, open(done_file, "a") as log:
    futures = [pool.submit(read_patch, t) for t in in_vol[:12]]
    nxt = 12
    for i in range(len(in_vol)):
        t, vol = futures[i].result()
        if nxt < len(in_vol):
            futures.append(pool.submit(read_patch, in_vol[nxt]))
            nxt += 1
        futures[i] = None
        core = tuple(slice(t[k] * CORE, min((t[k] + 1) * CORE, shape[k])) for k in range(3))
        n = [c.stop - c.start for c in core]
        if vol[MARGIN:MARGIN + n[0], MARGIN:MARGIN + n[1], MARGIN:MARGIN + n[2]].max() > 0:
            x = torch.from_numpy(vol)[None, None].cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                o = model(normalize_ct(x))
            c = (slice(MARGIN, MARGIN + n[0]), slice(MARGIN, MARGIN + n[1]), slice(MARGIN, MARGIN + n[2]))
            prob = torch.softmax(o["seg"].float(), 1)[0][(slice(None),) + c]
            d = torch.nn.functional.normalize(o["direction"].float()[0], dim=0)[(slice(None),) + c]
            root["vt"][core] = (prob[1] * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            root["hz"][core] = (prob[2] * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            root["dir"][(slice(None),) + core] = (d * 127).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        else:
            skipped_empty += 1
        log.write(",".join(map(str, t)) + "\n")
        log.flush()
        if (i + 1) % 25 == 0 or i + 1 == len(in_vol):
            el = time.time() - t0
            print(f"  tile {i + 1}/{len(in_vol)}  {el:.0f}s, ~{el / (i + 1) * (len(in_vol) - i - 1):.0f}s left",
                  flush=True)
print(f"done in {time.time() - t0:.0f}s ({skipped_empty} tiles outside the scan were skipped) -> {STORE}")
