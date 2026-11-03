"""Shared AFV prediction store: cached CT access and on-demand 96^3 tile prediction."""
import os
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path

import numcodecs
import numpy as np
import requests
import torch
import zarr

ROOT = Path(os.environ.get("VC3D_HOME", Path.home() / ".VC3D"))
VOLUME_ID = "20250728152254"
CT_URL = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes/"
          f"{VOLUME_ID}-9.362um-1.2m-113keV-masked.zarr")
STORE = ROOT / "afv_predictions" / f"PHerc0846A_{VOLUME_ID}.zarr"
CT_CACHE = ROOT / "afv_ct_cache" / VOLUME_ID
# folder with model.py, plans.json, dataset.json and checkpoint_final.pth
MODEL_DIR = Path(os.environ.get("AFV_MODEL_DIR", ROOT / "afv_model"))
CORE, MARGIN = 96, 16
PATCH = CORE + 2 * MARGIN


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
        for attempt in range(6):
            try:
                r = self._session().get(f"{self.url}/{key}", timeout=60)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 5:
                    raise
                self.local.s = requests.Session()
                time.sleep(2 + 2 * attempt)
                continue
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
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        tmp.write_bytes(r.content)
        try:
            os.replace(tmp, path)
        except PermissionError:                 # another process finished the same chunk first
            tmp.unlink(missing_ok=True)
        return r.content

    def __iter__(self):
        raise NotImplementedError("listing is not supported")

    def __len__(self):
        return 0

    def __setitem__(self, key, value):
        raise PermissionError("read-only store")

    def __delitem__(self, key):
        raise PermissionError("read-only store")


def open_ct():
    CT_CACHE.mkdir(parents=True, exist_ok=True)
    return zarr.open(zarr.storage.KVStore(CachedHttpStore(CT_URL, CT_CACHE)), mode="r")["0"]


def open_store(shape):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(STORE), mode="a")
    comp = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    for name in ("vt", "hz"):
        if name not in root:
            root.create_dataset(name, shape=shape, chunks=(CORE,) * 3, dtype="u1", fill_value=0,
                                compressor=comp, write_empty_chunks=False)
    if "dir" not in root:
        root.create_dataset("dir", shape=(3,) + tuple(shape), chunks=(3,) + (CORE,) * 3, dtype="i1",
                            fill_value=0, compressor=comp, write_empty_chunks=False)
    root.attrs.update({"model": "Qualzz20/afv_fiber_9um_v2", "ct_url": CT_URL, "core": CORE, "margin": MARGIN,
                       "vt_hz": "softmax probability * 255", "dir": "unit fibre axis (x,y,z) * 127, sign-free"})
    return root


class Predictor:
    """Predicts 96^3 store tiles on demand; tiles already listed in tiles_done.txt are skipped."""

    def __init__(self):
        self.ct = open_ct()
        self.shape = tuple(self.ct.shape)
        self.root = open_store(self.shape)
        self.done_file = STORE / "tiles_done.txt"
        self.model = None
        self.refresh()

    def refresh(self):
        """Re-read tiles finished by any process (e.g. a background band job)."""
        self.done = set()
        if self.done_file.exists():
            for line in self.done_file.read_text().split():
                self.done.add(tuple(map(int, line.split(","))))

    def in_volume(self, t):
        return all(0 <= t[i] * CORE < self.shape[i] for i in range(3))

    def read_patch(self, t):
        s = np.array(t) * CORE - MARGIN
        lo, hi = np.maximum(s, 0), np.minimum(s + PATCH, self.shape)
        vol = np.zeros((PATCH,) * 3, np.uint8)
        vol[tuple(slice(a - b, c - b) for a, b, c in zip(lo, s, hi))] = \
            self.ct[tuple(slice(a, c) for a, c in zip(lo, hi))]
        return vol

    def predict(self, t, vol):
        """Run the model on a prefetched patch and write the tile core to the store."""
        if self.model is None:
            import sys
            sys.path.insert(0, str(MODEL_DIR))
            from model import load_model
            self.model = load_model(MODEL_DIR, device="cuda")
        from model import normalize_ct
        core = tuple(slice(t[k] * CORE, min((t[k] + 1) * CORE, self.shape[k])) for k in range(3))
        n = [c.stop - c.start for c in core]
        c = (slice(MARGIN, MARGIN + n[0]), slice(MARGIN, MARGIN + n[1]), slice(MARGIN, MARGIN + n[2]))
        if vol[c].max() > 0:
            x = torch.from_numpy(vol)[None, None].cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                o = self.model(normalize_ct(x))
            prob = torch.softmax(o["seg"].float(), 1)[0][(slice(None),) + c]
            d = torch.nn.functional.normalize(o["direction"].float()[0], dim=0)[(slice(None),) + c]
            self.root["vt"][core] = (prob[1] * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            self.root["hz"][core] = (prob[2] * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            self.root["dir"][(slice(None),) + core] = \
                (d * 127).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        with open(self.done_file, "a") as f:
            f.write(",".join(map(str, t)) + "\n")
        self.done.add(tuple(t))

    def ensure(self, t):
        """Make sure tile t is in the store (predicting it now if needed). Returns True if predicted now."""
        t = tuple(int(v) for v in t)
        if t in self.done or not self.in_volume(t):
            return False
        self.refresh()
        if t in self.done:
            return False
        self.predict(t, self.read_patch(t))
        return True
