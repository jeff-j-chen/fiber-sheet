"""Lateral (in-sheet) wander check: residual of a fibre's sheet coordinate across the fibre from a smoothed copy of itself.

Usage: python lat_check.py <sheet name> <dir> [<dir> ...] [--sigma 200] [--thr 8]
In the flattened sheet horizontals run along rows and verticals along columns, so the across-fibre coordinate is nearly
a straight line; a strand hop or a wander off the strand shows as a residual of several vx. Prints the maximum residual
(vx) per fibre and the stretches above --thr.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_trace import Sheet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sheet")
ap.add_argument("dirs", nargs="+")
ap.add_argument("--sigma", type=float, default=200.0)
ap.add_argument("--thr", type=float, default=8.0)
args = ap.parse_args()
sh = Sheet(args.sheet)
for d in args.dirs:
    for p in sorted(Path(d).glob("*.path.npy")):
        n = p.name[:-9]
        A = np.load(p).astype(np.float64)[:, :3]
        uv, dist = sh.uv_of(A[::2])
        ok = dist < 25
        if ok.sum() < 100:
            continue
        uv = uv[ok]
        a, b = (uv[:, 0], uv[:, 1]) if n[0] == "h" else (uv[:, 1], uv[:, 0])
        o = np.argsort(b)
        a, b, ax = a[o], b[o], A[::2][ok][o][:, 0 if n[0] == "h" else 2]
        k = args.sigma / 2.0 / 20.0 / 0.5      # samples: fibre sampled every 2 vx, uv step ~ 20 vx per cell
        sm = gaussian_filter1d(a, max(args.sigma / 2.0, 1.0), mode="nearest")
        res = (a - sm) * 20.0
        bad = np.abs(res) > args.thr
        segs, i = [], 0
        while i < len(res):
            if bad[i]:
                j = i
                while j + 1 < len(res) and bad[j + 1]:
                    j += 1
                if ax[j] - ax[i] >= 20:
                    segs.append(f"{ax[i]:.0f}..{ax[j]:.0f}({res[i:j + 1][np.argmax(np.abs(res[i:j + 1]))]:+.0f})")
                i = j + 1
            else:
                i += 1
        print(f"{n:24s} max |res| {np.abs(res).max():5.1f} vx   " + " ".join(segs))
