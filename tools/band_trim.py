"""Cut a fibre to the stretch around an anchor where its own probability band is unbroken.

The fibre's family probability (hz for h, vt for v) above --thr is labelled into 3D connected components in a
box around the fibre. Walking out from the anchor (the axis position of a trusted crossing or the seed), the
fibre is cut at the first place where the band under it is missing for more than --gap vx or where it moves
onto a different component (a band change = a probable hop onto another layer).

Usage: python band_trim.py <in.path.npy> <out.path.npy> <anchor> [--lo X --hi X] [--thr 0.85] [--gap 6]
<anchor>: axis coordinate (x for h, z for v). --lo/--hi limit the analysed range (default: whole fibre).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from fiber_meta import FIBER_COORDS
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import CORE, Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("anchor", type=float)
ap.add_argument("--lo", type=float, default=-1e9)
ap.add_argument("--hi", type=float, default=1e9)
ap.add_argument("--thr", type=float, default=0.85)
ap.add_argument("--gap", type=float, default=6.0)
args = ap.parse_args()
name = Path(args.src).name[:-9]
ax = 0 if name[0] == "h" else 2
A = np.load(args.src).astype(np.float64)
if A[-1, ax] < A[0, ax]:
    A = A[::-1]
sel = (A[:, ax] >= args.lo) & (A[:, ax] <= args.hi)
P0 = A[sel]
arc0 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P0[:, :3], axis=0), axis=1))])
S = np.arange(0, arc0[-1], 1.0)
idx = np.searchsorted(arc0, S).clip(0, len(P0) - 1)
P = P0[idx]
# straightened tube: offsets in the two axes other than the fibre's main axis
R, RZ = 20, (6 if ax == 0 else 20)
u, w = (1, 2) if ax == 0 else (0, 1)
ou, ow = np.meshgrid(np.arange(-R, R + 1), np.arange(-RZ, RZ + 1), indexing="ij")
pr = Predictor()
arr = pr.root["hz" if ax == 0 else "vt"]
tube = np.zeros((len(P), *ou.shape), np.uint8)
for s0 in range(0, len(P), 200):
    blk = P[s0:s0 + 200, :3]
    lo = np.maximum(np.floor(blk.min(0)).astype(int) - R - 2, 0)
    hi = np.ceil(blk.max(0)).astype(int) + R + 3
    for kz in range(lo[2] // CORE, hi[2] // CORE + 1):
        for ky in range(lo[1] // CORE, hi[1] // CORE + 1):
            for kx in range(lo[0] // CORE, hi[0] // CORE + 1):
                pr.ensure((kz, ky, kx))
    vol = arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    for i, q in enumerate(blk):
        c = np.round(q).astype(int) - lo
        pts = np.repeat(c[None, None], ou.shape[0], 0).repeat(ou.shape[1], 1)
        pts[..., u] += ou
        pts[..., w] += ow
        pts = np.clip(pts, 0, np.array(vol.shape[::-1]) - 1)
        tube[s0 + i] = vol[pts[..., 2], pts[..., 1], pts[..., 0]]
lab, _ = ndimage.label(tube > args.thr * 255, structure=np.ones((3, 3, 3)))
core = lab[:, R - 1:R + 2, RZ - 1:RZ + 2].reshape(len(P), -1)
comp = np.array([np.bincount(r[r > 0]).argmax() if (r > 0).any() else 0 for r in core])
arc = S
k0 = int(np.argmin(np.abs(P[:, ax] - args.anchor)))


def walk(step):
    cur = comp[k0] if comp[k0] else None
    k, last_ok = k0, k0
    while 0 <= k + step < len(P):
        k += step
        c = comp[k]
        if c == 0:
            if abs(arc[k] - arc[last_ok]) > args.gap:
                return last_ok, f"band missing {abs(arc[k] - arc[last_ok]):.0f} vx at {P[last_ok, ax]:.0f}"
            continue
        if cur is None:
            cur = c
        if c != cur:
            return last_ok, f"band change at {P[k, ax]:.0f}" + (f" after a {abs(arc[k] - arc[last_ok]):.0f} vx gap"
                                                               if abs(k - last_ok) > 1 else "")
        last_ok = k
    return last_ok, "end of range"


a, why_a = walk(-1)
b, why_b = walk(+1)
keep = P0[idx[a]:idx[b] + 1]
# keep parts outside the analysed range attached to the kept stretch
if a == 0:
    keep = np.concatenate([A[A[:, ax] < args.lo], keep])
if b == len(P) - 1:
    keep = np.concatenate([keep, A[A[:, ax] > args.hi]])
keep[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(keep[:, :3], axis=0), axis=1))])
np.save(args.dst, keep.astype(np.float32))
arc2 = keep[:, 4]
ctrl = np.unique(np.concatenate([np.searchsorted(arc2, np.arange(0, arc2[-1], 60.0)), [len(arc2) - 1]]))
line = np.unique(np.concatenate([np.searchsorted(arc2, np.arange(0, arc2[-1], 4.0)), ctrl]))
Path(args.dst[:-9] + ".json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                                    "control_points": keep[ctrl, :3].round(3).tolist(),
                                                    "line_points": keep[line, :3].round(3).tolist()}, indent=1))
c = "xyz"[ax]
print(f"{name}: {c} {A[:, ax].min():.0f}..{A[:, ax].max():.0f} -> {keep[:, ax].min():.0f}..{keep[:, ax].max():.0f}"
      f"  (low side: {why_a}; high side: {why_b})")
