"""Side audit: every fibre's signed distance to a tifxyz surface built from the fibres.

Horizontals should sit on the +normal side of the surface (about +ply/2) and verticals on the -normal side
(about -ply/2). For each fibre the signed distance is reported per bin along the fibre (x for horizontals,
z for verticals); bins on the wrong side, or more than half a layer from where they should be, are flagged.

Usage: python fiber_side_audit.py <surface_tifxyz> <fiber_dir> [--ply 9] [--spacing 14] [--bin 200]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from fiber_meta import FIBER_COORDS
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("surface")
ap.add_argument("fibers")
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--spacing", type=float, default=14.0)
ap.add_argument("--bin", type=float, default=200.0)
ap.add_argument("--max-dist", type=float, default=40.0, help="ignore samples farther than this from the surface")
ap.add_argument("--trim-ends", action="store_true",
                help="cut flagged bins at either end of a fibre (in place; flagged bins in the middle are kept)")
args = ap.parse_args()

seg = Path(args.surface)
G = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
ok = (G != -1).all(-1)
H, W = ok.shape
uu, vv = np.meshgrid(np.arange(0, W - 1, 0.25, dtype=np.float32), np.arange(0, H - 1, 0.25, dtype=np.float32))
D = np.stack([cv2.remap(G[..., c], uu, vv, cv2.INTER_LINEAR) for c in range(3)], -1)
dok = cv2.remap(ok.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
n = np.cross(np.gradient(D, axis=1), np.gradient(D, axis=0))
n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
dok &= np.isfinite(n).all(-1)
tree, pts, nv = cKDTree(D[dok]), D[dok], n[dok]

bad_total = 0
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    name = p.name[:-9]
    a = np.load(p)[:, :3].astype(np.float64)
    d, k = tree.query(a)
    s = ((a - pts[k]) * nv[k]).sum(1)
    m = d <= args.max_dist
    want = args.ply / 2 if name.startswith("h") else -args.ply / 2
    ax = 0 if name.startswith("h") else 2
    flags, bins = [], []
    for lo in np.arange(np.floor(a[m, ax].min() / args.bin) * args.bin if m.any() else 0, a[m, ax].max() if m.any() else 0, args.bin):
        b = m & (a[:, ax] >= lo) & (a[:, ax] < lo + args.bin)
        if b.sum() < 10:
            continue
        med = float(np.median(s[b]))
        bad = np.sign(med) != np.sign(want) or abs(med - want) >= args.spacing / 2
        bins.append((lo, bad))
        if bad:
            flags.append(f"{'xyz'[ax]}{lo:.0f}:{med:+.0f}")
    if args.trim_ends and flags and len(bins) > 2:
        i, j = 0, len(bins) - 1
        while i <= j and bins[i][1]:
            i += 1
        while j >= i and bins[j][1]:
            j -= 1
        if i <= j and (i > 0 or j < len(bins) - 1):
            lo_c = bins[i][0] if i > 0 else -np.inf
            hi_c = bins[j][0] + args.bin if j < len(bins) - 1 else np.inf
            full = np.load(p)
            keep = (full[:, ax] >= lo_c) & (full[:, ax] <= hi_c)
            runs = np.split(np.arange(len(full)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
            run = max((r for r in runs if keep[r[0]]), key=len)
            full = full[run].copy()
            full[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(full[:, :3], axis=0), axis=1))])
            np.save(p, full)
            arc = full[:, 4]
            ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
            line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
            p.with_name(f"{name}.json").write_text(json.dumps(
                {"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": full[ctrl, :3].round(3).tolist(),
                 "line_points": full[line, :3].round(3).tolist()}, indent=1))
            flags.append(f"-> trimmed to {'xyz'[ax]} {full[:, ax].min():.0f}..{full[:, ax].max():.0f}")
    bad_total += len(flags)
    side = np.mean(np.sign(s[m]) == np.sign(want)) * 100 if m.any() else float("nan")
    print(f"{name:22s} median {np.median(s[m]) if m.any() else float('nan'):+5.1f} vx (want {want:+.1f}), "
          f"{side:5.1f}% on the correct side" + (f"  FLAGGED: {' '.join(flags)}" if flags else ""))
print(f"{bad_total} flagged bins")
