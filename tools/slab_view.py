"""Fixed-x slab in the (z, y) plane: CT (top) and max vertical-fibre probability (bottom) with fibres overlaid.

Usage: python slab_view.py <fiber_dir> <x0> <zlo:zhi> <ylo:yhi> <out.png> [--half 6] [--only a,b]
Verticals within --half of x0 are drawn as coloured dots, horizontals crossing x0 as labelled crosses.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import CORE, Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("x0", type=int)
ap.add_argument("zr")
ap.add_argument("yr")
ap.add_argument("out")
ap.add_argument("--half", type=int, default=6)
ap.add_argument("--only")
ap.add_argument("--fam", default="vt", choices=["vt", "hz"])
args = ap.parse_args()
z0, z1 = map(int, args.zr.split(":"))
y0, y1 = map(int, args.yr.split(":"))
xa, xb = args.x0 - args.half, args.x0 + args.half + 1

pr = Predictor()
for kz in range(z0 // CORE, (z1 - 1) // CORE + 1):
    for ky in range(y0 // CORE, (y1 - 1) // CORE + 1):
        for kx in range(xa // CORE, (xb - 1) // CORE + 1):
            pr.ensure((kz, ky, kx))
vt = pr.root[args.fam][z0:z1, y0:y1, xa:xb].astype(np.float32).max(-1) / 255     # (z, y)
ct = pr.ct[z0:z1, y0:y1, args.x0 - 1:args.x0 + 2].astype(np.float32).mean(-1)
lo, hi = np.percentile(ct, [1, 99])
ct = np.clip((ct - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
top = cv2.cvtColor(ct.T, cv2.COLOR_GRAY2BGR)                                         # rows = y, cols = z
bot = cv2.applyColorMap((vt.T * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
img = np.concatenate([top, np.full((4, top.shape[1], 3), 255, np.uint8), bot])
off = top.shape[0] + 4
rng = np.random.default_rng(1)
only = args.only.split(",") if args.only else None
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    n = p.name[:-9]
    if only and not any(n.startswith(o) for o in only):
        continue
    a = np.load(p)[:, :3]
    col = tuple(int(c) for c in rng.integers(80, 256, 3))
    if n[0] == "v":
        m = (np.abs(a[:, 0] - args.x0) <= args.half) & (a[:, 2] >= z0) & (a[:, 2] < z1) & (a[:, 1] >= y0) & (a[:, 1] < y1)
        if m.sum() < 3:
            continue
        for q in a[m][::3]:
            for o in (0, off):
                cv2.circle(img, (int(q[2] - z0), int(q[1] - y0) + o), 1, col, -1)
        q = a[m][len(a[m]) // 2]
        cv2.putText(img, n[:6], (int(q[2] - z0) + 4, int(q[1] - y0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    else:
        i = int(np.argmin(np.abs(a[:, 0] - args.x0)))
        q = a[i]
        if abs(q[0] - args.x0) > 3 or not (z0 <= q[2] < z1 and y0 <= q[1] < y1):
            continue
        for o in (0, off):
            cv2.drawMarker(img, (int(q[2] - z0), int(q[1] - y0) + o), col, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(img, n[:6], (int(q[2] - z0) + 6, int(q[1] - y0) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
for z in range((z0 // 100 + 1) * 100, z1, 100):
    cv2.putText(img, str(z), (z - z0 - 14, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
for y in range((y0 // 50 + 1) * 50, y1, 50):
    cv2.putText(img, str(y), (2, y - y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
cv2.imwrite(args.out, img)
print(f"wrote {args.out} ({img.shape[1]}x{img.shape[0]}): cols z {z0}..{z1}, rows y {y0}..{y1}, x {args.x0}+/-{args.half}")
