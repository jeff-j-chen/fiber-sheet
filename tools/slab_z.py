"""Fixed-z slab in the (x, y) plane: CT (top) and max horizontal-fibre probability (bottom) with fibres overlaid.

Usage: python slab_z.py <fiber_dir> <z0> <xlo:xhi> <ylo:yhi> <out.png> [--half 4] [--scale 1]
Horizontals within --half of z0 are drawn as dots, verticals passing z0 as labelled crosses.
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
ap.add_argument("z0", type=int)
ap.add_argument("xr")
ap.add_argument("yr")
ap.add_argument("out")
ap.add_argument("--half", type=int, default=4)
ap.add_argument("--scale", type=float, default=1.0)
args = ap.parse_args()
x0, x1 = map(int, args.xr.split(":"))
y0, y1 = map(int, args.yr.split(":"))
za, zb = args.z0 - args.half, args.z0 + args.half + 1

pr = Predictor()
for kz in range(za // CORE, (zb - 1) // CORE + 1):
    for ky in range(y0 // CORE, (y1 - 1) // CORE + 1):
        for kx in range(x0 // CORE, (x1 - 1) // CORE + 1):
            pr.ensure((kz, ky, kx))
hz = pr.root["hz"][za:zb, y0:y1, x0:x1].astype(np.float32).max(0) / 255
vt = pr.root["vt"][za:zb, y0:y1, x0:x1].astype(np.float32).max(0) / 255
ct = pr.ct[args.z0 - 1:args.z0 + 2, y0:y1, x0:x1].astype(np.float32).mean(0)
lo, hi = np.percentile(ct, [1, 99])
ct = np.clip((ct - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
top = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
bot = np.zeros_like(top)
bot[..., 2] = (hz * 255).astype(np.uint8)          # red = horizontal fibres
bot[..., 1] = (vt * 255).astype(np.uint8)          # green = vertical fibres
s = args.scale
top = cv2.resize(top, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
bot = cv2.resize(bot, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
img = np.concatenate([top, np.full((4, top.shape[1], 3), 255, np.uint8), bot])
off = top.shape[0] + 4
rng = np.random.default_rng(3)
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    n = p.name[:-9]
    a = np.load(p)[:, :3]
    col = tuple(int(c) for c in rng.integers(100, 256, 3))
    if n[0] == "h":
        m = (np.abs(a[:, 2] - args.z0) <= args.half) & (a[:, 0] >= x0) & (a[:, 0] < x1) & (a[:, 1] >= y0) & (a[:, 1] < y1)
        if m.sum() < 3:
            continue
        for q in a[m][::4]:
            for o in (0, off):
                cv2.circle(img, (int((q[0] - x0) * s), int((q[1] - y0) * s) + o), 1, col, -1)
        q = a[m][len(a[m]) // 2]
        cv2.putText(img, n[:6], (int((q[0] - x0) * s), int((q[1] - y0) * s) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    else:
        i = int(np.argmin(np.abs(a[:, 2] - args.z0)))
        q = a[i]
        if abs(q[2] - args.z0) > 3 or not (x0 <= q[0] < x1 and y0 <= q[1] < y1):
            continue
        pt = (int((q[0] - x0) * s), int((q[1] - y0) * s))
        cv2.drawMarker(img, pt, col, cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.putText(img, n[2:6], (pt[0] - 12, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
for x in range((x0 // 100 + 1) * 100, x1, 100):
    cv2.putText(img, str(x), (int((x - x0) * s) - 14, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
for y in range((y0 // 50 + 1) * 50, y1, 50):
    cv2.putText(img, str(y), (2, int((y - y0) * s) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
cv2.imwrite(args.out, img)
print(f"wrote {args.out} ({img.shape[1]}x{img.shape[0]}): cols x {x0}..{x1}, rows y {y0}..{y1}, z {args.z0}")
