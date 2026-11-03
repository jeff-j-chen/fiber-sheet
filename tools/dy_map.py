"""Map of y differences between two tifxyz surfaces over (x, z): dy = B - A at A's grid points.

Usage: python dy_map.py <A> <B> <out.png> [--paths DIR] [--fibers dir]
Prints a coarse table (x 200 x z 200 cells, median dy). Image: blue = B smaller y, red = B larger, +-30 vx.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("out")
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--fibers")
args = ap.parse_args()


def load(name):
    seg = Path(args.paths) / name
    X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    ok = (X != -1) & (Z > 0)
    return X[ok], Y[ok], Z[ok]


ax, ay, az = load(args.a)
bx, by, bz = load(args.b)
f = LinearNDInterpolator(np.c_[bx, bz], by)
dy = f(np.c_[ax, az]) - ay
x0, z0 = ax.min(), az.min()
W, H = int((ax.max() - x0) / 5) + 1, int((az.max() - z0) / 5) + 1
img = np.zeros((H, W, 3), np.uint8)
m = np.isfinite(dy)
c = np.clip(dy[m] / 30, -1, 1)
ix, iz = ((ax[m] - x0) / 5).astype(int), ((az[m] - z0) / 5).astype(int)
img[iz, ix, 2] = (np.maximum(c, 0) * 255).astype(np.uint8)
img[iz, ix, 0] = (np.maximum(-c, 0) * 255).astype(np.uint8)
img[iz, ix, 1] = ((1 - np.abs(c)) * 120).astype(np.uint8)
img = cv2.dilate(img, np.ones((3, 3), np.uint8))
if args.fibers:
    for p in Path(args.fibers).glob("*.path.npy"):
        a = np.load(p)[:, :3]
        for q in a[::10]:
            cv2.circle(img, (int((q[0] - x0) / 5), int((q[2] - z0) / 5)), 0, (255, 255, 255), -1)
for x in range(int(x0 // 500 + 1) * 500, int(ax.max()), 500):
    cv2.putText(img, str(x), (int((x - x0) / 5), H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
for z in range(int(z0 // 500 + 1) * 500, int(az.max()), 500):
    cv2.putText(img, str(z), (2, int((z - z0) / 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
cv2.imwrite(args.out, img)
print(f"dy = {args.b} - {args.a}: median {np.nanmedian(dy):+.1f}, |dy|>7 on {np.mean(np.abs(dy[m]) > 7) * 100:.0f}%")
gx = np.arange(int(x0 // 200) * 200, ax.max() + 1, 200)
gz = np.arange(int(z0 // 200) * 200, az.max() + 1, 200)
print("z\\x    " + " ".join(f"{x:5.0f}" for x in gx))
for z in gz:
    row = []
    for x in gx:
        s = m & (ax >= x) & (ax < x + 200) & (az >= z) & (az < z + 200)
        row.append(f"{np.median(dy[s]):+5.0f}" if s.sum() > 3 else "    .")
    print(f"{z:6.0f} " + " ".join(row))
