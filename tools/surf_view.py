"""Surface-aligned view: CT (and fibre probability) sampled on a sheet y=f(x,z) at a normal offset, with fibres overlaid.

Usage: python surf_view.py <surface> <out.png> --box x0:x1:z0:z1 [--fibers dir ...] [--off -8] [--scale 1]
       [--prob v|h] [--near 25] [--labels]
Columns = x, rows = z (z grows downward). Fibres within --near vx of the sheet are drawn (v: dots, h: dots), colour per fibre.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402
from fiber_vs_surface import surface_points  # noqa: E402

PATHS = Path(r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")


def height_field(surf, step=4.0):
    P = surface_points(PATHS / surf, step=step)
    return LinearNDInterpolator(P[:, [0, 2]], P[:, 1])


def normals(f, x, z, d=4.0):
    fx = (f(np.c_[x + d, z]) - f(np.c_[x - d, z])) / (2 * d)
    fz = (f(np.c_[x, z + d]) - f(np.c_[x, z - d])) / (2 * d)
    n = np.c_[fx, -np.ones_like(fx), fz]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("surface")
    ap.add_argument("out")
    ap.add_argument("--box", required=True)
    ap.add_argument("--fibers", action="append", default=[])
    ap.add_argument("--off", type=float, default=-8.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prob", choices=["v", "h"])
    ap.add_argument("--near", type=float, default=25.0)
    ap.add_argument("--labels", action="store_true")
    args = ap.parse_args()
    x0, x1, z0, z1 = map(float, args.box.split(":"))
    f = height_field(args.surface)
    xs, zs = np.arange(x0, x1, 1.0), np.arange(z0, z1, 1.0)
    XX, ZZ = np.meshgrid(xs, zs)
    x, z = XX.ravel(), ZZ.ravel()
    y = f(np.c_[x, z])
    ok = np.isfinite(y)
    n = np.zeros((len(x), 3))
    n[ok] = normals(f, x[ok], z[ok])
    P = np.c_[x, np.nan_to_num(y), z] + args.off * n
    # sample in row blocks so each block's CT slab stays small
    img = np.zeros(XX.shape, np.float32)
    rows = XX.shape[0]
    for r0 in range(0, rows, 100):
        sl = slice(r0 * XX.shape[1], min(r0 + 100, rows) * XX.shape[1])
        m = ok[sl]
        if m.any():
            v = np.zeros(m.shape)
            v[m] = core.sample_ct(P[sl][m])
            img[r0:r0 + 100] = v.reshape(-1, XX.shape[1])[: img[r0:r0 + 100].shape[0]]
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    top = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    panels = [top]
    if args.prob:
        pm = np.zeros(XX.shape, np.float32)
        for r0 in range(0, rows, 100):
            sl = slice(r0 * XX.shape[1], min(r0 + 100, rows) * XX.shape[1])
            m = ok[sl]
            if m.any():
                v = np.zeros(m.shape)
                v[m] = core.sample(args.prob, P[sl][m])
                pm[r0:r0 + 100] = v.reshape(-1, XX.shape[1])[: pm[r0:r0 + 100].shape[0]]
        b = np.zeros_like(top)
        b[..., 2 if args.prob == "h" else 1] = (np.clip(pm, 0, 1) * 255).astype(np.uint8)
        panels.append(b)
    s = args.scale
    panels = [cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) for p in panels]
    rng = np.random.default_rng(5)
    for d in args.fibers:
        for p in sorted(Path(d).glob("*.path.npy")):
            nme = p.name[:-9]
            a = np.load(p)[:, :3].astype(np.float64)
            m = (a[:, 0] >= x0) & (a[:, 0] < x1) & (a[:, 2] >= z0) & (a[:, 2] < z1)
            if m.sum() < 3:
                continue
            a = a[m]
            ys = f(np.c_[a[:, 0], a[:, 2]])
            nn = normals(f, a[:, 0], a[:, 2]) if np.isfinite(ys).all() else None
            if nn is None:
                ys = np.nan_to_num(ys, nan=1e9)
            sd = ((a[:, 1] - ys) if nn is None else (a - np.c_[a[:, 0], ys, a[:, 2]]).__mul__(nn).sum(1))
            sel = np.abs(sd) < args.near
            if sel.sum() < 3:
                continue
            col = tuple(int(c) for c in rng.integers(80, 256, 3))
            pts = a[sel]
            for q in pts[::6]:
                for pan in panels:
                    cv2.circle(pan, (int((q[0] - x0) * s), int((q[2] - z0) * s)), 2, col, -1)
            if args.labels or True:
                q = pts[len(pts) // 2]
                for pan in panels:
                    cv2.putText(pan, nme.split("_")[0] + nme.split("_")[1][:4] if nme.startswith(("v_j", "v_k", "v_L")) else nme[:11],
                                (int((q[0] - x0) * s) + 3, int((q[2] - z0) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    # axis ticks
    for pan in panels:
        for xv in range(int(np.ceil(x0 / 100) * 100), int(x1), 100):
            cv2.putText(pan, str(xv), (int((xv - x0) * s), 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        for zv in range(int(np.ceil(z0 / 100) * 100), int(z1), 100):
            cv2.putText(pan, str(zv), (0, int((zv - z0) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.imwrite(args.out, np.concatenate(panels, axis=1))
    print("wrote", args.out)
