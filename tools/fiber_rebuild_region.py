"""Rebuild horizontals through a hard bend from trusted bounding horizontals (bottom / top).

For each target (an existing horizontal, or a new one given by z):
 1. z track inside [lo, hi]: interpolated between the smoothed z tracks of bottom and top, at the target's
    relative position where it enters the stretch (straight, no lateral wandering).
 2. ply curve along that track: Viterbi on fibre probability + CT brightness with a wide depth band and
    the left end pinned to the reference surface at the entry (layer continuity from the trusted side).
    (Validated: following h3394's own track this way reproduces h3394 to 1-4 vx through region 1.)
 3. single-strand lock around the ply curve (narrow depth band, expensive sideways moves).
 4. joined to the target's own track outside the stretch; its old track beyond hi is kept only if it meets
    the rebuilt end within --join vx, else replaced by a straight-locked continuation guided by the
    surface (reported).

Usage: python fiber_rebuild_region.py <fiber_dir> <surface> <out_dir> --bottom h_a --top h_b --x lo:hi
       --targets h_c,h_d,z12985,z13170 [--entry lo+50] [--ct 0.5] [--join 8] [--extend-to X]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("surface")
ap.add_argument("out")
ap.add_argument("--bottom", required=True)
ap.add_argument("--top", required=True)
ap.add_argument("--x", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--entry", type=float, help="x where the target's relative position is measured (default lo+50)")
ap.add_argument("--ct", type=float, default=0.5)
ap.add_argument("--depth", type=int, default=35)
ap.add_argument("--join", type=float, default=8.0)
ap.add_argument("--extend-to", type=float, help="continue rebuilt fibres to this x when the old right part is dropped")
ap.add_argument("--lock-side", type=int, default=12)
ap.add_argument("--reverse", action="store_true", help="pin the right end instead (trusted side on the right)")
ap.add_argument("--both", action="store_true", help="pin both ends to the target's own points (both sides trusted)")
ap.add_argument("--vanchors", default="", help="verticals whose crossing (+ply) fixes the layer (name prefixes)")
ap.add_argument("--ply", type=float, default=9.0, help="horizontal = vertical + ply * crossing normal")
ap.add_argument("--aw", type=float, default=1.0, help="anchor cost per vx of depth error beyond --atol")
ap.add_argument("--atol", type=float, default=2.0)
ap.add_argument("--areach", type=int, default=12, help="guide points (2 vx) on each side of an anchor")
ap.add_argument("--no-pin", action="store_true", help="do not pin the start to the surface (use with --vanchors)")
args = ap.parse_args()
fd = Path(args.fibers)
core.set_lateral(fd)
lo, hi = map(float, args.x.split(":"))
entry = args.entry if args.entry is not None else (hi - 50 if args.reverse else lo + 50)

seg = Path(args.surface)
X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1) & (Y != -1)
gs = 1.0 / json.loads((seg / "meta.json").read_text())["scale"][0]
xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
f = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False, fill_value=None)


def load(prefix):
    p = next(fd.glob(prefix + "*.path.npy"))
    a = np.load(p).astype(np.float64)
    return p.name[:-9], a[np.argsort(a[:, 0])]


def sampler(pts):
    return (1 - args.ct) * core.sample("h", pts) + args.ct * core.sample_ct(pts)


VA = []
for q in [q for q in args.vanchors.split(",") if q]:
    for p in fd.glob(q + "*.path.npy"):
        VA.append((p.name[:-9], np.load(p).astype(np.float64)[:, :3]))


def anchor_prior(G, depth=None):
    """Depth cost from verticals: near each crossing the fibre must sit ply along the normal from the vertical."""
    if not VA:
        return None, []
    N, Sd = core.frame(G, "h")
    D = np.arange(-(depth or args.depth), (depth or args.depth) + 1, 1.0)
    pri = np.zeros((len(G), len(D)))
    rep = []
    for nm, V in VA:
        d2 = ((V[:, None, [0, 2]] - G[None, :, [0, 2]]) ** 2).sum(2)
        iv, ig = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if d2[iv, ig] > 15 ** 2 or ig < 2 or ig > len(G) - 3:
            continue
        th = G[min(ig + 5, len(G) - 1)] - G[max(ig - 5, 0)]
        tv = V[min(iv + 15, len(V) - 1)] - V[max(iv - 15, 0)]
        th /= np.linalg.norm(th)
        th = th * np.sign(th[0] or 1.0)
        tv = tv * np.sign(tv[2] or 1.0)
        tv = tv - (tv @ th) * th
        n = np.cross(th, tv)
        n /= np.linalg.norm(n)
        A = V[iv] + args.ply * n
        da = float((A - G[ig]) @ N[ig])
        a0, a1 = max(ig - args.areach, 0), min(ig + args.areach + 1, len(G))
        pri[a0:a1] += args.aw * np.maximum(np.abs(D - da) - args.atol, 0)[None]
        rep.append((nm.split('_')[1], ig, da))
    return pri, rep


x = np.arange(lo, hi + 0.1, 1.0)
_, B = load(args.bottom)
_, T = load(args.top)
zb = gaussian_filter1d(np.interp(x, B[:, 0], B[:, 2]), 30)
ztp = gaussian_filter1d(np.interp(x, T[:, 0], T[:, 2]), 30)
ie = int(entry - lo)
for tgt in args.targets.split(","):
    E = None
    if tgt.startswith("z"):
        z0 = float(tgt[1:])
        w = (z0 - zb[len(x) // 2]) / (ztp[len(x) // 2] - zb[len(x) // 2])
        name = None
    else:
        name, E = load(tgt)
        ze = np.interp(entry, E[:, 0], E[:, 2])
        w = (ze - zb[ie]) / (ztp[ie] - zb[ie])
    zz = zb + w * (ztp - zb)
    G = core.resample(np.column_stack([x, f(np.c_[zz, x]), zz]), 2.0)
    if args.reverse:
        G = G[::-1]
    if E is not None and E[0, 0] <= G[0, 0] <= E[-1, 0]:
        # start exactly on the target's own point where it enters the stretch
        P_s = np.array([np.interp(G[0, 0], E[:, 0], E[:, c]) for c in range(3)])
        G += np.linspace(1, 0, len(G))[:, None] * (P_s - G[0])
    pin_end = bool(args.both and E is not None and E[0, 0] <= G[-1, 0] <= E[-1, 0])
    if pin_end:
        P_e = np.array([np.interp(G[-1, 0], E[:, 0], E[:, c]) for c in range(3)])
        G += np.linspace(0, 1, len(G))[:, None] * (P_e - G[-1])
    zg = G[:, 2].copy()
    pri, arep = anchor_prior(G)
    pin_s = not args.no_pin
    C0, P0 = core.dp(G, "h", depth=args.depth, side=6, dcost=3.0, scost=0.2, sampler=sampler, pin_start=pin_s,
                     pin_end=pin_end, depth_prior=pri)
    C0[:, 2] = zg
    C1, P1 = core.lock_strand(C0, "h", sigma=100.0, depth=4, side=args.lock_side, scost=5.0, pin_start=pin_s,
                              pin_end=pin_end, prior_fn=(lambda g, d: anchor_prior(g, d)[0]) if VA else None)
    if arep:
        Ng, _ = core.frame(G, "h")
        from scipy.spatial import cKDTree
        t1 = cKDTree(C1)
        arep = [f"{nm}:{da:+.0f}/{(C0[ig] - G[ig]) @ Ng[ig]:+.0f}/{(C1[t1.query(G[ig])[1]] - G[ig]) @ Ng[ig]:+.0f}"
                for nm, ig, da in arep]
    if args.reverse:
        C1, P1 = C1[::-1], P1[::-1]
    if name is None:
        name = f"h_{x[len(x) // 2]:.0f}_{np.interp(x[len(x) // 2], C1[:, 0], C1[:, 1]):.0f}_{zz[len(x) // 2]:.0f}"
    note = "new"
    if E is not None:
        keep_side = E[E[:, 0] > lo, :3] if args.reverse else E[E[:, 0] < lo, :3]
        far = E[E[:, 0] < lo - 2, :3] if args.reverse else E[E[:, 0] > hi + 2, :3]
        if len(far):
            end = C1[0] if args.reverse else C1[-1]
            gap = float(np.linalg.norm(far[-1 if args.reverse else 0] - end))
            if gap <= args.join:
                note = f"old far part kept (gap {gap:.1f} vx)"
            else:
                note = f"old far part DROPPED (rebuilt end is {gap:.0f} vx from it)"
                far = far[:0]
        else:
            note = "no old far part"
        parts = [far, C1, keep_side] if args.reverse else [keep_side, C1, far]
        pp = [np.ones(len(parts[0])), P1, np.ones(len(parts[2]))]
        C1 = np.vstack(parts)
        P1 = np.concatenate(pp)
    core.save(args.out, name, C1, P1)
    inside = (C1[:, 0] >= lo) & (C1[:, 0] <= hi)
    zin = C1[inside, 2]
    print(f"{name:22s} w {w:.2f}: ply-curve p {P0.mean():.2f}, strand p {P1[inside].mean():.2f} "
          f"(weak {np.mean(P1[inside] < 0.3) * 100:.0f}%), z range in stretch {zin.min():.0f}..{zin.max():.0f}; {note}"
          + (f"; anchors {' '.join(arep)}" if arep else ""))
