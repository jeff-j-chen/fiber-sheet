"""Global (Viterbi) fibre tracing in a reference-surface frame.

The fibre is found as the most probable continuous path through the fibre-probability volume, inside a
band around a guide curve that lies on a reference surface. States are (depth along the surface normal,
sideways offset within the surface); a step may change each by at most one voxel, and each change costs,
so the path can only leave a layer slowly and pays for it, unlike the greedy tracer that hops when the
next layer is locally brighter. After the first pass the guide is re-centred on the smoothed result.

Usage:
  fiber_dp.py <ref_tifxyz> <out_dir> --fiber <fiber.path.npy>            re-trace an existing fibre
  fiber_dp.py <ref_tifxyz> <out_dir> --guide "x,z;x,z;..." --family h    trace a new fibre along a guide
options: --name, --lo/--hi (extend/crop the run along x for h, z for v), --png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402
from fiber_meta import FIBER_COORDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("--fiber", help="existing fibre to re-trace (its x,z path becomes the guide)")
ap.add_argument("--guide", help="'x,z;x,z;...' guide polyline (new fibre)")
ap.add_argument("--family", choices="hv")
ap.add_argument("--name")
ap.add_argument("--lo", type=float, help="run start along the primary axis (default: guide start)")
ap.add_argument("--hi", type=float, help="run end along the primary axis (default: guide end)")
ap.add_argument("--step", type=float, default=3.0, help="vx per DP step along the primary axis")
ap.add_argument("--depth", type=int, default=30, help="depth band +/- vx around the reference surface")
ap.add_argument("--side", type=int, default=24, help="sideways band +/- vx around the guide")
ap.add_argument("--passes", type=int, default=2, help="re-centre the guide on the result and re-run")
ap.add_argument("--guide-sigma", type=float, default=60.0, help="smoothing (vx) of a guide taken from a path")
ap.add_argument("--dcost", type=float, default=1.0, help="cost per vx of depth change")
ap.add_argument("--scost", type=float, default=0.3, help="cost per vx of sideways change")
ap.add_argument("--dprior", type=float, default=0.0, help="cost per step per vx of |depth| (pull to the surface)")
ap.add_argument("--sprior", type=float, default=0.0, help="cost per step per vx of |side| (pull to the guide)")
ap.add_argument("--floor", type=float, default=0.05, help="probability floor in -log(p + floor)")
ap.add_argument("--png", action="store_true", help="write a depth-section image")
args = ap.parse_args()

# ---- reference surface: y = f(x, z) on a regular grid
ref = Path(args.ref)
X, Y, Z = (cv2.imread(str(ref / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
valid = (X != -1) & (Y != -1)
gs = 1.0 / json.loads((ref / "meta.json").read_text())["scale"][0]
x0, z0 = X[valid].min(), Z[valid].min()
xs, zs = x0 + gs * np.arange(X.shape[1]), z0 + gs * np.arange(X.shape[0])
fy = RegularGridInterpolator((zs, xs), np.where(valid, Y, np.nan), bounds_error=False, fill_value=None)


def surf(x, z):
    return fy(np.column_stack([z, x]))


if args.fiber:
    src = np.load(args.fiber).astype(np.float64)
    name = args.name or Path(args.fiber).name[:-len(".path.npy")]
    fam = args.family or name[0]
else:
    src = None
    name, fam = args.name, args.family
pa = 0 if fam == "h" else 2                      # primary axis (x for horizontals, z for verticals)
sa = 2 if fam == "h" else 0


def guide_from(p_, q_, P_):
    o = np.argsort(p_)
    p_, q_ = p_[o], q_[o]
    u = np.arange(p_[0], p_[-1] + 1e-6, 1.0)
    q = gaussian_filter1d(np.interp(u, p_, q_), args.guide_sigma, mode="nearest")
    k = min(200.0, u[-1] - u[0])                  # extrapolate with the slope of the last 200 vx
    sl0 = (np.interp(u[0] + k, u, q) - q[0]) / k if k > 0 else 0.0
    sl1 = (q[-1] - np.interp(u[-1] - k, u, q)) / k if k > 0 else 0.0
    Q = np.interp(P_, u, q)
    Q = np.where(P_ < u[0], q[0] + (P_ - u[0]) * sl0, Q)
    return np.where(P_ > u[-1], q[-1] + (P_ - u[-1]) * sl1, Q)


if src is not None:
    gp, gq = src[:, pa], src[:, sa]
else:
    g = np.array([[float(v) for v in t.split(",")] for t in args.guide.split(";")])
    gp, gq = (g[:, 0], g[:, 1]) if fam == "h" else (g[:, 1], g[:, 0])
lo = args.lo if args.lo is not None else gp.min()
hi = args.hi if args.hi is not None else gp.max()
P = np.arange(lo, hi + 1e-6, args.step)
D = np.arange(-args.depth, args.depth + 1, 1.0)
Ss = np.arange(-args.side, args.side + 1, 1.0)
nD, nS, nP = len(D), len(Ss), len(P)

pred = Predictor()
own = pred.root["hz" if fam == "h" else "vt"]


def sample(arr, pts_zyx):
    lo_ = np.maximum(np.floor(pts_zyx.min(0)).astype(int) - 1, 0)
    hi_ = np.minimum(np.ceil(pts_zyx.max(0)).astype(int) + 2, pred.shape)
    blk = arr[tuple(slice(a, b) for a, b in zip(lo_, hi_))].astype(np.float32)
    return map_coordinates(blk, (pts_zyx - lo_).T, order=1, mode="nearest")


def frame(Q):
    gx, gz = (P, Q) if fam == "h" else (Q, P)
    G = np.column_stack([gx, surf(gx, gz), gz])
    h = 2.0
    fx = (surf(gx + h, gz) - surf(gx - h, gz)) / (2 * h)
    fz = (surf(gx, gz + h) - surf(gx, gz - h)) / (2 * h)
    N = np.column_stack([fx, -np.ones_like(fx), fz])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = np.gradient(G, axis=0)
    T -= (T * N).sum(1, keepdims=True) * N
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Sd = np.cross(T, N)
    Sd *= np.sign(Sd[:, sa])[:, None]             # side axis points along +z (h) / +x (v)
    return G, N, Sd


def probs(G, N, Sd):
    prob = np.zeros((nP, nD, nS), np.float32)
    CH = 40
    for c0 in range(0, nP, CH):
        c1 = min(c0 + CH, nP)
        pts = (G[c0:c1, None, None] + N[c0:c1, None, None] * D[None, :, None, None]
               + Sd[c0:c1, None, None] * Ss[None, None, :, None])
        flat = pts.reshape(-1, 3)
        t0 = np.floor(flat[:, ::-1].min(0) / 96).astype(int)
        t1 = np.floor(flat[:, ::-1].max(0) / 96).astype(int)
        for tz in range(t0[0], t1[0] + 1):
            for ty in range(t0[1], t1[1] + 1):
                for tx in range(t0[2], t1[2] + 1):
                    pred.ensure((tz, ty, tx))
        prob[c0:c1] = sample(own, flat[:, ::-1]).reshape(c1 - c0, nD, nS) / 255.0
    return prob


moves = [(dd, ds) for dd in (-1, 0, 1) for ds in (-1, 0, 1)]


def viterbi(prob):
    emis = (-np.log(prob + args.floor) + args.dprior * np.abs(D)[None, :, None]
            + args.sprior * np.abs(Ss)[None, None, :])
    cost = emis[0].copy()
    back = np.zeros((nP, nD, nS), np.int8)
    for i in range(1, nP):
        best = np.full((nD, nS), np.inf)
        arg = np.zeros((nD, nS), np.int8)
        for m, (dd, ds) in enumerate(moves):
            sh = np.full((nD, nS), np.inf)
            # state (d, s) comes from (d - dd, s - ds)
            sh[max(dd, 0):nD + min(dd, 0), max(ds, 0):nS + min(ds, 0)] = \
                cost[max(-dd, 0):nD - max(dd, 0), max(-ds, 0):nS - max(ds, 0)]
            sh += args.dcost * abs(dd) + args.scost * abs(ds)
            better = sh < best
            best[better], arg[better] = sh[better], m
        cost = best + emis[i]
        back[i] = arg
    di, si = np.unravel_index(int(np.argmin(cost)), cost.shape)
    path = np.zeros((nP, 2), int)
    for i in range(nP - 1, -1, -1):
        path[i] = di, si
        dd, ds = moves[back[i, di, si]]
        di, si = di - dd, si - ds
    return path


Q = guide_from(gp, gq, P)
for it in range(args.passes):
    G, N, Sd = frame(Q)
    prob = probs(G, N, Sd)
    path = viterbi(prob)
    pd, ps = D[path[:, 0]], Ss[path[:, 1]]
    pdS, psS = gaussian_filter1d(pd, 1.0), gaussian_filter1d(ps, 1.0)
    F = G + N * pdS[:, None] + Sd * psS[:, None]
    if it + 1 < args.passes:
        Q = guide_from(P, F[:, sa], P)
pp = prob[np.arange(nP), path[:, 0], path[:, 1]]

arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=1))])
s1 = np.arange(0, arc[-1], 1.0)
F1 = np.stack([np.interp(s1, arc, F[:, c]) for c in range(3)], 1)
p1 = np.interp(s1, arc, pp)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.save(out / f"{name}.path.npy", np.column_stack([F1, p1, s1]).astype(np.float32))
ctrl = np.unique(np.concatenate([np.arange(0, len(s1), 60), [len(s1) - 1]]))
line = np.unique(np.concatenate([np.arange(0, len(s1), 4), ctrl]))
(out / f"{name}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                              "control_points": F1[ctrl].round(3).tolist(),
                                              "line_points": F1[line].round(3).tolist()}, indent=1))


def runs_of(mask, min_len):
    out_, i = [], 0
    while i < nP:
        if mask[i]:
            j = i
            while j + 1 < nP and mask[j + 1]:
                j += 1
            if (j - i + 1) * args.step >= min_len:
                out_.append((i, j))
            i = j + 1
        else:
            i += 1
    return out_


ax = "xz"[pa == 2]
weak = pp < 0.3
msg = f"{name}: {P[0]:.0f}..{P[-1]:.0f}, mean p {pp.mean():.2f}, p<0.3 on {weak.mean() * 100:.0f}%"
dev, dorig, inside = None, None, None
if src is not None:
    o = np.argsort(src[:, pa])
    inside = (P >= src[o[0], pa]) & (P <= src[o[-1], pa])
    orig = np.stack([np.interp(P, src[o, pa], src[o, c]) for c in range(3)], 1)
    dorig = ((orig - G) * N).sum(1)
    dist = np.linalg.norm(orig - F, axis=1)
    dev = np.where(inside, dist, np.nan)
    big = inside & (dist > 4)
    msg += f"; vs original: median dist {np.nanmedian(dev):.1f}, >4 vx on {big.sum() / max(inside.sum(), 1) * 100:.0f}%"
    for i, j in runs_of(big, 15):
        dd = np.nanmedian(pdS[i:j + 1] - dorig[i:j + 1])
        msg += f"\n    differs {ax} {P[i]:.0f}..{P[j]:.0f}: dist {np.nanmedian(dev[i:j + 1]):.0f} vx (depth {dd:+.0f})"
for i, j in runs_of(weak, 30):
    msg += f"\n    weak {ax} {P[i]:.0f}..{P[j]:.0f}"
print(msg)

if args.png:
    band = np.zeros((nD, nP), np.float32)
    for i in range(nP):
        s = path[i, 1]
        band[:, i] = prob[i, :, max(s - 2, 0):s + 3].max(1)
    img = cv2.cvtColor((np.clip(band, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (nP, nD * 3), interpolation=cv2.INTER_NEAREST)
    for i in range(nP):
        img[min(int((pdS[i] + args.depth) * 3), nD * 3 - 1), i] = (0, 0, 255)
        if dorig is not None and inside[i]:
            r = int((dorig[i] + args.depth) * 3)
            if 0 <= r < nD * 3:
                img[r, i] = (255, 255, 0)
    img[args.depth * 3, ::6] = (0, 200, 0)
    lab = np.zeros((20, nP, 3), np.uint8)
    for c in range(0, nP, max(int(250 / args.step), 1)):
        cv2.putText(lab, f"{P[c]:.0f}", (c, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.imwrite(str(out / f"{name}_dp.png"), np.vstack([lab, img]))
