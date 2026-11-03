"""Re-trace and extend a fibre with Viterbi steps in the fibre's own frame (no reference surface).

Horizontals lie in planes of nearly constant z, so their depth axis is taken perpendicular to the fibre
within the x-y plane and their side axis is z; this follows tight bends that a smooth surface misses.
Verticals use depth = perpendicular to the fibre within the y-z plane and side = x.

1. refine: one global Viterbi pass over the whole fibre around its smoothed path (fixes local hops/wobble).
2. march: extend each end in short Viterbi windows (straight 50-vx look-ahead, 30 vx kept per window,
   start state pinned to the current end) until --lo/--hi or until the probability stays weak.
3. a final global refine pass over the extended fibre.

Usage: fiber_march.py <fiber.path.npy> <out_dir> [--lo X] [--hi X] [--name N] [--no-extend]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402
from fiber_meta import FIBER_COORDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fiber")
ap.add_argument("out")
ap.add_argument("--name")
ap.add_argument("--lo", type=float, help="extend down to this x (h) / z (v)")
ap.add_argument("--hi", type=float, help="extend up to this x (h) / z (v)")
ap.add_argument("--no-extend", action="store_true")
ap.add_argument("--bridge", help="other fibre: join the end of this fibre to the start of that one (pinned DP) "
                "and report how strong the joining strand is")
ap.add_argument("--at", help="a:b  bridge from primary coordinate a on this fibre to b on the other")
ap.add_argument("--no-refine", action="store_true")
ap.add_argument("--crop", help="lo:hi keep only this primary-axis range of the input before anything else")
ap.add_argument("--depth", type=int, default=14)
ap.add_argument("--side", type=int, default=8)
ap.add_argument("--dcost", type=float, default=1.0)
ap.add_argument("--scost", type=float, default=0.3)
ap.add_argument("--floor", type=float, default=0.05)
ap.add_argument("--guide-sigma", type=float, default=25.0)
ap.add_argument("--look", type=float, default=50.0)
ap.add_argument("--keep", type=float, default=30.0)
ap.add_argument("--weak", type=float, default=0.25, help="stop when a kept window's mean p is below this ...")
ap.add_argument("--weak-n", type=int, default=3, help="... for this many windows in a row")
ap.add_argument("--ref", help="tifxyz whose normal (made orthogonal to the fibre) is the depth axis "
                "(default: fibre-frame axes)")
args = ap.parse_args()

src = np.load(args.fiber).astype(np.float64)
name = args.name or Path(args.fiber).name[:-len(".path.npy")]
fam = Path(args.fiber).name[0]
pa = 0 if fam == "h" else 2
if args.crop:
    a_, b_ = map(float, args.crop.split(":"))
    src = src[(src[:, pa] >= a_) & (src[:, pa] <= b_)]
if src[-1, pa] < src[0, pa]:
    src = src[::-1]

pred = Predictor()
own = pred.root["hz" if fam == "h" else "vt"]
D = np.arange(-args.depth, args.depth + 1, 1.0)
Ss = np.arange(-args.side, args.side + 1, 1.0)
nD, nS = len(D), len(Ss)
UP = np.array([0.0, -1.0, 0.0])                  # depth axis sign: towards -y
SIDE = np.array([0.0, 0.0, 1.0]) if fam == "h" else np.array([1.0, 0.0, 0.0])
ref_normal = None
if args.ref:
    import cv2
    from scipy.interpolate import RegularGridInterpolator
    _X, _Y, _Z = (cv2.imread(str(Path(args.ref) / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    _ok = (_X != -1) & (_Y != -1)
    _gs = 1.0 / json.loads((Path(args.ref) / "meta.json").read_text())["scale"][0]
    _xs, _zs = _X[_ok].min() + _gs * np.arange(_X.shape[1]), _Z[_ok].min() + _gs * np.arange(_X.shape[0])
    _f = RegularGridInterpolator((_zs, _xs), np.where(_ok, _Y, np.nan), bounds_error=False, fill_value=None)

    def ref_normal(G):
        x, z = G[:, 0], G[:, 2]
        fx = (_f(np.c_[z, x + 3]) - _f(np.c_[z, x - 3])) / 6
        fz = (_f(np.c_[z + 3, x]) - _f(np.c_[z - 3, x])) / 6
        n = np.c_[fx, -np.ones_like(fx), fz]
        n[~np.isfinite(n).all(1)] = UP
        return n / np.linalg.norm(n, axis=1, keepdims=True)


def resample(C, step):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    s = np.arange(0, arc[-1] + 1e-6, step)
    return np.stack([np.interp(s, arc, C[:, c]) for c in range(3)], 1)


def frame(G):
    T = np.gradient(gaussian_filter1d(G, 3, axis=0, mode="nearest"), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    if ref_normal is not None:
        N = ref_normal(G)
        N -= (N * T).sum(1, keepdims=True) * T
    elif fam == "h":                              # depth: perpendicular to the fibre within the x-y plane
        N = np.column_stack([-T[:, 1], T[:, 0], np.zeros(len(T))])
    else:                                         # depth: perpendicular to the fibre within the y-z plane
        N = np.column_stack([np.zeros(len(T)), -T[:, 2], T[:, 1]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    N *= np.where((N @ UP) < 0, -1.0, 1.0)[:, None]
    Sd = SIDE[None] - (T @ SIDE)[:, None] * T - (N @ SIDE)[:, None] * N
    Sd /= np.linalg.norm(Sd, axis=1, keepdims=True)
    return N, Sd


def sample(pts_xyz):
    z = pts_xyz[:, ::-1]
    lo_ = np.maximum(np.floor(z.min(0)).astype(int) - 1, 0)
    hi_ = np.minimum(np.ceil(z.max(0)).astype(int) + 2, pred.shape)
    t0, t1 = lo_ // 96, (hi_ - 1) // 96
    for tz in range(t0[0], t1[0] + 1):
        for ty in range(t0[1], t1[1] + 1):
            for tx in range(t0[2], t1[2] + 1):
                pred.ensure((tz, ty, tx))
    blk = own[tuple(slice(a, b) for a, b in zip(lo_, hi_))].astype(np.float32)
    return map_coordinates(blk, (z - lo_).T, order=1, mode="nearest") / 255.0


moves = [(dd, ds) for dd in (-1, 0, 1) for ds in (-1, 0, 1)]


def dp(G, pin_start=False, pin_end=False):
    """Viterbi around guide G (n,3, ~2 vx apart). Returns points, per-point prob."""
    N, Sd = frame(G)
    n = len(G)
    prob = np.zeros((n, nD, nS), np.float32)
    CH = 50
    for c0 in range(0, n, CH):
        c1 = min(c0 + CH, n)
        pts = G[c0:c1, None, None] + N[c0:c1, None, None] * D[None, :, None, None] + Sd[c0:c1, None, None] * Ss[None, None, :, None]
        prob[c0:c1] = sample(pts.reshape(-1, 3)).reshape(c1 - c0, nD, nS)
    emis = -np.log(prob + args.floor)
    cost = emis[0].copy()
    if pin_start:
        cost[:] = np.inf
        cost[nD // 2, nS // 2] = 0.0
    back = np.zeros((n, nD, nS), np.int8)
    for i in range(1, n):
        best = np.full((nD, nS), np.inf)
        arg = np.zeros((nD, nS), np.int8)
        for m, (dd, ds) in enumerate(moves):
            sh = np.full((nD, nS), np.inf)
            sh[max(dd, 0):nD + min(dd, 0), max(ds, 0):nS + min(ds, 0)] = \
                cost[max(-dd, 0):nD - max(dd, 0), max(-ds, 0):nS - max(ds, 0)]
            sh += args.dcost * abs(dd) + args.scost * abs(ds)
            better = sh < best
            best[better], arg[better] = sh[better], m
        cost = best + emis[i]
        back[i] = arg
    if pin_end:
        di, si = nD // 2, nS // 2
    else:
        di, si = np.unravel_index(int(np.argmin(cost)), cost.shape)
    path = np.zeros((n, 2), int)
    for i in range(n - 1, -1, -1):
        path[i] = di, si
        dd, ds = moves[back[i, di, si]]
        di, si = di - dd, si - ds
    d = gaussian_filter1d(D[path[:, 0]], 1.0)
    s = gaussian_filter1d(Ss[path[:, 1]], 1.0)
    return G + N * d[:, None] + Sd * s[:, None], prob[np.arange(n), path[:, 0], path[:, 1]]


def refine(C):
    G = resample(C, 2.0)
    G = gaussian_filter1d(G, args.guide_sigma / 2.0, axis=0, mode="nearest")
    return dp(G)


def march(C, P, sign, limit):
    """Extend the end of C (sign=+1: last point, -1: first point) towards primary-axis `limit`."""
    C, P = (C, P) if sign > 0 else (C[::-1], P[::-1])
    weak_run, added = 0, 0
    while True:
        end = C[-1]
        if (end[pa] - limit) * (1 if (limit > C[0, pa]) else -1) >= 0:
            break
        tail = resample(C[-80:], 1.0)
        tail = tail[-60:] if len(tail) >= 60 else tail
        d = tail[-1] - tail[0]
        d /= np.linalg.norm(d) + 1e-9
        ov = resample(C[-40:], 2.0)[-15:]                         # ~30 vx overlap on the current path
        ext = end[None] + d[None] * np.arange(2.0, args.look + 1e-6, 2.0)[:, None]
        G = np.vstack([ov, ext])
        F, pp = dp(G, pin_start=True)
        k0 = len(ov)
        k1 = k0 + int(args.keep / 2.0)
        newF, newp = F[k0:k1], pp[k0:k1]
        C = np.vstack([C, newF])
        P = np.concatenate([P, newp])
        added += len(newF)
        weak_run = weak_run + 1 if newp.mean() < args.weak else 0
        if weak_run >= args.weak_n:
            cut = weak_run * len(newF)
            C, P = C[:-cut], P[:-cut]
            added -= cut
            break
        if added > 6000:
            break
    return (C, P) if sign > 0 else (C[::-1], P[::-1])


if args.bridge:
    oth = np.load(args.bridge).astype(np.float64)
    if oth[-1, pa] < oth[0, pa]:
        oth = oth[::-1]
    a_, b_ = map(float, args.at.split(":"))
    fwd = b_ > a_
    A = src[:, :3] if fwd else src[::-1, :3]
    B = oth[:, :3] if fwd else oth[::-1, :3]
    ia = int(np.argmin(np.abs(A[:, pa] - a_)))
    ib = int(np.argmin(np.abs(B[:, pa] - b_)))
    pA, pB = A[ia], B[ib]
    tA = A[ia] - A[max(ia - 40, 0)]
    tB = B[min(ib + 40, len(B) - 1)] - B[ib]
    L = np.linalg.norm(pB - pA)
    tA, tB = tA / np.linalg.norm(tA) * L, tB / np.linalg.norm(tB) * L
    u = np.linspace(0, 1, max(int(L / 1.0), 10))[:, None]
    H = ((2 * u ** 3 - 3 * u ** 2 + 1) * pA + (u ** 3 - 2 * u ** 2 + u) * tA
         + (-2 * u ** 3 + 3 * u ** 2) * pB + (u ** 3 - u ** 2) * tB)
    G = resample(H, 2.0)
    F, pp = dp(G, pin_start=True, pin_end=True)
    win = 10
    sm = np.convolve(pp, np.ones(win) / win, mode="valid") if len(pp) > win else pp
    k = int(np.argmin(sm))
    off = np.linalg.norm(F - G, axis=1)
    print(f"bridge {name} @{a_:.0f} -> {Path(args.bridge).name[:-9]} @{b_:.0f}: {len(G) * 2} vx, mean p {pp.mean():.2f}, "
          f"weakest 20-vx window p {sm.min():.2f} near {'xz'[pa == 2]}={F[k + win // 2, pa]:.0f}, "
          f"max offset from the Hermite guide {off.max():.0f} vx")
    nm = args.name or f"{name}_to_{Path(args.bridge).name[:-9]}"
    J = np.vstack([A[:ia], F, B[ib + 1:]])
    if J[-1, pa] < J[0, pa]:
        J = J[::-1]
    arcJ = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(J, axis=0), axis=1))])
    s1 = np.arange(0, arcJ[-1], 1.0)
    F1 = np.stack([np.interp(s1, arcJ, J[:, c]) for c in range(3)], 1)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.out) / f"{nm}.path.npy", np.column_stack([F1, np.ones(len(s1)), s1]).astype(np.float32))
    sys.exit(0)

C0 = src[:, :3]
P0 = src[:, 3] if src.shape[1] > 3 else np.ones(len(src))
if args.no_refine:
    C, P = resample(C0, 2.0), np.interp(np.linspace(0, 1, len(resample(C0, 2.0))), np.linspace(0, 1, len(P0)), P0)
else:
    C, P = refine(C0)
if not args.no_extend:
    if args.hi is not None:
        C, P = march(C, P, +1, args.hi)
    if args.lo is not None:
        C, P = march(C, P, -1, args.lo)
    if not args.no_refine:
        C, P = refine(C)

arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
s1 = np.arange(0, arc[-1], 1.0)
F1 = np.stack([np.interp(s1, arc, C[:, c]) for c in range(3)], 1)
p1 = np.interp(s1, arc, P)
if F1[-1, pa] < F1[0, pa]:
    F1, p1 = F1[::-1], p1[::-1]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.save(out / f"{name}.path.npy", np.column_stack([F1, p1, s1]).astype(np.float32))
ctrl = np.unique(np.concatenate([np.arange(0, len(s1), 60), [len(s1) - 1]]))
line = np.unique(np.concatenate([np.arange(0, len(s1), 4), ctrl]))
(out / f"{name}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                              "control_points": F1[ctrl].round(3).tolist(),
                                              "line_points": F1[line].round(3).tolist()}, indent=1))

# report: distance to the input path, weak stretches
ax = "xz"[pa == 2]
o = np.argsort(src[:, pa])
inside = (F1[:, pa] >= src[o[0], pa]) & (F1[:, pa] <= src[o[-1], pa])
orig = np.stack([np.interp(F1[:, pa], src[o, pa], src[o, c]) for c in range(3)], 1)
dist = np.linalg.norm(orig - F1, axis=1)
msg = (f"{name}: {ax} {F1[0, pa]:.0f}..{F1[-1, pa]:.0f} (input {src[o[0], pa]:.0f}..{src[o[-1], pa]:.0f}), "
       f"mean p {p1.mean():.2f}, p<0.3 on {(p1 < 0.3).mean() * 100:.0f}%, vs input median {np.median(dist[inside]):.1f} vx")


def runs(mask, min_len):
    r, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                r.append((i, j))
            i = j + 1
        else:
            i += 1
    return r


for i, j in runs(inside & (dist > 4), 20):
    msg += f"\n    differs {ax} {F1[i, pa]:.0f}..{F1[j, pa]:.0f}: {np.median(dist[i:j + 1]):.0f} vx"
for i, j in runs(gaussian_filter1d(p1, 5) < 0.3, 30):
    msg += f"\n    weak {ax} {F1[i, pa]:.0f}..{F1[j, pa]:.0f}"
print(msg)
