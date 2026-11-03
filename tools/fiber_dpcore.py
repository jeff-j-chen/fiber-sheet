"""Shared Viterbi fibre machinery: probability sampling, local frames, pinned DP and lateral straightening.

frame: horizontals use depth = perpendicular to the fibre in the x-y plane (side ~ z); verticals use the
reference-surface normal (or the y-z plane perpendicular) for depth and side = T x N.
straighten(): removes wiggles across the fibre within the sheet (side direction) while keeping depth detail,
then re-runs the DP with a tight side band, so fibres cannot zig-zag onto neighbouring physical fibres.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from afv_core import Predictor

_pred = None
LATFN = {}


def set_lateral(fiber_dir):
    """Use the other family's fibre tangents as each family's in-sheet lateral axis."""
    LATFN["h"] = tangent_field(fiber_dir, "v")
    LATFN["v"] = tangent_field(fiber_dir, "h")


def pred():
    global _pred
    if _pred is None:
        _pred = Predictor()
    return _pred


def resample(C, step):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    s = np.arange(0, arc[-1] + 1e-6, step)
    return np.stack([np.interp(s, arc, C[:, c]) for c in range(3)], 1)


def ref_normal_fn(ref):
    import cv2
    from scipy.interpolate import RegularGridInterpolator
    ref = Path(ref)
    X, Y, Z = (cv2.imread(str(ref / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
    ok = (X != -1) & (Y != -1)
    gs = 1.0 / json.loads((ref / "meta.json").read_text())["scale"][0]
    xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
    f = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False, fill_value=None)

    def fn(G):
        x, z = G[:, 0], G[:, 2]
        fx = (f(np.c_[z, x + 3]) - f(np.c_[z, x - 3])) / 6
        fz = (f(np.c_[z + 3, x]) - f(np.c_[z - 3, x])) / 6
        n = np.c_[fx, -np.ones_like(fx), fz]
        n[~np.isfinite(n).all(1)] = (0, -1, 0)
        return n / np.linalg.norm(n, axis=1, keepdims=True)
    return fn


def tangent_field(fiber_dir, fam_other, k=8):
    """Unit tangent of the nearest fibres of family fam_other (inverse-distance weighted), +x / +z oriented."""
    from scipy.spatial import cKDTree
    P, T = [], []
    for p in Path(fiber_dir).glob(f"{fam_other}_*.path.npy"):
        a = np.load(p)[:, :3].astype(np.float64)
        if len(a) < 30:
            continue
        t = np.gradient(gaussian_filter1d(a, 5, axis=0, mode="nearest"), axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        ax = 0 if fam_other == "h" else 2
        t *= np.where(t[:, ax] < 0, -1.0, 1.0)[:, None]
        P.append(a[::3])
        T.append(t[::3])
    P, T = np.concatenate(P), np.concatenate(T)
    tree = cKDTree(P)

    def fn(G):
        d, i = tree.query(G, k=k)
        w = 1.0 / (d + 5.0)
        t = (T[i] * w[..., None]).sum(1)
        return t / np.linalg.norm(t, axis=1, keepdims=True)
    return fn


def frame(G, fam, refn=None, smooth=3.0, latfn=None):
    T = np.gradient(gaussian_filter1d(G, smooth, axis=0, mode="nearest"), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    latfn = latfn or LATFN.get(fam)
    if latfn is not None:
        L = latfn(G)
        L = L - (L * T).sum(1, keepdims=True) * T
        L /= np.linalg.norm(L, axis=1, keepdims=True)
        N = np.cross(T, L)
    elif refn is not None:
        N = refn(G)
        N = N - (N * T).sum(1, keepdims=True) * T
    elif fam == "h":
        N = np.column_stack([-T[:, 1], T[:, 0], np.zeros(len(T))])
    else:
        N = np.column_stack([np.zeros(len(T)), -T[:, 2], T[:, 1]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    N *= np.where(N[:, 1] > 0, -1.0, 1.0)[:, None]
    Sd = np.cross(T, N)
    ax = 2 if fam == "h" else 0
    Sd *= np.where(Sd[:, ax] < 0, -1.0, 1.0)[:, None]
    return N, Sd


def sample(fam, pts_xyz):
    p = pred()
    arr = p.root["hz" if fam == "h" else "vt"]
    z = pts_xyz[:, ::-1]
    lo = np.maximum(np.floor(z.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(z.max(0)).astype(int) + 2, p.shape)
    for tz in range(lo[0] // 96, (hi[0] - 1) // 96 + 1):
        for ty in range(lo[1] // 96, (hi[1] - 1) // 96 + 1):
            for tx in range(lo[2] // 96, (hi[2] - 1) // 96 + 1):
                p.ensure((tz, ty, tx))
    blk = arr[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32)
    return map_coordinates(blk, (z - lo).T, order=1, mode="nearest") / 255.0


MOVES = [(dd, ds) for dd in (-1, 0, 1) for ds in (-1, 0, 1)]


def sample_ct(pts_xyz, lo_pct=None):
    """CT intensity at points, scaled to 0..1 by the 5th/99th percentile of the sampled block."""
    p = pred()
    z = pts_xyz[:, ::-1]
    lo = np.maximum(np.floor(z.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(z.max(0)).astype(int) + 2, p.shape)
    blk = p.ct[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32)
    a, b = np.percentile(blk, [5, 99])
    v = map_coordinates(blk, (z - lo).T, order=1, mode="nearest")
    return np.clip((v - a) / (b - a + 1e-6), 0, 1)


def dp(G, fam, depth=6, side=6, dcost=1.5, scost=3.0, floor=0.05, refn=None, pin_start=False, pin_end=False,
       side_prior=0.0, sampler=None, depth_prior=None):
    """Viterbi around guide G (~2 vx spacing). Returns smoothed points and per-point probability.
    depth_prior: optional (len(G), 2*depth+1) cost added per guide point and depth offset."""
    sampler = sampler or (lambda pts: sample(fam, pts))
    N, Sd = frame(G, fam, refn)
    D = np.arange(-depth, depth + 1, 1.0)
    S = np.arange(-side, side + 1, 1.0)
    nD, nS, n = len(D), len(S), len(G)
    prob = np.zeros((n, nD, nS), np.float32)
    for c0 in range(0, n, 50):
        c1 = min(c0 + 50, n)
        pts = G[c0:c1, None, None] + N[c0:c1, None, None] * D[None, :, None, None] + Sd[c0:c1, None, None] * S[None, None, :, None]
        prob[c0:c1] = sampler(pts.reshape(-1, 3)).reshape(c1 - c0, nD, nS)
    emis = -np.log(prob + floor) + side_prior * np.abs(S)[None, None, :]
    if depth_prior is not None:
        emis = emis + depth_prior[:, :, None]
    cost = emis[0].copy()
    if pin_start:
        cost[:] = np.inf
        cost[nD // 2, nS // 2] = 0.0
    back = np.zeros((n, nD, nS), np.int8)
    for i in range(1, n):
        best = np.full((nD, nS), np.inf)
        arg = np.zeros((nD, nS), np.int8)
        for m, (dd, ds) in enumerate(MOVES):
            sh = np.full((nD, nS), np.inf)
            sh[max(dd, 0):nD + min(dd, 0), max(ds, 0):nS + min(ds, 0)] = \
                cost[max(-dd, 0):nD - max(dd, 0), max(-ds, 0):nS - max(ds, 0)]
            sh += dcost * abs(dd) + scost * abs(ds)
            better = sh < best
            best[better], arg[better] = sh[better], m
        cost = best + emis[i]
        back[i] = arg
    di, si = (nD // 2, nS // 2) if pin_end else np.unravel_index(int(np.argmin(cost)), cost.shape)
    path = np.zeros((n, 2), int)
    for i in range(n - 1, -1, -1):
        path[i] = di, si
        dd, ds = MOVES[back[i, di, si]]
        di, si = di - dd, si - ds
    d = gaussian_filter1d(D[path[:, 0]], 1.0)
    s = gaussian_filter1d(S[path[:, 1]], 1.0)
    return G + N * d[:, None] + Sd * s[:, None], prob[np.arange(n), path[:, 0], path[:, 1]]


def lateral_smooth(C, fam, sigma, refn=None):
    """Remove the side (within-sheet, across-fibre) component of C's deviation from its sigma-smoothed self."""
    G0 = resample(C, 2.0)
    Gs = gaussian_filter1d(G0, sigma / 2.0, axis=0, mode="nearest")
    _, Sd = frame(Gs, fam, refn)
    d = ((G0 - Gs) * Sd).sum(1, keepdims=True)
    return G0 - d * Sd


def straighten(C, fam, sigma=60.0, iters=2, depth=4, side=2, refn=None, pin_start=False, pin_end=False):
    P = None
    for _ in range(iters):
        G = lateral_smooth(C, fam, sigma, refn)
        if pin_start:
            G[0] = C[0]
        if pin_end:
            G[-1] = C[-1]
        C, P = dp(G, fam, depth=depth, side=side, dcost=1.5, scost=4.0, refn=refn,
                  pin_start=pin_start, pin_end=pin_end)
    return C, P


def lock_strand(C, fam, sigma=150.0, depth=4, side=24, scost=5.0, refn=None, pin_start=False, pin_end=False,
                prior_fn=None):
    """Follow one physical strand: very straight guide, wide side band, expensive sideways moves.
    prior_fn(G, depth) -> optional depth prior for each guide."""
    pf = prior_fn or (lambda G, d: None)
    G = lateral_smooth(C, fam, sigma, refn)
    if pin_start:
        G[0] = C[0]
    if pin_end:
        G[-1] = C[-1]
    C1, P1 = dp(G, fam, depth=depth, side=side, dcost=1.5, scost=scost, refn=refn,
                pin_start=pin_start, pin_end=pin_end, depth_prior=pf(G, depth))
    G = lateral_smooth(C1, fam, 40.0, refn)
    if pin_start:
        G[0] = C[0]
    if pin_end:
        G[-1] = C[-1]
    return dp(G, fam, depth=depth, side=3, dcost=1.5, scost=scost, refn=refn, pin_start=pin_start, pin_end=pin_end,
              depth_prior=pf(G, depth))


def shakiness(C, fam, sigma=60.0, refn=None):
    """RMS of the side component of the deviation from the sigma-smoothed path (vx)."""
    G0 = resample(C, 2.0)
    Gs = gaussian_filter1d(G0, sigma / 2.0, axis=0, mode="nearest")
    _, Sd = frame(Gs, fam, refn)
    d = ((G0 - Gs) * Sd).sum(1)
    k = max(int(sigma), 1)
    d = d[k:-k] if len(d) > 3 * k else d
    return float(np.sqrt(np.mean(d ** 2))), float(np.percentile(np.abs(d), 95))


def save(out_dir, name, C, P):
    from fiber_meta import FIBER_COORDS
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    s1 = np.arange(0, arc[-1], 1.0)
    F = np.stack([np.interp(s1, arc, C[:, c]) for c in range(3)], 1)
    p1 = np.interp(s1, arc, P)
    ax = 0 if name[0] == "h" else 2
    if F[-1, ax] < F[0, ax]:
        F, p1 = F[::-1], p1[::-1]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.path.npy", np.column_stack([F, p1, s1]).astype(np.float32))
    ctrl = np.unique(np.concatenate([np.arange(0, len(s1), 60), [len(s1) - 1]]))
    line = np.unique(np.concatenate([np.arange(0, len(s1), 4), ctrl]))
    (out / f"{name}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                                  "control_points": F[ctrl].round(3).tolist(),
                                                  "line_points": F[line].round(3).tolist()}, indent=1))
