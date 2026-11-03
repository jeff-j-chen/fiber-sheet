"""Vertical fibres traced along the sheet's canonical vertical direction (from trusted verticals), swept across x.

The guide for each seed follows the tangent field of trusted verticals (so it cannot cut across strands at
the wrong angle), sits ply behind the reference surface (vertical = surface + ply * (-n)), and is refined by
a narrow Viterbi pass plus single-strand lock. Seeds are swept along a z level every --step vx; the best
non-overlapping candidates (mean vertical-fibre probability) are kept.

Usage: python canon_trace.py <trusted_v_dir> <surface> <out_dir> --x lo:hi --z0 Z [--step 8] [--zr lo:hi]
       [--min-p 0.75] [--sep 45] [--seeds x:y:z,...]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("trusted")
ap.add_argument("surface")
ap.add_argument("out")
ap.add_argument("--x", default="")
ap.add_argument("--z0", type=float, default=12500)
ap.add_argument("--zr", default="10700:13850")
ap.add_argument("--step", type=float, default=8)
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--min-p", type=float, default=0.75)
ap.add_argument("--sep", type=float, default=45)
ap.add_argument("--seeds", default="")
ap.add_argument("--prefix", default="v_k")
ap.add_argument("--run-thr", type=float, default=0.3)
ap.add_argument("--run-win", type=int, default=80)
ap.add_argument("--join-gap", type=int, default=0, help="merge strong runs separated by at most this many samples (2 vx)")
ap.add_argument("--anchors", help="dir whose horizontals fix the layer at each crossing (v = h - ply*n)")
ap.add_argument("--aw", type=float, default=2.0)
args = ap.parse_args()
zlo, zhi = map(float, args.zr.split(":"))
tf = core.tangent_field(args.trusted, "v")
core.LATFN["v"] = core.tangent_field(args.trusted, "h") if list(Path(args.trusted).glob("h_*.path.npy")) else None
if core.LATFN["v"] is None:
    del core.LATFN["v"]

seg = Path(args.surface)
X, Y, Z = (cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64) for c in "xyz")
ok = (X != -1) & (Y != -1)
gs = 1.0 / json.loads((seg / "meta.json").read_text())["scale"][0]
xs, zs = X[ok].min() + gs * np.arange(X.shape[1]), Z[ok].min() + gs * np.arange(X.shape[0])
fy = RegularGridInterpolator((zs, xs), np.where(ok, Y, np.nan), bounds_error=False, fill_value=np.nan)


def surf_pt(x, z):
    y = fy([[z, x]])[0]
    if not np.isfinite(y):
        return None
    gx = (fy([[z, x + 3]])[0] - fy([[z, x - 3]])[0]) / 6
    gz = (fy([[z + 3, x]])[0] - fy([[z - 3, x]])[0]) / 6
    n = np.array([gx, -1.0, gz])
    n /= np.linalg.norm(n)              # points to -y (same side as crossing normals)
    return np.array([x, y, z]) - args.ply * n


def guide(x0, z0):
    pts = []
    for sgn in (1, -1):
        x, z = x0, z0
        run = []
        while zlo <= z <= zhi:
            p = surf_pt(x, z)
            if p is None or not np.isfinite(p).all():
                break
            run.append(p)
            t = tf(p[None])[0]
            if abs(t[2]) < 0.3:
                break
            x += sgn * 2.0 * t[0] / t[2]
            z += sgn * 2.0
        pts.append(run if sgn == 1 else run[::-1][:-1])
    G = np.array(pts[1] + pts[0])
    return G if len(G) > 50 else None


HS = [np.load(p)[:, :3].astype(np.float64)[::2] for p in Path(args.anchors).glob("h_*.path.npy")] if args.anchors else []


def anchor_prior(G, depth):
    if not HS:
        return None
    N, _ = core.frame(G, "v")
    D = np.arange(-depth, depth + 1, 1.0)
    pri = np.zeros((len(G), len(D)))
    gt = cKDTree(G)
    for Q in HS:
        if Q[:, 2].max() < G[:, 2].min() - 30 or Q[:, 2].min() > G[:, 2].max() + 30:
            continue
        dq, jg = gt.query(Q)
        iq = int(np.argmin(dq))
        ig = int(jg[iq])
        if dq[iq] > 30:
            continue
        tv = G[min(ig + 5, len(G) - 1)] - G[max(ig - 5, 0)]
        th = Q[min(iq + 10, len(Q) - 1)] - Q[max(iq - 10, 0)]
        th, tv = th * (np.sign(th[0]) or 1.0), tv * (np.sign(tv[2]) or 1.0)
        n = np.cross(th, tv)
        n /= np.linalg.norm(n)
        da = float((Q[iq] - args.ply * n - G[ig]) @ N[ig])
        a0, a1 = max(ig - 12, 0), min(ig + 13, len(G))
        pri[a0:a1] += args.aw * np.maximum(np.abs(D - da) - 2.0, 0)[None]
    return pri


def trace(G):
    sampler = lambda q: 0.8 * core.sample("v", q) + 0.2 * core.sample_ct(q)
    C, P = core.dp(G, "v", depth=args.depth, side=3, dcost=1.5, scost=4.0, sampler=sampler,
                   depth_prior=anchor_prior(G, args.depth))
    C, P = core.lock_strand(C, "v", sigma=150.0, depth=3, side=4, scost=6.0,
                            prior_fn=lambda g, d: anchor_prior(g, d))
    return C, P


def runs(P, thr=0.35, win=25, min_len=150):
    """All stretches whose running mean probability stays above thr (at least min_len samples)."""
    m = np.convolve(P, np.ones(win) / win, mode="same") > thr
    out, s0 = [], None
    for i, v in enumerate(np.append(m, False)):
        if v and s0 is None:
            s0 = i
        elif not v and s0 is not None:
            if i - s0 >= min_len:
                out.append((s0, i))
            s0 = None
    return out


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cands = []
if args.seeds:
    seeds = [tuple(map(float, s.split(":"))) for s in args.seeds.split(",")]
    starts = [(s[0], s[2]) for s in seeds]
else:
    lo, hi = map(float, args.x.split(":"))
    starts = [(x, args.z0) for x in np.arange(lo, hi + 0.1, args.step)]
for x0, z0 in starts:
    G = guide(x0, z0)
    if G is None:
        continue
    C0, P0 = trace(G)
    rr = runs(P0, args.run_thr, args.run_win, 60 if args.join_gap else 150)
    merged = []
    for a, b in rr:
        if merged and a - merged[-1][1] <= args.join_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    for a, b in [r for r in merged if r[1] - r[0] >= 150]:
        C, P = C0[a:b], P0[a:b]
        sh = core.shakiness(C, "v")[0]
        cands.append((P.mean(), x0, z0, C, P, sh))
        print(f"  seed x {x0:.0f}: z {C[:, 2].min():.0f}..{C[:, 2].max():.0f}, p {P.mean():.2f}, "
              f"weak {np.mean(P < 0.3) * 100:.0f}%, shake {sh:.1f}")
cands.sort(key=lambda c: -c[0] * np.sqrt(len(c[3])))
kept = []
for c in cands:
    if c[0] < args.min_p:
        continue
    if any(abs(np.median(c[3][:, 0]) - np.median(k[3][:, 0])) < args.sep and
           min(c[3][:, 2].max(), k[3][:, 2].max()) - max(c[3][:, 2].min(), k[3][:, 2].min()) > 100 for k in kept):
        continue
    kept.append(c)
for c in kept:
    C, P = c[3], c[4]
    name = f"{args.prefix}{c[1]:.0f}_{np.median(C[:, 2]):.0f}"
    core.save(out, name, C, P)
    print(f"{name}: z {C[:, 2].min():.0f}..{C[:, 2].max():.0f}, p {c[0]:.2f}, shake {c[5]:.1f}")
