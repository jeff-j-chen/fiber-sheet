"""Re-trace fibre stretches through a bend so they follow a reference sheet (hand-made tifxyz) instead of cutting it.

Each fibre's points get (u, v) on the reference grid (nearest point). Inside --box the stretch is replaced: its uv
track is bridged by a cubic Hermite curve between the uv just outside the box (tangents from the fibre's own uv
track), mapped back onto the reference sheet, shifted by the fibre's own signed offset from the sheet (median
outside the box), then refined by a narrow pinned Viterbi (fibre probability + CT) in the sheet's frame and
single-strand locked. Fibres that end inside the box are extended along their uv direction (one end pinned).

Usage: python fiber_uvguide.py <fiber_dir> <ref_surface> <out_dir> --box x0:x1:z0:z1 [--names a,b]
       [--depth 5] [--ct 0.3] [--margin 30] [--extend-to 0]
       [--new h:FRAC:A:B ...]  new horizontal/vertical between fibres A and B at FRAC (0=A, 1=B) of their
                               uv separation, built the same way over the box (uv zipper)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402
from fiber_vs_surface import surface_grid, surface_points, uv_to_xyz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("--paths", default=r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")
ap.add_argument("--box", required=True)
ap.add_argument("--names", default="")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--side", type=int, default=4)
ap.add_argument("--ct", type=float, default=0.3)
ap.add_argument("--margin", type=int, default=30, help="points beyond the box also re-traced")
ap.add_argument("--min-outside", type=int, default=60)
ap.add_argument("--no-lock", action="store_true")
ap.add_argument("--new", action="append", default=[])
ap.add_argument("--extend", action="store_true", help="fibres ending inside the box continue to the box edge")
ap.add_argument("--whole", action="store_true", help="re-trace in-box parts along their own uv track (needs --off-h/--off-v)")
ap.add_argument("--through", action="append", default=[], help="fam:x,y,z new fibre through the point, zippered between its neighbours")
ap.add_argument("--off-h", type=float, help="fixed offset of horizontals from the sheet (default: per fibre)")
ap.add_argument("--off-v", type=float)
ap.add_argument("--anchors", default="", help="other-family fibres (prefixes) whose crossings fix the layer")
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--aw", type=float, default=2.0)
ap.add_argument("--atol", type=float, default=2.0)
ap.add_argument("--areach", type=int, default=12)
args = ap.parse_args()
bx = list(map(float, args.box.split(":")))
ref = Path(args.paths) / args.ref
X, Y, Z, ok = surface_grid(ref)
Su, UV = surface_points(ref, step=1.5, return_uv=True)
tree = cKDTree(Su)
# grid normals (unit, pointing to -y like core.frame)
gi = [np.gradient(np.where(ok, A, np.nan), axis=0) for A in (X, Y, Z)]
gj = [np.gradient(np.where(ok, A, np.nan), axis=1) for A in (X, Y, Z)]
NG = np.cross(np.stack(gi, -1), np.stack(gj, -1))
NG /= np.linalg.norm(NG, axis=-1, keepdims=True)
NG *= np.where(NG[..., 1:2] > 0, -1.0, 1.0)
NG = np.nan_to_num(NG)


def uv_of(P):
    d, i = tree.query(P)
    return UV[i], d


def xyz(uv):
    return np.column_stack([uv_to_xyz(A, uv[:, 0], uv[:, 1]) for A in (X, Y, Z)])


def normal(uv):
    n = np.column_stack([uv_to_xyz(NG[..., c], uv[:, 0], uv[:, 1]) for c in range(3)])
    bad = np.linalg.norm(n, axis=1) < 1e-6
    n[bad] = (0.0, -1.0, 0.0)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


QOK = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]


def on_sheet(uv):
    i = np.clip(np.floor(uv[:, 0]).astype(int), 0, QOK.shape[0] - 1)
    j = np.clip(np.floor(uv[:, 1]).astype(int), 0, QOK.shape[1] - 1)
    return QOK[i, j]


def refn(G):
    return normal(uv_of(G)[0])


def inbox(P):
    return (P[:, 0] >= bx[0]) & (P[:, 0] <= bx[1]) & (P[:, 2] >= bx[2]) & (P[:, 2] <= bx[3])


def hermite(p0, m0, p1, m1, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * p0 + (t ** 3 - 2 * t ** 2 + t) * m0
            + (-2 * t ** 3 + 3 * t ** 2) * p1 + (t ** 3 - t ** 2) * m1)


def sampler_for(fam):
    return lambda pts: (1 - args.ct) * core.sample(fam, pts) + args.ct * core.sample_ct(pts)


def signed_offset(P):
    uv, _ = uv_of(P)
    return float(np.median(((P - xyz(uv)) * normal(uv)).sum(1)))


def anchor_prior(G, fam, depth):
    """Depth cost: at each crossing with an anchor fibre the trace must sit ply from it (h - v = +ply along th x tv)."""
    if not ANCH[fam]:
        return None
    N, _ = core.frame(G, fam, refn)
    D = np.arange(-depth, depth + 1, 1.0)
    pri = np.zeros((len(G), len(D)))
    for nm, Q in ANCH[fam]:
        d2 = ((Q[:, None] - G[None]) ** 2).sum(2)
        iq, ig = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if d2[iq, ig] > 25 ** 2:
            continue
        tg = G[min(ig + 5, len(G) - 1)] - G[max(ig - 5, 0)]
        tq = Q[min(iq + 10, len(Q) - 1)] - Q[max(iq - 10, 0)]
        th, tv = (tg, tq) if fam == "h" else (tq, tg)
        th, tv = th * (np.sign(th[0]) or 1.0), tv * (np.sign(tv[2]) or 1.0)
        n = np.cross(th, tv)
        n /= np.linalg.norm(n)
        A = Q[iq] + (args.ply if fam == "h" else -args.ply) * n
        da = float((A - G[ig]) @ N[ig])
        a0, a1 = max(ig - args.areach, 0), min(ig + args.areach + 1, len(G))
        pri[a0:a1] += args.aw * np.maximum(np.abs(D - da) - args.atol, 0)[None]
    return pri


def trace(Guv, off, fam, pin_s, pin_e, Ps=None, Pe=None):
    """uv guide -> sheet points shifted by off -> pinned narrow DP -> strand lock."""
    G = xyz(Guv) + off * normal(Guv)
    G = core.resample(G, 2.0)
    w = np.clip(np.arange(len(G)) / 20.0, 0, 1)        # blend toward pins over 40 vx only
    if Ps is not None:
        trace.mis_s = float(np.linalg.norm(Ps - G[0]))
        G += (1 - w)[:, None] * (Ps - G[0])
    if Pe is not None:
        trace.mis_e = float(np.linalg.norm(Pe - G[-1]))
        G += (1 - w[::-1])[:, None] * (Pe - G[-1])
    C, P = core.dp(G, fam, depth=args.depth, side=args.side, dcost=1.5, scost=3.0, refn=refn,
                   pin_start=pin_s, pin_end=pin_e, sampler=sampler_for(fam),
                   depth_prior=anchor_prior(G, fam, args.depth))
    if not args.no_lock:
        C, P = core.lock_strand(C, fam, sigma=100.0, depth=3, side=8, scost=5.0, refn=refn,
                                pin_start=pin_s, pin_end=pin_e, prior_fn=lambda g, d: anchor_prior(g, fam, d))
    return C, P, G


fd, out = Path(args.fibers), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ANCH = {"h": [], "v": []}
for q in [q for q in args.anchors.split(",") if q]:
    for p in fd.glob(q + "*.path.npy"):
        ANCH["v" if p.name[0] == "h" else "h"].append((p.name[:-9], np.load(p)[:, :3].astype(np.float64)[::2]))
names = [n for n in args.names.split(",") if n]
for p in sorted(fd.glob("*.path.npy")):
    name = p.name[:-9]
    if names and not any(name.startswith(n) for n in names):
        continue
    A = np.load(p).astype(np.float64)
    fam = name[0]
    ax = 0 if fam == "h" else 2
    A = A[np.argsort(A[:, ax])]
    F = A[:, :3]
    m = inbox(F)
    if m.sum() < 5:
        continue
    idx = np.nonzero(m)[0]
    i0, i1 = max(idx[0] - args.margin, 0), min(idx[-1] + args.margin, len(F) - 1)
    left, right = F[:i0 + 1], F[i1:]
    has_l, has_r = len(left) > args.min_outside, len(right) > args.min_outside
    if not (has_l or has_r) and not args.whole:
        print(f"{name:22s} no trusted part outside the box, skipped")
        continue
    out_pts = np.vstack([left[:-1]] * has_l + [right[1:]] * has_r) if (has_l or has_r) else F[i0:i1 + 1]
    off = signed_offset(out_pts)
    fixed = args.off_h if fam == "h" else args.off_v
    if fixed is not None:
        off = fixed
    uvF, dF = uv_of(F)
    k = 40
    if has_l:
        u0 = uvF[i0]
        t0 = (uvF[i0] - uvF[max(i0 - k, 0)]) / max(min(k, i0), 1)
    if has_r:
        u1 = uvF[i1]
        t1 = (uvF[min(i1 + k, len(F) - 1)] - uvF[i1]) / max(min(k, len(F) - 1 - i1), 1)
    if args.whole:
        pass
    elif not has_l:
        u0 = u1 - t1 * (i1 - i0)
        t0 = t1
    if not args.whole and not has_r:
        u1 = u0 + t0 * (i1 - i0)
        t1 = t0
    n = max(i1 - i0, 10)
    if args.extend and not args.whole and not (has_l and has_r):
        lo, hi = (bx[0], bx[1]) if fam == "h" else (bx[2], bx[3])
        rate = abs(F[min(i1 + k, len(F) - 1), ax] - F[i1, ax]) / max(min(k, len(F) - 1 - i1), 1) if has_r else \
            abs(F[i0, ax] - F[max(i0 - k, 0), ax]) / max(min(k, i0), 1)
        n = max(int(((F[i1, ax] - lo) if has_r else (hi - F[i0, ax])) / max(rate, 0.2)), 10)
        if has_r:
            u0 = u1 - t1 * n
        else:
            u1 = u0 + t0 * n
    if args.whole:
        Guv = gaussian_filter1d(uvF[i0:i1 + 1], 4, axis=0, mode="nearest")
    else:
        Guv = hermite(u0, t0 * n, u1, t1 * n, n + 1)
    Guv[:, 0] = np.clip(Guv[:, 0], 0, X.shape[0] - 1.001)
    Guv[:, 1] = np.clip(Guv[:, 1], 0, X.shape[1] - 1.001)
    if args.extend and not (has_l and has_r) and not on_sheet(Guv).all():
        bad = ~on_sheet(Guv)
        Guv = Guv[np.nonzero(bad)[0].max() + 1:] if has_r else Guv[:np.nonzero(bad)[0].min()]
        if len(Guv) < 20:
            print(f"{name:22s} extension leaves the reference sheet at once, skipped")
            continue
    if args.whole:
        Guv = Guv[on_sheet(Guv)]
        if len(Guv) < 20:
            print(f"{name:22s} off the reference sheet, skipped")
            continue
    if not on_sheet(Guv).all():
        print(f"{name:22s} bridge leaves the reference sheet ({np.mean(~on_sheet(Guv)) * 100:.0f}% off), skipped")
        continue
    trace.mis_s = trace.mis_e = float("nan")
    C, P, G = trace(Guv, off, fam, has_l, has_r, F[i0] if has_l else None, F[i1] if has_r else None)
    dold = np.median(dF[i0:i1 + 1])
    dnew = np.median(tree.query(C)[0])
    C = np.vstack([left[:-1]] * has_l + [C] + [right[1:]] * has_r)
    PP = np.concatenate([A[:i0, 3]] * has_l + [P] + [A[i1 + 1:, 3]] * has_r)
    core.save(out, name, C, PP)
    print(f"{name:22s} offset {off:+5.1f}  stretch {F[i0, ax]:.0f}..{F[i1, ax]:.0f}  dist to ref {dold:5.1f} -> "
          f"{dnew:5.1f}  p {P.mean():.2f} weak {np.mean(P < 0.3) * 100:.0f}%  pin gap {trace.mis_s:.0f}/{trace.mis_e:.0f}")

for spec in args.through:
    fam, pt = spec.split(":")
    q = np.array(pt.split(","), float)
    tax = 0 if fam == "h" else 2
    uvp = uv_of(q[None])[0][0]
    cand = []
    for p in fd.glob(f"{fam}_*.path.npy"):
        Fc = np.load(p)[:, :3].astype(np.float64)
        Fc = Fc[np.argsort(Fc[:, tax])]
        if Fc[0, tax] <= q[tax] <= Fc[-1, tax]:
            at = np.array([np.interp(q[tax], Fc[:, tax], Fc[:, c]) for c in range(3)])
            cand.append((p.name[:-9], uv_of(at[None])[0][0]))
    cand.sort(key=lambda c: np.linalg.norm(c[1] - uvp))
    nb = next((c for c in cand[1:] if (uvp - cand[0][1]) @ (c[1] - cand[0][1]) > 0), None) if cand else None
    if nb is None:
        print(f"{spec}: no fibres on both sides of the point")
        continue
    d = nb[1] - cand[0][1]
    args.new.append(f"{fam}:{((uvp - cand[0][1]) @ d) / (d @ d):.3f}:{cand[0][0]}:{nb[0]}")
    print(f"{spec}: between {cand[0][0]} and {nb[0]}, frac {args.new[-1].split(':')[1]}")

for spec in args.new:
    fam, frac, a, b = spec.split(":")
    frac = float(frac)
    ax = 0 if fam == "h" else 2
    Fa = np.load(next(fd.glob(a + "*.path.npy")))[:, :3].astype(np.float64)
    Fb = np.load(next(fd.glob(b + "*.path.npy")))[:, :3].astype(np.float64)
    Fa, Fb = Fa[np.argsort(Fa[:, ax])], Fb[np.argsort(Fb[:, ax])]
    lo, hi = (bx[0], bx[1]) if fam == "h" else (bx[2], bx[3])
    s = np.arange(lo, hi + 0.1, 2.0)
    s = s[(s >= max(Fa[0, ax], Fb[0, ax])) & (s <= min(Fa[-1, ax], Fb[-1, ax]))]
    if len(s) < 20:
        print(f"{spec}: bounding fibres do not share the box span")
        continue
    ua = uv_of(np.column_stack([np.interp(s, Fa[:, ax], Fa[:, c]) for c in range(3)]))[0]
    ub = uv_of(np.column_stack([np.interp(s, Fb[:, ax], Fb[:, c]) for c in range(3)]))[0]
    Guv = gaussian_filter1d((1 - frac) * ua + frac * ub, 5, axis=0, mode="nearest")
    off = (1 - frac) * signed_offset(Fa) + frac * signed_offset(Fb)
    fixed = args.off_h if fam == "h" else args.off_v
    if fixed is not None:
        off = fixed
    C, P, G = trace(Guv, off, fam, False, False)
    mid = C[len(C) // 2]
    name = f"{fam}_{mid[0]:.0f}_{mid[1]:.0f}_{mid[2]:.0f}"
    core.save(out, name, C, P)
    print(f"{name:22s} new {frac:.2f} between {a} and {b}: offset {off:+.1f}, p {P.mean():.2f} "
          f"weak {np.mean(P < 0.3) * 100:.0f}%, dist to ref {np.median(tree.query(C)[0]):.1f}")
