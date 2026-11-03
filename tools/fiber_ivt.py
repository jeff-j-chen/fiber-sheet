"""Bounded z-zipper ("IVT" trace): march the sheet between two trusted horizontals, from each side.

Over [lo, hi] the bottom and top trusted horizontals give z tracks zb(x), zt(x). A curve starts on the bottom
fibre and is moved in small steps to track zb + w (zt - zb), w = 0..1; at each step a Viterbi pass with a
narrow depth band (fibre probability + CT) re-snaps it to the sheet, so it keeps its layer by continuity.
The same is done downwards from the top fibre. If the sheet is continuous and the trace keeps its layer, the
upward curve lands on the top fibre and the two curves agree at every w; disagreement shows where they part.

Usage: python fiber_ivt.py <fiber_dir> <out_dir> --bottom h_a --top h_b --x lo:hi [--dz 4] [--depth 5]
       [--ct 0.5] [--save 0.1,0.2,...] [--tag r1]
Writes <tag>_up_w<w>.path.npy and <tag>_dn_w<w>.path.npy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--bottom", required=True)
ap.add_argument("--top", required=True)
ap.add_argument("--x", required=True)
ap.add_argument("--dz", type=float, default=4.0)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--dcost", type=float, default=2.0)
ap.add_argument("--ct", type=float, default=0.5)
ap.add_argument("--zwin", type=int, default=6)
ap.add_argument("--save", default="")
ap.add_argument("--tag", default="ivt")
ap.add_argument("--only", choices=["up", "dn", "both"], default="both")
ap.add_argument("--wmax", type=float, default=1.0, help="stop the upward march at this w (top fibre optional)")
args = ap.parse_args()
fd = Path(args.fibers)
lo, hi = map(float, args.x.split(":"))


def load(prefix):
    p = next(fd.glob(prefix + "*.path.npy"))
    a = np.load(p).astype(np.float64)
    return a[np.argsort(a[:, 0])]


def sampler(pts):
    v = core.sample("h", pts)
    return (1 - args.ct) * v + args.ct * core.sample_ct(pts) if args.ct > 0 else v


B, T = load(args.bottom), load(args.top)
xg = np.arange(lo, hi + 0.1, 1.0)
zb = gaussian_filter1d(np.interp(xg, B[:, 0], B[:, 2]), 20)
zt = gaussian_filter1d(np.interp(xg, T[:, 0], T[:, 2]), 20)
span = float(np.median(zt - zb))
n = max(int(abs(span) * args.wmax / args.dz), 1)
saves = sorted(float(s) for s in args.save.split(",") if s)


def track(x, w):
    return np.interp(x, xg, zb) + w * (np.interp(x, xg, zt) - np.interp(x, xg, zb))


def start(F):
    m = (F[:, 0] >= lo) & (F[:, 0] <= hi)
    return core.resample(F[m, :3], 2.0)


def march(cur, ws):
    out, ps = {}, []
    si = 0
    for w in ws[1:]:
        G = cur.copy()
        G[:, 2] = track(G[:, 0], w)
        C, P = core.dp(G, "h", depth=args.depth, side=args.zwin, dcost=args.dcost, scost=0.2, sampler=sampler)
        C[:, 2] = G[:, 2]
        C[:, :2] = gaussian_filter1d(C[:, :2], 1.5, axis=0, mode="nearest")
        cur = core.resample(C, 2.0)
        ps.append(P.mean())
        for s in saves:
            if s not in out and abs(w - s) <= 0.5 / n + 1e-9:
                out[s] = (cur.copy(), P)
    return cur, out, ps


def compare(C, F):
    """median / 90% |y| distance of curve C from fibre F at the same x (fibre's own z ignored)."""
    m = (C[:, 0] >= F[0, 0]) & (C[:, 0] <= F[-1, 0])
    d = np.abs(C[m, 1] - np.interp(C[m, 0], F[:, 0], F[:, 1]))
    return np.median(d), np.percentile(d, 90), C[m, 0][np.argmax(d)] if len(d) else 0


ws_up = np.linspace(0, args.wmax, n + 1)
res = {}
if args.only in ("up", "both"):
    endu, saved_u, pu = march(start(B), ws_up)
    for w, (C, P) in saved_u.items():
        core.save(args.out, f"h_{args.tag}up{int(round(w * 100)):03d}", C, np.interp(np.arange(len(C)), np.arange(len(P)), P))
    res["up"] = saved_u
    print(f"up: {n} steps, mean p per step min {min(pu):.2f} median {np.median(pu):.2f}")
    if args.wmax >= 1:
        md, p90, xw = compare(endu, T)
        print(f"   up end vs top fibre: median |dy| {md:.1f}, 90% {p90:.1f} vx (worst at x {xw:.0f})")
if args.only in ("dn", "both"):
    endd, saved_d, pd = march(start(T), ws_up[::-1])
    for w, (C, P) in saved_d.items():
        core.save(args.out, f"h_{args.tag}dn{int(round(w * 100)):03d}", C, np.interp(np.arange(len(C)), np.arange(len(P)), P))
    res["dn"] = saved_d
    print(f"down: mean p per step min {min(pd):.2f} median {np.median(pd):.2f}")
    md, p90, xw = compare(endd, B)
    print(f"   down end vs bottom fibre: median |dy| {md:.1f}, 90% {p90:.1f} vx (worst at x {xw:.0f})")
if "up" in res and "dn" in res:
    for w in saves:
        if w in res["up"] and w in res["dn"]:
            Cu, Cd = res["up"][w][0], res["dn"][w][0]
            xs = np.arange(max(Cu[0, 0], Cd[0, 0]), min(Cu[-1, 0], Cd[-1, 0]), 50.0)
            dy = np.interp(xs, Cu[:, 0], Cu[:, 1]) - np.interp(xs, Cd[:, 0], Cd[:, 1])
            print(f"   w {w:.2f} (z~{np.median(Cu[:, 2]):.0f}): up-down dy " + " ".join(f"{x:.0f}:{d:+.0f}" for x, d in zip(xs[::2], dy[::2])))
