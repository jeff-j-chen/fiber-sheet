"""March a horizontal-ply curve through z ("z-zipper") starting from a trusted horizontal.

The start fibre's track over [lo, hi] is moved by one z step at a time; at each step a Viterbi pass with a
narrow depth band re-snaps it to the horizontal-fibre ply, so the curve keeps the same layer by continuity
in z (sheets are continuous in z; a single step cannot reach the next layer). Every --save-every vx a
pseudo-fibre hs<z>.path.npy is written; existing horizontals are compared with the curve at their own z.

Usage: python fiber_zmarch.py <fiber_dir> <out_dir> --start h_3394 --x lo:hi --to Z [--step 10]
       [--depth 6] [--save-every 20] [--check h_a,h_b]
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
ap.add_argument("--start", required=True)
ap.add_argument("--x", required=True)
ap.add_argument("--to", type=float, required=True, help="march until the curve's median z reaches this")
ap.add_argument("--step", type=float, default=10.0)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--dcost", type=float, default=2.0)
ap.add_argument("--save-every", type=float, default=20.0)
ap.add_argument("--check", default="")
ap.add_argument("--tag", default="hs")
ap.add_argument("--zwin", type=int, default=8)
args = ap.parse_args()
fd = Path(args.fibers)
lo, hi = map(float, args.x.split(":"))


def load(prefix):
    p = next(fd.glob(prefix + "*.path.npy"))
    a = np.load(p).astype(np.float64)
    return p.name[:-9], a[np.argsort(a[:, 0])]


_, S = load(args.start)
S = S[(S[:, 0] >= lo) & (S[:, 0] <= hi), :3]
cur = core.resample(S, 2.0)
checks = [load(c) for c in args.check.split(",") if c]
sign = 1.0 if args.to > np.median(cur[:, 2]) else -1.0
nsteps = int(abs(args.to - np.median(cur[:, 2])) / args.step)
z_last_save = None
for k in range(1, nsteps + 1):
    G = cur.copy()
    G[:, 2] += sign * args.step
    G = core.resample(G, 2.0)
    # wide, cheap z band: the ply is made of discrete strands in z, so look at the nearest strands around z
    C, P = core.dp(G, "h", depth=args.depth, side=args.zwin, dcost=args.dcost, scost=0.2)
    C[:, 2] = G[:, 2]                                    # keep the curve exactly one step away in z
    C[:, :2] = gaussian_filter1d(C[:, :2], 1.5, axis=0, mode="nearest")
    cur = core.resample(C, 2.0)
    zm = float(np.median(cur[:, 2]))
    if z_last_save is None or abs(zm - z_last_save) >= args.save_every - 1e-6 or k == nsteps:
        z_last_save = zm
        core.save(args.out, f"{args.tag}{zm:.0f}", cur, np.interp(np.arange(len(cur)), np.arange(len(P)), P))
        msg = f"z {zm:6.0f}: mean p {P.mean():.2f}, weak {np.mean(P < 0.3) * 100:3.0f}%"
        for nm, A in checks:
            zA = np.interp(cur[:, 0], A[:, 0], A[:, 2], left=np.nan, right=np.nan)
            near = np.abs(zA - cur[:, 2]) < args.step
            if near.sum() > 20:
                yA = np.interp(cur[near, 0], A[:, 0], A[:, 1])
                dy = yA - cur[near, 1]
                msg += f" | {nm[:6]} dy med {np.median(dy):+5.1f} (n{near.sum()}, x {cur[near, 0].min():.0f}..{cur[near, 0].max():.0f})"
        print(msg)
