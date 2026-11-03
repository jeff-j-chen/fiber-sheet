"""Report lateral shakiness of fibres, and optionally straighten them.

Usage: python fiber_straight.py <fiber_dir> [--ref tifxyz] [--names a,b] [--out dir --fix]
       [--sigma 60] [--depth 4] [--side 2]
Without --fix: prints RMS / 95% of the across-fibre (in-sheet) wiggle around a sigma-smoothed path.
With --fix: re-traces each named fibre with fiber_dpcore.straighten and writes it to --out, printing
shakiness and mean probability before/after.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--ref")
ap.add_argument("--names", default="")
ap.add_argument("--out")
ap.add_argument("--fix", action="store_true")
ap.add_argument("--sigma", type=float, default=60.0)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--side", type=int, default=2)
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--search", type=float, default=0.0,
                help="also try the straightened guide shifted sideways by up to +/- this (vx); keep the strongest")
ap.add_argument("--lateral-from", help="fibre dir whose other-family tangents define the in-sheet lateral axis")
ap.add_argument("--lock", type=float, default=0.0, help="single-strand mode: guide smoothing sigma (vx), e.g. 150")
ap.add_argument("--lock-side", type=int, default=24)
ap.add_argument("--scost", type=float, default=5.0)
args = ap.parse_args()
if args.lateral_from:
    core.set_lateral(args.lateral_from)
refn = core.ref_normal_fn(args.ref) if args.ref else None
want = [n for n in args.names.split(",") if n]
rows = []
for p in sorted(Path(args.fibers).glob("*.path.npy")):
    name = p.name[:-9]
    if want and not any(name.startswith(w) for w in want):
        continue
    a = np.load(p).astype(np.float64)
    fam = name[0]
    rn = refn if fam == "v" else None
    rms, p95 = core.shakiness(a[:, :3], fam, args.sigma, rn)
    if not args.fix:
        rows.append((rms, name, p95, float(a[:, 3].mean())))
        continue
    best = None
    offs = np.arange(-args.search, args.search + 0.1, 5.0) if args.search > 0 else [0.0]
    G0 = core.lateral_smooth(a[:, :3], fam, args.sigma, rn)
    _, Sd0 = core.frame(G0, fam, rn, smooth=10)
    for off in offs:
        if args.lock:
            Cc, Pc = core.lock_strand(G0 + off * Sd0, fam, args.lock, args.depth, args.lock_side, args.scost, rn)
        else:
            Cc, Pc = core.straighten(G0 + off * Sd0, fam, args.sigma, args.iters, args.depth, args.side, rn)
        score = float(np.mean(Pc > 0.5))
        if best is None or score > best[0]:
            best = (score, off, Cc, Pc)
    _, off, C, P = best
    rms2, p952 = core.shakiness(C, fam, args.sigma, rn)
    core.save(args.out, name, C, P)
    dist = np.median(np.min(np.linalg.norm(core.resample(a[:, :3], 4.0)[:, None] - C[None, ::2], axis=2), axis=1))
    print(f"{name:22s} shake {rms:4.1f}/{p95:4.1f} -> {rms2:4.1f}/{p952:4.1f} vx, mean p {P.mean():.2f}, "
          f"weak {np.mean(P < 0.3) * 100:3.0f}%, side shift {off:+.0f}, moved median {dist:.1f} vx")
for rms, name, p95, mp in sorted(rows, reverse=True):
    print(f"{name:22s} shake rms {rms:4.1f}  95% {p95:4.1f} vx")
