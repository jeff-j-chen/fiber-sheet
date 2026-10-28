"""Crossing-verified same-sheet coverage of each horizontal fibre.

For every horizontal, crossings with the verticals are ordered by x; a crossing is good when its offset is
within half a layer of the canonical ply offset. The verified span is the longest run of consecutive good
crossings (x of first .. x of last), so a fibre only counts where verticals confirm it on both sides.
Usage: python fiber_coverage.py <fiber_dir> [--ply 9 --spacing 14] [--zat 4400]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--spacing", type=float, default=14.0)
ap.add_argument("--zat", type=float, default=4400.0, help="report each fibre's z at this x")
args = ap.parse_args()
fdir = Path(args.fibers)

rows = [r for r in csv.DictReader(open(fdir / "crossings.csv")) if r["verdict"] != "no overlap"]
by_h = {}
for r in rows:
    at = [float(v) for v in r["at"].strip("[]").split()]
    good = abs(float(r["normal_offset_vx"]) - args.ply) < args.spacing / 2
    by_h.setdefault(r["horizontal"], []).append((at[0], good, float(r["normal_offset_vx"]), r["vertical"]))

out = []
for h, cs in by_h.items():
    cs.sort()
    best, run = (0.0, None, None), []
    for c in cs + [(None, False, 0, "")]:
        if c[1]:
            run.append(c)
            continue
        if run:
            span = run[-1][0] - run[0][0]
            if span > best[0] or best[1] is None:
                best = (span, run[0][0], run[-1][0])
        run = []
    P = np.load(fdir / f"{h}.path.npy")[:, :3]
    o = np.argsort(P[:, 0])
    z = np.interp(args.zat, P[o, 0], P[o, 2]) if P[:, 0].min() <= args.zat <= P[:, 0].max() else float("nan")
    marks = " ".join(f"{'+' if g else 'x'}{v.split('_')[1]}:{off:+.0f}" for _, g, off, v in cs)
    out.append((z, h, best, P[:, 0].min(), P[:, 0].max(), marks))

for z, h, (span, a, b), x0, x1, marks in sorted(out, key=lambda t: -t[0] if np.isfinite(t[0]) else 0):
    ver = f"verified x {a:5.0f}..{b:5.0f} ({span:4.0f})" if a is not None else "verified -"
    print(f"{h:20s} z@{args.zat:.0f} {z:7.0f}  trace x {x0:5.0f}..{x1:5.0f}  {ver}  | {marks}")
