"""Crop every fibre to its verified span: first..last same-sheet crossing, plus a margin.

Usage: python fiber_trim_verified.py <fiber_dir> <out_dir> [--margin 200] [--ply 9 --spacing 14] [--min-good 2]
       [--keep-names a,b]   (copied unchanged)
Fibres with fewer than --min-good good crossings are dropped (listed).
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--margin", type=float, default=200.0)
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--spacing", type=float, default=14.0)
ap.add_argument("--min-good", type=int, default=2)
ap.add_argument("--keep-names", default="")
args = ap.parse_args()
src, dst = Path(args.src), Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
good = defaultdict(list)
for r in csv.DictReader(open(src / "crossings.csv")):
    if r["verdict"] == "no overlap" or abs(float(r["normal_offset_vx"]) - args.ply) >= args.spacing / 2:
        continue
    at = [float(v) for v in r["at"].strip("[]").split()]
    good[r["horizontal"]].append(at[0])
    good[r["vertical"]].append(at[2])
keep = set(n for n in args.keep_names.split(",") if n)
for p in sorted(src.glob("*.path.npy")):
    name = p.name[:-9]
    a = np.load(p).astype(np.float64)
    if name in keep:
        shutil.copy(p, dst / p.name)
        continue
    g = good.get(name, [])
    if len(g) < args.min_good:
        print(f"{name:22s} dropped ({len(g)} good crossings)")
        continue
    ax = 0 if name[0] == "h" else 2
    lo, hi = min(g) - args.margin, max(g) + args.margin
    m = (a[:, ax] >= lo) & (a[:, ax] <= hi)
    runs = np.split(np.arange(len(a)), np.flatnonzero(np.diff(m.astype(int))) + 1)
    run = max((r for r in runs if m[r[0]]), key=len)
    b = a[run].copy()
    b[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(b[:, :3], axis=0), axis=1))])
    np.save(dst / p.name, b.astype(np.float32))
    if len(run) < len(a):
        print(f"{name:22s} {'xz'[ax == 2]} {a[:, ax].min():.0f}..{a[:, ax].max():.0f} -> {b[:, ax].min():.0f}..{b[:, ax].max():.0f}")
if (src / "layers.json").exists():
    shutil.copy(src / "layers.json", dst / "layers.json")
