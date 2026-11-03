"""Repair isolated wrong crossings by re-tracing the local stretch of one fibre with a pinned Viterbi bridge.

A crossing that is off while the fibre's crossings on both sides are fine usually means a short excursion
onto a neighbouring layer. For each such crossing, the stretch between the neighbouring good crossings is
re-traced (fiber_march.py --bridge, both ends pinned, higher depth-change cost) on whichever of the two
fibres has the isolated error; the repair is kept if the crossing becomes good and the strand stays strong.

Usage: python fiber_fix_excursions.py <fiber_dir> [--ply 9 --spacing 14] [--dcost 2.0] [--min-p 0.5]
Modifies fibres in place (originals saved as <name>.path.npy.bak); re-run fiber_crossings_fn.py afterwards.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--ply", type=float, default=9.0)
ap.add_argument("--spacing", type=float, default=14.0)
ap.add_argument("--dcost", type=float, default=2.0)
ap.add_argument("--min-p", type=float, default=0.5)
ap.add_argument("--pad", type=float, default=15.0)
args = ap.parse_args()
d = Path(args.fibers)
here = Path(__file__).resolve().parent
py = sys.executable


def good(o):
    return abs(o - args.ply) < args.spacing / 2


def load_crossings(dd):
    out = []
    for r in csv.DictReader(open(dd / "crossings.csv")):
        if r["verdict"] == "no overlap":
            continue
        at = [float(v) for v in r["at"].strip("[]").split()]
        out.append((r["horizontal"], r["vertical"], at[0], at[2], float(r["normal_offset_vx"])))
    return out


def offset_between(fa, fb):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in (fa, fb):
            shutil.copy(f, td / Path(f).name)
        subprocess.run([py, str(here / "fiber_crossings_fn.py"), str(td)], capture_output=True)
        cr = load_crossings(td)
        return cr[0][4] if cr else None


cross = load_crossings(d)
seq = {}
for k, (h, v, x, z, o) in enumerate(cross):
    seq.setdefault(h, []).append((x, k))
    seq.setdefault(v, []).append((z, k))
seq = {n: sorted(s) for n, s in seq.items()}
fixed = 0
for k, (h, v, x, z, o) in enumerate(cross):
    if good(o):
        continue
    best = None
    for me, other, pos in ((h, v, x), (v, h, z)):
        s = seq[me]
        i = [kk for _, kk in s].index(k)
        if i == 0 or i == len(s) - 1 or not good(cross[s[i - 1][1]][4]) or not good(cross[s[i + 1][1]][4]):
            continue
        a, b = s[i - 1][0] + args.pad, s[i + 1][0] - args.pad
        if b - a < 40:
            continue
        src = d / f"{me}.path.npy"
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([py, str(here / "fiber_march.py"), str(src), td, "--bridge", str(src),
                                "--at", f"{a:.0f}:{b:.0f}", "--depth", "16", "--dcost", str(args.dcost),
                                "--name", me], capture_output=True, text=True)
            m = re.search(r"mean p ([\d.]+), weakest 20-vx window p ([\d.]+)", r.stdout)
            newp = Path(td) / f"{me}.path.npy"
            if not m or not newp.exists():
                continue
            meanp, minp = float(m.group(1)), float(m.group(2))
            o2 = offset_between(newp, d / f"{other}.path.npy")
            ok = o2 is not None and good(o2) and minp >= args.min_p * 0.5 and meanp >= args.min_p
            print(f"{h} x {v}: {o:+.0f} -> re-trace {me} {a:.0f}..{b:.0f}: offset {o2 if o2 is None else round(o2)}, "
                  f"mean p {meanp:.2f}, min p {minp:.2f} {'OK' if ok else 'no'}")
            if ok and (best is None or meanp > best[0]):
                keep = Path(tempfile.gettempdir()) / f"fix_{me}.path.npy"
                shutil.copy(newp, keep)
                best = (meanp, me, keep)
    if best:
        _, me, keep = best
        bak = d / f"{me}.path.npy.bak"
        if not bak.exists():
            shutil.copy(d / f"{me}.path.npy", bak)
        shutil.copy(keep, d / f"{me}.path.npy")
        fixed += 1
print(f"repaired {fixed} crossings")
