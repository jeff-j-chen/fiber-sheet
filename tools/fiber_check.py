"""Confidence report for candidate fibres against the sheet, the fibre probability and the crossing model.

Usage: python fiber_check.py <sheet name> <cand_dir> --base <dir> [--base <dir>...] [--bin 150] [--names a,b]
       [--min-p 0.55] [--dev 6] [--write <out_dir>]
Per candidate and bin along its main axis: mean probability, median sheet deviation (target 0 for h, -9 for v), crossings
with the base fibres (good = within half a layer of the ply: |offset - 9| < 6). A bin is confident when p >= --min-p,
|deviation| <= --dev and none of its crossings is bad. The longest run of confident bins (single bad bins inside are not
bridged) is the kept span; --write saves the candidates cropped to it (with 1 bin margin dropped at run ends not verified).
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_trace import Sheet  # noqa: E402

PY = sys.executable
M = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("sheet")
ap.add_argument("cand")
ap.add_argument("--base", action="append", default=[], help="advisory fibres: crossings listed but not used for confidence")
ap.add_argument("--trusted", action="append", default=[], help="trusted fibres: crossings decide confidence")
ap.add_argument("--bin", type=float, default=150.0)
ap.add_argument("--names", default="")
ap.add_argument("--min-p", type=float, default=0.5)
ap.add_argument("--dev", type=float, default=6.0)
ap.add_argument("--write")
args = ap.parse_args()

cand = {p.name[:-9]: p for p in sorted(Path(args.cand).glob("*.path.npy"))}
if args.names:
    cand = {n: p for n, p in cand.items() if any(n.startswith(k) for k in args.names.split(","))}
tmp = Path(tempfile.mkdtemp(prefix="fchk_"))
trusted_names = set()
for d in args.base + args.trusted:
    for p in Path(d).glob("*.path.npy"):
        if p.name[:-9] not in cand:
            shutil.copy(p, tmp / p.name)
            if d in args.trusted:
                trusted_names.add(p.name[:-9])
if not args.trusted:
    trusted_names = None
for n, p in cand.items():
    shutil.copy(p, tmp / p.name)
subprocess.run([PY, str(M / "fiber_crossings_fn.py"), str(tmp)], capture_output=True)
rows = [r for r in csv.DictReader(open(tmp / "crossings.csv")) if r["verdict"] != "no overlap"]
sheet = Sheet(args.sheet)
out = Path(args.write) if args.write else None
if out:
    out.mkdir(parents=True, exist_ok=True)

for n, p in cand.items():
    A = np.load(p).astype(np.float64)
    fam = n[0]
    ax = 0 if fam == "h" else 2
    tgt = 0.0 if fam == "h" else -9.0
    dev = sheet.signed_offset(A[:, :3]) - tgt
    edges = np.arange(np.floor(A[:, ax].min() / args.bin) * args.bin, A[:, ax].max() + args.bin, args.bin)
    X, ADV = [], []
    for r in rows:
        if r["horizontal"] != n and r["vertical"] != n:
            continue
        at = [float(v) for v in r["at"].strip("[]").split()]
        other = r["vertical"] if fam == "h" else r["horizontal"]
        if trusted_names is not None and other not in trusted_names:
            ADV.append((at[ax], other, float(r["normal_offset_vx"])))
            continue
        X.append((at[ax], other, float(r["normal_offset_vx"])))
    status, line = [], []
    for lo in edges[:-1]:
        m = (A[:, ax] >= lo) & (A[:, ax] < lo + args.bin)
        if m.sum() < 20:
            status.append(None)
            continue
        pm, dm = float(np.median(A[m, 3])), float(np.median(dev[m]))
        xs = [(o, off) for pos, o, off in X if lo <= pos < lo + args.bin]
        bad = [(o, off) for o, off in xs if abs(off - 9) >= 9]
        good = [1 for o, off in xs if abs(off - 9) < 6]
        ok = (pm >= args.min_p or (good and pm >= 0.05)) and abs(dm) <= args.dev and not bad
        status.append(ok)
        line.append(f"{lo:5.0f}:{'+' if ok else '-'} p{pm:.2f} d{dm:+.0f} x{len(xs) - len(bad)}/{len(xs)}")
    runs, s0 = [], None
    for i, s in enumerate(status + [False]):
        if s and s0 is None:
            s0 = i
        elif not s and s0 is not None:
            runs.append((s0, i))
            s0 = None
    print(f"== {n}  ({A[:, ax].min():.0f}..{A[:, ax].max():.0f}) mean p {A[:, 3].mean():.2f}, sheet dev median {np.median(dev):+.1f}, "
          f"crossings {sum(abs(o - 9) < 6 for _, _, o in X)}/{len(X)} good")
    print("   " + " | ".join(line))
    badx = [f"{o}@{pos:.0f}:{off:+.0f}{chr(33) if abs(off - 9) >= 9 else chr(63)}" for pos, o, off in sorted(X) if abs(off - 9) >= 6]
    adv = [f"{o}@{pos:.0f}:{off:+.0f}" for pos, o, off in sorted(ADV) if abs(off - 9) >= 9]
    if adv:
        print("   advisory (untrusted fibres) disagree: " + " ".join(adv))
    if badx:
        print("   bad crossings: " + " ".join(badx))
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        lo, hi = edges[a], edges[b]
        full = status.count(True)
        print(f"   confident span {lo:.0f}..{hi:.0f} ({b - a} of {len(status)} bins; {full} confident bins in total)")
        if out:
            k = (A[:, ax] >= lo) & (A[:, ax] <= hi)
            B = A[k]
            np.save(out / p.name, B.astype(np.float32))
    else:
        print("   NO confident bins")
shutil.rmtree(tmp, ignore_errors=True)




