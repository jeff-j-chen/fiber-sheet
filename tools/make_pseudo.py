"""Pseudo horizontals from the hand-fixed sheet, for TPS support where the fibres give none near the edited area.

Usage: python make_pseudo.py <fiber_dir> <out_dir> --user <surface> --orig <surface> [--gap 120] [--zstep 60]
Copies <fiber_dir> to <out_dir> and adds h_s<z>_<x> fibres on the user sheet (z line every --zstep) wherever the
sheet differs from --orig by > --dy (dilated by --dilate vx) and no fibre sample is within --gap vx in x-z.
Only for building the TPS; the exported fibre set stays untouched.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surf_view import height_field  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--user", required=True)
ap.add_argument("--orig", required=True)
ap.add_argument("--gap", type=float, default=120.0)
ap.add_argument("--zstep", type=float, default=60.0)
ap.add_argument("--dy", type=float, default=3.0)
ap.add_argument("--dilate", type=float, default=250.0)
ap.add_argument("--all", action="store_true", help="support everywhere the fibres give none, not only near edits")
args = ap.parse_args()
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(args.fibers, out)

fu, fo = height_field(args.user, 6.0), height_field(args.orig, 6.0)
S = []
for p in Path(args.fibers).glob("*.path.npy"):
    S.append(np.load(p)[::4, [0, 2]])
tree = cKDTree(np.vstack(S))

xs = np.arange(1200.0, 7600.0, 8.0)
added = 0
for z in np.arange(9900.0, 14700.0, args.zstep):
    P = np.c_[xs, np.full_like(xs, z)]
    yu, yo = fu(P), fo(P)
    ok = np.isfinite(yu) & np.isfinite(yo)
    edited = ok & (np.abs(yu - yo) > args.dy)
    # dilate along x only, but test edited neighbourhood in z too
    near_edit = np.zeros(len(xs), bool)
    for dz in (-args.dilate, -args.dilate / 2, 0, args.dilate / 2, args.dilate):
        Q = np.c_[xs, np.full_like(xs, z + dz)]
        a, b = fu(Q), fo(Q)
        e = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > args.dy)
        k = int(args.dilate / 8)
        near_edit |= np.convolve(e.astype(float), np.ones(2 * k + 1), mode="same") > 0
    free = tree.query(P)[0] > args.gap
    if args.all:
        near_edit[:] = True
    sel = ok & near_edit & free
    i = 0
    while i < len(xs):
        if sel[i]:
            j = i
            while j + 1 < len(xs) and sel[j + 1]:
                j += 1
            if xs[j] - xs[i] >= 240:
                x = xs[i:j + 1]
                xyz = np.c_[x, yu[i:j + 1], np.full(len(x), z)]
                arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
                A = np.c_[xyz, np.ones(len(x)), arc]
                np.save(out / f"h_s{z:.0f}_{x[0]:.0f}.path.npy", A)
                added += 1
            i = j + 1
        else:
            i += 1
print(f"added {added} pseudo horizontals -> {out}")
