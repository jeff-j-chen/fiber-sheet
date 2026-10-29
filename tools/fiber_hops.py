"""Hop detector: find the fewest layer changes ('hops') along fibres that explain every crossing.

Each crossing c between vertical v and horizontal h measures k_c = round((offset - p) / S), i.e. the horizontal
is k_c layers above the vertical there. Every crossing gets an integer X_c (the vertical's layer at c; the
horizontal's is X_c + k_c). Along each fibre, consecutive crossings on different layers cost one hop. The
assignment with the fewest hops (iterated conditional modes, many random restarts) places the hops; a hop is
located between the two crossings, and hops are blamed on whichever fibre the optimiser finds cheapest.
Crossings whose offset is ambiguous (more than --amb layers from an integer) are ignored.

Usage: python fiber_hops.py <fiber_dir> [--zmin 10900 --zmax 12600] [--exclude regex] [--vcost 1 --hcost 1]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--zmin", type=float, default=-1e9)
ap.add_argument("--zmax", type=float, default=1e9)
ap.add_argument("--exclude")
ap.add_argument("--p", type=float, default=9.0)
ap.add_argument("--S", type=float, default=15.0)
ap.add_argument("--amb", type=float, default=0.3)
ap.add_argument("--vcost", type=float, default=1.0, help="cost of a hop on a vertical")
ap.add_argument("--hcost", type=float, default=1.0, help="cost of a hop on a horizontal")
ap.add_argument("--restarts", type=int, default=300)
ap.add_argument("--cut", help="write fibres trimmed to their longest layer-0 run of crossings into this folder")
ap.add_argument("--min-run", type=int, default=2, help="--cut: crossings needed in the kept run")
args = ap.parse_args()
ex = re.compile(args.exclude) if args.exclude else None

C = []      # (v, h, k, x, z)
for r in csv.DictReader(open(Path(args.dir) / "crossings.csv")):
    if r["verdict"] == "no overlap" or (ex and (ex.search(r["vertical"]) or ex.search(r["horizontal"]))):
        continue
    x, _, z = (float(t) for t in r["at"].strip("[]").split())
    if not args.zmin <= z <= args.zmax:
        continue
    L = (float(r["normal_offset_vx"]) - args.p) / args.S
    k = int(np.round(L))
    if abs(L - k) <= args.amb:
        C.append((r["vertical"], r["horizontal"], k, x, z))
n = len(C)
chains = defaultdict(list)          # fibre -> crossing indices in order along it
for i, (v, h, k, x, z) in enumerate(C):
    chains[v].append(i)
    chains[h].append(i)
for f, idx in chains.items():
    idx.sort(key=lambda i: C[i][4] if f[0] == "v" else C[i][3])
cost_of = {f: (args.vcost if f[0] == "v" else args.hcost) for f in chains}
nbrs = [[] for _ in range(n)]       # (j, fibre, offset of i on that fibre, offset of j on that fibre)
for f, idx in chains.items():
    for a, b in zip(idx, idx[1:]):
        oa = 0 if f[0] == "v" else C[a][2]
        ob = 0 if f[0] == "v" else C[b][2]
        nbrs[a].append((b, f, oa, ob))
        nbrs[b].append((a, f, ob, oa))


def total(X):
    t = 0.0
    for f, idx in chains.items():
        for a, b in zip(idx, idx[1:]):
            la = X[a] + (0 if f[0] == "v" else C[a][2])
            lb = X[b] + (0 if f[0] == "v" else C[b][2])
            t += cost_of[f] * (la != lb)
    return t


rng = np.random.default_rng(0)
best, bestX = np.inf, None
for rs in range(args.restarts):
    X = np.zeros(n, int) if rs == 0 else rng.integers(-2, 3, n)
    for _ in range(50):
        changed = False
        for i in rng.permutation(n):
            cand = {X[j] + oj - oi for j, f, oi, oj in nbrs[i]} | {X[i]}
            def local(val):
                return sum(cost_of[f] * (val + oi != X[j] + oj) for j, f, oi, oj in nbrs[i])
            new = min(sorted(cand, key=lambda c: (abs(c - X[i]) > 0, abs(c))), key=local)
            if local(new) < local(X[i]):
                X[i], changed = new, True
        if not changed:
            break
    t = total(X)
    if t < best:
        best, bestX = t, X.copy()
X = bestX - int(np.round(np.median(bestX)))
print(f"{n} crossings, {len(chains)} fibres; fewest hops: {best:g}")
hops = []
for f, idx in sorted(chains.items()):
    seq = [X[i] + (0 if f[0] == "v" else C[i][2]) for i in idx]
    for a, b, la, lb in zip(idx, idx[1:], seq, seq[1:]):
        if la != lb:
            ax = 4 if f[0] == "v" else 3
            other_a = C[a][1] if f[0] == "v" else C[a][0]
            other_b = C[b][1] if f[0] == "v" else C[b][0]
            hops.append(f)
            print(f"  HOP {f:24s} {lb - la:+d} layer(s) between {'xz'[ax == 4]}={C[a][ax]:.0f} ({other_a[:6]}) "
                  f"and {C[b][ax]:.0f} ({other_b[:6]})")
print("\nlayer of each fibre along its crossings (fibres without hops: single value)")
for f, idx in sorted(chains.items()):
    seq = [int(X[i] + (0 if f[0] == "v" else C[i][2])) for i in idx]
    runs, prev = [], None
    for s in seq:
        if s != prev:
            runs.append([s, 1])
            prev = s
        else:
            runs[-1][1] += 1
    print(f"  {f:24s} " + " -> ".join(f"L{s:+d}x{c}" for s, c in runs))

if args.cut:
    import json
    import shutil

    from fiber_meta import FIBER_COORDS
    out = Path(args.cut)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.dir).glob("*.path.npy")):
        f = p.name[:-9]
        if f not in chains:
            continue
        idx = chains[f]
        ax = 2 if f[0] == "v" else 0
        seq = [int(X[i] + (0 if f[0] == "v" else C[i][2])) for i in idx]
        best, start = None, None
        for k, s in enumerate(seq):
            if s != 0:
                start = None
                continue
            start = k if start is None else start
            if best is None or k - start > best[1] - best[0]:
                best = [start, k]
        if best is None or best[1] - best[0] + 1 < args.min_run:
            print(f"  drop {f}: no run of {args.min_run} layer-0 crossings")
            continue
        pos = [C[i][4 if ax == 2 else 3] for i in idx]
        lo = -np.inf if best[0] == 0 else (pos[best[0] - 1] + pos[best[0]]) / 2
        hi = np.inf if best[1] == len(pos) - 1 else (pos[best[1]] + pos[best[1] + 1]) / 2
        a = np.load(p)
        keep = a[(a[:, ax] >= lo) & (a[:, ax] <= hi)]
        keep[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(keep[:, :3], axis=0), axis=1))])
        np.save(out / p.name, keep)
        arc = keep[:, 4]
        ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
        line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
        (out / f"{f}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                                  "control_points": keep[ctrl, :3].round(3).tolist(),
                                                  "line_points": keep[line, :3].round(3).tolist()}, indent=1))
    print(f"wrote trimmed fibres to {out}")
