"""Integer layer assignment over fibre pieces that maximises agreeing crossings.

Verticals are split at --zsplit into an upper piece (fixed to layer 0) and a lower piece; horizontals are whole.
Each crossing with |frac| <= 0.3 gives an edge L_h - L_v = k. Coordinate descent from L=0, many restarts.
Usage: python fiber_lsolve.py <dir> [--zsplit 11800] [--zmin 10900] [--only regex]
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--zsplit", type=float, default=11800)
ap.add_argument("--zmin", type=float, default=10900)
ap.add_argument("--zmax", type=float, default=12600)
ap.add_argument("--only", help="regex of horizontals to use")
ap.add_argument("--p", type=float, default=9.0)
ap.add_argument("--S", type=float, default=15.0)
args = ap.parse_args()
pat = re.compile(args.only) if args.only else None
edges = []
for r in csv.DictReader(open(Path(args.dir) / "crossings.csv")):
    if r["verdict"] == "no overlap":
        continue
    z = float(r["at"].strip("[]").split()[2])
    if not args.zmin <= z <= args.zmax:
        continue
    h = r["horizontal"]
    if pat and not pat.search(h):
        continue
    L = (float(r["normal_offset_vx"]) - args.p) / args.S
    k = int(np.round(L))
    if abs(L - k) > 0.3:
        continue
    v = r["vertical"] + ("_lo" if z < args.zsplit else "_hi")
    edges.append((h, v, k))
nodes = sorted({e[0] for e in edges} | {e[1] for e in edges})
fixed = {n: 0 for n in nodes if n.endswith("_hi")}
adj = defaultdict(list)
for h, v, k in edges:
    adj[h].append((v, k, +1))
    adj[v].append((h, k, -1))


def score(L):
    return sum(L[h] - L[v] == k for h, v, k in edges)


best, bestL = -1, None
rng = np.random.default_rng(0)
for restart in range(200):
    L = {n: fixed.get(n, 0 if restart == 0 else int(rng.integers(-2, 3))) for n in nodes}
    for _ in range(30):
        changed = False
        for n in rng.permutation(nodes) if restart else nodes:
            if n in fixed:
                continue
            votes = Counter()
            for m, k, sgn in adj[n]:
                votes[L[m] + k if sgn > 0 else L[m] - k] += 1
            if votes:
                top = max(votes.values())
                cand = [c for c, n_ in votes.items() if n_ == top]
                new = L[n] if L[n] in cand else min(cand, key=abs)
                if new != L[n]:
                    L[n], changed = new, True
        if not changed:
            break
    s = score(L)
    if s > best:
        best, bestL = s, dict(L)
print(f"{best}/{len(edges)} crossings agree")
for n in nodes:
    if n in fixed:
        continue
    bad = [(m, k) for m, k, sgn in adj[n] if (bestL[n] - bestL[m] if sgn > 0 else bestL[m] - bestL[n]) != k]
    print(f"  {n:28s} L={bestL[n]:+d}  {len(adj[n]) - len(bad)}/{len(adj[n])} agree"
          + ("  disagree: " + " ".join(m[:6] for m, _ in bad) if bad else ""))
