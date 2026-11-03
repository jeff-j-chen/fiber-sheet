"""Trace a fibre through anchor points placed one ply away from every crossing fibre of the other family.

For a vertical at x = X (or a horizontal at z = Z), every fibre of the other family that covers X (Z) gives an
anchor: its point there moved by the ply along the crossing normal (fiber_seed2.py convention). Near each
anchor the strongest strands of the wanted family (within +/-dwin in depth, +/-search sideways) are candidates;
a Viterbi pass over the anchors picks one per anchor with little sideways jumping. The polyline through the
picks is then re-traced by fiber_march.py with a narrow depth band, so the fibre cannot leave the layer the
anchors define. Anchors with no strand are reported and skipped.

Usage: python fiber_anchor_trace.py <fiber_dir> <out_dir> --family v --at X [--ply -9] [--ref tifxyz]
       [--extend 150] [--depth 5] [--anchors a,b,...] [--name N]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--family", choices="hv", required=True)
ap.add_argument("--at", type=float, default=0.0, help="x for a new vertical, z for a new horizontal")
ap.add_argument("--ply", type=float, default=-9.0)
ap.add_argument("--dwin", type=float, default=3.0)
ap.add_argument("--search", type=float, default=30.0)
ap.add_argument("--jump", type=float, default=0.04, help="cost per vx of sideways change between anchors")
ap.add_argument("--min-p", type=float, default=0.3)
ap.add_argument("--anchors", default="", help="only these anchor fibres (name prefixes)")
ap.add_argument("--ref")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--extend", type=float, default=0.0)
ap.add_argument("--name")
ap.add_argument("--chain", action="store_true", help="pinned Viterbi between consecutive anchors (zipper)")
ap.add_argument("--chain-side", type=int, default=8)
ap.add_argument("--lateral-from")
ap.add_argument("--track", help="existing fibre whose path gives where each anchor fibre is crossed")
ap.add_argument("--track-near", type=float, default=60.0)
args = ap.parse_args()
fdir, out = Path(args.fibers), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fam = args.family
afam = "h" if fam == "v" else "v"
apa = 0 if afam == "h" else 2              # anchor fibres' primary axis = where the new fibre sits
pred = Predictor()
arr = pred.root["vt" if fam == "v" else "hz"]
pref = [p for p in args.anchors.split(",") if p]
track = None
if args.track:
    from scipy.spatial import cKDTree
    track = np.load(args.track).astype(np.float64)[:, :3]
    ttree = cKDTree(track)

anchors = []
for p in sorted(fdir.glob(f"{afam}_*.path.npy")):
    name = p.name[:-9]
    if pref and not any(name.startswith(q) for q in pref):
        continue
    a = np.load(p).astype(np.float64)[:, :3]
    a = a[np.argsort(a[:, apa])]
    at = args.at
    if track is not None:
        dist, _ = ttree.query(a)
        k = int(np.argmin(dist))
        if dist[k] > args.track_near:
            continue
        at = a[k, apa]
    if not (a[0, apa] + 20 <= at <= a[-1, apa] - 20):
        continue
    i = int(np.argmin(np.abs(a[:, apa] - at)))
    t = a[min(i + 15, len(a) - 1)] - a[max(i - 15, 0)]
    t /= np.linalg.norm(t)
    if track is not None:
        _, j = ttree.query(a[i])
        tt = track[min(j + 20, len(track) - 1)] - track[max(j - 20, 0)]
        tt /= np.linalg.norm(tt)
    if afam == "h":
        th, tv = t, (tt if track is not None else np.array([0.0, 0.0, 1.0]))
        tv = tv * np.sign(tv[2] or 1.0)
        tv = tv - (tv @ th) * th
    else:
        tv, th = t, (tt if track is not None else np.array([1.0, 0.0, 0.0]))
        th = th * np.sign(th[0] or 1.0)
        th = th - (th @ tv) * tv
    n = np.cross(th, tv)
    n /= np.linalg.norm(n)
    along = tv if fam == "v" else th
    side = th if fam == "v" else tv
    along = along - (along @ n) * n
    along /= np.linalg.norm(along)
    side = side - (side @ n) * n - (side @ along) * along
    side /= np.linalg.norm(side)
    base = a[i] + (args.ply if fam == "v" else -args.ply) * n
    dd = np.arange(-args.dwin, args.dwin + 0.1, 1.0)
    ss = np.arange(-args.search, args.search + 0.1, 1.0)
    ll = np.arange(-8, 8.1, 2.0)
    P = (base[None, None, None] + dd[:, None, None, None] * n + ss[None, :, None, None] * side
         + ll[None, None, :, None] * along)
    flat = P.reshape(-1, 3)[:, ::-1]
    lo = np.maximum(np.floor(flat.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(flat.max(0)).astype(int) + 2, pred.shape)
    for tz in range(lo[0] // 96, (hi[0] - 1) // 96 + 1):
        for ty in range(lo[1] // 96, (hi[1] - 1) // 96 + 1):
            for tx in range(lo[2] // 96, (hi[2] - 1) // 96 + 1):
                pred.ensure((tz, ty, tx))
    blk = arr[tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))].astype(np.float32) / 255
    v = map_coordinates(blk, (flat - lo).T, order=1).reshape(P.shape[:3]).mean(2)     # depth, side
    prof = v.max(0)
    pk, _ = find_peaks(np.concatenate([[0], prof, [0]]), height=args.min_p, distance=6)
    pk = pk - 1
    cands = []
    for k in sorted(pk, key=lambda k: -prof[k])[:4]:
        j = int(np.argmax(v[:, k]))
        cands.append((base + dd[j] * n + ss[k] * side, float(prof[k])))
    anchors.append((name, a[i], cands))

pa_new = 2 if fam == "v" else 0            # new fibre's primary axis
lat = 0 if fam == "v" else 2               # new fibre's sideways world axis
anchors.sort(key=lambda r: r[1][pa_new])
use = [r for r in anchors if r[2]]
for name, A, c in anchors:
    if not c:
        print(f"  anchor {name}: no strand of the wanted family at the ply (p < {args.min_p})")
if len(use) < 2:
    sys.exit("fewer than two usable anchors")

# Viterbi over anchors: one candidate each, penalising sideways jumps
cost = [np.array([-np.log(p) for _, p in use[0][2]])]
back = []
for i in range(1, len(use)):
    prev, cur = use[i - 1][2], use[i][2]
    c = np.zeros(len(cur))
    b = np.zeros(len(cur), int)
    for j, (P, p) in enumerate(cur):
        tr = [cost[-1][k] + args.jump * abs(P[lat] - Q[lat]) for k, (Q, _) in enumerate(prev)]
        b[j] = int(np.argmin(tr))
        c[j] = tr[b[j]] - np.log(p)
    cost.append(c)
    back.append(b)
sel = [int(np.argmin(cost[-1]))]
for b in reversed(back):
    sel.append(int(b[sel[-1]]))
sel = sel[::-1]
pts = np.array([use[i][2][s][0] for i, s in enumerate(sel)])
for i, s in enumerate(sel):
    P, p = use[i][2][s]
    print(f"  anchor {use[i][0]:22s} -> ({P[0]:.0f}, {P[1]:.0f}, {P[2]:.0f}) p {p:.2f}")

# polyline through the picks, densified
seg = []
for i in range(len(pts) - 1):
    m = max(int(np.linalg.norm(pts[i + 1] - pts[i])), 2)
    seg.append(pts[i] + (pts[i + 1] - pts[i]) * np.linspace(0, 1, m, endpoint=False)[:, None])
poly = np.vstack(seg + [pts[-1:]])
arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
name = args.name or f"{fam}_{pts[len(pts) // 2][0]:.0f}_{pts[len(pts) // 2][1]:.0f}_{pts[len(pts) // 2][2]:.0f}"
if args.chain:
    # zipper: every stretch between consecutive anchors is a separate Viterbi pass pinned at both anchors
    import fiber_dpcore as core
    if args.lateral_from:
        core.set_lateral(args.lateral_from)
    CC, PP = [], []
    for i in range(len(pts) - 1):
        G = core.resample(np.vstack([pts[i], pts[i + 1]]), 2.0)
        if len(G) < 3:
            continue
        G[-1] = pts[i + 1]
        Ci, Pi = core.dp(G, fam, depth=args.depth, side=args.chain_side, dcost=2.0, scost=4.0,
                         pin_start=True, pin_end=True)
        CC.append(Ci if i == 0 else Ci[1:])
        PP.append(Pi if i == 0 else Pi[1:])
    C, P = np.vstack(CC), np.concatenate(PP)
    core.save(out, name, C, P)
    print(f"{name}: chain of {len(pts) - 1} pinned stretches, mean p {P.mean():.2f}, weak {np.mean(P < 0.3) * 100:.0f}%")
    sys.exit(0)
guide = out / f"{name}_guide.path.npy"
np.save(guide, np.column_stack([poly, np.ones(len(poly)), arc]).astype(np.float32))
cmd = [sys.executable, str(Path(__file__).resolve().parent / "fiber_march.py"), str(guide), str(out), "--name", name,
       "--depth", str(args.depth), "--guide-sigma", "20"]
if args.ref:
    cmd += ["--ref", args.ref]
if args.extend > 0:
    cmd += ["--lo", str(pts[0, pa_new] - args.extend), "--hi", str(pts[-1, pa_new] + args.extend)]
else:
    cmd += ["--no-extend"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-800:])
guide.unlink()
