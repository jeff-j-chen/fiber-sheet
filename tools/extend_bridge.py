"""Wind horizontal fibres back to where they are still clean on the +x side, then extend them rightward
with verified pieces only.

1. Wind back: the first stretch past the seed where the path wobbles sideways or loses probability is cut
   (minus --backoff vx).
2. Extend: trace one way with a straightness prior. Only clean runs are kept; a disturbed stretch is
   bridged (smooth curve, prob 0) only if the clean piece after it lies where the clean fibre before it
   extrapolates to (within --bridge-normal / --bridge-side vx, i.e. no layer change).
3. Hop: when the trace stops or diverges, look further ahead along the extrapolated fibre for a confident
   horizontal voxel, trace from there, and accept it under the same alignment test.

Usage: python extend_bridge.py <src_dir> <out_dir> <segment_dir> [--names h_a,h_b] [--xmax 6600]
Copies verticals unchanged. Writes <out_dir>/<name>.json/.path.npy and extend_report.csv.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fiber_meta import FIBER_COORDS
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import CORE, Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("segment")
ap.add_argument("--names", default="")
ap.add_argument("--xmax", type=float, default=6600.0, help="stop extending past this x")
ap.add_argument("--sigma", type=float, default=30.0, help="smoothing (vx) the wobble is measured against")
ap.add_argument("--win", type=int, default=50, help="window (vx) for wobble RMS and mean probability")
ap.add_argument("--rel", type=float, default=2.5)
ap.add_argument("--floor", type=float, default=1.2, help="min sideways-wobble threshold (vx RMS)")
ap.add_argument("--normal-floor", type=float, default=5.0,
                help="fibres undulate along the normal where they pass over/under crossings, so this is loose")
ap.add_argument("--min-prob", type=float, default=0.6)
ap.add_argument("--persist", type=int, default=40, help="vx a wonky stretch must last to count")
ap.add_argument("--backoff", type=float, default=50.0)
ap.add_argument("--trace-args", default="--inertia 0.8 --reacquire-ahead 60 --seed-radius 2")
ap.add_argument("--min-run", type=int, default=100, help="shortest clean run (vx) worth keeping")
ap.add_argument("--max-bridge", type=int, default=1100, help="longest disturbed stretch (vx) to bridge")
ap.add_argument("--bridge-normal", type=float, default=5.0, help="max normal mismatch across a gap (vx)")
ap.add_argument("--bridge-side", type=float, default=8.0, help="max sideways mismatch across a gap (vx)")
ap.add_argument("--hop", default="60,1021,60", help="look-ahead distances (vx) min,max,step for hops")
ap.add_argument("--max-hops", type=int, default=6)
ap.add_argument("--probe-len", type=float, default=400.0, help="how far a hop probe is traced (vx)")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()
src, out = Path(args.src), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent

seg = Path(args.segment)
sx, sy = json.loads((seg / "meta.json").read_text())["scale"]
g = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (g != -1).all(-1)
uu, vv = np.meshgrid(np.arange(0, (g.shape[1] - 1) / sx, 4.0) * sx, np.arange(0, (g.shape[0] - 1) / sy, 4.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(g[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
ntree, nvec = cKDTree(sp[ok]), nrm[ok]
pred = Predictor()
hz = pred.root["hz"]


def resample(xyz, prob):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(arc) > 1e-6])
    xyz, prob, arc = xyz[keep], prob[keep], arc[keep]
    s = np.arange(0, arc[-1] + 1e-6, 1.0)
    return np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1), np.interp(s, arc, prob)


def normal_at(p, t):
    n = nvec[ntree.query(p, k=8)[1]].mean(0)
    n -= (n @ t) * t
    return n / (np.linalg.norm(n) + 1e-9)


def wobble(xyz, prob):
    """Per-sample windowed RMS of sideways / normal deviation from the smoothed path, and windowed prob."""
    sm = gaussian_filter1d(xyz, args.sigma, axis=0, mode="nearest")
    t = np.gradient(sm, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = nvec[ntree.query(sm, k=8)[1]].mean(1)
    n -= (n * t).sum(1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    r = xyz - sm
    rs, rn = (r * np.cross(t, n)).sum(1), (r * n).sum(1)
    w = args.win
    return (np.sqrt(uniform_filter1d(rs ** 2, w, mode="nearest").clip(0)),
            np.sqrt(uniform_filter1d(rn ** 2, w, mode="nearest").clip(0)),
            uniform_filter1d(prob, w, mode="nearest"))


def first_wonky(ws, wn, wp, start, ts, tn):
    bad = (ws > ts) | (wn > tn) | (wp < args.min_prob)
    run = 0
    for i in range(start, len(bad)):
        run = run + 1 if bad[i] else 0
        if run >= args.persist:
            j = slice(i - run + 1, i + 1)
            why = "/".join(k for k, b in (("side", (ws[j] > ts).any()), ("normal", (wn[j] > tn).any()),
                                          ("prob", (wp[j] < args.min_prob).any())) if b)
            return i - run + 1, why
    return len(bad), "-"


def end_fit(P, k=400):
    """Quadratic fit (per coordinate, against sample index) of the last k samples; returns (coefs, tangent)."""
    idx = np.arange(max(0, len(P) - k), len(P))
    deg = 2 if len(idx) >= 150 else 1
    co = [np.polyfit(idx, P[idx, c], deg) for c in range(3)]
    t = np.array([np.polyval(np.polyder(c), len(P) - 1) for c in co])
    return co, t / np.linalg.norm(t)


def bridge(allx, allp, start, ts, tn):
    """Keep clean runs after `start`; fill a disturbed gap with a smooth curve only when the next clean run
    lies where the fibre before the gap extrapolates to. Returns (xyz, prob, note)."""
    ws, wn, wp = wobble(allx, allp)
    good = ~((ws > ts) | (wn > tn) | (wp < args.min_prob))
    good[:start] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], good.astype(int), [0]])))
    runs = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if s < start <= e or e - s >= args.min_run]
    allx, allp = allx.copy(), allp.copy()
    s_cur, e_cur = runs[0]
    notes = []
    for s_nxt, e_nxt in runs[1:]:
        gap = s_nxt - e_cur
        if gap > args.max_bridge:
            notes.append(f"{gap} vx disturbed at x {allx[e_cur - 1, 0]:.0f}")
            break
        # both clean pieces extrapolate half-way; a layer change shows up as a normal mismatch
        ia = np.arange(max(s_cur, e_cur - 400), e_cur)
        ib = np.arange(s_nxt, min(e_nxt, s_nxt + 400))
        fa = [np.polyfit(ia, allx[ia, c], 2 if len(ia) >= 250 else 1) for c in range(3)]
        fb = [np.polyfit(ib, allx[ib, c], 2 if len(ib) >= 250 else 1) for c in range(3)]
        mid = (e_cur - 1 + s_nxt) / 2
        d = np.array([np.polyval(fa[c], mid) - np.polyval(fb[c], mid) for c in range(3)])
        ta = np.array([np.polyval(np.polyder(c), e_cur - 1) for c in fa])
        ta /= np.linalg.norm(ta)
        tb = np.array([np.polyval(np.polyder(c), s_nxt) for c in fb])
        tb /= np.linalg.norm(tb)
        tm = (ta + tb) / np.linalg.norm(ta + tb)
        n = normal_at(np.array([np.polyval(c, mid) for c in fa]), tm)
        dn, ds = d @ n, d @ np.cross(tm, n)
        if abs(dn) > args.bridge_normal or abs(ds) > args.bridge_side:
            notes.append(f"misaligned after x {allx[e_cur - 1, 0]:.0f} (normal {dn:+.1f}, side {ds:+.1f})")
            break
        p0, p1 = allx[e_cur - 1], allx[s_nxt]
        L = np.linalg.norm(p1 - p0)
        m0, m1 = ta * L, tb * L
        u = (np.arange(1, gap + 1) / (gap + 1))[:, None]
        allx[e_cur:s_nxt] = ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * m0 +
                             (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * m1)
        allp[e_cur:s_nxt] = 0.0
        notes.append(f"bridged {gap} vx at x {allx[e_cur - 1, 0]:.0f} (normal {dn:+.1f}, side {ds:+.1f})")
        s_cur, e_cur = s_nxt, e_nxt
    x, p = resample(allx[:e_cur], allp[:e_cur])
    return x, p, "; ".join(notes)


def trace_from(pt, axis, max_len=None):
    tmp = out / "_tmp_ext.json"
    cmd = [sys.executable, str(here / "trace_fiber_3d.py"), *map(str, pt), "h", str(tmp), "--one-way",
           "--axis", *map(str, axis), *args.trace_args.split()]
    if max_len:
        cmd += ["--max-length", str(2 * max_len)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
    npy = tmp.with_suffix(".path.npy")
    if not npy.exists():
        return None
    e = np.load(npy).astype(np.float64)[:, :4]
    npy.unlink()
    tmp.unlink()
    return e if len(e) > 2 else None


def extend(P, Pp, e, ts, tn):
    """Append trace e to P (gap between them marked prob 0) and keep what passes the bridge test."""
    chord = np.linalg.norm(e[0, :3] - P[-1])
    allx, allp = resample(np.concatenate([P, e[:, :3]]), np.concatenate([Pp, e[:, 3]]))
    allp[len(P):len(P) - 1 + int(round(chord))] = 0.0
    return bridge(allx, allp, len(P), ts, tn)


def find_seeds(q, r=10, k=2):
    """Up to k confident horizontal voxels within r vx of q, nearest first, at least 4 vx apart."""
    lo = np.floor(q[::-1] - r - 1).astype(int)
    hi = lo + 2 * r + 3
    for c in {tuple(np.where(b, hi - 1, lo) // CORE) for b in np.ndindex(2, 2, 2)}:
        pred.ensure(c)
    blk = hz[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32) / 255
    zz, yy, xx = np.nonzero(blk >= 0.8)
    pts = np.stack([xx, yy, zz], 1) + lo[::-1]
    d = np.linalg.norm(pts - q, axis=1)
    pts = pts[d <= r][np.argsort(d[d <= r])].astype(float)
    picked = []
    for p in pts:
        if all(np.linalg.norm(p - o) >= 4 for o in picked):
            picked.append(p)
            if len(picked) == k:
                break
    return picked


def hop(P, Pp, ts, tn):
    co, t = end_fit(P)
    for dist in range(*map(int, args.hop.split(","))):
        q = np.array([np.polyval(c, len(P) - 1 + dist) for c in co])
        for s in find_seeds(q):
            e = trace_from(s, t, args.probe_len)
            if e is None or len(e) * 2 < args.min_run + 50:
                continue
            P2, Pp2, note = extend(P, Pp, e, ts, tn)
            if args.verbose:
                print(f"    hop {dist}: seed {s.tolist()} probe {len(e) * 2} vx, kept {len(P2) - len(P)} vx: {note}")
            if len(P2) < len(P) + 0.8 * dist + args.min_run:
                continue
            e2 = trace_from(P2[-1], end_fit(P2)[1])
            if e2 is not None:
                P3, Pp3, n3 = extend(P2, Pp2, e2, ts, tn)
                if len(P3) > len(P2):
                    return P3, Pp3, f"hop {dist} vx at x {P[-1, 0]:.0f}" + (f" ({note})" if note else "") + \
                        (f"; {n3}" if n3 else "")
            return P2, Pp2, f"hop {dist} vx at x {P[-1, 0]:.0f}" + (f" ({note})" if note else "")
    return None


def write_fiber(path_out, xyz, prob):
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
    line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
    path_out.write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": xyz[ctrl].round(3).tolist(),
                                    "line_points": xyz[line].round(3).tolist()}, indent=1))
    np.save(path_out.with_suffix(".path.npy"), np.column_stack([xyz, prob, arc]).astype(np.float32))


names = [n for n in args.names.split(",") if n] or sorted(p.name[:-9] for p in src.glob("h_*.path.npy"))
for p in src.glob("v_*.path.npy"):
    shutil.copy(p, out / p.name)
    shutil.copy(src / f"{p.name[:-9]}.json", out / f"{p.name[:-9]}.json")

report = ["name,orig_xmax,cut_x,cut_why,final_xmax,bridged_vx,log"]
for name in names:
    raw = np.load(src / f"{name}.path.npy").astype(np.float64)
    xyz, prob = raw[:, :3], raw[:, 3]
    if xyz[-1, 0] < xyz[0, 0]:
        xyz, prob = xyz[::-1], prob[::-1]
    xyz, prob = resample(xyz, prob)
    ws, wn, wp = wobble(xyz, prob)
    seed = np.array([float(v) for v in name.split("_")[1:4]])
    i_seed = int(np.argmin(np.linalg.norm(xyz - seed, axis=1)))
    mid = slice(len(xyz) // 4, 3 * len(xyz) // 4)
    ts = max(args.floor, args.rel * np.median(ws[mid]))
    tn = max(args.normal_floor, args.rel * np.median(wn[mid]))
    i_bad, why = first_wonky(ws, wn, wp, i_seed + args.win, ts, tn)
    i_cut = max(i_seed, min(i_bad, len(xyz) - 1) - int(args.backoff))
    P, Pp = xyz[:i_cut + 1], prob[:i_cut + 1]
    log = []
    e = trace_from(P[-1], end_fit(P)[1])
    if e is not None:
        P2, Pp2, note = extend(P, Pp, e, ts, tn)
        if len(P2) > len(P):
            P, Pp = P2, Pp2
            log.append(f"traced to x {P[-1, 0]:.0f}" + (f" ({note})" if note else ""))
        elif note:
            log.append(note)
    for _ in range(args.max_hops):
        if P[-1, 0] >= args.xmax:
            break
        h = hop(P, Pp, ts, tn)
        if h is None:
            log.append(f"no aligned piece within {args.hop.split(',')[1]} vx ahead of x {P[-1, 0]:.0f}")
            break
        P, Pp, note = h
        log.append(f"{note} -> x {P[-1, 0]:.0f}")
    write_fiber(out / f"{name}.json", P, Pp)
    bridged = int((Pp[i_cut:] == 0).sum())
    report.append(f"{name},{xyz[:, 0].max():.0f},{xyz[i_cut, 0]:.0f},{why},{P[:, 0].max():.0f},{bridged},"
                  f"\"{' | '.join(log)}\"")
    print(f"{name:20s} orig x<= {xyz[:, 0].max():5.0f}  cut at {xyz[i_cut, 0]:5.0f} ({why})  "
          f"-> final x<= {P[:, 0].max():5.0f}  ({bridged} vx bridged)\n      " + "\n      ".join(log), flush=True)
(out / "extend_report.csv").write_text("\n".join(report) + "\n")
