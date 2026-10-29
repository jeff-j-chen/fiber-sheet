"""Crossing offsets from the fibres themselves: the local sheet normal is t_h x t_v at the closest approach.

No reference surface is needed, so offsets don't depend on a (possibly wrong) sheet's orientation.
Writes <fiber_dir>/crossings.csv with the same columns as fiber_crossings.py.
Usage: python fiber_crossings_fn.py <fiber_dir> [--near 25]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--near", type=float, default=25.0, help="max closest-approach distance (vx) to count as overlapping")
args = ap.parse_args()
d = Path(args.fibers)


def resample(xyz, step=1.0):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    s = np.arange(0, arc[-1] + 1e-9, step)
    return np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1)


F = {}
for p in sorted(d.glob("*.path.npy")):
    pts = resample(np.load(p)[:, :3].astype(np.float64))
    if len(pts) > 20:
        F[p.name[:-9]] = (pts, cKDTree(pts))


def tangent(pts, i, k=10):
    t = pts[min(i + k, len(pts) - 1)] - pts[max(i - k, 0)]
    return t / (np.linalg.norm(t) + 1e-9)


rows = []
for vn, (vp, vt) in F.items():
    if vn[0] != "v":
        continue
    for hn, (hp, ht) in F.items():
        if hn[0] != "h":
            continue
        dist, j = vt.query(hp)
        i = int(np.argmin(dist))
        if dist[i] > args.near:
            rows.append([vn, hn, "no overlap", f"{dist[i]:.2f}", "nan", "[nan nan nan]", "0", "0"])
            continue
        jv = int(j[i])
        th, tv = tangent(hp, i), tangent(vp, jv)
        th *= np.sign(th[0]) or 1.0
        tv *= np.sign(tv[2]) or 1.0
        n = np.cross(th, tv)
        n /= np.linalg.norm(n) + 1e-9
        off = float((hp[i] - vp[jv]) @ n)
        at = (hp[i] + vp[jv]) / 2
        verdict = "crosses" if dist[i] <= 5 else "unclear" if dist[i] <= 8 else "misses"
        rows.append([vn, hn, verdict, f"{dist[i]:.2f}", f"{off:.2f}", "[" + " ".join(f"{c:.1f}" for c in at) + "]",
                     f"{min(jv, len(vp) - 1 - jv):.0f}", f"{min(i, len(hp) - 1 - i):.0f}"])
with open(d / "crossings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["vertical", "horizontal", "verdict", "distance_vx", "normal_offset_vx", "at", "v_from_end_vx",
                "h_from_end_vx"])
    w.writerows(rows)
ov = [float(r[4]) for r in rows if r[2] != "no overlap"]
print(f"{len(ov)} overlapping pairs; offsets sorted: " + " ".join(f"{o:+.0f}" for o in sorted(ov)))
