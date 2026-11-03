"""Layer-consistency audit of traced fibres and a search for same-layer horizontal seeds.

Part A (no model assumptions): for every vertical, the difference between the crossing offsets of two
horizontals is (L_a - L_b) * S, independent of the ply offset p and of the sign convention. Comparing
that difference across verticals shows which fibre changes layer between which crossings.
Part B: along chosen verticals, sample the horizontal-fibre probability along the sheet normal over a
z range and list hz peaks; peaks in the same offset band as the trusted horizontals are seed candidates.

Usage: python fiber_consistency.py <fiber_dir> <segment_dir> [--probe v_3631_3498_13153,v_4393_3359_12670]
       [--zmin 11850 --zmax 12180] [--band 3,17]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--probe", default="v_3631_3498_13153,v_4393_3359_12670")
ap.add_argument("--zmin", type=float, default=11850)
ap.add_argument("--zmax", type=float, default=12180)
ap.add_argument("--band", default="3,17", help="same-layer offset band (vx) for seed candidates")
args = ap.parse_args()
fdir = Path(args.fibers)

rows = [r for r in csv.DictReader(open(fdir / "crossings.csv"))
        if r["verdict"] != "no overlap" and float(r["distance_vx"]) <= 60]
V = sorted({r["vertical"] for r in rows})
H = sorted({r["horizontal"] for r in rows}, key=lambda n: int(n.split("_")[-1]))
off = {(r["vertical"], r["horizontal"]): float(r["normal_offset_vx"]) for r in rows}
dist = {(r["vertical"], r["horizontal"]): float(r["distance_vx"]) for r in rows}
short = {n: n.split("_")[0] + "_" + (n.split("_")[1] if n[0] == "v" else n.split("_")[-1]) for n in V + H}

print("A. signed crossing offsets (vx, + = horizontal on +normal side); '(d)' = 3D gap")
print(" " * 9 + "".join(f"{short[v]:>15s}" for v in V))
for h in H:
    cells = [f"{off[v, h]:+6.1f} ({dist[v, h]:4.1f})" if (v, h) in off else " " * 13 for v in V]
    print(f"{short[h]:9s}" + "".join(f"{c:>15s}" for c in cells))

ref = "h_3297_3319_12216"
print(f"\nA2. offset of each horizontal minus {short[ref]} at the same vertical "
      "(equal along a row = consistent; a step of ~15 = layer change):")
print(" " * 9 + "".join(f"{short[v]:>10s}" for v in V))
for h in H:
    cells = [f"{off[v, h] - off[v, ref]:+6.1f}" if (v, h) in off and (v, ref) in off else "" for v in V]
    print(f"{short[h]:9s}" + "".join(f"{c:>10s}" for c in cells))

vals = np.array(list(off.values()))
for lo, hi, lab in ((3.5, 11.5, "|o-7.5|<=4 (current rule)"), (0, 15, "0<o<15 (midpoint rule)")):
    print(f"same-layer by {lab}: {((vals > lo) & (vals < hi)).sum()}/{len(vals)}")

# ---------------------------------------------------------------- Part B: seed search along verticals
import zarr  # noqa: E402

seg = Path(args.segment)
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (grid != -1).all(-1)
gh, gw = gv.shape
uu, vv = np.meshgrid(np.arange(0, (gw - 1) / sx, 2.0) * sx, np.arange(0, (gh - 1) / sy, 2.0) * sy)
sp = np.stack([cv2.remap(grid[..., i], uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR)
               for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
vi, ui = np.nonzero(ok)
tree, sheet_n = cKDTree(sp[vi, ui]), nrm[vi, ui]

hz = zarr.open_group(str(Path.home() / ".VC3D/afv_predictions/PHerc0846A_20250728152254.zarr"), mode="r")["hz"]
blo, bhi = map(float, args.band.split(","))
T = np.arange(-30, 31, 1.0)
for vname in args.probe.split(","):
    path = np.load(fdir / f"{vname}.path.npy")[:, :3]
    sel = (path[:, 2] >= args.zmin) & (path[:, 2] <= args.zmax)
    if sel.sum() < 5:
        print(f"\n{vname}: no samples in z range")
        continue
    pts = path[sel]
    lo = np.floor(pts.min(0) - 40).astype(int)
    hi = np.ceil(pts.max(0) + 40).astype(int)
    box = hz[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32) / 255  # z, y, x
    prof = []
    for p in pts:
        n = sheet_n[tree.query(p)[1]]
        q = p[None] + T[:, None] * n[None] - lo
        # max over a small in-plane cross (fibre centreline jitter)
        tang = np.cross(n, [0, 0, 1.0])
        tang /= np.linalg.norm(tang) + 1e-9
        best = np.zeros(len(T))
        for d in (-2, 0, 2):
            qq = q + d * tang
            best = np.maximum(best, map_coordinates(box, [qq[:, 2], qq[:, 1], qq[:, 0]], order=1))
        prof.append(best)
    prof = np.array(prof)
    print(f"\nB. {vname}: hz peaks along the normal, z {args.zmin:.0f}-{args.zmax:.0f} "
          f"(* = in same-layer band {blo:+.0f}..{bhi:+.0f})")
    # group consecutive samples by z blocks of 6 vx and take peaks of the block mean
    zz = pts[:, 2]
    for z0 in np.arange(args.zmin, args.zmax, 6):
        m = (zz >= z0) & (zz < z0 + 6)
        if not m.any():
            continue
        pr = prof[m].max(0)
        pk, _ = find_peaks(pr, height=0.35, distance=4)
        if not len(pk):
            continue
        txt = "  ".join(f"{'*' if blo <= T[k] <= bhi else ' '}{T[k]:+.0f}:{pr[k]:.2f}" for k in pk)
        c = pts[m].mean(0)
        n = sheet_n[tree.query(c)[1]]
        cand = [c + T[k] * n for k in pk if blo <= T[k] <= bhi]
        seed = f"  seed {cand[0].round().astype(int).tolist()}" if cand else ""
        print(f"  z {z0:6.0f}  {txt}{seed}")
