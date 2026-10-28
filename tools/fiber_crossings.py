"""Crossing analysis for traced fibres: which vertical/horizontal pairs truly intersect.

Usage: python fiber_crossings.py <fiber_dir> <segment_dir> [--xmax 5100]
Reads <fiber_dir>/{v,h}_*.path.npy written by trace_fiber_3d.py.
Writes crossings.csv, groups.json and a summary to stdout.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

CROSS_VX, MISS_VX, NEAR_VX, END_VX = 5.0, 8.0, 60.0, 20.0

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--xmax", type=float, default=5100.0)
args = ap.parse_args()
fdir = Path(args.fibers)


def resample(xyz, step=1.0):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    s = np.arange(0, arc[-1] + 1e-9, step)
    return np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1), s


fibers = {}
for p in sorted(fdir.glob("*.path.npy")):
    name = p.name[: -len(".path.npy")]
    xyz = np.load(p)[:, :3].astype(np.float64)
    keep = xyz[:, 0] <= args.xmax
    if keep.sum() < 10:
        continue
    # keep the longest run under xmax that contains the seed region
    runs = np.split(np.arange(len(xyz)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
    runs = [r for r in runs if keep[r[0]]]
    xyz = xyz[max(runs, key=len)]
    pts, arc = resample(xyz)
    fibers[name] = {"family": name[0], "pts": pts, "arc": arc, "tree": cKDTree(pts)}

# sheet normals for signed offsets
seg = Path(args.segment)
meta = json.loads((seg / "meta.json").read_text())
sx, sy = meta["scale"]
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (grid != -1).all(-1)
gh, gw = gv.shape
uu, vv = np.meshgrid(np.arange(0, (gw - 1) / sx, 2.0) * sx, np.arange(0, (gh - 1) / sy, 2.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(grid[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
vi, ui = np.nonzero(ok)
sheet_tree = cKDTree(sp[vi, ui])
sheet_n = nrm[vi, ui]

# how closely the existing sheet follows each fibre
for name, f in fibers.items():
    d, _ = sheet_tree.query(f["pts"])
    f["on_sheet"] = float((d <= 4.0).mean())
    f["sheet_median"] = float(np.median(d))

rows = []
verticals = [n for n in fibers if fibers[n]["family"] == "v"]
horizontals = [n for n in fibers if fibers[n]["family"] == "h"]
for vn in verticals:
    V = fibers[vn]
    for hn in horizontals:
        H = fibers[hn]
        d, j = H["tree"].query(V["pts"])
        i = int(np.argmin(d))
        j = int(j[i])
        dist = float(d[i])
        pv, ph = V["pts"][i], H["pts"][j]
        v_end = min(V["arc"][i], V["arc"][-1] - V["arc"][i])
        h_end = min(H["arc"][j], H["arc"][-1] - H["arc"][j])
        _, k = sheet_tree.query((pv + ph) / 2)
        n = sheet_n[k]
        signed = float((ph - pv) @ n)                    # + : horizontal lies on the +normal side
        if v_end < END_VX or h_end < END_VX:
            verdict = "no overlap"
        elif dist <= CROSS_VX:
            verdict = "crosses"
        elif dist <= MISS_VX:
            verdict = "unclear"
        elif dist <= NEAR_VX:
            verdict = "misses"
        else:
            verdict = "no overlap"
        rows.append(dict(vertical=vn, horizontal=hn, verdict=verdict, distance_vx=round(dist, 1),
                         normal_offset_vx=round(signed, 1), at=((pv + ph) / 2).round(1).tolist(),
                         v_from_end_vx=round(float(v_end)), h_from_end_vx=round(float(h_end))))

# groups: connected components over "crosses" edges
parent = {n: n for n in fibers}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for r in rows:
    if r["verdict"] == "crosses":
        parent[find(r["vertical"])] = find(r["horizontal"])
groups = {}
for n in fibers:
    groups.setdefault(find(n), []).append(n)
groups = sorted(groups.values(), key=lambda g: -len(g))

with open(fdir / "crossings.csv", "w") as fh:
    keys = list(rows[0].keys())
    fh.write(",".join(keys) + "\n")
    for r in rows:
        fh.write(",".join(str(r[k]).replace(",", " ") for k in keys) + "\n")
(fdir / "groups.json").write_text(json.dumps(
    [{"fibers": g, "on_sheet_fraction": {n: round(fibers[n]["on_sheet"], 2) for n in g}} for g in groups],
    indent=1))

# ---- summary --------------------------------------------------------------------------------------
print(f"{len(verticals)} vertical x {len(horizontals)} horizontal fibres (clipped at x <= {args.xmax:.0f})\n")
print("fibre lengths / how well the current sheet follows each fibre:")
for n, f in fibers.items():
    print(f"  {n:22s} {f['arc'][-1]:6.0f} vx ({f['arc'][-1] * 9.362 / 1000:5.1f} mm)  "
          f"sheet within 4 vx: {f['on_sheet'] * 100:5.1f}%  median sheet dist {f['sheet_median']:5.1f} vx")
sym = {"crosses": "X", "misses": "-", "unclear": "?", "no overlap": "."}
print("\ncrossing matrix (X crosses, - misses = different layers, ? unclear, . never meet)")
print(" " * 22 + " ".join(f"V{i + 1}" for i in range(len(verticals))))
for hn in horizontals:
    cells = [sym[next(r["verdict"] for r in rows if r["vertical"] == vn and r["horizontal"] == hn)]
             for vn in verticals]
    print(f"  {hn:20s} " + "  ".join(cells))
for i, vn in enumerate(verticals):
    print(f"  V{i + 1} = {vn}")
print("\nmisses (different layers): signed offset along the sheet normal (+ = horizontal is on the +normal side)")
for r in rows:
    if r["verdict"] in ("misses", "unclear"):
        print(f"  {r['vertical']:20s} vs {r['horizontal']:20s} {r['verdict']:8s} gap {r['distance_vx']:5.1f} vx, "
              f"normal offset {r['normal_offset_vx']:+6.1f} vx at {r['at']}")
print("\ngroups of mutually crossing fibres (each group = one sheet/layer):")
for k, g in enumerate(groups):
    on = np.mean([fibers[n]["on_sheet"] for n in g])
    print(f"  group {k + 1} ({len(g)} fibres, current sheet follows them {on * 100:.0f}% of the time): {', '.join(g)}")
