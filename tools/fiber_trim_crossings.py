"""Trim every fibre to the stretch its crossings confirm.

For each fibre, crossings are ordered along the fibre (x for horizontals, z for verticals). A wrong
crossing is blamed on the fibre whose own neighbouring crossings (one before, one after along that fibre)
are worse; with a tie both fibres take the blame. An isolated borderline wrong crossing (less than a full
layer off, good crossings on both sides) is treated as noise. The longest run of crossings that are not
the fibre's fault is kept and the fibre is cut halfway to the neighbouring blamed crossing.
Fibres without a single same-sheet crossing are reported and copied unchanged unless --drop-unverified.

Usage: python fiber_trim_crossings.py <fiber_dir> <out_dir> [--ply 9 --spacing 14] [--drop-unverified]
Needs <fiber_dir>/crossings.csv (fiber_crossings.py). Writes .path.npy + v1 .json per fibre.
Run it repeatedly (recomputing crossings in between) until nothing is cut any more.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from fiber_meta import FIBER_COORDS

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--ply", type=float)
ap.add_argument("--spacing", type=float)
ap.add_argument("--drop-unverified", action="store_true")
ap.add_argument("--family", choices=["h", "v", "both"], default="both", help="which fibres may be trimmed")
args = ap.parse_args()
src, dst = Path(args.fibers), Path(args.out)
dst.mkdir(parents=True, exist_ok=True)
lay = src / "layers.json"
L = json.loads(lay.read_text()) if lay.exists() else {}
p = args.ply if args.ply is not None else L.get("ply_offset_vx", 9.0)
S = args.spacing if args.spacing is not None else L.get("spacing_vx", 14.0)

cl = []
for r in csv.DictReader(open(src / "crossings.csv")):
    if r["verdict"] == "no overlap":
        continue
    at = [float(v) for v in r["at"].strip("[]").split()]
    o = float(r["normal_offset_vx"])
    cl.append({"h": r["horizontal"], "v": r["vertical"], "x": at[0], "z": at[2], "o": o,
               "good": abs(o - p) < S / 2})
order = {}
for k, c in enumerate(cl):
    order.setdefault(c["h"], []).append((c["x"], k))
    order.setdefault(c["v"], []).append((c["z"], k))
order = {n: [k for _, k in sorted(v)] for n, v in order.items()}


def nbr_good(name, k):
    """How many of the crossings next to crossing k along fibre `name` are good (0..2), and how many exist."""
    seq = order[name]
    i = seq.index(k)
    nb = [seq[j] for j in (i - 1, i + 1) if 0 <= j < len(seq)]
    return sum(cl[j]["good"] for j in nb), len(nb)


for k, c in enumerate(cl):
    c["fault"] = set()
    if c["good"]:
        continue
    gh, nh = nbr_good(c["h"], k)
    gv, nv = nbr_good(c["v"], k)
    if gh == nh and nh == 2 and abs(c["o"] - p) < S and gv == nv and nv == 2:
        continue                                       # isolated borderline crossing on two clean fibres
    rh, rv = gh / max(nh, 1), gv / max(nv, 1)
    if rh == rv:
        # tie: the fibre that is wrong more often overall takes the blame
        rh = np.mean([cl[j]["good"] for j in order[c["h"]]])
        rv = np.mean([cl[j]["good"] for j in order[c["v"]]])
    if rh > rv:
        c["fault"] = {c["v"]}
    elif rv > rh:
        c["fault"] = {c["h"]}
    else:
        c["fault"] = {c["h"], c["v"]}
cross = {n: [((cl[k]["x"] if n.startswith("h") else cl[k]["z"]), n not in cl[k]["fault"], cl[k]["o"])
             for k in ks] for n, ks in order.items()}


def write(name, a):
    a = a.copy()
    a[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1))])
    np.save(dst / f"{name}.path.npy", a.astype(np.float32))
    arc = a[:, 4]
    ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
    line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
    (dst / f"{name}.json").write_text(json.dumps(
        {"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": a[ctrl, :3].round(3).tolist(),
         "line_points": a[line, :3].round(3).tolist()}, indent=1))


for pth in sorted(src.glob("*.path.npy")):
    name = pth.name[:-9]
    a = np.load(pth).astype(np.float64)
    cs = cross.get(name, [])
    ax = 0 if name.startswith("h") else 2
    if args.family != "both" and name[0] != args.family:
        if src.resolve() != dst.resolve():
            shutil.copy(pth, dst / pth.name)
            if (src / f"{name}.json").exists():
                shutil.copy(src / f"{name}.json", dst / f"{name}.json")
        continue
    if not any(g for _, g, _ in cs):
        print(f"{name:22s} no same-sheet crossing ({len(cs)} crossings)" +
              (" -> dropped" if args.drop_unverified else " -> kept unchanged"))
        if not args.drop_unverified and src.resolve() != dst.resolve():
            shutil.copy(pth, dst / pth.name)
            if (src / f"{name}.json").exists():
                shutil.copy(src / f"{name}.json", dst / f"{name}.json")
        continue
    best, i = None, 0
    while i < len(cs):
        if not cs[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < len(cs) and cs[j + 1][1]:
            j += 1
        if best is None or (j - i, cs[j][0] - cs[i][0]) > (best[1] - best[0], cs[best[1]][0] - cs[best[0]][0]):
            best = (i, j)
        i = j + 1
    i, j = best
    lo = (cs[i - 1][0] + cs[i][0]) / 2 if i > 0 else -np.inf
    hi = (cs[j][0] + cs[j + 1][0]) / 2 if j + 1 < len(cs) else np.inf
    keep = (a[:, ax] >= lo) & (a[:, ax] <= hi)
    runs = np.split(np.arange(len(a)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
    run = max((r for r in runs if keep[r[0]]), key=len)
    write(name, a[run])
    lab = "xyz"[ax]
    cut = (f"{lab} {a[run, ax].min():.0f}..{a[run, ax].max():.0f}" if (np.isfinite(lo) or np.isfinite(hi))
           else "uncut")
    print(f"{name:22s} {j - i + 1}/{len(cs)} crossings kept ({sum(g for _, g, _ in cs)} not its fault); {cut}")
print(f"wrote {dst}")
