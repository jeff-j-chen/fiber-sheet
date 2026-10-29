"""Copy a fibre set, cutting fibres to given coordinate ranges and dropping unwanted fibres.

Usage: python fiber_trim.py <src_dir> <dst_dir> [--cut h_a:4400,...] [--keep v_b:z:11500:13550,...] [--drop h_c,h_d]
  --cut name:xmax keeps x <= xmax; --keep name:axis:lo:hi keeps lo <= axis <= hi.
Writes <name>.path.npy and a v1 vc3d_fiber <name>.json for every kept fibre.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from fiber_meta import FIBER_COORDS

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--cut", default="")
ap.add_argument("--keep", default="")
ap.add_argument("--drop", default="")
args = ap.parse_args()
src, dst = Path(args.src), Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
cuts = {k: (0, -np.inf, float(v)) for k, v in (c.split(":") for c in args.cut.split(",") if c)}
for c in filter(None, args.keep.split(",")):
    n, axis, lo, hi = c.split(":")
    cuts[n] = ("xyz".index(axis), float(lo), float(hi))
drop = set(filter(None, args.drop.split(",")))

for p in sorted(src.glob("*.path.npy")):
    name = p.name[:-9]
    if name in drop:
        continue
    if name not in cuts:
        if src.resolve() != dst.resolve():
            shutil.copy(p, dst / p.name)
            shutil.copy(src / f"{name}.json", dst / f"{name}.json")
        continue
    a = np.load(p)
    ax, lo, hi = cuts[name]
    keep = (a[:, ax] >= lo) & (a[:, ax] <= hi)
    runs = np.split(np.arange(len(a)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
    a = a[max((r for r in runs if keep[r[0]]), key=len)].copy()
    a[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1))])
    np.save(dst / p.name, a)
    arc = a[:, 4]
    ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
    line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
    (dst / f"{name}.json").write_text(json.dumps(
        {"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": a[ctrl, :3].round(3).tolist(),
         "line_points": a[line, :3].round(3).tolist()}, indent=1))
    print(f"{name}: kept {'xyz'[ax]} {lo:.0f}..{hi:.0f} -> {arc[-1]:.0f} vx, "
          f"{'xyz'[ax]} {a[:, ax].min():.0f}..{a[:, ax].max():.0f}")
print(f"wrote {dst}")
