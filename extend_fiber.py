"""Extend a traced horizontal fibre with the x-monotonic, z-locked tracer.

Horizontal fibres in this region are nearly straight in z and only bend in y as the sheet curves, while
the free 3D tracer tends to drift up/down or hop layers where the fibre gets faint. This script

  1. winds the fibre back on the chosen side to the last clean point: the first stretch past the seed
     where the path wobbles sideways or loses probability is found automatically (or give --from-x),
     and --backoff vx are removed before it;
  2. traces onward from there with --xmono: x advances monotonically, y follows the fibre with a slope
     prior (so it can follow the sheet's curvature), z is allowed to move only --z-slack vx per step;
  3. splices kept part + extension, writes <name>.json / <name>.path.npy (the original is kept as
     <name>.path.npy.orig / <name>.json.orig), and, if the folder has verticals, prints the fibre's crossing
     verdicts so you can see where the extension stays on the sheet.

Example
  python extend_fiber.py data/grid1_final7/h_3394_3491_13385.path.npy --segment <tifxyz folder>
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fiber", help="<name>.path.npy of a horizontal fibre (written by analyze_fiber.py / the tracer)")
ap.add_argument("--segment", required=True, help="tifxyz sheet folder; only its orientation is used (layer normal)")
ap.add_argument("--direction", choices=["+x", "-x", "both"], default="+x")
ap.add_argument("--from-x", type=float, help="restart from this x instead of the automatic clean point")
ap.add_argument("--backoff", type=float, default=50.0, help="vx removed before the first wobbly stretch")
ap.add_argument("--stop-x", type=float, help="cut the result at this x (on the extended side)")
ap.add_argument("--y-slack", type=float, default=2.0, help="max y re-centring per 2-vx step")
ap.add_argument("--z-slack", type=float, default=0.25, help="max z re-centring per 2-vx step")
ap.add_argument("--reacquire-ahead", type=float, default=60.0, help="how far ahead (vx) to look across faint gaps")
ap.add_argument("--out", help="output folder (default: the fibre's folder, original kept as <name>.*.orig)")
args = ap.parse_args()

src = Path(args.fiber)
name = src.name[:-len(".path.npy")]
fdir = src.parent
out = Path(args.out) if args.out else fdir
out.mkdir(parents=True, exist_ok=True)

# layer normals from the sheet, for splitting wobble into sideways / normal parts
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


def resample(xyz, prob):
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(arc) > 1e-6])
    xyz, prob, arc = xyz[keep], prob[keep], arc[keep]
    s = np.arange(0, arc[-1] + 1e-6, 1.0)
    return np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1), np.interp(s, arc, prob)


def first_wonky(xyz, prob, start, step):
    """Index of the first sustained wobble/low-probability stretch walking from start in direction step."""
    sm = gaussian_filter1d(xyz, 30, axis=0, mode="nearest")
    t = np.gradient(sm, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = nvec[ntree.query(sm, k=8)[1]].mean(1)
    n -= (n * t).sum(1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    side = ((xyz - sm) * np.cross(t, n)).sum(1)
    ws = np.sqrt(uniform_filter1d(side ** 2, 50, mode="nearest").clip(0))
    wp = uniform_filter1d(prob, 50, mode="nearest")
    mid = slice(len(xyz) // 4, 3 * len(xyz) // 4)
    bad = (ws > max(1.2, 2.5 * np.median(ws[mid]))) | (wp < 0.6)
    run = 0
    idx = range(start, len(xyz)) if step > 0 else range(start, -1, -1)
    for i in idx:
        run = run + 1 if bad[i] else 0
        if run >= 40:
            return i - step * (run - 1)
    return idx[-1] if len(idx) else start


raw = np.load(src).astype(np.float64)
xyz, prob = resample(raw[:, :3], raw[:, 3])
if xyz[-1, 0] < xyz[0, 0]:
    xyz, prob = xyz[::-1].copy(), prob[::-1].copy()
try:
    seed = np.array([float(v) for v in name.split("_")[1:4]])
    i_seed = int(np.argmin(np.linalg.norm(xyz - seed, axis=1)))
except ValueError:
    i_seed = len(xyz) // 2
print(f"{name}: {len(xyz)} vx, x {xyz[:, 0].min():.0f}..{xyz[:, 0].max():.0f}")

lo, hi = 0, len(xyz) - 1
pieces = {}
for d in (["+x", "-x"] if args.direction == "both" else [args.direction]):
    step = 1 if d == "+x" else -1
    if args.from_x is not None:
        i_cut = int(np.argmin(np.abs(xyz[:, 0] - args.from_x)))
    else:
        i_bad = first_wonky(xyz, prob, i_seed + step * 50, step)
        i_cut = int(np.clip(i_bad - step * args.backoff, min(i_seed, i_bad), max(i_seed, i_bad)))
    if d == "+x":
        hi = i_cut
    else:
        lo = i_cut
    tmp = out / f"_{name}_ext{'p' if step > 0 else 'm'}.json"
    cmd = [sys.executable, str(TOOLS / "trace_fiber_3d.py"), *map(str, xyz[i_cut]), "h", str(tmp),
           "--xmono", "--one-way", "--axis", str(float(step)), "0", "0", "--seed-radius", "2",
           "--y-slack", str(args.y_slack), "--z-slack", str(args.z_slack),
           "--reacquire-ahead", str(args.reacquire_ahead)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    npy = tmp.with_suffix(".path.npy")
    if not npy.exists():
        msg = (r.stdout + r.stderr).strip().splitlines()
        print(f"  {d}: no extension ({msg[-1] if msg else 'tracer produced no output'})")
        continue
    e = np.load(npy).astype(np.float64)
    npy.unlink()
    tmp.unlink()
    end = next((ln.strip() for ln in r.stdout.splitlines() if "end at" in ln), "")
    print(f"  {d}: restart at x {xyz[i_cut, 0]:.0f} (was {'clean' if args.from_x is None else 'chosen'}), "
          f"extension reaches x {e[-1, 0]:.0f}  [{end}]")
    pieces[d] = e

parts_xyz, parts_p = [xyz[lo:hi + 1]], [prob[lo:hi + 1]]
if "-x" in pieces:
    parts_xyz.insert(0, pieces["-x"][1:, :3][::-1])
    parts_p.insert(0, pieces["-x"][1:, 3][::-1])
if "+x" in pieces:
    parts_xyz.append(pieces["+x"][1:, :3])
    parts_p.append(pieces["+x"][1:, 3])
P, pr = np.concatenate(parts_xyz), np.concatenate(parts_p)
if args.stop_x is not None:
    m = P[:, 0] <= args.stop_x if args.direction != "-x" else P[:, 0] >= args.stop_x
    P, pr = P[m], pr[m]

if out == fdir:
    for suf in (".path.npy", ".json"):
        if (fdir / f"{name}{suf}").exists() and not (fdir / f"{name}{suf}.orig").exists():
            shutil.copy(fdir / f"{name}{suf}", fdir / f"{name}{suf}.orig")
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
np.save(out / f"{name}.path.npy", np.column_stack([P, pr, arc]).astype(np.float32))
ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
(out / f"{name}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1,
                                              "control_points": P[ctrl].round(3).tolist(),
                                              "line_points": P[line].round(3).tolist()}, indent=1))
print(f"wrote {out / name}.json/.path.npy: {arc[-1]:.0f} vx ({arc[-1] * 9.362 / 1000:.1f} mm), "
      f"x {P[:, 0].min():.0f}..{P[:, 0].max():.0f}")

# crossing verdicts along the new fibre
if list(out.glob("v_*.path.npy")):
    subprocess.run([sys.executable, str(TOOLS / "fiber_crossings.py"), str(out), str(seg), "--xmax", "1e9"],
                   capture_output=True, text=True)
    lay = out / "layers.json"
    S, p = (json.loads(lay.read_text())["spacing_vx"], json.loads(lay.read_text())["ply_offset_vx"]) \
        if lay.exists() else (14.0, 9.0)
    rows = sorted((r for r in csv.DictReader(open(out / "crossings.csv"))
                   if r["horizontal"] == name and r["verdict"] != "no overlap"),
                  key=lambda r: float(r["at"].strip("[]").split()[0]))
    print(f"crossings (same sheet = offset within {S / 2:g} vx of {p:+g}):")
    for r in rows:
        o = float(r["normal_offset_vx"])
        dl = int(round((o - p) / S))
        x = float(r["at"].strip("[]").split()[0])
        print(f"  x {x:6.0f}  {r['vertical']:22s} offset {o:+6.1f} -> "
              + ("same sheet" if dl == 0 else f"{abs(dl)} layer(s) {'+' if dl > 0 else '-'}normal"))
