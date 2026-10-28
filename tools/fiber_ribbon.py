"""On-fibre view: render the papyrus at each fibre's own depth, walking along the fibre.

Usage: python fiber_ribbon.py <fiber_dir> <segment_dir> [--half 30] [--names a,b,...]
For every <name>.path.npy in fiber_dir writes <name>_ribbon.png:
  rows of CT at depth -15, -7.5, 0, +7.5, +15 vx along the local layer normal (0 = the fibre itself),
  then fibre-probability rows (red = this fibre's family, green = the crossing family) at depth 0 and at
  the depth where the same sheet's other fibre family should be (ply offset from layers.json).
Columns are 1 vx of arc length; rows are -half..+half vx sideways within the layer.
Crossing marks above the strip use crossings.csv / layers.json when present (green = fits the layer model).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--half", type=int, default=30)
ap.add_argument("--names", default="")
args = ap.parse_args()
fdir = Path(args.fibers)

layers = json.loads((fdir / "layers.json").read_text()) if (fdir / "layers.json").exists() else None
ply = layers["ply_offset_vx"] if layers else 7.5
crossings = list(csv.DictReader(open(fdir / "crossings.csv"))) if (fdir / "crossings.csv").exists() else []

# layer normals from the sheet's orientation (raw orientation, same sign convention as fiber_crossings.py)
seg = Path(args.segment)
sx, sy = json.loads((seg / "meta.json").read_text())["scale"]
g = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
gv = (g != -1).all(-1)
uu, vv = np.meshgrid(np.arange(0, (g.shape[1] - 1) / sx, 2.0) * sx, np.arange(0, (g.shape[0] - 1) / sy, 2.0) * sy)
uu, vv = uu.astype(np.float32), vv.astype(np.float32)
sp = np.stack([cv2.remap(g[..., i], uu, vv, cv2.INTER_LINEAR) for i in range(3)], -1)
ok = cv2.remap(gv.astype(np.float32), uu, vv, cv2.INTER_LINEAR) > 0.999
nrm = np.cross(np.gradient(sp, axis=1), np.gradient(sp, axis=0))
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(nrm).all(-1)
ntree, nvec = cKDTree(sp[ok]), nrm[ok]

pred = Predictor()
ct, hz_arr, vt_arr = pred.ct, pred.root["hz"], pred.root["vt"]
depths = [-15.0, -ply, 0.0, ply, 15.0]
across = np.arange(-args.half, args.half + 1, 1.0)


def sample_block(arr, pts_zyx, scale=1.0):
    """Trilinear samples of a zarr array at zyx points, reading one bounding block."""
    lo = np.maximum(np.floor(pts_zyx.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(pts_zyx.max(0)).astype(int) + 2, pred.shape)
    blk = arr[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32) * scale
    return map_coordinates(blk, (pts_zyx - lo).T, order=1, mode="nearest")


def render(name):
    path = np.load(fdir / f"{name}.path.npy").astype(np.float64)
    xyz, arc = path[:, :3], path[:, 4]
    s = np.arange(0, arc[-1], 1.0)
    F = np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], 1)
    t = gaussian_filter1d(np.gradient(F, axis=0), 5, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    _, k = ntree.query(F, k=12)
    n = gaussian_filter1d(nvec[k].mean(1), 15, axis=0)
    n -= (n * t).sum(1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    side = np.cross(t, n)

    family = name[0]
    other_ply = -ply if family == "h" else ply        # where this sheet's other fibre family sits
    W, Hh = len(F), len(across)
    ct_rows = {d: np.zeros((Hh, W), np.float32) for d in depths}
    prob_rows = {d: np.zeros((Hh, W, 2), np.float32) for d in (0.0, other_ply)}
    prof_d = np.arange(-22.0, 22.01, 1.0)
    prof_side = np.arange(-8.0, 8.01, 2.0)
    profile = np.zeros((len(prof_d), W), np.float32)               # crossing-family prob vs depth
    for a0 in range(0, W, 120):
        a1 = min(a0 + 120, W)
        base = F[a0:a1, None, :] + side[a0:a1, None, :] * across[None, :, None]       # cols, rows, 3
        for t_ in {tuple(v) for v in (np.floor(base.reshape(-1, 3)[:, ::-1] / 96)).astype(int)}:
            pred.ensure(t_)
        oth_arr = vt_arr if family == "h" else hz_arr
        core = F[a0:a1, None, None, :] + side[a0:a1, None, None, :] * prof_side[None, None, :, None] \
            + n[a0:a1, None, None, :] * prof_d[None, :, None, None]
        vals = sample_block(oth_arr, core.reshape(-1, 3)[:, ::-1], 1 / 255).reshape(a1 - a0, len(prof_d), len(prof_side))
        profile[:, a0:a1] = vals.mean(2).T
        for d in depths:
            acc = 0
            for dd in (-1.0, 0.0, 1.0):                                               # 3-voxel slab
                pts = (base + n[a0:a1, None, :] * (d + dd)).reshape(-1, 3)[:, ::-1]
                acc = acc + sample_block(ct, pts)
            ct_rows[d][:, a0:a1] = (acc / 3).reshape(a1 - a0, Hh).T
        for d in prob_rows:
            pts = (base + n[a0:a1, None, :] * d).reshape(-1, 3)[:, ::-1]
            own, oth = (hz_arr, vt_arr) if family == "h" else (vt_arr, hz_arr)
            prob_rows[d][:, a0:a1, 0] = sample_block(own, pts, 1 / 255).reshape(a1 - a0, Hh).T
            prob_rows[d][:, a0:a1, 1] = sample_block(oth, pts, 1 / 255).reshape(a1 - a0, Hh).T

    lo, hi = np.percentile(np.concatenate([r.ravel() for r in ct_rows.values()]), [1, 99.5])
    clahe = cv2.createCLAHE(2.5, (8, 8))
    margin, gapr = 210, 4
    panels = []
    for d in depths:
        img = clahe.apply(np.clip((ct_rows[d] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if d == 0.0:
            img[Hh // 2, ::4] = (0, 220, 255)                                           # fibre centre line
        tag = "fibre depth" if d == 0 else ("same-sheet other ply" if abs(d - other_ply) < 1e-6 else
                                            ("next sheet?" if abs(d) >= 15 else "opposite side"))
        panels.append((f"{d:+.1f} vx  {tag}", img))
    for d in (0.0, other_ply):
        pr = np.zeros((Hh, W, 3), np.uint8)
        pr[..., 2] = (prob_rows[d][..., 0] * 255).astype(np.uint8)
        pr[..., 1] = (prob_rows[d][..., 1] * 255).astype(np.uint8)
        panels.append((f"pred @ {d:+.1f} (R own, G other)", pr))

    # crossing marks
    top = np.zeros((34, W, 3), np.uint8)
    fit = []
    if crossings:
        tree = cKDTree(F)
        lay = layers["fibre_layer"] if layers else {}
        for r in crossings:
            if r["verdict"] == "no overlap" or name not in (r["vertical"], r["horizontal"]):
                continue
            at = np.array([float(v) for v in r["at"].strip("[]").split()])
            col = int(tree.query(at)[1])
            other = r["horizontal"] if name == r["vertical"] else r["vertical"]
            off = float(r["normal_offset_vx"])
            exp = ((lay.get(r["horizontal"], 0) - lay.get(r["vertical"], 0)) * layers["spacing_vx"] + ply) if layers else ply
            good = abs(off - exp) <= 4
            colour = (0, 200, 0) if good else (0, 0, 255)
            cv2.line(top, (col, 20), (col, 33), colour, 2)
            cv2.putText(top, other.split("_")[0] + other.split("_")[1] + f" {off:+.0f}", (max(col - 30, 0), 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
            fit.append((col, other, off, good))

    rows = [np.hstack([np.zeros((34, margin, 3), np.uint8), top])]
    for label, img in panels:
        lab = np.zeros((Hh, margin, 3), np.uint8)
        cv2.putText(lab, label, (4, Hh // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([lab, img]))
        rows.append(np.zeros((gapr, margin + W, 3), np.uint8))
    out = np.vstack(rows)
    for c in range(0, W, 200):                                                          # arc ruler
        cv2.putText(out, f"{c}", (margin + c, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (160, 160, 160), 1, cv2.LINE_AA)
    cv2.imwrite(str(fdir / f"{name}_ribbon.png"), out)

    # crossing-family peak depth per 50-vx window
    win = 50
    rows_csv, flags = [], []
    for c0 in range(0, W - win + 1, win):
        prof = profile[:, c0:c0 + win].mean(1)
        j = int(np.argmax(prof))
        peak, strength = float(prof_d[j]), float(prof[j])
        p = F[c0 + win // 2]
        status = "weak" if strength < 0.08 else ("ok" if abs(peak - other_ply) <= 4 else "OFF")
        rows_csv.append((c0, *p.round(1), peak, round(strength, 3), status))
        flags.append(status)
    with open(fdir / f"{name}_plydepth.csv", "w") as fh:
        fh.write("arc_vx,x,y,z,crossing_family_peak_depth_vx,peak_prob,status\n")
        for rr in rows_csv:
            fh.write(",".join(str(v) for v in rr) + "\n")
    okn, offn, weakn = flags.count("ok"), flags.count("OFF"), flags.count("weak")
    off_runs = []
    for rr in rows_csv:
        if rr[-1] == "OFF":
            if off_runs and off_runs[-1][1] == rr[0] - win:
                off_runs[-1][1] = rr[0]
                off_runs[-1][2].append(rr[4])
            else:
                off_runs.append([rr[0], rr[0], [rr[4]], rr[1:4]])
    print(f"{name}: {arc[-1]:.0f} vx; crossing family at the expected ply depth ({other_ply:+.1f}) in "
          f"{okn}/{okn + offn} windows ({weakn} too weak to tell)"
          + (f"; crossings fitting the layer model {sum(f[3] for f in fit)}/{len(fit)}" if fit else ""))
    for a, b, peaks, xyz0 in off_runs:
        print(f"    off: arc {a}-{b + win} vx near x,y,z {list(xyz0)}  peak depth {np.median(peaks):+.0f} vx")


names = [n for n in args.names.split(",") if n] or sorted(p.name[:-len(".path.npy")] for p in fdir.glob("*.path.npy"))
for nm in names:
    render(nm)
