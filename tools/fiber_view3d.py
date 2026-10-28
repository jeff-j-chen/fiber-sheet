"""Interactive 3D view (HTML) and a static overview (PNG) of traced fibres, crossings and the sheet.

Usage: python fiber_view3d.py <fiber_dir> <segment_dir> [--xmax 5100]
Needs crossings.csv / groups.json from fiber_crossings.py. Writes fibers_3d.html and fibers_overview.png.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("segment")
ap.add_argument("--xmax", type=float, default=5100.0)
args = ap.parse_args()
fdir = Path(args.fibers)

groups = json.loads((fdir / "groups.json").read_text())
group_of = {n: k for k, g in enumerate(groups) for n in g["fibers"]}
layer_info = json.loads((fdir / "layers.json").read_text()) if (fdir / "layers.json").exists() else None
if layer_info:
    group_of = {n: int(l) for n, l in layer_info["fibre_layer"].items()}
palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
rows = list(csv.DictReader(open(fdir / "crossings.csv")))
if layer_info:
    # same sheet = horizontal on the canonical (+normal) side of the vertical, closer than the next sheet;
    # the opposite side is where the touching sheet's vertical sits, so it never counts
    S, p = layer_info["spacing_vx"], layer_info["ply_offset_vx"]
    for r in rows:
        if r["verdict"] != "no overlap":
            r["verdict"] = "crosses" if abs(float(r["normal_offset_vx"]) - p) < S / 2 else "misses"
label_word = "layer" if layer_info else "group"

fibers = {}
for p in sorted(fdir.glob("*.path.npy")):
    name = p.name[: -len(".path.npy")]
    xyz = np.load(p)[:, :3]
    fibers[name] = xyz[xyz[:, 0] <= args.xmax]

# sheet, decimated
seg = Path(args.segment)
grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
step = max(1, grid.shape[1] // 120)
g = grid[::step, ::step]
gv = (g != -1).all(-1)
g = np.where(gv[..., None], g, np.nan)

fig = go.Figure()
fig.add_trace(go.Surface(x=g[..., 0], y=g[..., 1], z=g[..., 2], opacity=0.25, showscale=False,
                         colorscale=[[0, "lightgray"], [1, "lightgray"]], name="current sheet",
                         hoverinfo="skip", showlegend=True))
for name, xyz in fibers.items():
    k = group_of.get(name, -1)
    colour = palette[k % len(palette)] if k >= 0 else "black"
    dash = "solid" if name.startswith("v") else "dot"
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                               line=dict(color=colour, width=6, dash=dash),
                               name=f"{name} ({label_word} {k})", hovertext=name, hoverinfo="text"))
for verdict, colour, label in (("crosses", "green", "same-sheet crossing"), ("misses", "red", "passes (different sheet)"),
                               ("unclear", "orange", "unclear")):
    sel = [r for r in rows if r["verdict"] == verdict]
    if not sel:
        continue
    at = np.array([[float(v) for v in r["at"].strip("[]").split()] for r in sel])
    fig.add_trace(go.Scatter3d(x=at[:, 0], y=at[:, 1], z=at[:, 2], mode="markers",
                               marker=dict(size=5 if verdict == "crosses" else 4, color=colour),
                               name=label,
                               hovertext=[f"{r['vertical']} x {r['horizontal']}: {r['distance_vx']} vx apart, "
                                          f"normal offset {r['normal_offset_vx']}" for r in sel],
                               hoverinfo="text"))
fig.update_layout(title=f"Traced fibres (solid = vertical, dotted = horizontal; colour = {label_word})",
                  scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                  legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
fig.write_html(fdir / "fibers_3d.html", include_plotlyjs=True)

# static overview: three orthogonal projections
fig2, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (i, j, lab) in zip(axes, ((0, 2, "x-z"), (0, 1, "x-y"), (1, 2, "y-z"))):
    ax.scatter(g[..., i].ravel(), g[..., j].ravel(), s=0.3, c="lightgray", rasterized=True)
    for name, xyz in fibers.items():
        k = group_of.get(name, -1)
        ax.plot(xyz[:, i], xyz[:, j], ls="-" if name.startswith("v") else ":",
                color=palette[k % len(palette)] if k >= 0 else "black", lw=1.6)
    for verdict, colour in (("crosses", "green"), ("misses", "red")):
        at = np.array([[float(v) for v in r["at"].strip("[]").split()] for r in rows if r["verdict"] == verdict])
        if len(at):
            ax.scatter(at[:, i], at[:, j], s=25, c=colour, zorder=5)
    ax.set_title(f"{lab} projection (gray = current sheet)")
    ax.set_xlabel("xyz"[i]); ax.set_ylabel("xyz"[j]); ax.set_aspect("equal")
axes[0].invert_yaxis()
fig2.tight_layout()
fig2.savefig(fdir / "fibers_overview.png", dpi=110)
print("wrote", fdir / "fibers_3d.html", "and", fdir / "fibers_overview.png")
