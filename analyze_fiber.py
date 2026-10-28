"""Trace one fibre from an (x, y, z) point and analyse it against a fibre set and a sheet.

Steps
  1. Decide the fibre family (horizontal / vertical) from the model's probabilities near the point,
     unless --family is given. Missing prediction tiles are computed on the fly (needs the GPU model).
  2. Trace the fibre. Horizontals use the x-monotonic, z-locked tracer (--xmono); verticals use the
     free 3D tracer. The result is written into --fibers as <name>.json (vc3d_fiber v1, importable in
     VC3D's Fibers panel) and <name>.path.npy (columns x, y, z, prob, arc).
  3. If the folder holds fibres of the other family, compute all crossings, fit the layer model and print
     where the new fibre sits relative to every crossing fibre (same sheet or how many layers off), its
     crossing-verified same-sheet span, and rebuild fibers_3d.html.

Example
  python analyze_fiber.py 3394 3491 13385 --fibers data/my_fibers --segment <tifxyz folder>
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
sys.path.insert(0, str(TOOLS))


def run(script, *a, show=True):
    r = subprocess.run([sys.executable, str(TOOLS / script), *map(str, a)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{script} failed:\n{r.stdout}\n{r.stderr}")
    if show:
        print(r.stdout.rstrip())
    return r.stdout


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("x", type=float)
ap.add_argument("y", type=float)
ap.add_argument("z", type=float)
ap.add_argument("--fibers", required=True, help="fibre-set folder; the new fibre is written here (created if missing)")
ap.add_argument("--segment", required=True, help="tifxyz sheet folder (x.tif, y.tif, z.tif, meta.json); "
                                                 "its orientation defines the layer normal")
ap.add_argument("--family", choices=["auto", "h", "v"], default="auto")
ap.add_argument("--name", help="output name (default <family>_<x>_<y>_<z>)")
ap.add_argument("--seed-radius", type=float, default=4.0, help="how far (vx) from the point to look for the fibre")
ap.add_argument("--free", action="store_true", help="trace horizontals with the free 3D tracer instead of --xmono")
ap.add_argument("--xmax", type=float, default=6700.0, help="ignore fibre parts beyond this x in the analysis")
args = ap.parse_args()
fdir = Path(args.fibers)
fdir.mkdir(parents=True, exist_ok=True)

# 1. family -------------------------------------------------------------------------------------
fam = args.family
if fam == "auto":
    from afv_core import CORE, Predictor
    pred = Predictor()
    p = np.array([args.x, args.y, args.z])
    lo = np.floor(p - args.seed_radius).astype(int)
    hi = np.ceil(p + args.seed_radius).astype(int) + 1
    for t in {tuple(np.where(b, hi - 1, lo)[::-1] // CORE) for b in np.ndindex(2, 2, 2)}:
        pred.ensure(t)
    sl = tuple(slice(a, b) for a, b in zip(lo[::-1], hi[::-1]))
    ph = pred.root["hz"][sl].max() / 255
    pv = pred.root["vt"][sl].max() / 255
    fam = "h" if ph >= pv else "v"
    print(f"family: horizontal-prob {ph:.2f}, vertical-prob {pv:.2f} within {args.seed_radius:g} vx -> "
          f"{'horizontal' if fam == 'h' else 'vertical'}")
name = args.name or f"{fam}_{args.x:.0f}_{args.y:.0f}_{args.z:.0f}"

# 2. trace --------------------------------------------------------------------------------------
extra = ["--xmono", "--reacquire-ahead", "60"] if fam == "h" and not args.free else []
out = run("trace_fiber_3d.py", args.x, args.y, args.z, fam, fdir / f"{name}.json",
          "--seed-radius", args.seed_radius, *extra, show=False)
for line in out.splitlines():
    if line.startswith(("seed", "  +", "  -", "traced")):
        print(line)
P = np.load(fdir / f"{name}.path.npy")
print(f"x {P[:, 0].min():.0f}..{P[:, 0].max():.0f}, y {P[:, 1].min():.0f}..{P[:, 1].max():.0f}, "
      f"z {P[:, 2].min():.0f}..{P[:, 2].max():.0f}")

# 3. crossings / layers -----------------------------------------------------------------------------
other = list(fdir.glob(("v" if fam == "h" else "h") + "_*.path.npy"))
if not other:
    print(f"\nno {'vertical' if fam == 'h' else 'horizontal'} fibres in {fdir}: nothing to cross-check yet")
    sys.exit()
run("fiber_crossings.py", fdir, args.segment, "--xmax", args.xmax, show=False)
run("fiber_layers.py", fdir, "--min-spacing", 14, "--max-spacing", 20, show=False)
L = json.loads((fdir / "layers.json").read_text())
S, p = L["spacing_vx"], L["ply_offset_vx"]
print(f"\nlayer model: spacing {S:g} vx, ply offset {p:+g} vx (horizontal on the +normal side of its vertical)")
print(f"crossings of {name} (offset = horizontal minus vertical along the sheet normal):")
rows = [r for r in csv.DictReader(open(fdir / "crossings.csv"))
        if name in (r["vertical"], r["horizontal"]) and r["verdict"] != "no overlap"]
if not rows:
    print("  none (the fibre never comes within 60 vx of a fibre of the other family)")
for r in sorted(rows, key=lambda r: float(r["at"].strip("[]").split()[0])):
    o = float(r["normal_offset_vx"])
    dl = int(round((o - p) / S))
    verdict = "same sheet" if dl == 0 else f"{abs(dl)} layer(s) {'+' if dl > 0 else '-'}normal"
    partner = r["vertical"] if fam == "h" else r["horizontal"]
    print(f"  {partner:22s} at {r['at']:28s} offset {o:+6.1f} vx -> {verdict}")
lay = L["fibre_layer"].get(name)
main = max(L["layers"].items(), key=lambda kv: len(kv[1]))
print(f"layer of {name}: {lay}; the main sheet (most fibres) is layer {main[0]} with {len(main[1])} fibres")
if fam == "h":
    cov = run("fiber_coverage.py", fdir, "--ply", p, "--spacing", S, show=False)
    print(next((ln for ln in cov.splitlines() if ln.startswith(name)), "").split("|")[0].rstrip())
run("fiber_view3d.py", fdir, args.segment, "--xmax", args.xmax)
