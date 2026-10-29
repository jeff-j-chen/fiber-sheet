"""Extend a fibre past one of its ends with the monotonic tracer (--xmono for h, --zmono for v).

The fibre is cut --backoff vx before the chosen end, traced onward one way from there, and the extension
is spliced on. The original is kept as <name>.path.npy.orig / <name>.json.orig.

Usage: python extend_mono.py <fibre>.path.npy --end {+,-} [--backoff 50] [--extra "..."]
  --end + extends towards larger x (h) / larger z (v); - towards smaller.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from fiber_meta import FIBER_COORDS

ap = argparse.ArgumentParser()
ap.add_argument("fiber")
ap.add_argument("--end", choices=["+", "-"], required=True)
ap.add_argument("--backoff", type=float, default=50.0)
ap.add_argument("--extra", default="--reacquire-ahead 60 --seed-radius 2")
args = ap.parse_args()
src = Path(args.fiber)
name = src.name[:-len(".path.npy")]
fam = name[0]
ax = 0 if fam == "h" else 2
a = np.load(src).astype(np.float64)
if a[-1, ax] < a[0, ax]:
    a = a[::-1]
a[:, 4] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1))])
sign = 1 if args.end == "+" else -1
if sign > 0:
    i_cut = int(np.searchsorted(a[:, 4], a[-1, 4] - args.backoff))
else:
    i_cut = int(np.searchsorted(a[:, 4], args.backoff))
axis = ["0", "0", "0"]
axis[ax] = str(float(sign))
tmp = src.parent / f"_{name}_ext.json"
cmd = [sys.executable, str(Path(__file__).resolve().parent / "trace_fiber_3d.py"), *map(str, a[i_cut, :3]), fam, str(tmp),
       "--xmono" if fam == "h" else "--zmono", "--one-way", "--axis", *axis, *args.extra.split()]
r = subprocess.run(cmd, capture_output=True, text=True)
npy = tmp.with_suffix(".path.npy")
if not npy.exists():
    sys.exit(f"no extension: {(r.stdout + r.stderr).strip()[-300:]}")
e = np.load(npy).astype(np.float64)
npy.unlink()
tmp.unlink()
keep = a[:i_cut + 1] if sign > 0 else a[i_cut:]
P = np.concatenate([keep[:, :4], e[1:, :4]]) if sign > 0 else np.concatenate([e[1:, :4][::-1], keep[:, :4]])
for suf in (".path.npy", ".json"):
    f = src.parent / f"{name}{suf}"
    if f.exists() and not Path(str(f) + ".orig").exists():
        shutil.copy(f, str(f) + ".orig")
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1))])
np.save(src, np.column_stack([P, arc]).astype(np.float32))
ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
(src.parent / f"{name}.json").write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                                                     "control_points": P[ctrl, :3].round(3).tolist(),
                                                     "line_points": P[line, :3].round(3).tolist()}, indent=1))
end = next((ln.strip() for ln in r.stdout.splitlines() if "end at" in ln), "")
print(f"{name}: {'xyz'[ax]} {a[:, ax].min():.0f}..{a[:, ax].max():.0f} -> {P[:, ax].min():.0f}..{P[:, ax].max():.0f}  [{end}]")
