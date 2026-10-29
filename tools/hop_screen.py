"""Band-continuity hop screen for many fibres: reports gaps / merged layers / component changes along each.

Usage: python hop_screen.py <fiber_dir> <names regex> <lo:hi> [--thr 0.85] [--fam auto]
For each fibre: the axis positions where the hz/vt band under it breaks (no voxel > thr), where its band merges
with a neighbouring layer (two runs in the same column, horizontals only) and where the component changes.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("names")
ap.add_argument("rng")
ap.add_argument("--thr", default="0.85")
args = ap.parse_args()
here = Path(__file__).resolve().parent
pat = re.compile(args.names)
for p in sorted(Path(args.dir).glob("*.path.npy")):
    n = p.name[:-9]
    if not pat.search(n):
        continue
    fam = "hz" if n[0] == "h" else "vt"
    r = subprocess.run([sys.executable, str(here / "band_check3d.py"), str(p), args.rng, "--thr", args.thr,
                        "--fam", fam, "--every", "5"], capture_output=True, text=True)
    events, prev, gap0 = [], None, None
    for ln in r.stdout.splitlines():
        m = re.match(r"\S\s+(-?\d+) .*component (\d+) \(size (\d+)\)(.*)", ln)
        if not m:
            continue
        pos, comp, size, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
        if comp == 0:
            gap0 = pos if gap0 is None else gap0
            continue
        if gap0 is not None:
            events.append(f"gap {gap0}-{pos}")
            gap0 = None
        if prev is not None and comp != prev:
            events.append(f"band change @{pos}")
        if rest.count(",") >= 1 and "same component at y [" in rest and rest.strip().endswith("]"):
            ys = rest.split("[")[1].rstrip("]").split(",")
            if len(ys) >= 2:
                events.append(f"merged @{pos}")
        prev = comp
    # collapse consecutive identical event kinds
    out, last = [], None
    for e in events:
        k = e.split()[0]
        if k == "merged" and last == "merged":
            continue
        out.append(e)
        last = k
    print(f"{n:24s} {'clean' if not out else ' | '.join(out)}")
