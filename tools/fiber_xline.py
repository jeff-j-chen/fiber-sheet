import csv
import sys
from collections import defaultdict

# per-fibre crossing strings ordered along the fibre: fiber_xline.py <crossings.csv> [p] [S] [h|v]
rows = list(csv.DictReader(open(sys.argv[1])))
p = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0
S = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
fam = sys.argv[4] if len(sys.argv) > 4 else "h"
by = defaultdict(list)
for r in rows:
    if r["verdict"] == "no overlap":
        continue
    at = [float(v) for v in r["at"].strip("[]").split()]
    off = float(r["normal_offset_vx"])
    me, other, pos = (r["horizontal"], r["vertical"], at[0]) if fam == "h" else (r["vertical"], r["horizontal"], at[2])
    by[me].append((pos, other.split("_")[1], off))
for me in sorted(by, key=lambda n: n.split("_")[1]):
    xs = sorted(by[me])
    good = sum(abs(o - p) < S / 2 for _, _, o in xs)
    s = " ".join(f"{o2}@{pos:.0f}:{o:+.0f}{'' if abs(o - p) < S / 2 else '!'}" for pos, o2, o in xs)
    print(f"{me[:18]:18s} {good:2d}/{len(xs):2d}  {s}")
