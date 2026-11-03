"""Brute-force candidate fibres at many levels and score each by its crossings.

Usage: python brute_levels.py <base_dir> <out_dir> --levels a,b,c :: <generator command with {z} and {out}>
For each level the generator command (e.g. fiber_rebuild_region.py ... --targets z{z}) is run with {out}
replaced by out_dir/z<level>; its fibre(s) are added to a copy of base_dir, crossings are computed, and the
candidate's crossing line (good/total, offsets) and generator summary are printed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
py = sys.executable
i = sys.argv.index("::")
base, out = Path(sys.argv[1]), Path(sys.argv[2])
levels = sys.argv[sys.argv.index("--levels") + 1].split(",")
gen = sys.argv[i + 1:]
for z in levels:
    od = out / f"z{z}"
    if od.exists():
        shutil.rmtree(od)
    cmd = [a.replace("{z}", z).replace("{out}", str(od)) for a in gen]
    r = subprocess.run([py] + cmd, capture_output=True, text=True)
    summary = " | ".join(l.strip() for l in r.stdout.splitlines() if "p " in l)[-160:]
    cands = list(od.glob("*.path.npy")) if od.exists() else []
    if not cands:
        print(f"z{z}: no fibre ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else summary})")
        continue
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for p in base.glob("*.path.npy"):
            shutil.copy(p, td / p.name)
        for p in cands:
            shutil.copy(p, td / p.name)
        subprocess.run([py, str(here / "fiber_crossings_fn.py"), str(td)], capture_output=True)
        x = subprocess.run([py, str(here / "fiber_xline.py"), str(td / "crossings.csv"), "9", "14",
                            cands[0].name[0]], capture_output=True, text=True).stdout
        names = [c.name[:-9][:18] for c in cands]
        lines = [l for l in x.splitlines() if any(l.startswith(n) for n in names)]
        print(f"z{z}: {summary}\n      " + ("\n      ".join(lines) if lines else "(no crossings)"))
