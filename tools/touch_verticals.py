"""Re-trace each vertical from its crossing with a given horizontal (seed = point on the horizontal, tool shifts it to the -9 plane).

Usage: python touch_verticals.py <sheet> <horizontal.path.npy> <vfull_dir> <nbr_dir> <out_dir>
Prints, per vertical, the crossing seed and how far the new trace is from the already-written fibre where they overlap.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = r"C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
HERE = Path(__file__).resolve().parent


def load(p, ax):
    A = np.load(p).astype(float)
    return A[np.argsort(A[:, ax])]


def main():
    sheet, hp, vfull, nbr, out = sys.argv[1:6]
    H = load(hp, 0)
    for vp in sorted(Path(vfull).glob("v_*.path.npy"), key=lambda p: int(p.name[2:6])):
        V = load(vp, 2)
        xs = np.arange(H[0, 0], H[-1, 0], 1.0)
        hz = np.interp(xs, H[:, 0], H[:, 2])
        f = np.interp(hz, V[:, 2], V[:, 0]) - xs
        s = np.where(np.diff(np.sign(f)) != 0)[0]
        if not len(s):
            print(vp.name[:6], "no intersection with the horizontal")
            continue
        i = s[np.argmin(np.abs(xs[s] - V[:, 0].mean()))] if len(s) > 1 else s[0]
        seed = np.array([xs[i], np.interp(xs[i], H[:, 0], H[:, 1]), hz[i]])
        name = f"v_{seed[0]:.0f}_{seed[1]:.0f}_{seed[2]:.0f}"
        r = subprocess.run([PY, str(HERE / "surf_trace.py"), sheet, out, "--fam", "v", "--seed", ",".join(f"{c:.1f}" for c in seed),
                            "--nbr", nbr, "--exclude", vp.name[:6], "--ldepth", "2", "--name", name],
                           capture_output=True, text=True, cwd=HERE)
        print(vp.name[:6], "seed", seed.round(0), r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        new = load(Path(out) / f"{name}.path.npy", 2)
        cur_p = Path(nbr) / vp.name
        if cur_p.exists():
            cur = load(cur_p, 2)
            zz = np.arange(cur[0, 2] + 20, min(cur[-1, 2], new[-1, 2]) - 20, 20.0)
            d = np.linalg.norm(np.stack([np.interp(zz, new[:, 2], new[:, c]) - np.interp(zz, cur[:, 2], cur[:, c]) for c in range(3)], 1), axis=1)
            print("   vs written over z %.0f..%.0f: median %.1f max %.1f vx; new z range %.0f..%.0f" % (zz[0], zz[-1], np.median(d), d.max(), new[0, 2], new[-1, 2]))


main()
