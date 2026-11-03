"""Per-vertical direction and shake report: are verticals following the sheet's canonical vertical direction?

Usage: python vert_audit.py <fiber_dir> [--good-shake 2.5]
Canonical direction = slope dx/dz (and dy/dz) of the low-shake verticals, interpolated across x by a median of
the nearest good ones. Prints per vertical: fitted dx/dz, canonical dx/dz there, deviation, shake, and the
max lateral deviation from its own straight fit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("--good-shake", type=float, default=2.5)
args = ap.parse_args()
rows = []
for p in sorted(Path(args.fibers).glob("v_*.path.npy")):
    a = np.load(p).astype(np.float64)
    C = a[:, :3]
    if len(C) < 100:
        continue
    k = np.polyfit(C[:, 2], C[:, 0], 1)
    curv = np.polyfit(C[:, 2], C[:, 0], 2)[0] * 1e4
    res = C[:, 0] - np.polyval(k, C[:, 2])
    sh = core.shakiness(C, "v")[0]
    rows.append((p.name[:-9], np.median(C[:, 0]), k[0], curv, np.abs(res).max(), sh, a[:, 3].mean() if a.shape[1] > 3 else np.nan,
                 C[:, 2].min(), C[:, 2].max()))
good = [r for r in rows if r[5] < args.good_shake]
gx = np.array([r[1] for r in good])
gk = np.array([r[2] for r in good])
print(f"{len(good)} good verticals (shake < {args.good_shake}); their dx/dz: " + " ".join(f"{k:+.2f}" for k in gk))
for r in rows:
    near = np.argsort(np.abs(gx - r[1]))[1:5] if r in good else np.argsort(np.abs(gx - r[1]))[:4]
    can = np.median(gk[near]) if len(near) else np.nan
    flag = "  <-- off" if abs(r[2] - can) > 0.08 or r[5] >= args.good_shake else ""
    print(f"{r[0]:22s} x~{r[1]:5.0f} z {r[7]:.0f}..{r[8]:.0f}  dx/dz {r[2]:+.3f} canon {can:+.3f}  curv {r[3]:+.2f}  "
          f"maxdev {r[4]:5.1f}  shake {r[5]:.1f}  p {r[6]:.2f}{flag}")
