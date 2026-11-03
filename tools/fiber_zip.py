"""Zipper a horizontal through a hard stretch between two trusted horizontals (bottom/top).

Inside [lo, hi] (x), the guide is interpolated between the trusted fibres: both are parametrised by arc
fraction u over the stretch, and guide(u) = B(u) + w(u) (T(u) - B(u)). w comes from where the fibre being
rebuilt (--fiber) sits relative to B and T at the ends of the stretch (or from --z when it is new); ends the
fibre covers are pinned. The strand is then found with the single-strand Viterbi (narrow depth band, wide but
expensive side band), so it follows the layer the trusted fibres define and cannot hop to a neighbour strand.

Usage: python fiber_zip.py <fiber_dir> <out_dir> --bottom h_a --top h_b --range lo:hi
       (--fiber h_c | --z Z --name h_new) [--depth 5] [--side 20] [--scost 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fibers")
ap.add_argument("out")
ap.add_argument("--bottom", required=True)
ap.add_argument("--top", required=True)
ap.add_argument("--range", required=True)
ap.add_argument("--fiber")
ap.add_argument("--z", type=float)
ap.add_argument("--name")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--side", type=int, default=20)
ap.add_argument("--scost", type=float, default=5.0)
ap.add_argument("--no-pin", default="", help="'lo', 'hi' or 'lo,hi': ends not to pin even if covered")
args = ap.parse_args()
fd = Path(args.fibers)
core.set_lateral(fd)
lo, hi = map(float, args.range.split(":"))


def load(prefix):
    p = next(fd.glob(prefix + "*.path.npy"))
    a = np.load(p).astype(np.float64)
    return p.name[:-9], a[np.argsort(a[:, 0])]


def stretch(a, x0, x1, n):
    m = (a[:, 0] >= x0) & (a[:, 0] <= x1)
    s = a[m, :3]
    if len(s) < 10 or s[0, 0] > x0 + 15 or s[-1, 0] < x1 - 15:
        sys.exit(f"trusted fibre does not cover {x0:.0f}..{x1:.0f}")
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))])
    u = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(u, arc, s[:, c]) for c in range(3)], 1)


_, B = load(args.bottom)
_, T = load(args.top)
n = int((hi - lo) / 2) + 1
Bs, Ts = stretch(B, lo, hi, n), stretch(T, lo, hi, n)
nopin = args.no_pin.split(",")
E = None
if args.fiber:
    name, E = load(args.fiber)
else:
    name = args.name


def w_at(P, i):
    d = Ts[i] - Bs[i]
    return float((P - Bs[i]) @ d / (d @ d))


pins = {}
if E is not None:
    for key, x, i in (("lo", lo, 0), ("hi", hi, n - 1)):
        if key not in nopin and E[0, 0] <= x <= E[-1, 0]:
            pins[key] = np.array([np.interp(x, E[:, 0], E[:, c]) for c in range(3)])
if args.z is not None:
    w0 = w1 = (args.z - Bs[n // 2, 2]) / (Ts[n // 2, 2] - Bs[n // 2, 2])
else:
    wl = w_at(pins["lo"], 0) if "lo" in pins else None
    wh = w_at(pins["hi"], n - 1) if "hi" in pins else None
    if wl is None and wh is None:
        zz = np.interp((lo + hi) / 2, E[:, 0], E[:, 2])
        wl = wh = (zz - Bs[n // 2, 2]) / (Ts[n // 2, 2] - Bs[n // 2, 2])
    w0 = wl if wl is not None else wh
    w1 = wh if wh is not None else wl
u = np.linspace(0, 1, n)[:, None]
G = Bs + (w0 + (w1 - w0) * u) * (Ts - Bs)
if "lo" in pins:
    G += (1 - u) * (pins["lo"] - G[0])
if "hi" in pins:
    G += u * (pins["hi"] - G[-1])
G = core.resample(G, 2.0)
C, P = core.lock_strand(G, "h", sigma=80.0, depth=args.depth, side=args.side, scost=args.scost,
                        pin_start="lo" in pins, pin_end="hi" in pins)
if E is not None:
    left = E[E[:, 0] < lo, :3]
    right = E[E[:, 0] > hi, :3]
    pl = E[E[:, 0] < lo, 3] if E.shape[1] > 3 else np.ones(len(left))
    pr = E[E[:, 0] > hi, 3] if E.shape[1] > 3 else np.ones(len(right))
    C = np.vstack([left, C, right])
    P = np.concatenate([pl, P, pr])
core.save(args.out, name, C, P)
weak = P < 0.3
print(f"{name}: zipped {lo:.0f}..{hi:.0f} between {args.bottom} / {args.top} (w {w0:.2f}->{w1:.2f}, "
      f"pins {','.join(pins) or 'none'}), stretch mean p {P[(C[:, 0] >= lo) & (C[:, 0] <= hi)].mean():.2f}, "
      f"weak {np.mean(weak[(C[:, 0] >= lo) & (C[:, 0] <= hi)]) * 100:.0f}%")
