"""Trace a new fibre that sticks to a (trusted) sheet: uv guide between neighbouring fibres, narrow DP, strand lock.

The guide runs along the sheet's own grid (uv of a flattened tifxyz): for a horizontal, row v as a function of column u is
blended between the two neighbouring horizontals bracketing the seed (or parallel to the nearest one); verticals likewise
with u(v). The 3D guide is sheet(uv) + off * normal, with off = 0 for horizontals and -9 for verticals, and the path is
refined by a Viterbi pass (fibre probability + CT) with a narrow depth band, so it cannot leave the sheet's layer.

Usage: python surf_trace.py <sheet name> <out_dir> --fam h|v --seed x,y,z --nbr <dir> [--nbr <dir>...]
       [--name N] [--off 0] [--depth 4] [--ct 0.3] [--urange lo:hi]
Prints extent, probability, deviation from the sheet and the seed's offset from it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fiber_dpcore as core  # noqa: E402
from fiber_vs_surface import surface_grid, surface_points, uv_to_xyz  # noqa: E402

PATHS = Path(r"C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths")


class Sheet:
    def __init__(self, name):
        ref = PATHS / name
        self.X, self.Y, self.Z, self.ok = surface_grid(ref)
        self.Su, self.UV = surface_points(ref, step=1.5, return_uv=True)
        self.tree = cKDTree(self.Su)
        gi = [np.gradient(np.where(self.ok, A, np.nan), axis=0) for A in (self.X, self.Y, self.Z)]
        gj = [np.gradient(np.where(self.ok, A, np.nan), axis=1) for A in (self.X, self.Y, self.Z)]
        NG = np.cross(np.stack(gi, -1), np.stack(gj, -1))
        NG /= np.linalg.norm(NG, axis=-1, keepdims=True)
        NG *= np.where(NG[..., 1:2] > 0, -1.0, 1.0)
        self.NG = np.nan_to_num(NG)
        self.QOK = self.ok[:-1, :-1] & self.ok[1:, :-1] & self.ok[:-1, 1:] & self.ok[1:, 1:]

    def uv_of(self, P):
        d, i = self.tree.query(P)
        return self.UV[i], d

    def xyz(self, uv):
        return np.column_stack([uv_to_xyz(A, uv[:, 0], uv[:, 1]) for A in (self.X, self.Y, self.Z)])

    def normal(self, uv):
        n = np.column_stack([uv_to_xyz(self.NG[..., c], uv[:, 0], uv[:, 1]) for c in range(3)])
        bad = np.linalg.norm(n, axis=1) < 1e-6
        n[bad] = (0.0, -1.0, 0.0)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def on_sheet(self, uv):
        i = np.clip(np.floor(uv[:, 0]).astype(int), 0, self.QOK.shape[0] - 1)
        j = np.clip(np.floor(uv[:, 1]).astype(int), 0, self.QOK.shape[1] - 1)
        inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= self.QOK.shape[0]) & (uv[:, 1] <= self.QOK.shape[1])
        return self.QOK[i, j] & inside

    def signed_offset(self, P):
        uv, _ = self.uv_of(P)
        return ((P - self.xyz(uv)) * self.normal(uv)).sum(1)


class Line:
    """Cubic fit a(b) of one neighbouring fibre in sheet coordinates, linear beyond its ends."""

    def __init__(self, a, b):
        o = np.argsort(b)
        a, b = a[o], b[o]
        self.lo, self.hi = float(b[0]), float(b[-1])
        self.c = np.polyfit(b, a, 3)
        d = np.polyder(self.c)
        self.vlo, self.slo = np.polyval(self.c, self.lo), np.polyval(d, self.lo)
        self.vhi, self.shi = np.polyval(self.c, self.hi), np.polyval(d, self.hi)

    def __call__(self, b):
        b = np.asarray(b, float)
        v = np.polyval(self.c, np.clip(b, self.lo, self.hi))
        v = np.where(b < self.lo, self.vlo + self.slo * (b - self.lo), v)
        return np.where(b > self.hi, self.vhi + self.shi * (b - self.hi), v)


def neighbour_lines(sheet, dirs, fam):
    lines = []
    for d in dirs:
        for p in sorted(Path(d).glob(f"{fam}_*.path.npy")):
            A = np.load(p)[:, :3].astype(np.float64)[::3]
            uv, dist = sheet.uv_of(A)
            m = dist < 20
            if m.sum() < 60:
                continue
            uv = uv[m]
            a, b = (uv[:, 0], uv[:, 1]) if fam == "h" else (uv[:, 1], uv[:, 0])
            if np.ptp(b) < 60:
                continue
            lines.append((p.name[:-9], Line(a, b)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("out")
    ap.add_argument("--fam", choices=["h", "v"], required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--nbr", action="append", required=True)
    ap.add_argument("--name")
    ap.add_argument("--off", type=float)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ldepth", type=int, default=3, help="depth band (vx) of the strand lock around the sheet plane")
    ap.add_argument("--side", type=int, default=3)
    ap.add_argument("--ct", type=float, default=0.3)
    ap.add_argument("--exclude", default="", help="neighbour names to ignore")
    args = ap.parse_args()
    fam = args.fam
    off = args.off if args.off is not None else (0.0 if fam == "h" else -9.0)
    sheet = Sheet(args.sheet)
    seed = np.array(args.seed.split(","), float)
    uv0, d0 = sheet.uv_of(seed[None])
    uv0 = uv0[0]
    soff = float(sheet.signed_offset(seed[None])[0])
    print(f"seed {seed.round(0)}: sheet distance {d0[0]:.1f}, signed offset {soff:+.1f}, uv (row {uv0[0]:.0f}, col {uv0[1]:.0f})")
    a0, b0 = (uv0[0], uv0[1]) if fam == "h" else (uv0[1], uv0[0])
    ex = [e for e in args.exclude.split(",") if e]
    lines = [(n, L) for n, L in neighbour_lines(sheet, args.nbr, fam) if not any(n.startswith(e) for e in ex)]
    vals = sorted(((float(L(b0)), n, L) for n, L in lines), key=lambda t: t[0])
    below = [t for t in vals if t[0] <= a0]
    above = [t for t in vals if t[0] > a0]
    if below and above:
        (va, na, La), (vb, nb, Lb) = below[-1], above[0]
        w = (a0 - va) / (vb - va)
        guide = lambda b: (1 - w) * La(b) + w * Lb(b)
        print(f"guide between {na} and {nb} at fraction {w:.2f} (rows {va:.1f} / {vb:.1f})")
    else:
        v, n, L = (below[-1] if below else above[0])
        dv = a0 - v
        guide = lambda b: L(b) + dv
        print(f"guide parallel to {n} (row offset {dv:+.1f})")

    def walk(sign):
        bs, pts = [], []
        b = b0
        while True:
            a = float(guide(b))
            uv = np.array([[a, b]] if fam == "h" else [[b, a]])
            if not sheet.on_sheet(uv)[0]:
                break
            bs.append(b)
            b += sign * 0.1
            if len(bs) > 6000:
                break
        return bs

    bs_f, bs_b = walk(1), walk(-1)
    refn = lambda P: sheet.normal(sheet.uv_of(P)[0])
    # a seed clicked on the sheet marks where the fibre is along the sheet; its depth is the target plane
    start = (sheet.xyz(uv0[None]) + off * sheet.normal(uv0[None]))[0] if abs(soff - off) > 3 else seed.copy()
    print(f"start {start.round(0)} ({'on target plane' if abs(soff - off) > 3 else 'at seed'})")

    def trace_half(bs_half):
        a = guide(np.array(bs_half))
        uv = np.column_stack([a, bs_half]) if fam == "h" else np.column_stack([bs_half, a])
        G = core.resample(sheet.xyz(uv) + off * sheet.normal(uv), 2.0)
        # the seed is the strand: start exactly there (shifted onto the target plane), follow it outwards
        G = G + (start - G[0]) * np.clip(1 - np.arange(len(G)) / 40.0, 0, 1)[:, None]
        G[0] = start
        return core.lock_strand(G, fam, sigma=150.0, depth=args.ldepth, side=24, scost=5.0, refn=refn, pin_start=True)

    Cf, Pf = trace_half(bs_f) if len(bs_f) > 20 else (seed[None], np.ones(1))
    Cb, Pb = trace_half(bs_b) if len(bs_b) > 20 else (seed[None], np.ones(1))
    C = np.vstack([Cb[::-1], Cf[1:]])
    P = np.concatenate([Pb[::-1], Pf[1:]])
    so = sheet.signed_offset(C) - off
    name = args.name or f"{fam}_{seed[0]:.0f}_{seed[1]:.0f}_{seed[2]:.0f}"
    core.save(args.out, name, C, P)
    ax = 0 if fam == "h" else 2
    dseed = np.linalg.norm(C - seed, axis=1).min()
    bad = np.abs(so) > 6
    print(f"{name}: {ax and 'z' or 'x'} {C[:, ax].min():.0f}..{C[:, ax].max():.0f}, {len(C)} pts, mean p {P.mean():.2f}, weak {np.mean(P < 0.3) * 100:.0f}%, "
          f"offset-from-target median {np.median(so):+.1f} max |.| {np.abs(so).max():.1f} ({bad.mean() * 100:.0f}% beyond 6), seed distance {dseed:.1f}")


if __name__ == "__main__":
    main()

