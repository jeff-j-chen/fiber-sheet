"""Trace one fibre in 3D through the AFV prediction store, then measure a sheet against it.

Usage:
  python trace_fiber_3d.py <x> <y> <z> <h|v> <out.json> [--segment DIR] [--threshold 0.35]
         [--max-gap 15] [--step 2] [--cp-spacing 60]

The trace uses only the predictions (probability ridge + predicted fibre axis), never the sheet.
With --segment, every fibre point gets its distance to the sheet, written to <out>.sheet_dist.csv,
and a summary of where the sheet stays on / leaves the fibre is printed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fiber_meta import FIBER_COORDS
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("x", type=float)
ap.add_argument("y", type=float)
ap.add_argument("z", type=float)
ap.add_argument("family", choices=["h", "v"])
ap.add_argument("out")
ap.add_argument("--segment")
ap.add_argument("--threshold", type=float, default=0.35)
ap.add_argument("--max-gap", type=int, default=15, help="steps allowed below threshold before stopping")
ap.add_argument("--step", type=float, default=2.0)
ap.add_argument("--cp-spacing", type=float, default=60.0)
ap.add_argument("--max-length", type=float, default=20000.0)
ap.add_argument("--reacquire-radius", type=float, default=6.0,
                help="sideways search radius (vx) when the ridge is lost; 0 disables")
ap.add_argument("--reacquire-ahead", type=float, default=20.0, help="how far ahead (vx) to search")
ap.add_argument("--seed-radius", type=float, default=4.0, help="how far from the given point to look for the fibre")
ap.add_argument("--lock-layer", action="store_true",
                help="use the sheet's orientation as the layer normal and limit each step's motion along it")
ap.add_argument("--normal-slack", type=float, default=1.0, help="max re-centring along the normal per step (vx)")
ap.add_argument("--side-slack", type=float, default=3.0, help="max sideways re-centring within the layer per step (vx)")
ap.add_argument("--axis", type=float, nargs=3, help="initial walking direction (x y z); sign matters with --one-way")
ap.add_argument("--one-way", action="store_true", help="walk only along --axis, not both ways")
ap.add_argument("--inertia", type=float, default=0.0, help="0..1 weight of the previous heading (straighter path)")
ap.add_argument("--xmono", action="store_true",
                help="horizontals: advance monotonically in x, follow y with a slope prior, keep z nearly still")
ap.add_argument("--y-slack", type=float, default=1.0, help="--xmono: max y re-centring per step (vx)")
ap.add_argument("--z-slack", type=float, default=0.25, help="--xmono: max z re-centring per step (vx)")
ap.add_argument("--zmono", action="store_true",
                help="verticals: advance monotonically in z, follow y with a slope prior, keep x nearly still")
ap.add_argument("--x-slack", type=float, default=0.5, help="--zmono: max x re-centring per step (vx)")
ap.add_argument("--slope-rate", type=float, default=0.1,
                help="--xmono/--zmono: how fast the free-axis slope prior follows the fibre (0..1; higher = sharper bends)")
ap.add_argument("--guide", help="--xmono/--zmono: height-field tifxyz y=f(x,z) (e.g. fiber_surface_tps.py output) "
                "whose local slope replaces the y slope prior, so the walk bends with the sheet")
args = ap.parse_args()
if args.lock_layer and not args.segment:
    raise SystemExit("--lock-layer needs --segment (its orientation gives the layer normal)")

predictor = Predictor()
root = predictor.root
prob_arr = root["hz" if args.family == "h" else "vt"]
dir_arr = root["dir"]
C = prob_arr.chunks[0]
cache = {}
new_tiles = [0]


def chunk(key):
    """Cached (prob, dir) for a 96^3 chunk, predicting it first if it isn't in the store yet."""
    if key not in cache:
        if len(cache) > 400:
            cache.pop(next(iter(cache)))
        if predictor.ensure(key):
            new_tiles[0] += 1
        p = np.zeros((C, C, C), np.float32)
        d = np.zeros((3, C, C, C), np.float32)
        if predictor.in_volume(key):
            sl = tuple(slice(k * C, (k + 1) * C) for k in key)
            pp = prob_arr[sl].astype(np.float32) / 255.0      # edge tiles are smaller than C
            p[:pp.shape[0], :pp.shape[1], :pp.shape[2]] = pp
            d[:, :pp.shape[0], :pp.shape[1], :pp.shape[2]] = dir_arr[(slice(None),) + sl].astype(np.float32) / 127.0
        cache[key] = (p, d)
    return cache[key]


def sample(xyz, want_dir=False):
    """Trilinear probability (and nearest-voxel direction) at xyz points (n,3)."""
    xyz = np.atleast_2d(xyz)
    zyx = xyz[:, ::-1]
    base = np.floor(zyx).astype(np.int64)
    frac = zyx - base
    out = np.zeros(len(xyz), np.float32)
    for corner in np.ndindex(2, 2, 2):
        c = base + np.array(corner)
        w = np.prod(np.where(np.array(corner), frac, 1 - frac), axis=1)
        keys = c // C
        for key in {tuple(k) for k in keys}:
            m = (keys == key).all(1)
            p, _ = chunk(key)
            loc = c[m] - np.array(key) * C
            out[m] += w[m] * p[loc[:, 0], loc[:, 1], loc[:, 2]]
    if not want_dir:
        return out
    near = np.rint(zyx).astype(np.int64)
    dirs = np.zeros((len(xyz), 3), np.float32)
    keys = near // C
    for key in {tuple(k) for k in keys}:
        m = (keys == key).all(1)
        _, d = chunk(key)
        loc = near[m] - np.array(key) * C
        dirs[m] = d[:, loc[:, 0], loc[:, 1], loc[:, 2]].T
    return out, dirs


def local_axis(pos, prev):
    """Predicted fibre axis near pos, sign-aligned with prev and probability-weighted."""
    offs = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3), -1).reshape(-1, 3).astype(np.float32)
    p, d = sample(pos[None] + offs, want_dir=True)
    good = np.linalg.norm(d, axis=1) > 0.3
    if not good.any():
        return prev
    d = d[good] * np.sign(d[good] @ prev)[:, None]
    a = (d * (p[good][:, None] + 1e-3)).sum(0)
    n = np.linalg.norm(a)
    return prev if n < 1e-6 else a / n


def perp_basis(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


normal_tree = normal_vecs = None
if args.lock_layer:
    _seg = Path(args.segment)
    _sx, _sy = json.loads((_seg / "meta.json").read_text())["scale"]
    _g = np.stack([cv2.imread(str(_seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
    _gv = (_g != -1).all(-1)
    _uu, _vv = np.meshgrid(np.arange(0, (_g.shape[1] - 1) / _sx, 4.0) * _sx,
                           np.arange(0, (_g.shape[0] - 1) / _sy, 4.0) * _sy)
    _uu, _vv = _uu.astype(np.float32), _vv.astype(np.float32)
    _sp = np.stack([cv2.remap(_g[..., i], _uu, _vv, cv2.INTER_LINEAR) for i in range(3)], -1)
    _ok = cv2.remap(_gv.astype(np.float32), _uu, _vv, cv2.INTER_LINEAR) > 0.999
    _n = np.cross(np.gradient(_sp, axis=1), np.gradient(_sp, axis=0))
    _n /= np.linalg.norm(_n, axis=-1, keepdims=True) + 1e-9
    _ok &= np.isfinite(_n).all(-1)
    normal_tree, normal_vecs = cKDTree(_sp[_ok]), _n[_ok]


def layer_basis(pos, a):
    """(in-layer sideways, layer normal) unit vectors, both perpendicular to the fibre axis a."""
    if normal_tree is None:
        return perp_basis(a)
    _, k = normal_tree.query(pos, k=12)
    n = normal_vecs[k].mean(0)
    n -= (n @ a) * a
    if np.linalg.norm(n) < 1e-6:
        return perp_basis(a)
    n /= np.linalg.norm(n)
    return np.cross(a, n), n


def offsets(side, normal, step_side, step_normal):
    """Candidate (sideways, normal) offsets on an ellipse."""
    ii, jj = np.meshgrid(np.arange(-side, side + 1e-6, step_side), np.arange(-normal, normal + 1e-6, step_normal))
    o = np.stack([ii.ravel(), jj.ravel()], 1)
    return o[(o[:, 0] / max(side, 1e-6)) ** 2 + (o[:, 1] / max(normal, 1e-6)) ** 2 <= 1.0 + 1e-6]


if args.lock_layer:
    disc = offsets(args.side_slack, args.normal_slack, 0.5, 0.5)
else:
    disc = np.array([(i, j) for i in np.arange(-3, 3.01, 0.5) for j in np.arange(-3, 3.01, 0.5) if i * i + j * j <= 9.0])


def recentre(pos, a):
    """Probability-weighted centre of the fibre core in the plane perpendicular to a."""
    e1, e2 = layer_basis(pos, a)
    cand = pos[None] + disc[:, :1] * e1 + disc[:, 1:] * e2
    p = sample(cand)
    pmax = float(p.max())
    if pmax <= 0:
        return pos, 0.0
    w = np.clip(p - 0.5 * pmax, 0, None)
    return (cand * w[:, None]).sum(0) / w.sum(), pmax


r = args.reacquire_radius
if r <= 0:
    wide = np.zeros((0, 2))
elif args.lock_layer:
    wide = offsets(r, 1.5 * args.normal_slack, 1.0, 0.5)
else:
    wide = np.array([(i, j) for i in np.arange(-r, r + 0.01, 1.0) for j in np.arange(-r, r + 0.01, 1.0)
                     if i * i + j * j <= r * r])


def reacquire(pos, a):
    """Search further ahead and wider for the ridge; prefer candidates near the straight extension."""
    if len(wide) == 0:
        return None
    e1, e2 = layer_basis(pos, a)
    for dist in np.arange(2 * args.step, args.reacquire_ahead + 0.01, args.step):
        cand = pos[None] + a * dist + wide[:, :1] * e1 + wide[:, 1:] * e2
        p = sample(cand)
        score = p - 0.04 * np.linalg.norm(wide, axis=1)
        k = int(np.argmax(score))
        if p[k] >= args.threshold + 0.15:
            new, pmax = recentre(cand[k], a)
            if (new - pos) @ a < 0.5 * dist:
                continue                                # recentred onto something beside/behind us
            if local_axis(new, a) @ a < np.cos(np.radians(30)):
                continue                                # a crossing or neighbouring fibre, not this one
            return new, pmax, dist, float(np.linalg.norm(wide[k]))
    return None


# ---- seed -----------------------------------------------------------------------------------
target = np.array([args.x, args.y, args.z])
R = args.seed_radius
g = np.stack(np.meshgrid(*[np.arange(-R, R + 0.01, 0.5 if R <= 6 else 1.0)] * 3), -1).reshape(-1, 3)
g = g[np.linalg.norm(g, axis=1) <= R]
g = g[np.argsort(np.linalg.norm(g, axis=1))]
ps = sample(target + g)
strong = np.flatnonzero(ps >= max(0.6, args.threshold))     # nearest confident voxel, not the strongest one
if len(strong) == 0:
    other = root["vt" if args.family == "h" else "hz"]
    zyx = np.rint(target[::-1] + g[:, ::-1]).astype(int)
    lo, hi = zyx.min(0), zyx.max(0) + 1
    blk = other[tuple(slice(a, b) for a, b in zip(lo, hi))]
    op = blk[tuple((zyx - lo).T)] / 255.0
    raise SystemExit(f"no {args.family} fibre within {R:g} vx of the point (max {args.family}-prob "
                     f"{ps.max():.2f}; the other family peaks at {op.max():.2f}, "
                     f"{np.linalg.norm(g[np.argmax(op)]):.1f} vx away)")
seed = target + g[strong[0]]
init = np.array([1.0, 0, 0]) if args.family == "h" else np.array([0, 0, 1.0])
if args.axis:
    init = np.array(args.axis, float) / np.linalg.norm(args.axis)
axis0 = local_axis(seed, init)
seed, _ = recentre(seed, axis0)
print(f"seed {seed.round(1).tolist()} ({np.linalg.norm(seed - target):.1f} vx from the point), "
      f"{args.family}-prob {sample(seed)[0]:.2f}, axis {axis0.round(2).tolist()}")

# ---- walk both ways ---------------------------------------------------------------------------
halves, reasons, jumps = [], [], []
other = np.zeros((0, 3))
turn_steps = int(round(100 / args.step))


guide = None
if args.guide:
    from scipy.interpolate import LinearNDInterpolator
    _G = np.stack([cv2.imread(str(Path(args.guide) / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1)
    _G = _G[(_G != -1).all(-1)].astype(np.float64)
    guide = LinearNDInterpolator(_G[:, [0, 2]], _G[:, 1])


def guide_slope(pos, step, free):
    """Free-axis change of the guide surface over one step (None outside it)."""
    a, b = pos.copy(), pos + step
    g = guide(np.array([[a[0], a[2]], [b[0], b[2]]]))
    return None if not np.isfinite(g).all() else float(g[1] - g[0])


def walk_xmono(sign, adv=0, free=1, lock=2, free_slack=None, lock_slack=None):
    """Step along +/- the adv axis; re-centre in the free axis (slope prior) and, barely, the lock axis."""
    free_slack = args.y_slack if free_slack is None else free_slack
    lock_slack = args.z_slack if lock_slack is None else lock_slack
    sx = sign * (1.0 if axis0[adv] >= 0 else -1.0)
    ys = axis0[free] / axis0[adv] if abs(axis0[adv]) > 0.3 else 0.0
    zs = 0.0
    # centre is measured over a fibre-sized window, but the applied move per step is clamped
    dy, dz = np.meshgrid(np.arange(-4, 4.01, 0.5), np.arange(-4, 4.01, 0.5))
    grid = np.zeros((dy.size, 3))
    grid[:, free], grid[:, lock] = dy.ravel(), dz.ravel()

    def centre(c):
        p = sample(c + grid)
        pmax = float(p.max())
        w = np.clip(p - 0.5 * pmax, 0, None)
        return ((grid * w[:, None]).sum(0) / w.sum() if pmax > 0 else np.zeros(3)), pmax

    pos = seed.copy()
    for _ in range(5):
        off, _ = centre(pos)
        pos = pos + off
    r = max(args.reacquire_radius, free_slack)
    wy, wz = np.meshgrid(np.arange(-r, r + 1e-6, 0.5), np.arange(-2 * lock_slack, 2 * lock_slack + 1e-6, 0.25))
    wide_x = np.zeros((wy.size, 3))
    wide_x[:, free], wide_x[:, lock] = wy.ravel(), wz.ravel()
    pts, probs, gap, travelled, reason = [], [], 0, 0.0, "max length"
    while travelled < args.max_length / 2:
        step = np.zeros(3)
        step[adv], step[free], step[lock] = sx * args.step, ys * sx * args.step, zs * sx * args.step
        if guide is not None:
            gs = guide_slope(pos, np.where(np.arange(3) == free, 0.0, step), free)
            if gs is not None:
                step[free] = gs
        off, pmax = centre(pos + step)
        if pmax >= args.threshold:
            off[free] = np.clip(off[free], -free_slack, free_slack)
            off[lock] = np.clip(off[lock], -lock_slack, lock_slack)
            new = pos + step + off
            gap = 0
        else:
            new = None
            for ahead in np.arange(2 * args.step, args.reacquire_ahead + 0.01, args.step):
                c2 = pos + step / args.step * ahead + wide_x
                p2 = sample(c2)
                k = int(np.argmax(p2 - 0.04 * np.abs(wide_x[:, free])))
                if p2[k] >= args.threshold + 0.15:
                    new, pmax = c2[k], float(p2[k])
                    jumps.append((sign, new.copy(), ahead, float(abs(wide_x[k, free]))))
                    gap = 0
                    break
            if new is None:
                gap += 1
                new = pos + step
                if gap > args.max_gap:
                    reason = f"probability below {args.threshold} for {args.max_gap} steps"
                    break
        ddx = new[adv] - pos[adv]
        ys = (1 - args.slope_rate) * ys + args.slope_rate * (new[free] - pos[free]) / ddx
        zs = 0.98 * zs + 0.02 * (new[lock] - pos[lock]) / ddx
        travelled += float(np.linalg.norm(new - pos))
        pos = new
        pts.append(pos.copy())
        probs.append(pmax)
    return pts, probs, gap, reason, pos


for sign in ((1,) if args.one_way else (1, -1)):
    if args.xmono or args.zmono:
        pts, probs, gap, reason, pos = (walk_xmono(sign) if args.xmono else
                                        walk_xmono(sign, 2, 1, 0, args.y_slack, args.x_slack))
        k = len(pts) - gap
        halves.append((np.array(pts[:max(k, 0)]).reshape(-1, 3), np.array(probs[:max(k, 0)])))
        reasons.append(f"{'+' if sign > 0 else '-'} end at {pos.round(1).tolist()}: {reason}")
        continue
    pos, a = seed.copy(), axis0 * sign
    pts, probs, gap, length = [], [], 0, 0.0
    best_progress, a_prev = 0.0, a.copy()
    reason = "max length"
    while length < args.max_length / 2:
        guess = pos + a * args.step
        new, pmax = recentre(guess, a)
        if pmax < args.threshold:
            found = reacquire(pos, a)
            if found is not None:
                new, pmax, ahead, side = found
                jumps.append((sign, new.copy(), ahead, side))
                gap = 0
            else:
                gap += 1
                new = guess                            # coast straight across a faint stretch
                if gap > args.max_gap:
                    reason = f"probability below {args.threshold} for {args.max_gap} steps"
                    break
        else:
            gap = 0
        move = new - pos
        if np.linalg.norm(move) < 0.25 * args.step or move @ a <= 0:
            reason = f"stalled/reversed (step {np.linalg.norm(move):.2f} vx, prob {pmax:.2f})"
            break
        length += np.linalg.norm(move)
        a = local_axis(new, move / np.linalg.norm(move))
        if args.inertia > 0:
            a = args.inertia * a_prev + (1 - args.inertia) * a
            a /= np.linalg.norm(a)
        a_prev = a
        pos = new
        pts.append(pos.copy())
        probs.append(pmax)
        if args.one_way:
            progress = (pos - seed) @ axis0
            best_progress = max(best_progress, progress)
            if progress < best_progress - 30:
                reason = "heading back against the start direction"
                cut = next(i for i, q in enumerate(pts) if (q - seed) @ axis0 >= best_progress - 1e-6)
                del pts[cut + 1:], probs[cut + 1:]
                gap = 0
                break
        if len(pts) >= 2 * turn_steps:
            d1 = pts[-1] - pts[-turn_steps]
            d0 = pts[-turn_steps] - pts[-2 * turn_steps]
            cosang = d1 @ d0 / (np.linalg.norm(d1) * np.linalg.norm(d0) + 1e-9)
            if cosang < np.cos(np.radians(45)):
                reason = f"turned {np.degrees(np.arccos(np.clip(cosang, -1, 1))):.0f} deg within 200 vx"
                del pts[-turn_steps:], probs[-turn_steps:]
                gap = 0
                break
        if len(pts) % 10 == 0:
            older = np.array(pts[:-int(60 / args.step)]) if len(pts) > 60 / args.step else np.zeros((0, 3))
            near = np.concatenate([older, other])
            if len(near) and np.min(np.linalg.norm(near - pos, axis=1)) < 6.0:
                reason = "looped back onto the traced fibre"
                break
    k = len(pts) - gap                                  # drop the trailing faint stretch
    halves.append((np.array(pts[:max(k, 0)]).reshape(-1, 3), np.array(probs[:max(k, 0)])))
    other = halves[-1][0]
    reasons.append(f"{'+' if sign > 0 else '-'} end at {pos.round(1).tolist()}: {reason}")
print("\n".join("  " + r for r in reasons))
for sign, p, ahead, side in jumps:
    print(f"  re-acquired ({'+' if sign > 0 else '-'} side) at {p.round(1).tolist()}: "
          f"{ahead:.0f} vx ahead, {side:.1f} vx sideways  <- review this spot")

fwd, back = halves[0], (halves[1] if len(halves) > 1 else (np.zeros((0, 3)), np.zeros(0)))
xyz = np.concatenate([back[0][::-1], seed[None], fwd[0]])
prob = np.concatenate([back[1][::-1], [sample(seed)[0]], fwd[1]])
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
seed_i = len(back[0])
print(f"traced {arc[-1]:.0f} vx ({arc[-1] * 9.362 / 1000:.2f} mm): {arc[seed_i]:.0f} vx one way, "
      f"{arc[-1] - arc[seed_i]:.0f} vx the other; mean prob {prob.mean():.2f}, "
      f"{(prob < args.threshold).mean() * 100:.1f}% of steps bridged; {new_tiles[0]} new tiles predicted")

# ---- vc3d_fiber (version 1; VC3D imports it and exports version 4) --------------------------------
ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], args.cp_spacing)),
                                 [len(arc) - 1, seed_i]]))
line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
cps, lps = xyz[ctrl].round(3).tolist(), xyz[line].round(3).tolist()
cps[int(np.where(ctrl == seed_i)[0][0])] = target.tolist()
lps[int(np.where(line == seed_i)[0][0])] = target.tolist()
out = Path(args.out)
out.write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS, "control_points": cps, "line_points": lps}, indent=1))
np.save(out.with_suffix(".path.npy"), np.column_stack([xyz, prob, arc]).astype(np.float32))
print(f"wrote {out} ({len(cps)} control points, {len(lps)} line points)")

# ---- how the sheet relates to this fibre ---------------------------------------------------------
if args.segment:
    seg = Path(args.segment)
    sm = json.loads((seg / "meta.json").read_text())
    ssx, ssy = sm["scale"]
    grid = np.stack([cv2.imread(str(seg / f"{c}.tif"), cv2.IMREAD_UNCHANGED) for c in "xyz"], -1).astype(np.float32)
    gv = (grid != -1).all(-1)
    gh, gw = gv.shape
    uu, vv = np.meshgrid(np.arange(0, (gw - 1) / ssx, 1.0) * ssx, np.arange(0, (gh - 1) / ssy, 1.0) * ssy)
    sp = np.stack([cv2.remap(grid[..., i], uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR)
                   for i in range(3)], -1)
    sok = cv2.remap(gv.astype(np.float32), uu.astype(np.float32), vv.astype(np.float32), cv2.INTER_LINEAR) > 0.999
    vv_i, uu_i = np.nonzero(sok)
    tree = cKDTree(sp[vv_i, uu_i])
    dist, idx = tree.query(xyz)
    su, sv = uu_i[idx], vv_i[idx]
    rows = np.column_stack([arc, xyz, prob, dist, su, sv])
    np.savetxt(out.with_suffix(".sheet_dist.csv"), rows, delimiter=",", fmt="%.2f",
               header="arc_vx,x,y,z,prob,dist_to_sheet_vx,sheet_u,sheet_v", comments="")
    on = dist <= 4.0
    print(f"sheet vs fibre: {on.mean() * 100:.1f}% of the fibre within 4 vx of the sheet; "
          f"median {np.median(dist):.1f} vx, max {dist.max():.1f} vx")
    # contiguous off-sheet stretches
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~on).astype(int), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if arc[e - 1] - arc[s] >= 20:
            print(f"  off sheet for {arc[e - 1] - arc[s]:.0f} vx at arc {arc[s]:.0f}-{arc[e - 1]:.0f} "
                  f"(max {dist[s:e].max():.1f} vx away; sheet px u {su[s]}-{su[e - 1]}, v {sv[s]}-{sv[e - 1]})")
    print(f"wrote {out.with_suffix('.sheet_dist.csv')}")
