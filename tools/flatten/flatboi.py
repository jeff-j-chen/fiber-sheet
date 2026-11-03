"""Windows stand-in for VC3D's flatboi: libigl SLIM with the same setup and outputs.

Usage: python flatboi.py <mesh.obj> [iters=50] [energy=symmetric_dirichlet|conformal] [--tol=<x>]
Writes <mesh>_flatboi.obj next to the input and copies <mesh>.obj.griduv if present.
"""
import shutil
import sys
from pathlib import Path

import igl
import numpy as np

ENERGIES = {
    "symmetric_dirichlet": igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
    "symmetric-dirichlet": igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
    "sd": igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
    "conformal": igl.MappingEnergyType.CONFORMAL,
}


def area_stats(V, F, uv):
    """Per-triangle 2D/3D area ratio, normalised so a perfectly equal-area map gives 1 everywhere."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    p, q, r = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    area2 = 0.5 * ((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (r[:, 0] - p[:, 0]) * (q[:, 1] - p[:, 1]))
    ok = area3 > 1e-12
    ratio = (np.abs(area2[ok]) / np.abs(area2[ok]).sum()) / (area3[ok] / area3[ok].sum())
    flipped = int(np.count_nonzero(area2[ok] * np.sign(area2[ok].sum()) < 0))
    return np.percentile(ratio, [5, 50, 95]), flipped


def main(argv):
    pos = [a for a in argv[1:] if not a.startswith("--")]
    tol = next((float(a.split("=", 1)[1]) for a in argv[1:] if a.startswith("--tol=")), 0.0)
    if not pos or len(pos) > 3 or not pos[0].lower().endswith(".obj"):
        print(__doc__, file=sys.stderr)
        return 1
    obj = Path(pos[0])
    iters = int(pos[1]) if len(pos) > 1 else 50
    energy_name = (pos[2] if len(pos) > 2 else "symmetric_dirichlet").lower()
    if energy_name not in ENERGIES:
        print(f"Unknown energy '{energy_name}'. Supported: symmetric_dirichlet, conformal", file=sys.stderr)
        return 1

    V, TC, _N, F, FTC, _FN = igl.readOBJ(str(obj))
    if F.shape[1] != 3 or len(V) == 0:
        print("Expecting a non-empty triangle mesh.", file=sys.stderr)
        return 1

    bnd = igl.boundary_loop(F)
    if len(bnd) == 0:
        print("Mesh has no boundary (closed surface).", file=sys.stderr)
        return 1

    if TC.size and FTC.size:
        # Same "original" initial condition as flatboi: per-vertex UVs from the grid UVs.
        uv = np.zeros((len(V), 2))
        uv[F.ravel()] = TC[FTC.ravel(), :2]
        print("Initial condition: grid UVs from vc_tifxyz2obj")
    else:
        bnd_circle = igl.map_vertices_to_circle(V, bnd)
        uv = igl.harmonic(V, F, bnd, bnd_circle, 1)
        print("No per-corner UVs in input; using harmonic initial condition.")

    pct, flips = area_stats(V, F, uv)
    print(f"start  area ratio p5/p50/p95 = {pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}  flipped={flips}")

    F32 = np.asfortranarray(F, dtype=np.int32)
    bnd32 = np.ascontiguousarray(bnd, dtype=np.int32)
    # soft_p=0 like flatboi: boundary is free, so the map is not pinned to the grid outline.
    data = igl.slim_precompute(np.asfortranarray(V, dtype=np.float64), F32,
                               np.asfortranarray(uv, dtype=np.float64), ENERGIES[energy_name],
                               bnd32, np.asfortranarray(uv[bnd], dtype=np.float64), 0.0)

    prev_err = None
    for it in range(1, iters + 1):
        uv = np.asarray(igl.slim_solve(data, 1))[:, :2]
        if not np.isfinite(uv).all():
            print(f"SLIM diverged at iter {it}/{iters}; stopping.", file=sys.stderr)
            return 2
        print(f"PROGRESS {it}/{iters}", flush=True)
        if it % 10 == 0 or it == iters:
            pct, flips = area_stats(V, F, uv)
            err = pct[2] / pct[0]
            print(f"[it {it}] area ratio p5/p50/p95 = {pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}  flipped={flips}", flush=True)
            if tol > 0 and prev_err is not None and abs(prev_err - err) / prev_err < tol:
                print(f"Converged at iter {it}")
                break
            prev_err = err

    uv = uv - uv.min(axis=0)
    out = obj.with_name(obj.stem + "_flatboi.obj")
    corner_uv = uv[F.ravel()]
    ftc = np.arange(F.size, dtype=np.int64).reshape(F.shape)
    igl.writeOBJ(str(out), V, F.astype(np.int64), TC=corner_uv, FTC=ftc)
    print(f"Wrote: {out}")

    griduv = Path(str(obj) + ".griduv")
    if griduv.exists():
        shutil.copyfile(griduv, str(out) + ".griduv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
