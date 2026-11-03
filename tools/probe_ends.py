"""Probability profile ahead of each end of a traced fibre (diagnostic)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from afv_core import Predictor  # noqa: E402

path = np.load(sys.argv[1])
family = sys.argv[2] if len(sys.argv) > 2 else "h"
xyz = path[:, :3].astype(np.float64)
pred = Predictor()
arr = pred.root["hz" if family == "h" else "vt"]


def block_max(centre, axis, dist, radius=8):
    """Max probability in a small disc perpendicular to axis at centre + axis * dist."""
    t = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(axis, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    best = 0.0
    for i in np.arange(-radius, radius + 0.1, 1.0):
        for j in np.arange(-radius, radius + 0.1, 1.0):
            p = centre + axis * dist + i * e1 + j * e2
            z, y, x = np.rint(p[::-1]).astype(int)
            pred.ensure((z // 96, y // 96, x // 96))
            best = max(best, arr[z, y, x] / 255.0)
    return best


for name, end, inner in (("+ end", xyz[-1], xyz[-40]), ("- end", xyz[0], xyz[39])):
    axis = (end - inner) / np.linalg.norm(end - inner)
    prof = [block_max(end, axis, d) for d in range(0, 201, 10)]
    print(f"{name} {end.round(0).tolist()} axis {axis.round(2).tolist()}")
    print("   dist ahead: " + " ".join(f"{d:4d}" for d in range(0, 201, 10)))
    print("   max prob  : " + " ".join(f"{p:4.2f}" for p in prof))
