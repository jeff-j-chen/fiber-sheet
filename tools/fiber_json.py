"""(Re)write the VC3D v1 .json for every <name>.path.npy in a folder (control points every 60 vx, line every 4 vx).

Usage: python fiber_json.py <fiber_dir> [--missing-only]
"""
import json
import sys
from pathlib import Path

import numpy as np

from fiber_meta import FIBER_COORDS

d = Path(sys.argv[1])
only_missing = "--missing-only" in sys.argv
n = 0
for p in sorted(d.glob("*.path.npy")):
    out = p.with_name(p.name[:-len(".path.npy")] + ".json")
    if only_missing and out.exists():
        continue
    a = np.load(p).astype(np.float64)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1))])
    ctrl = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 60.0)), [len(arc) - 1]]))
    line = np.unique(np.concatenate([np.searchsorted(arc, np.arange(0, arc[-1], 4.0)), ctrl]))
    out.write_text(json.dumps({"type": "vc3d_fiber", "version": 1, **FIBER_COORDS,
                               "control_points": a[ctrl, :3].round(3).tolist(),
                               "line_points": a[line, :3].round(3).tolist()}, indent=1))
    n += 1
print(f"wrote {n} json files in {d}")
