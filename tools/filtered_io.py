"""Read/write VC3D fibre collections (filtered.json) as <name>.path.npy files and back.

Usage: python filtered_io.py export <filtered.json> <out_dir>       one <name>.path.npy (x,y,z,p=1,arc) per fibre
       python filtered_io.py update <filtered.json> <src_dir> name[,name...]
          replace/add the named fibres from <src_dir>/<name>.path.npy (backup written once as <file>.bak-<date>)
"""
import copy
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np


def line_of(F):
    ctrl = np.unique(np.concatenate([np.arange(0, len(F), 60), [len(F) - 1]]))
    line = np.unique(np.concatenate([np.arange(0, len(F), 4), ctrl]))
    return F[ctrl], F[line]


def main():
    cmd, jp = sys.argv[1], Path(sys.argv[2])
    j = json.loads(jp.read_text())
    cols = j["point_collections"]
    if cmd == "export":
        out = Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        for c in cols:
            L = np.array(c["line_points"], float)
            arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))])
            s = np.arange(0, arc[-1], 1.0)
            F = np.stack([np.interp(s, arc, L[:, k]) for k in range(3)], 1)
            ax = 0 if c["name"][0] == "h" else 2
            if F[-1, ax] < F[0, ax]:
                F = F[::-1]
            np.save(out / f"{c['name']}.path.npy", np.column_stack([F, np.ones(len(F)), s]).astype(np.float32))
            print(c["name"], len(F))
    elif cmd == "remove":
        shutil.copy(jp, jp.with_name(jp.name + time.strftime(".bak-%Y%m%d-%H%M%S")))
        gone = set(sys.argv[3].split(","))
        j["point_collections"] = [c for c in cols if c["name"] not in gone]
        jp.write_text(json.dumps(j, indent=2))
        print("removed", sorted(gone & {c["name"] for c in cols}))
    elif cmd == "update":
        src = Path(sys.argv[3])
        shutil.copy(jp, jp.with_name(jp.name + time.strftime(".bak-%Y%m%d-%H%M%S")))
        for name in sys.argv[4].split(","):
            F = np.load(src / f"{name}.path.npy")[:, :3].astype(np.float64)
            ctrl, line = line_of(F)
            hit = [c for c in cols if c["name"] == name]
            if hit:
                c = hit[0]
            else:
                c = copy.deepcopy(next((x for x in cols if x["name"][0] == name[0]), cols[0]))
                c["name"] = name
                c["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "afv-" + name))
                c["filename"] = c["uuid"] + ".json"
                cols.append(c)
            c["control_points"] = [{"position": [round(v, 3) for v in p]} for p in ctrl.tolist()]
            c["control_point_ids"] = list(range(len(ctrl)))
            c["line_points"] = [[round(v, 3) for v in p] for p in line.tolist()]
            print("updated" if hit else "added", name, len(ctrl), "control points,", len(line), "line points")
        jp.write_text(json.dumps(j, indent=2))


main()

