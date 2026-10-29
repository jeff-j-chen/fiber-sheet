"""Coordinate binding written into every vc3d_fiber JSON, so VC3D doesn't ask for the fibre units.

Same fields VC3D itself writes: coordinates are full-resolution (level 0) voxels of volume 20250728152254.
Run directly to add them to existing files: python fiber_meta.py <dir_or_json> [...]
"""
import json
import sys
from pathlib import Path

FIBER_COORDS = {
    "vc_open_data_coordinate_space": "PHerc0846A/20250728152254@L0",
    "vc_open_data_source_coordinate_level": 0,
    "vc_open_data_source_coordinate_scale_factor": 1,
    "vc_open_data_source_original_resolution": 9.362,
    "vc_open_data_source_path": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/"
                                "volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr",
}

if __name__ == "__main__":
    n = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        for f in ([p] if p.is_file() else sorted(p.glob("*.json"))):
            try:
                d = json.loads(f.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(d, dict) or d.get("type") != "vc3d_fiber":
                continue
            if all(d.get(k) == v for k, v in FIBER_COORDS.items()):
                continue
            d.update(FIBER_COORDS)
            f.write_text(json.dumps(d, indent=1))
            n += 1
    print(f"stamped {n} fibre file(s)")
