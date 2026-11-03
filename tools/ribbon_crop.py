import sys

import cv2
import numpy as np

# crop <name>_ribbon.png to the arc columns whose primary coordinate lies in [lo, hi]
# usage: ribbon_crop.py <fiber_dir> <name> <lo> <hi> <out.png>
d, name, lo, hi, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
a = np.load(f"{d}/{name}.path.npy").astype(np.float64)
ax = 0 if name[0] == "h" else 2
arc = a[:, 4]
s = np.arange(0, arc[-1], 1.0)
pos = np.interp(s, arc, a[:, ax])
cols = np.flatnonzero((pos >= lo) & (pos <= hi))
img = cv2.imread(f"{d}/{name}_ribbon.png")
margin = 210
keep = np.concatenate([np.arange(margin), margin + cols[cols < img.shape[1] - margin]])
crop = np.ascontiguousarray(img[:, keep])
for c in range(margin, crop.shape[1], 100):
    p = pos[cols[0] + c - margin]
    cv2.putText(crop, f"{p:.0f}", (c, crop.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, crop)
print(out, crop.shape)
