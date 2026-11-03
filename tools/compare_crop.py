import sys
from pathlib import Path

import cv2
import numpy as np

pred = Path(sys.argv[1])
render = Path(sys.argv[2])
crop_x, crop_y = int(sys.argv[3]), int(sys.argv[4])

vt = cv2.imread(str(pred / "vt_prob.tif"), cv2.IMREAD_UNCHANGED)
hz = cv2.imread(str(pred / "hz_prob.tif"), cv2.IMREAD_UNCHANGED)
ct = np.mean([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted(render.glob("*.tif"))], 0)
h, w = ct.shape
ct8 = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

sl = (slice(crop_y, crop_y + h), slice(crop_x, crop_x + w))
pred_rgb = np.zeros((h, w, 3), np.uint8)
pred_rgb[..., 2] = hz[sl]
pred_rgb[..., 1] = vt[sl]
ct_rgb = cv2.cvtColor(ct8, cv2.COLOR_GRAY2BGR)

print("vertical  prob in crop: mean %.3f, >0.5: %.1f%%" % (vt[sl].mean() / 255, (vt[sl] > 127).mean() * 100))
print("horizontal prob in crop: mean %.3f, >0.5: %.1f%%" % (hz[sl].mean() / 255, (hz[sl] > 127).mean() * 100))

stacked = np.vstack([ct_rgb, pred_rgb, cv2.addWeighted(ct_rgb, 0.6, pred_rgb, 0.8, 0)])
cv2.imwrite(str(pred / "compare_point.png"), stacked)
print("wrote", pred / "compare_point.png")
