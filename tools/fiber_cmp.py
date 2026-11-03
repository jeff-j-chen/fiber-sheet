import sys

import numpy as np

# compare fibres along the primary axis: fiber_cmp.py <step> a.path.npy b.path.npy ...  (h: by x; v: by z)
step = float(sys.argv[1])
paths = sys.argv[2:]
fam = paths[0].replace("\\", "/").split("/")[-1][0]
pa = 0 if fam == "h" else 2
A = []
for p in paths:
    a = np.load(p).astype(np.float64)
    A.append(a[np.argsort(a[:, pa])])
lo = min(a[0, pa] for a in A)
hi = max(a[-1, pa] for a in A)
print("pos   " + " | ".join(p.replace("\\", "/").split("/")[-1][:-9][-14:].rjust(20) for p in paths))
for t in np.arange(np.ceil(lo / step) * step, hi, step):
    cells = []
    for a in A:
        if a[0, pa] <= t <= a[-1, pa]:
            q = [np.interp(t, a[:, pa], a[:, c]) for c in range(4)]
            cells.append(f"y{q[1]:7.1f} {'zx'[pa == 2]}{q[2 if pa == 0 else 0]:7.1f} p{q[3]:.2f}")
        else:
            cells.append(" " * 20)
    print(f"{t:5.0f} " + " | ".join(cells))
