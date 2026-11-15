import cv2, numpy as np, sys, importlib, time
from synth import render
mod = importlib.import_module(sys.argv[1])
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
kinds = ["wood", "white_desk", "dark", "pattern", "beige"]
res = {k: [] for k in kinds}; t = 0; fails = []
for i in range(N):
    kind = kinds[i % 5]
    img, gt, _ = render(BASE + i, kind, shadow=(i % 3 != 2), clutter=(i % 4 == 1), finger=(i % 7 == 3))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    if hasattr(mod, "detect_color"):
        small = cv2.resize(img, None, fx=640 / 2592, fy=640 / 2592, interpolation=cv2.INTER_AREA); q = mod.detect_color(small)
    else:
        small = cv2.resize(g, None, fx=800 / 2592, fy=800 / 2592, interpolation=cv2.INTER_AREA); q = mod.detect(small)
    t += time.time() - t0
    diag = np.linalg.norm((gt[2] - gt[0]) * [1944, 2592])
    err = 1.0 if q is None else max(np.linalg.norm((q[j] - gt[j]) * [1944, 2592]) for j in range(4)) / diag
    res[kind].append(err)
    if err > 0.02: fails.append((i, kind, round(err, 3)))
allerr = sum(res.values(), [])
print(f"{sys.argv[1]}: success(<2% corner error) {sum(e < 0.02 for e in allerr)}/{len(allerr)}  "
      f"median err {np.median(allerr)*100:.2f}%  avg time {t/N*1000:.0f} ms")
for k in kinds: print(f"  {k:11s} {sum(e < 0.02 for e in res[k])}/{len(res[k])}")
print("fails:", fails[:20])
