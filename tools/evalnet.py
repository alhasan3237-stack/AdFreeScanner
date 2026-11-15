"""Evaluate detector with the trained network on all three test sets."""
import cv2, numpy as np, sys, time, v3, os
SURE = float(os.environ.get("SURE", "0.85")); v3.NET_W = float(os.environ.get("NETW", "1.0")); v3.NET_CAND = os.environ.get("CAND", "1") == "1"
from synth import render
from hard import render_hard
net = cv2.dnn.readNetFromONNX(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "none" else None
def prob_for(img):
    if net is None: return None
    blob = cv2.dnn.blobFromImage(img, 1.0, (96, 128), (0, 0, 0), False, False)
    net.setInput(blob); return net.forward()[0, 0]
kinds = ["wood", "white_desk", "dark", "pattern", "beige"]
sets = {"standard": [(lambda i: render(1000 + i, kinds[i % 5], shadow=(i % 3 != 2), clutter=(i % 4 == 1), finger=(i % 7 == 3))[:2], 60)],
        "unseen": [(lambda i: render(7000 + i, kinds[i % 5], shadow=(i % 3 != 2), clutter=(i % 4 == 1), finger=(i % 7 == 3))[:2], 40)],
        "hard": [(lambda i: render_hard(i)[:2], 36)]}
tot_ok = tot = 0; unsure_wrong = unsure_right = wrong = 0; t = 0
for name, [(fn, n)] in sets.items():
    ok = 0
    for i in range(n):
        img, gt = fn(i)
        s = cv2.resize(img, None, fx=640 / 2592, fy=640 / 2592, interpolation=cv2.INTER_AREA)
        t0 = time.time(); p = prob_for(s); v3.LAST_CONF = None
        q = v3.detect_color(s, prob=None if p is None else cv2.resize(p, (s.shape[1], s.shape[0])) if False else p); t += time.time() - t0
        diag = np.linalg.norm((gt[2] - gt[0]) * [1944, 2592])
        err = 1.0 if q is None else max(np.linalg.norm((q[j] - gt[j]) * [1944, 2592]) for j in range(4)) / diag
        good = err < 0.02; ok += good
        c = v3.LAST_CONF or (0, 1, 0)
        unsure = c[0] < 0.2 or c[1] > 0.6 or (c[2] is not None and c[2] < SURE)
        if not good: wrong += 1; unsure_wrong += unsure
        else: unsure_right += unsure
    print(f"  {name}: {ok}/{n}"); tot_ok += ok; tot += n
print(f"TOTAL {tot_ok}/{tot}  wrong {wrong}: flagged {unsure_wrong}; needless checks {unsure_right}  avg {t / tot * 1000:.0f} ms")
