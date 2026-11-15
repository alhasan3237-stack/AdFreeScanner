import cv2, numpy as np, glob, os
def metric(gray):
    # Crete-Roffet style: how much gradient energy a further blur removes (sharp = big loss).
    g = gray.astype(np.float32)
    b = cv2.blur(g, (1, 9)); b2 = cv2.blur(g, (9, 1))
    def var(a, ax): return np.abs(np.diff(a, axis=ax))
    dv = var(g, 0); dbv = var(b, 0); dh = var(g, 1); dbh = var(b2, 1)
    vv = np.maximum(0, dv - dbv); vh = np.maximum(0, dh - dbh)
    sv = dv.sum(); sh = dh.sum()
    bv = (sv - vv.sum()) / max(sv, 1); bh = (sh - vh.sum()) / max(sh, 1)
    return 1 - max(bv, bh)   # 1 = sharp, 0 = very blurry
fs = [f for f in sorted(glob.glob(os.path.expanduser("~/real/set/r*"))) if os.path.basename(f)[:3] not in {"r08","r23","r24"}]
print("name  res   sharp  g1.5   g3   motion12  (at 1600 work size)")
for f in fs:
    im = cv2.imread(f, 0); h, w = im.shape
    row = []
    for kind in ["none", "g1.5", "g3", "m"]:
        x = im.copy()
        sc = max(h, w) / 4000  # express blur in a 4000px photo
        if kind == "g1.5": x = cv2.GaussianBlur(x, (0, 0), 1.5 * sc)
        if kind == "g3": x = cv2.GaussianBlur(x, (0, 0), 3 * sc)
        if kind == "m":
            k = max(3, int(12 * sc)); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k; x = cv2.filter2D(x, -1, ker)
        s = 1600 / max(h, w)
        if s < 1: x = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        c = x[x.shape[0] // 6: -x.shape[0] // 6, x.shape[1] // 6: -x.shape[1] // 6]
        row.append(metric(c))
    print(os.path.basename(f)[:3], f"{max(h,w):5d}", " ".join(f"{v:.3f}" for v in row))
