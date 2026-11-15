"""Exact Python copy of the v2 Kotlin DocumentDetector + filters (baseline)."""
import cv2, numpy as np, math

def order(pts):
    pts = np.array(pts, float); s = pts.sum(1); d = pts[:, 0] - pts[:, 1]
    tl, br, tr, bl = pts[s.argmin()], pts[s.argmax()], pts[d.argmax()], pts[d.argmin()]
    q = np.array([tl, tr, br, bl])
    if len({tuple(p) for p in q}) < 4:
        c = pts.mean(0); a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]); q = pts[np.argsort(a)]
    return q

def convex(q):
    sign = 0
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-6: return False
        s = 1 if cr > 0 else -1
        if sign == 0: sign = s
        elif s != sign: return False
    return True

def angle(a, b, c):
    v1 = a - b; v2 = c - b; d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0 if d == 0 else math.degrees(math.acos(np.clip(v1 @ v2 / d, -1, 1)))

def detect(gray):
    h, w = gray.shape; area = w * h
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    k = max(5, max(w, h) // 45); k += (k % 2 == 0)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2)
    d3 = np.ones((3, 3), np.uint8)
    es = cv2.dilate(cv2.Canny(closed, 30, 90), d3); ew = cv2.dilate(cv2.Canny(closed, 8, 28), d3)
    emap = cv2.dilate(es | ew, np.ones((5, 5), np.uint8))
    _, otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands = []
    for b, mode in [(es, cv2.RETR_LIST), (ew, cv2.RETR_LIST), (otsu, cv2.RETR_EXTERNAL)]:
        cs, _ = cv2.findContours(b.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
        big = sorted([(c, cv2.contourArea(c)) for c in cs if cv2.contourArea(c) > area * 0.04], key=lambda t: -t[1])[:8]
        for c, _ in big:
            hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)
            if len(hull) < 4: continue
            per = cv2.arcLength(hull, True); q = None
            for f in [0.015, 0.02, 0.03, 0.04, 0.055, 0.07, 0.09]:
                ap = cv2.approxPolyDP(hull, f * per, True)
                if len(ap) == 4: q = ap.reshape(4, 2); break
                if len(ap) < 4: break
            if q is None: continue
            sc = score(q, area, w, h, emap, blurred)
            if sc > 0: cands.append((sc, q))
    if not cands: return None
    q = order(max(cands, key=lambda t: t[0])[1])
    return q / [w, h] if convex(q) else None

def score(raw, area, w, h, emap, gray):
    q = order(raw)
    if not convex(q): return 0
    a = cv2.contourArea(q.astype(np.float32)); r = a / area
    if r < 0.10 or r > 0.985: return 0
    ms = min(w, h) * 0.12
    for i in range(4):
        if np.linalg.norm(q[(i + 1) % 4] - q[i]) < ms: return 0
        an = angle(q[(i + 3) % 4], q[i], q[(i + 1) % 4])
        if an < 40 or an > 140: return 0
    hits = tot = 0
    for i in range(4):
        a_, b_ = q[i], q[(i + 1) % 4]; sh = 0
        for s in range(2, 38):
            t = s / 40; x, y = np.round(a_ + (b_ - a_) * t).astype(int)
            if 0 <= x < w and 0 <= y < h and emap[y, x]: sh += 1
        if sh < 36 * 0.25: return 0
        hits += sh; tot += 36
    sup = hits / tot
    c = q.mean(0); outer = c + (q - c) * 1.07
    ins = []
    for i in range(1, 10):
        for j in range(1, 10):
            u, v = i / 10, j / 10
            top = q[0] + (q[1] - q[0]) * u; bot = q[3] + (q[2] - q[3]) * u; p = np.round(top + (bot - top) * v).astype(int)
            if 0 <= p[0] < w and 0 <= p[1] < h: ins.append(gray[p[1], p[0]])
    outs = []
    for i in range(4):
        a_, b_ = outer[i], outer[(i + 1) % 4]
        for s in range(1, 10):
            p = np.round(a_ + (b_ - a_) * s / 10).astype(int)
            if 0 <= p[0] < w and 0 <= p[1] < h: outs.append(gray[p[1], p[0]])
    con = 1.0 if len(outs) < 6 else 1 + np.clip((np.mean(ins) - np.mean(outs)) / 255, -0.3, 0.4) * 1.5
    return math.sqrt(r) * sup * sup * con

def remove_shadows(ch):
    L = max(ch.shape); s = min(1, 500 / L)
    small = cv2.resize(ch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small = cv2.dilate(small, np.ones((9, 9), np.uint8))
    small = cv2.medianBlur(small, 25)
    bg = cv2.resize(small, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = cv2.divide(ch, bg, scale=255)
    return cv2.convertScaleAbs(out, alpha=1.2, beta=-51)

def auto(bgr): return cv2.merge([remove_shadows(c) for c in cv2.split(bgr)])
