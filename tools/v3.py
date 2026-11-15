"""v3 detector: colour edges, line + contour candidates, gradient-direction scoring, side refinement, inset."""
import cv2, numpy as np, math, itertools
from v2 import order, convex, angle

INSET = 0.0
TEXT = None
PROB = None   # page probability from the network, at working size
NET_W = 1.0
NET_CAND = True
LAST_CONF = None
SOFT_SUP = 0.9
MAXL = 8

def prep(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    chroma = cv2.magnitude(a.astype(np.float32) - 128, b.astype(np.float32) - 128)
    chroma = np.clip(chroma * 3, 0, 255).astype(np.uint8)  # boost colour differences
    return L, chroma

def close_text(img, k):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

class Fields:
    def __init__(self, chans):
        gx = gy = mag = None
        for c in chans:
            cx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3); cy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
            m = cv2.magnitude(cx, cy)
            if mag is None: gx, gy, mag = cx, cy, m
            else:
                sel = m > mag; gx = np.where(sel, cx, gx); gy = np.where(sel, cy, gy); mag = np.maximum(m, mag)
        self.gx, self.gy, self.mag = gx, gy, mag
        self.h, self.w = mag.shape
        nz = mag[mag > 0]
        self.T = 15.0  # text is removed before this, so a fixed low threshold keeps faint page edges

    def side_support(self, a, b, n=32):
        """Fraction of points along a->b that sit on a strong edge running the same way."""
        d = b - a; ln = np.hypot(*d)
        if ln < 1: return 0.0
        nx, ny = -d[1] / ln, d[0] / ln
        t = (np.arange(n) + 0.5) / n
        best = np.zeros(n, bool)
        for off in (-2, -1, 0, 1, 2):
            xs = np.round(a[0] + d[0] * t + nx * off).astype(int); ys = np.round(a[1] + d[1] * t + ny * off).astype(int)
            ok = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h)
            xs, ys = np.clip(xs, 0, self.w - 1), np.clip(ys, 0, self.h - 1)
            m = self.mag[ys, xs]
            al = np.abs(self.gx[ys, xs] * nx + self.gy[ys, xs] * ny) / (m + 1e-6)
            best |= ok & (m > self.T) & (al > 0.85)
        return float(best.mean())

def line_quality(seg, chans, w, h):
    """How 'page-edge-like' a line is: one side consistently differs from the other along its length."""
    A, B = seg[0], seg[1]; d = B - A; ln = np.hypot(*d); nrm = np.array([-d[1], d[0]]) / (ln + 1e-9)
    off = 0.015 * max(w, h) + 2; best = 0
    for ch in chans:
        a = band(ch, A, B, 20, off, nrm); b = band(ch, A, B, 20, -off, nrm); ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 6: continue
        dl = a[ok] - b[ok]
        best = max(best, min(1, np.mean(np.abs(dl)) / 14) * abs(np.mean(np.sign(dl) * (np.abs(dl) > 4))))
    return best

def lines_from(edges, w, h):
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=int(0.12 * min(w, h)), maxLineGap=int(0.03 * max(w, h)))
    if segs is None: return []
    out = []
    for x1, y1, x2, y2 in segs[:, 0]:
        th = math.atan2(y2 - y1, x2 - x1)
        out.append((np.array([x1, y1], float), np.array([x2, y2], float), math.hypot(x2 - x1, y2 - y1), th))
    # merge near-duplicates: same angle (±3°) and close distance
    out.sort(key=lambda s: -s[2]); merged = []
    for s in out:
        dup = False
        for m in merged:
            dth = abs(((s[3] - m[3]) + np.pi / 2) % np.pi - np.pi / 2)
            if dth < math.radians(4):
                # distance from s midpoint to line m
                p = (s[0] + s[1]) / 2; d = m[1] - m[0]; n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
                if abs((p - m[0]) @ n) < 0.02 * max(w, h): dup = True; break
        if not dup: merged.append(s)
    return merged[:60]

def fit_segment(seg, F, edges, w, h):
    """Re-fit a detected line to the edge pixels along its whole length (fixes small angle errors)."""
    A, B = seg[0], seg[1]; d = B - A; ln = np.hypot(*d)
    if ln < 1: return seg
    u = d / ln; n = np.array([-u[1], u[0]])
    ys, xs = np.nonzero(edges)
    rel = np.stack([xs, ys], 1) - A
    dist = rel @ n; along = rel @ u
    sel = (np.abs(dist) < 4) & (along > -0.5 * ln) & (along < 1.5 * ln)
    if sel.sum() < 20: return seg
    xs2, ys2 = xs[sel], ys[sel]
    al = np.abs(F.gx[ys2, xs2] * n[0] + F.gy[ys2, xs2] * n[1]) / (F.mag[ys2, xs2] + 1e-6)
    keep = al > 0.85
    if keep.sum() < 20: return seg
    pts = np.stack([xs2[keep], ys2[keep]], 1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    p0 = np.array([x0, y0], float); dv = np.array([vx, vy], float)
    a2 = p0 + dv * ((A - p0) @ dv); b2 = p0 + dv * ((B - p0) @ dv)
    return (a2, b2, seg[2], math.atan2(b2[1] - a2[1], b2[0] - a2[0]))

def intersect(l1, l2):
    p, r = l1[0], l1[1] - l1[0]; q, s = l2[0], l2[1] - l2[0]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9: return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    return p + t * r

def band(img, A, B, n, off, nrm):
    t = (np.arange(n) + 0.5) / n
    xs = np.round(A[0] + (B - A)[0] * t + nrm[0] * off).astype(int); ys = np.round(A[1] + (B - A)[1] * t + nrm[1] * off).astype(int)
    ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    v = np.full(n, np.nan); v[ok] = img[ys[ok], xs[ok]]
    return v

def separation(q, chans, w, h):
    """Per side: does it consistently split 'inside' from 'outside' along its whole length?"""
    c = q.mean(0); off = 0.015 * max(w, h) + 2; seps = []
    for i in range(4):
        A, B = q[i], q[(i + 1) % 4]; d = B - A; ln = np.hypot(*d); nrm = np.array([-d[1], d[0]]) / ln
        if (c - (A + B) / 2) @ nrm < 0: nrm = -nrm  # point inwards
        best = None
        for ci, ch in enumerate(chans):
            vin = band(ch, A, B, 24, off, nrm); vout = band(ch, A, B, 24, -off, nrm)
            ok = ~np.isnan(vin) & ~np.isnan(vout)
            if ok.sum() < 6: best = 0.3 if best is None else best; continue
            dl = vin[ok] - vout[ok]
            if ci == 1: dl = -dl  # colour channel: paper is LESS colourful than the background
            strength = min(1.0, np.mean(np.abs(dl)) / 14)
            consist = max(0.0, np.mean(np.sign(dl) * (np.abs(dl) > 4)))  # paper must be on the inside
            sc = strength * consist
            best = sc if best is None else max(best, sc)
        seps.append(best)
    return np.array(seps)

def score_quad(q, F, L, w, h, area, chans=None):
    if not convex(q): return 0, None
    a = cv2.contourArea(q.astype(np.float32)); r = a / area
    if r < 0.08 or r > 0.995: return 0, None
    ms = min(w, h) * 0.12
    for i in range(4):
        if np.linalg.norm(q[(i + 1) % 4] - q[i]) < ms: return 0, None
        an = angle(q[(i + 3) % 4], q[i], q[(i + 1) % 4])
        if an < 45 or an > 135: return 0, None
    # corners may be slightly outside the frame (page cut off), but not far
    if np.any(q < [-0.03 * w, -0.03 * h]) or np.any(q > [1.03 * w, 1.03 * h]): return 0, None
    sups = []; nborder = 0
    for i in range(4):
        A, B = q[i], q[(i + 1) % 4]
        on_border = all(min(P[0], P[1], w - 1 - P[0], h - 1 - P[1]) < 3 for P in (A, B))
        nborder += on_border
        sups.append(0.35 if on_border else F.side_support(A, B))
    if nborder > 1: return 0, None
    sups = np.array(sups)
    if sups.min() < 0.30: return 0, None
    # polarity: paper (inside) brighter than just outside
    c = q.mean(0); inner = c + (q - c) * 0.94; outer = c + (q - c) * 1.06
    def samp(poly):
        v = []
        for i in range(4):
            A, B = poly[i], poly[(i + 1) % 4]
            for t in np.linspace(0.1, 0.9, 9):
                x, y = np.round(A + (B - A) * t).astype(int)
                v.append(L[y, x] if 0 <= x < w and 0 <= y < h else np.nan)
        return np.array(v, float)
    vi, vo = samp(inner), samp(outer); ok = ~np.isnan(vo)
    pol = 1.0 if ok.sum() < 8 else 1 + np.clip(np.mean(vi[ok] - vo[ok]) / 255, -0.25, 0.3) * 1.2
    # A real corner: the page edge stops there. If a side's edge carries on past the corner,
    # this shape is only part of the page (e.g. cut by a shadow line).
    ext = []
    for i in range(4):
        A, B = q[i], q[(i + 1) % 4]; d = (B - A)
        ext.append(F.side_support(B + d * 0.03, B + d * 0.18, 12))
        ext.append(F.side_support(A - d * 0.03, A - d * 0.18, 12))
    ext = np.array(ext)
    sep = separation(q, chans if chans is not None else [L], w, h)
    # A crisp, continuous edge with similar brightness on both sides (page lying on another white
    # sheet) is still allowed, with a low value.
    sep = np.where((sep < 0.15) & (sups >= SOFT_SUP), 0.3, sep)
    if sep.min() < 0.15: return 0, None
    s = math.sqrt(r) * (sups.mean() ** 2) * math.sqrt(sups.min()) * pol * sep.mean() * math.sqrt(sep.min())
    s *= (1 - 0.6 * np.sort(ext)[-2:].mean())
    iou = net_iou(q, w, h)
    if iou is not None:
        s *= (0.15 + 0.85 * iou ** 3) ** NET_W
    if TEXT is not None:  # prefer the sheet with printing on it over a blank one
        ins = 0; tot = 0
        for i in range(2, 19):
            for j in range(2, 19):
                u, v = i / 20, j / 20
                top = q[0] + (q[1] - q[0]) * u; bot = q[3] + (q[2] - q[3]) * u; p = np.round(top + (bot - top) * v).astype(int)
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    tot += 1; ins += TEXT[p[1], p[0]]
        dens = ins / max(tot, 1)
        s *= 0.55 + 0.45 * min(1.0, dens / 0.06)
    return s, sups

def refine(q, F, edges, w, h):
    """Fit a straight line to the edge pixels along each side, then intersect neighbours."""
    ys, xs = np.nonzero(edges)
    if len(xs) == 0: return q
    pts = np.stack([xs, ys], 1).astype(np.float32)
    lines = []
    for i in range(4):
        A, B = q[i], q[(i + 1) % 4]; d = B - A; ln = np.hypot(*d); n = np.array([-d[1], d[0]]) / ln
        rel = pts - A; along = rel @ (d / ln); dist = rel @ n
        sel = (np.abs(dist) < 0.012 * max(w, h) + 2) & (along > 0.05 * ln) & (along < 0.95 * ln)
        # only edge pixels running the same way as this side (ignores wood grain, text, patterns)
        gxs = F.gx[ys, xs]; gys = F.gy[ys, xs]; ms = F.mag[ys, xs] + 1e-6
        sel &= np.abs(gxs * n[0] + gys * n[1]) / ms > 0.85
        if sel.sum() < 15: lines.append((A, B)); continue
        vx, vy, x0, y0 = cv2.fitLine(pts[sel], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        p0 = np.array([x0, y0]); dv = np.array([vx, vy])
        lines.append((p0, p0 + dv))
    out = []
    for i in range(4):
        p = intersect(lines[(i + 3) % 4], lines[i])
        if p is None: return q
        out.append(p)
    out = np.array(out)
    return out if convex(out) and np.max(np.linalg.norm(out - q, axis=1)) < 0.02 * max(w, h) else q

def contour_quads(binary, area):
    cs, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    big = sorted([c for c in cs if cv2.contourArea(cv2.convexHull(c)) > area * 0.06], key=lambda c: -cv2.contourArea(cv2.convexHull(c)))[:10]
    out = []
    for c in big:
        hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32); per = cv2.arcLength(hull, True)
        for f in [0.01, 0.02, 0.03, 0.045, 0.06, 0.08]:
            ap = cv2.approxPolyDP(hull, f * per, True)
            if len(ap) == 4: out.append(ap.reshape(4, 2).astype(float)); break
            if len(ap) < 4: break
    return out

def net_iou(q, w, h):
    """Overlap between an outline and the network's page area (0..1)."""
    if PROB is None: return None
    ph, pw = PROB.shape
    m = np.zeros((ph, pw), np.uint8)
    cv2.fillConvexPoly(m, np.int32(np.round(q * [pw / w, ph / h] * 8)), 1, shift=3)
    a = m > 0; b = PROB > 0.5
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0

def detect_color(bgr, gray_only=False, prob=None):
    global PROB
    PROB = prob
    h, w = bgr.shape[:2]; area = w * h
    if gray_only or bgr.ndim == 2:
        L = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); chans = [L]
    else:
        L, C = prep(bgr); chans = [L, C]
    k = max(5, max(w, h) // 60) | 1
    sm = [close_text(cv2.GaussianBlur(c, (5, 5), 0), k) for c in chans]
    global TEXT
    # printing = dark details that the text-erasing step removed
    TEXT = cv2.dilate(((sm[0].astype(np.int16) - chans[0].astype(np.int16)) > 25).astype(np.uint8), np.ones((3, 3), np.uint8))
    F = Fields(sm)
    edges = np.zeros_like(sm[0])
    for c in sm:
        edges |= cv2.Canny(c, 10, 30)  # text is already removed, so low thresholds are safe
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    cands = []
    # 1) contours of edge blobs and of bright regions
    _, otsu = cv2.threshold(sm[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for b in (edges_d, otsu):
        cands += contour_quads(b, area)
    if PROB is not None and NET_CAND:  # the network's page area, as an outline
        pm = cv2.resize((PROB > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        cands += contour_quads((pm > 127).astype(np.uint8) * 255, area)
    # 2) line combinations
    segs = [fit_segment(sg, F, edges, w, h) for sg in lines_from(edges, w, h)]
    segs = sorted(segs, key=lambda sg: -sg[2] * (0.15 + line_quality(sg, sm, w, h)))
    border = [(np.array([0., 0]), np.array([w - 1., 0]), w, 0.0), (np.array([0., h - 1]), np.array([w - 1., h - 1]), w, 0.0),
              (np.array([0., 0]), np.array([0., h - 1]), h, np.pi / 2), (np.array([w - 1., 0]), np.array([w - 1., h - 1]), h, np.pi / 2)]
    hor = [s for s in segs if abs(math.sin(s[3])) < 0.7] + border[:2]
    ver = [s for s in segs if abs(math.sin(s[3])) >= 0.7] + border[2:]
    hor = hor[:MAXL]; ver = ver[:MAXL]
    for h1, h2 in itertools.combinations(hor, 2):
        for v1, v2 in itertools.combinations(ver, 2):
            ps = [intersect(h1, v1), intersect(h1, v2), intersect(h2, v1), intersect(h2, v2)]
            if any(p is None for p in ps): continue
            cands.append(np.array(ps))
    scored = []
    for c in cands:
        q = order(c)
        s, _ = score_quad(q, F, sm[0], w, h, area, sm)
        if s > 0: scored.append((s, q))
    if not scored: return None
    scored.sort(key=lambda t: -t[0])
    best, bq = scored[0]
    diag = math.hypot(w, h)
    rival = next((s for s, q in scored[1:] if np.max(np.linalg.norm(q - bq, axis=1)) > 0.05 * diag), 0.0)
    global LAST_CONF
    LAST_CONF = (best, rival / best, net_iou(bq, w, h))
    bq = refine(bq, F, edges, w, h)
    return bq / [w, h]

def detect(gray):
    return detect_color(gray, gray_only=True)
