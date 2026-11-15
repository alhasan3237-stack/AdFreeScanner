"""Fast, very varied training scenes for the page-finding network (low resolution).
Output: image (H=160, W=120, BGR uint8) and mask of the TOP page's full outline (amodal)."""
import cv2, numpy as np

OH, OW = 128, 96           # network input (portrait 4:3)
RH, RW = 480, 360          # render size, then shrink (anti-aliasing, realistic blur)

def rand_color(rng, lo=0, hi=255): return rng.integers(lo, hi, 3).astype(np.float32)

def texture(rng, h, w):
    kind = rng.integers(0, 9)
    if kind == 0:   # wood grain
        x = np.linspace(0, 1, w)[None, :]; y = np.linspace(0, 1, h)[:, None]
        g = np.sin((x * rng.uniform(20, 60) + np.sin(y * rng.uniform(3, 9)) * 2) * np.pi) * 0.5 + 0.5
        c1, c2 = rand_color(rng, 40, 140), rand_color(rng, 80, 200)
        img = c1 * (1 - g[..., None]) + c2 * g[..., None]
        if rng.random() < 0.5: img = np.rot90(img).copy() if img.shape[0] == img.shape[1] else img
    elif kind == 1:  # checker / plaid
        s = rng.integers(8, 40); c = ((np.arange(w)[None, :] // s + np.arange(h)[:, None] // s) % 2)[..., None]
        img = np.where(c, rand_color(rng), rand_color(rng))
    elif kind == 2:  # stripes
        s = rng.integers(5, 30); c = ((np.arange(w)[None, :] + np.arange(h)[:, None] * rng.uniform(-1, 1)) // s % 2)[..., None]
        img = np.where(c, rand_color(rng), rand_color(rng))
    elif kind == 3:  # blobs (floral cloth, carpet)
        n = cv2.resize(rng.random((rng.integers(4, 20), rng.integers(4, 20), 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        img = np.clip(n, 0, 1) * rand_color(rng, 60, 255)
    elif kind == 4:  # light desk, low contrast with paper
        img = np.ones((h, w, 3), np.float32) * rand_color(rng, 170, 235)
    elif kind == 5:  # dark desk
        img = np.ones((h, w, 3), np.float32) * rand_color(rng, 10, 70)
    elif kind == 6:  # bed sheet / beige
        img = np.ones((h, w, 3), np.float32) * np.array([rng.uniform(180, 230)] * 3) * np.array([1, rng.uniform(0.9, 1), rng.uniform(0.75, 0.95)])
    elif kind == 7:  # marble / granite noise
        n = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), rng.uniform(1, 6))
        img = (0.6 + 0.8 * (n[..., None] - 0.5)) * rand_color(rng, 100, 230)
    else:            # gradient floor
        g = np.linspace(0, 1, h)[:, None, None]
        img = rand_color(rng) * (1 - g) + rand_color(rng) * g * np.ones((1, w, 1))
    img = np.broadcast_to(img, (h, w, 3)).astype(np.float32)
    return img + rng.normal(0, rng.uniform(1, 8), (h, w, 1)).astype(np.float32)

def page_content(rng, h, w, paper):
    p = np.ones((h, w, 3), np.float32) * paper
    kind = rng.integers(0, 6)
    ink = rand_color(rng, 0, 60) if rng.random() < 0.85 else np.array([150, 40, 30], np.float32)
    m = int(w * rng.uniform(0.06, 0.12)); y = int(h * rng.uniform(0.04, 0.1))
    lh = max(3, int(h * rng.uniform(0.012, 0.03)))
    if kind == 5:  # blank / almost blank
        return p
    if rng.random() < 0.6:  # heading
        cv2.rectangle(p, (m, y), (m + int(w * rng.uniform(0.3, 0.7)), y + lh * 2), ink.tolist(), -1); y += lh * 3
    if rng.random() < 0.4:  # coloured bar
        cv2.rectangle(p, (m, y), (w - m, y + lh * 2), rand_color(rng).tolist(), -1); y += lh * 3
    while y < h - m:
        r = rng.random()
        if r < 0.75:   # text line as dashes
            x = m
            while x < w - m - 4:
                ww = int(rng.integers(2, 9) * lh * 0.5)
                cv2.rectangle(p, (x, y), (min(x + ww, w - m), y + max(1, lh // 2)), (ink * rng.uniform(0.8, 1.2)).tolist(), -1)
                x += ww + max(2, lh // 2)
            y += lh
        elif r < 0.85:  # table
            rows, cols = rng.integers(2, 6), rng.integers(2, 5); th = rows * lh * 2
            for i in range(rows + 1): cv2.line(p, (m, y + i * lh * 2), (w - m, y + i * lh * 2), ink.tolist(), 1)
            for j in range(cols + 1):
                xx = m + j * (w - 2 * m) // cols; cv2.line(p, (xx, y), (xx, y + th), ink.tolist(), 1)
            y += th + lh
        elif r < 0.92:  # photo / dark box
            bh = int(h * rng.uniform(0.08, 0.25)); bw = int(w * rng.uniform(0.2, 0.6))
            box = cv2.resize(rng.random((4, 4, 3)).astype(np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC) * 255 if rng.random() < 0.6 else np.ones((bh, bw, 3), np.float32) * rand_color(rng, 0, 80)
            yy = min(y, h - bh - 1); xx = int(rng.uniform(m, max(m + 1, w - m - bw)))
            p[yy:yy + bh, xx:xx + bw] = np.clip(box, 0, 255); y += bh + lh
        else:
            y += lh * 2
    return p

def quad_for(rng, W, H, fill, out_ok):
    pw, ph = rng.uniform(0.6, 0.85), 1.0  # A4-ish or receipt/ID ratios
    r = rng.random()
    if r < 0.12: pw = rng.uniform(0.25, 0.4)       # receipt
    elif r < 0.22: pw, ph = 1.0, 0.63               # ID card / landscape
    elif r < 0.32: pw, ph = 1.0, 0.75               # landscape page
    s = fill * min(W / pw, H / ph)
    base = np.array([[-pw, -ph], [pw, -ph], [pw, ph], [-pw, ph]]) * s / 2
    ang = np.deg2rad(rng.choice([rng.uniform(-15, 15), rng.uniform(-40, 40), rng.uniform(-180, 180)], p=[0.7, 0.2, 0.1]))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]); q = base @ R.T
    q += rng.normal(0, 0.04, (4, 2)) * s
    k = rng.uniform(0, 0.3) if rng.random() < 0.4 else 0
    top = np.argsort(q[:, 1])[:2]; q[top, 0] *= (1 - k)            # perspective: far side narrower
    c = np.array([W / 2, H / 2]) + rng.normal(0, 0.08, 2) * [W, H]
    q += c
    if not out_ok:  # keep inside the frame
        mn, mx = q.min(0), q.max(0)
        q += np.clip(-mn + 2, 0, None) - np.clip(mx - [W - 2, H - 2], 0, None)
    return q.astype(np.float32)

def scene(seed):
    rng = np.random.default_rng(seed)
    W, H = RW, RH
    img = texture(rng, H, W)
    for _ in range(rng.integers(0, 6)):   # clutter: books, phones, keyboards, other papers
        box = cv2.boxPoints(((rng.uniform(0, W), rng.uniform(0, H)), (rng.uniform(30, 250), rng.uniform(30, 200)), rng.uniform(0, 180))).astype(np.int32)
        col = rand_color(rng) if rng.random() < 0.7 else rand_color(rng, 200, 250)
        cv2.fillConvexPoly(img, box, col.tolist())
    fill = rng.choice([rng.uniform(0.35, 0.75), rng.uniform(0.75, 0.97), rng.uniform(0.9, 1.1)], p=[0.55, 0.35, 0.1])
    q = quad_for(rng, W, H, fill, out_ok=fill > 0.95)
    paper = np.array([rng.uniform(205, 255)] * 3, np.float32)
    if rng.random() < 0.15: paper = rand_color(rng, 180, 255)
    if rng.random() < 0.15:  # another sheet under the page
        q2 = quad_for(rng, W, H, fill * rng.uniform(0.8, 1.1), False) + rng.normal(0, 40, 2)
        cv2.fillConvexPoly(img, np.int32(q2), (paper * rng.uniform(0.95, 1.03)).tolist())
    ph_, pw_ = 400, 300
    content = page_content(rng, ph_, pw_, paper)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw_, 0], [pw_, ph_], [0, ph_]]), q)
    warped = cv2.warpPerspective(content, M, (W, H))
    pm = cv2.warpPerspective(np.ones((ph_, pw_), np.float32), M, (W, H))
    img = img * (1 - pm[..., None]) + warped * pm[..., None]
    if rng.random() < 0.08:  # thin shadow line at the edge of a sheet on a sheet
        cv2.polylines(img, [np.int32(q)], True, (paper * 0.85).tolist(), 1)
    if rng.random() < 0.25:  # hands / fingers over edges
        skin = np.array([150, 175, 215.]) * rng.uniform(0.5, 1.05)  # BGR skin
        for _ in range(rng.integers(1, 3)):
            i = rng.integers(0, 4); a, b = q[i], q[(i + 1) % 4]; mid = a + (b - a) * rng.uniform(0.2, 0.8)
            out = mid - q.mean(0); out /= np.linalg.norm(out) + 1e-6
            cv2.ellipse(img, tuple(np.int32(mid + out * rng.uniform(20, 60))), (int(rng.uniform(25, 60)), int(rng.uniform(40, 90))), rng.uniform(0, 180), 0, 360, skin.tolist(), -1)
    # lighting: gradient, colour cast, shadows, glare
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1 + rng.uniform(-0.35, 0.35) * (xx / W - 0.5) + rng.uniform(-0.35, 0.35) * (yy / H - 0.5)
    light *= rng.uniform(0.35, 1.1)
    for _ in range(rng.integers(0, 3)):
        poly = np.int32(rng.uniform(-0.3, 1.3, (5, 2)) * [W, H]); sh = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(sh, cv2.convexHull(poly), 1.0); sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(1, 25))
        light *= 1 - sh * rng.uniform(0.2, 0.65)
    img = img * light[..., None] * rng.uniform(0.75, 1.15, 3).astype(np.float32)
    if rng.random() < 0.2:
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(20, 120)
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(30, 120))[..., None]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    img = img + rng.normal(0, rng.uniform(1, 10), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    small = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_AREA)
    mask = np.zeros((OH, OW), np.uint8); cv2.fillConvexPoly(mask, np.int32(np.round(q * [OW / W, OH / H] * 8)), 1, shift=3)
    return small, mask, q / [W, H]

if __name__ == "__main__":
    import time; t = time.time(); tiles = []
    for i in range(24):
        s, m, _ = scene(i); v = s.copy(); v[m > 0] = (v[m > 0] * 0.6 + np.array([0, 120, 0]) * 0.4).astype(np.uint8); tiles.append(np.hstack([s, v]))
    print("per scene ms", (time.time() - t) / 24 * 1000)
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]
    cv2.imwrite("/root/net/peek.png", cv2.resize(np.vstack(rows), None, fx=1.0, fy=1.0))
