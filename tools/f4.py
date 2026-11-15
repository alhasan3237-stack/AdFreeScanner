"""v4 filters: paper-mask lighting estimate (keeps photos, boxes, colours), luminance-only tone curve."""
import cv2, numpy as np

WORK = 1400  # lighting is estimated on a copy this size
SHARP = 1.9
NEIGH = 12.0
MATCH = 0.10
DIL = 5

def odd(n): return int(n) | 1

def push_pull(val, w):
    """Fill holes: smooth average of val where w>0, filling gaps from coarser levels."""
    vs, ws = [val * (w[..., None] if val.ndim == 3 else w)], [w]
    while min(vs[-1].shape[:2]) > 8:
        vs.append(cv2.pyrDown(vs[-1])); ws.append(cv2.pyrDown(ws[-1]))
    est = vs[-1] / np.maximum(ws[-1], 1e-6)[..., None] if val.ndim == 3 else vs[-1] / np.maximum(ws[-1], 1e-6)
    for l in range(len(vs) - 2, -1, -1):
        up = cv2.pyrUp(est, dstsize=(vs[l].shape[1], vs[l].shape[0]))
        wl = ws[l][..., None] if val.ndim == 3 else ws[l]
        a = np.clip(wl * 4, 0, 1)
        est = a * (vs[l] / np.maximum(wl, 1e-6)) + (1 - a) * up
    return est

def grad(g, sigma):
    b = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))

def paper_mask(small):
    """Pixels that are bare paper (lit or shadowed)."""
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 1) printed edges are sharp; shadow edges are soft. Keep only sharp, clear edges.
    fine, coarse = grad(g, 0.7), grad(g, 3.0)
    sharp = fine / (coarse + 1.0)
    rel = fine / (cv2.GaussianBlur(g, (0, 0), 8) + 10)
    edge = ((rel > 0.22) & (sharp > SHARP)).astype(np.uint8)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL, DIL)))
    # 2) not far darker than the paper nearby (black boxes / dark photos are, shadows are not)
    env = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (odd(max(g.shape) / 4),) * 2))
    env = cv2.GaussianBlur(env, (0, 0), max(g.shape) / 20)
    cand = ((edge == 0) & (g > 0.30 * env)).astype(np.uint8)
    # 3) regions enclosed by sharp edges (table cells, highlighter, bars, photos...).
    #    Start from the big paper area; then repeatedly accept a region if its colour matches the
    #    accepted paper right next to it. Shadowed cells match their shadowed neighbours; grey-filled
    #    cells, highlighter, coloured bars and photos don't, so they keep their look.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.int64); area[0] = 0
    accepted = area >= max(area.max() * 0.25, cand.size * 0.02)
    accepted[0] = False
    if not accepted.any(): return cand > 0
    min_area = cand.size * 0.0003
    sf = small.astype(np.float32)
    for _ in range(4):
        m = accepted[lab].astype(np.float32)
        wsum = cv2.GaussianBlur(m, (0, 0), NEIGH)
        local = cv2.GaussianBlur(sf * m[..., None], (0, 0), NEIGH) / np.maximum(wsum, 1e-4)[..., None]
        dev = np.abs(sf / np.maximum(local, 1) - 1).max(axis=2)
        valid = (wsum >= 0.05).ravel()           # only judge pixels with accepted paper nearby
        cnt = np.bincount(lab.ravel(), weights=valid.astype(np.float64), minlength=n)
        sums = np.bincount(lab.ravel(), weights=(dev.ravel() * valid), minlength=n)
        mean_dev = sums / np.maximum(cnt, 1)
        new = (~accepted) & (area >= min_area) & (cnt >= np.minimum(area, 30)) & (mean_dev < MATCH)
        new[0] = False
        if not new.any(): break
        accepted |= new
    return accepted[lab] & (cand > 0)

def lighting(bgr):
    """Per-channel paper colour/brightness everywhere (full size, float)."""
    H, W = bgr.shape[:2]; s = min(1.0, WORK / max(H, W))
    small = cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    m = paper_mask(small).astype(np.float32)
    bg = push_pull(small.astype(np.float32), m)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.2)
    return cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR), m

def flatten(bgr):
    bg, _ = lighting(bgr)
    return np.clip(bgr.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)

def tone_L(flat, white=0.94, black_pct=0.5, gamma=1.0, sharpen=0.6, chroma=1.0):
    """Tone curve on lightness only (colours stay true), then sharpen lightness."""
    lab = cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    blk = np.percentile(L, black_pct); blk = min(blk, 120)
    wht = 255 * white
    Ln = np.clip((L - blk) / max(1, wht - blk), 0, 1) ** gamma * 255
    if sharpen:
        Ln = np.clip(Ln + sharpen * (Ln - cv2.GaussianBlur(Ln, (0, 0), 1.0)), 0, 255)
    lab[..., 0] = Ln
    if chroma != 1.0:
        lab[..., 1:] = (lab[..., 1:] - 128) * chroma + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

def auto(bgr): return tone_L(flatten(bgr), white=0.93, gamma=1.1, sharpen=0.7, chroma=1.1)

GAMMA, BLACK, WHITE, SHARPEN, CHROMA = 1.0, 0.5, 0.92, 0.9, 1.1
def auto(bgr): return tone_L(flatten(bgr), white=WHITE, black_pct=BLACK, gamma=GAMMA, sharpen=SHARPEN, chroma=CHROMA)

def flat_gray(bgr): return cv2.cvtColor(flatten(bgr).astype(np.uint8), cv2.COLOR_BGR2GRAY)
def levels(g, black, white, gamma=1.0):
    lut = (np.clip((np.arange(256) - black) / max(1, white - black), 0, 1) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(g, lut)
def bp(g, pct=0.5): return min(120.0, float(np.percentile(g, pct)))
def magic(bgr): return tone_L(flatten(bgr), white=0.88, black_pct=1.5, gamma=1.15, sharpen=1.0, chroma=1.45)
def gray(bgr):
    g = flat_gray(bgr); out = levels(g, bp(g), 0.92 * 255).astype(np.float32)
    return np.clip(out + 0.9 * (out - cv2.GaussianBlur(out, (0, 0), 1.0)), 0, 255).astype(np.uint8)
def bw(bgr):
    g = cv2.GaussianBlur(flat_gray(bgr), (3, 3), 0)
    blk = odd(max(15, max(g.shape) // 50))
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 15)
    t[g > 215] = 255          # clean paper stays white
    t[g < 90] = 0             # solid dark areas stay solid (not hollow outlines)
    return cv2.medianBlur(t, 3)
def bw_soft(bgr):
    g = flat_gray(bgr); return levels(g, min(140, bp(g) + 60), 205, 1.6)
