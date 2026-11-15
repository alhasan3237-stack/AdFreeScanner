"""v3 filters."""
import cv2, numpy as np

def odd(n): return n | 1

def paper_map(ch):
    """Estimate the bare-paper brightness everywhere (text removed, lighting and shadow edges kept)."""
    H, W = ch.shape; L = max(H, W); s = min(1.0, 1000 / L)
    small = cv2.resize(ch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else ch.copy()
    k = odd(max(7, int(max(small.shape) / 70)))
    # closing removes dark text but keeps the outline of shadows in place (no halos)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, odd(k + 2))
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    return cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR) if s < 1 else bg

def flatten(ch):
    bg = paper_map(ch)
    return cv2.divide(ch, np.maximum(bg, 1), scale=255)

def levels(img, black, white, gamma=1.0):
    lut = np.clip((np.arange(256) - black) / max(1, white - black), 0, 1) ** gamma * 255
    return cv2.LUT(img, lut.astype(np.uint8))

def black_point(gray):
    return float(np.clip(np.percentile(gray, 0.8), 0, 90))

def sharpen(img, amount=0.7, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def saturate(bgr, factor):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def auto(bgr):
    """Clean white paper, no shadows, natural ink colours, crisp text."""
    flat = cv2.merge([flatten(c) for c in cv2.split(bgr)])
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    out = levels(flat, black_point(g) * 0.8, 232, 1.15)
    out = sharpen(out, 0.6, 1.0)
    return saturate(out, 1.2)

def magic(bgr):
    """Punchier: stronger contrast and colours (stamps, highlighters, coloured forms)."""
    flat = cv2.merge([flatten(c) for c in cv2.split(bgr)])
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    out = levels(flat, black_point(g), 222, 1.35)
    out = sharpen(out, 0.9, 1.0)
    return saturate(out, 1.6)

def lighten(bgr):
    """For photos / colourful pages: brighter, balanced, no hard cleaning."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab); L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    out = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_Lab2BGR)
    # white balance from the brightest 5% (the paper)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY); thr = np.percentile(g, 95); m = g >= thr
    gains = [230.0 / max(1, out[..., c][m].mean()) for c in range(3)]
    out = cv2.merge([cv2.convertScaleAbs(out[..., c], alpha=min(1.6, gains[c])) for c in range(3)])
    return sharpen(out, 0.4, 1.2)

def gray(bgr):
    g = flatten(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return sharpen(levels(g, black_point(g) * 0.8, 230, 1.2), 0.6, 1.0)

def bw(bgr):
    """Pure black & white: text only, paper pure white, specks removed."""
    g = flatten(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    blk = odd(max(15, max(g.shape) // 50))
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 15)
    # anything clearly paper-white stays white (stops speckles in empty areas)
    t[g > 215] = 255
    # remove tiny dots
    inv = 255 - t
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < max(3, (max(g.shape) // 1000) ** 2 * 3)
    small[0] = False
    inv[small[lab]] = 0
    return 255 - inv

def bw_soft(bgr):
    """Black & white with smooth edges: easier to read, still prints cleanly."""
    g = flatten(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return levels(g, min(140, black_point(g) + 60), 205, 1.6)

def original(bgr): return bgr
