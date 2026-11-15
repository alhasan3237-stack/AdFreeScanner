"""Realistic training scenes for the page finder (fine-tuning DocQuadNet-256).

Each scene: a phone-camera-like photo of ONE top document (sheet, notebook page, receipt, card,
form, magazine page...) on a realistic surface, letterboxed to 256x256 with black bars (exactly
like the app), plus its 4 corners in 256-space (TL, TR, BR, BL, canonical order).

Focus on what failed on real photos: strongly patterned surfaces (carpets, floral bedsheets,
lattice, tiles, lace), white-on-white, notebooks with spiral/stack edges, sheets on sheets,
fingers, shadows, glare, dim warm light, blur, noise, JPEG.
"""
import cv2, numpy as np, glob, os, math
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
def _fonts():
    """fonts.txt (one font file per line) if present, else the Latin fonts installed on this computer."""
    f = os.path.join(HERE, "fonts.txt")
    if os.path.exists(f): return [l.strip() for l in open(f) if l.strip()]
    dirs = [os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"), "/usr/share/fonts"]
    skip = ("symbol", "wingding", "webding", "marlett", "holomdl2", "segmdl2", "emoji", "mtextra", "bssym", "segoeicons", "fluent")
    found = [p for d in dirs for p in glob.glob(os.path.join(d, "**", "*.tt[fc]"), recursive=True)]
    return [p for p in sorted(found) if not any(k in os.path.basename(p).lower() for k in skip) and _usable(p)]


def _usable(path):
    """The font draws text without errors and has real Latin letters (not the same box for each)."""
    try:
        f = ImageFont.truetype(path, 24)
        a, b = f.getmask("a"), f.getmask("b")
        f.getmask("Total 12.50 Tk")
        return a.size != b.size or bytes(a) != bytes(b)
    except Exception:
        return False


FONTS = _fonts()
_PAGES = None
_PHOTOS = None
WORDS = ("the of and to in is for on that with as by this from at are be or an was have it not "
         "invoice total amount date name address phone account bank payment receipt tax due "
         "student university exam result marks class roll number certificate government office "
         "dhaka road house tk price quantity item description signature notice meeting report "
         "chapter section figure table summary introduction method results discussion").split()


def pages():
    global _PAGES
    if _PAGES is None:
        _PAGES = []
        for f in sorted(glob.glob(os.path.join(HERE, "pages", "*"))):
            im = cv2.imread(f)
            if im is None or min(im.shape[:2]) < 100: continue
            s = 700 / max(im.shape[:2])
            _PAGES.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return _PAGES


def photos():
    global _PHOTOS
    if _PHOTOS is None:
        _PHOTOS = []
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(os.environ.get("DQ_PHOTOS", "~/assets/ocv/samples/data")), "*"))):
            if not f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif")): continue
            im = cv2.imread(f)
            if im is None or min(im.shape[:2]) < 120 or im.ndim != 3: continue
            if np.std(im) < 25: continue
            s = 600 / max(im.shape[:2])
            _PHOTOS.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im)
    return _PHOTOS


def rc(rng, lo=0, hi=255):
    return rng.integers(lo, hi, 3).astype(np.float32)


PALETTES = [  # carpets / bedsheets / cloth seen in homes
    [(30, 30, 150), (40, 60, 180), (200, 220, 235), (60, 40, 30), (40, 120, 200)],     # red persian (BGR)
    [(90, 40, 20), (160, 120, 60), (210, 220, 230), (40, 40, 140), (30, 90, 170)],     # navy/cream
    [(60, 80, 30), (120, 160, 80), (220, 230, 235), (40, 50, 120), (30, 140, 210)],    # green/ochre
    [(150, 90, 200), (90, 170, 240), (240, 240, 245), (170, 120, 60), (60, 60, 200)],  # pink floral
    [(200, 200, 200), (120, 120, 120), (245, 245, 245), (60, 60, 60), (170, 170, 170)],  # grey
    [(40, 160, 220), (20, 90, 160), (230, 240, 250), (120, 60, 30), (60, 200, 230)],   # orange/yellow
]


def pal(rng):
    p = np.array(PALETTES[rng.integers(len(PALETTES))], np.float32)
    if rng.random() < 0.4: p = np.clip(p * rng.uniform(0.7, 1.2, (1, 3)), 0, 255)
    if rng.random() < 0.3: p = rng.permutation(p)
    return p


def tile_pattern(rng, h, w):
    """Carpet / kilim / lattice / tiles: a symmetric motif repeated, with a palette."""
    p = pal(rng)
    t = int(rng.integers(14, 70))
    kind = rng.integers(0, 4)
    n = rng.random((t, t)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), rng.uniform(0.8, 3.0))
    n = (n + n[:, ::-1] + n[::-1, :] + n[::-1, ::-1]) / 4  # mirror symmetry like woven motifs
    if kind == 1: n = (n + n.T) / 2
    if kind == 2:  # lattice lines
        yy, xx = np.mgrid[0:t, 0:t]
        n = np.minimum(np.abs(xx - yy) % t, np.abs(xx + yy - t) % t).astype(np.float32) / t
    q = np.clip(((n - n.min()) / (np.ptp(n) + 1e-6) * len(p)).astype(int), 0, len(p) - 1)
    tile = p[q]
    reps_y, reps_x = h // t + 2, w // t + 2
    img = np.tile(tile, (reps_y, reps_x, 1))
    oy, ox = rng.integers(0, t), rng.integers(0, t)
    img = img[oy:oy + h, ox:ox + w]
    if rng.random() < 0.5:  # carpet with medallion / border bands
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        cv2.ellipse(img, (int(cx), int(cy)), (int(w * rng.uniform(0.15, 0.4)), int(h * rng.uniform(0.15, 0.4))), 0, 0, 360, p[rng.integers(len(p))].tolist(), -1)
        for k in range(rng.integers(1, 4)):
            col = p[rng.integers(len(p))].tolist(); b = int(rng.integers(3, 12))
            off = int(rng.integers(0, max(1, min(h, w) // 3)))
            cv2.rectangle(img, (off, off), (w - off, h - off), col, b)
    if rng.random() < 0.6:  # woven texture
        img = img + rng.normal(0, rng.uniform(4, 14), (h, w, 1)).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.2))
    return img


def floral(rng, h, w):
    p = pal(rng)
    img = np.ones((h, w, 3), np.float32) * p[2]
    for _ in range(int(rng.integers(30, 160))):
        x, y = rng.uniform(0, w), rng.uniform(0, h); r = rng.uniform(4, 22); col = p[rng.integers(len(p))].tolist()
        for k in range(int(rng.integers(4, 7))):
            a = k * 2 * math.pi / 6 + rng.uniform(0, 0.5)
            cv2.ellipse(img, (int(x + math.cos(a) * r), int(y + math.sin(a) * r)), (int(r * 0.7), int(r * 0.4)), math.degrees(a), 0, 360, col, -1)
        cv2.circle(img, (int(x), int(y)), int(r * 0.4), p[rng.integers(len(p))].tolist(), -1)
        if rng.random() < 0.5:
            cv2.line(img, (int(x), int(y)), (int(x + rng.uniform(-40, 40)), int(y + rng.uniform(-40, 40))), (40, 110, 40), 2)
    return img


def wood(rng, h, w):
    x = np.linspace(0, 1, w)[None, :]; y = np.linspace(0, 1, h)[:, None]
    noise = cv2.resize(rng.random((8, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    g = np.sin((x * rng.uniform(15, 70) + noise * rng.uniform(0.3, 2.5) + np.sin(y * rng.uniform(1, 4)) * 0.6) * math.pi) * 0.5 + 0.5
    g = g ** rng.uniform(0.6, 2.0)
    base = np.array([rng.uniform(20, 90), rng.uniform(50, 130), rng.uniform(90, 200)], np.float32)
    if rng.random() < 0.2: base = rc(rng, 150, 230)  # light / painted wood
    img = base * (0.75 + 0.35 * g[..., None])
    if rng.random() < 0.5: img = np.ascontiguousarray(np.rot90(img)) if h == w else cv2.resize(np.ascontiguousarray(np.rot90(img)), (w, h))
    if rng.random() < 0.4:  # planks
        step = int(rng.integers(40, 160))
        for k in range(0, w, step): cv2.line(img, (k, 0), (k, h), (base * 0.5).tolist(), 1)
    return img


def stone(rng, h, w):
    n = sum(cv2.resize(rng.random((s, s)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) / (i + 1) for i, s in enumerate([4, 12, 40, 120]))
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    if rng.random() < 0.5:  # marble veins
        n = np.abs(np.sin(n * rng.uniform(8, 25)))
    c1, c2 = rc(rng, 120, 250), rc(rng, 60, 200)
    img = c1 * n[..., None] + c2 * (1 - n[..., None])
    if rng.random() < 0.4:  # floor tiles grout
        s = int(rng.integers(60, 200))
        for k in range(0, max(h, w), s):
            cv2.line(img, (k, 0), (k, h), (c2 * 0.6).tolist(), 2); cv2.line(img, (0, k), (w, k), (c2 * 0.6).tolist(), 2)
    return img


def plain(rng, h, w):
    kind = rng.integers(0, 4)
    if kind == 0: c = rc(rng, 175, 245)          # white/light table: low contrast with paper
    elif kind == 1: c = rc(rng, 5, 70)           # dark desk / black cloth
    elif kind == 2: c = rc(rng, 40, 220)         # any colour
    else: c = np.array([rng.uniform(150, 230)] * 3, np.float32) * np.array([rng.uniform(0.85, 1), rng.uniform(0.9, 1), 1], np.float32)
    img = np.ones((h, w, 3), np.float32) * c
    g = np.linspace(0, 1, h)[:, None, None]
    return img * (1 + (g - 0.5) * rng.uniform(-0.3, 0.3))


def photo_bg(rng, h, w):
    ph = photos()[rng.integers(len(photos()))]
    s = max(h / ph.shape[0], w / ph.shape[1]) * rng.uniform(1.0, 1.8)
    big = cv2.resize(ph, None, fx=s, fy=s)
    y = rng.integers(0, big.shape[0] - h + 1); x = rng.integers(0, big.shape[1] - w + 1)
    return big[y:y + h, x:x + w].astype(np.float32)


def background(rng, h, w):
    r = rng.random()
    if r < 0.24: img = tile_pattern(rng, h, w)
    elif r < 0.36: img = floral(rng, h, w)
    elif r < 0.52: img = wood(rng, h, w)
    elif r < 0.62: img = stone(rng, h, w)
    elif r < 0.84: img = plain(rng, h, w)
    else: img = photo_bg(rng, h, w)
    if rng.random() < 0.2:  # cloth folds
        f = cv2.GaussianBlur(rng.random((h // 8 + 1, w // 8 + 1)).astype(np.float32), (0, 0), 2)
        img = img * (0.8 + 0.4 * cv2.resize(f, (w, h)))[..., None]
    return np.ascontiguousarray(img[:h, :w]).astype(np.float32)


# ---------------- page content ----------------

def font(rng, size):
    for _ in range(4):
        try: return ImageFont.truetype(FONTS[rng.integers(len(FONTS))], max(6, int(size)))
        except Exception: pass
    return ImageFont.load_default()


def sentence(rng, n):
    return " ".join(WORDS[i] for i in rng.integers(0, len(WORDS), n))


def printed_page(rng, w, h, paper):
    im = Image.new("RGB", (w, h), tuple(int(v) for v in paper[::-1]))
    d = ImageDraw.Draw(im)
    ink = tuple(int(v) for v in (rng.integers(0, 60, 3)))
    m = int(w * rng.uniform(0.06, 0.12)); y = int(h * rng.uniform(0.04, 0.1))
    fs = h * rng.uniform(0.016, 0.03)
    if rng.random() < 0.4:  # letterhead band / logo
        col = tuple(int(v) for v in rng.integers(0, 200, 3))
        d.rectangle([0, 0, w, int(h * rng.uniform(0.03, 0.09))], fill=col) if rng.random() < 0.5 else d.ellipse([m, y, m + fs * 3, y + fs * 3], fill=col)
        y += int(fs * 3.5)
    d.text((m, y), sentence(rng, rng.integers(2, 6)).title(), fill=ink, font=font(rng, fs * rng.uniform(1.4, 2.2))); y += int(fs * 3)
    f = font(rng, fs)
    while y < h - m:
        r = rng.random()
        if r < 0.7:
            for _ in range(int(rng.integers(2, 9))):
                if y > h - m: break
                d.text((m, y), sentence(rng, int((w - 2 * m) / fs / 2.6)), fill=ink, font=f); y += int(fs * 1.5)
            y += int(fs * 0.8)
        elif r < 0.82:  # table
            rows, cols = int(rng.integers(2, 7)), int(rng.integers(2, 5)); rh = int(fs * 1.8)
            for i in range(rows + 1): d.line([m, y + i * rh, w - m, y + i * rh], fill=ink, width=1)
            for j in range(cols + 1):
                xx = m + j * (w - 2 * m) // cols; d.line([xx, y, xx, y + rows * rh], fill=ink, width=1)
            y += rows * rh + int(fs)
        elif r < 0.9 and photos():  # picture
            ph = photos()[rng.integers(len(photos()))]; bw = int(w * rng.uniform(0.25, 0.6)); bh = int(bw * rng.uniform(0.5, 1.0))
            if y + bh < h - m:
                im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(ph, (bw, bh)), cv2.COLOR_BGR2RGB)), (int(rng.uniform(m, w - m - bw)), y)); y += bh + int(fs)
            else: y += int(fs)
        else:  # signature / stamp
            x0 = int(rng.uniform(m, w * 0.6))
            pts = [(x0 + k * fs * 0.6, y + fs * math.sin(k * rng.uniform(0.5, 2))) for k in range(int(rng.integers(6, 14)))]
            d.line(pts, fill=(20, 40, 150), width=2)
            if rng.random() < 0.5: d.ellipse([x0 + fs * 6, y - fs, x0 + fs * 10, y + fs * 3], outline=(150, 40, 120), width=2)
            y += int(fs * 3)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR).astype(np.float32)


def notebook_page(rng, w, h, paper):
    img = np.ones((h, w, 3), np.float32) * paper
    gap = int(h * rng.uniform(0.025, 0.04)); line_col = np.array([rng.uniform(150, 220), rng.uniform(120, 180), rng.uniform(80, 140)], np.float32)
    if rng.random() < 0.8:
        for y in range(int(h * 0.08), h, gap): cv2.line(img, (0, y), (w, y), line_col.tolist(), 1)
    if rng.random() < 0.6: cv2.line(img, (int(w * 0.12), 0), (int(w * 0.12), h), (90, 90, 220), 1)
    if rng.random() < 0.15:  # squared paper
        for x in range(0, w, gap): cv2.line(img, (x, 0), (x, h), line_col.tolist(), 1)
    ink = (rng.uniform(100, 180), rng.uniform(20, 60), rng.uniform(0, 30)) if rng.random() < 0.6 else (30, 30, 30)
    y = int(h * 0.08) + gap - 3
    while y < h * rng.uniform(0.6, 0.97):
        x = int(w * rng.uniform(0.14, 0.2))
        while x < w * rng.uniform(0.7, 0.95):
            ww = int(rng.uniform(gap * 0.8, gap * 4))
            pts = np.array([(x + t, y - abs(math.sin(t * rng.uniform(0.3, 0.9))) * gap * 0.6) for t in range(0, ww, 2)], np.int32)
            cv2.polylines(img, [pts], False, ink, int(rng.integers(1, 3)))
            x += ww + int(gap * 0.5)
        y += gap
    return img


def receipt(rng, w, h, paper):
    im = Image.new("RGB", (w, h), tuple(int(v) for v in paper[::-1])); d = ImageDraw.Draw(im)
    f = font(rng, w / 22); y = int(h * 0.03)
    while y < h * 0.95:
        d.text((int(w * 0.06), y), sentence(rng, int(rng.integers(2, 5))).upper() + ("   %d.%02d" % (rng.integers(1, 999), rng.integers(0, 99)) if rng.random() < 0.6 else ""), fill=(40, 40, 40), font=f)
        y += int(w / 14)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR).astype(np.float32)


def card(rng, w, h):
    base = rc(rng, 120, 250)
    img = np.ones((h, w, 3), np.float32) * base
    g = np.linspace(0, 1, w)[None, :, None]; img = img * (1 - g * 0.3) + rc(rng) * g * 0.3
    if photos():
        ph = photos()[rng.integers(len(photos()))]; pw, phh = int(w * 0.28), int(h * 0.55)
        img[int(h * 0.25):int(h * 0.25) + phh, int(w * 0.05):int(w * 0.05) + pw] = cv2.resize(ph, (pw, phh))
    for k in range(5):
        cv2.rectangle(img, (int(w * 0.4), int(h * (0.25 + k * 0.12))), (int(w * rng.uniform(0.6, 0.95)), int(h * (0.25 + k * 0.12) + h * 0.05)), (40, 40, 40), -1)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.15)), rc(rng, 0, 200).tolist(), -1)
    return img


def page_image(rng, kind, w, h, paper):
    if kind == "notebook": return notebook_page(rng, w, h, paper)
    if kind == "receipt": return receipt(rng, w, h, paper)
    if kind == "card": return card(rng, w, h)
    if kind == "scan" and pages():
        p = pages()[rng.integers(len(pages()))].astype(np.float32)
        p = cv2.resize(p, (w, h))
        if rng.random() < 0.5: p = np.clip(p * (paper / 250.0), 0, 255)
        return p
    return printed_page(rng, w, h, paper)


# ---------------- geometry ----------------

def page_quad(rng, W, H, aspect, fill):
    """Corners of a page with this width/height aspect, filling `fill` of the frame, in pixels."""
    pw, ph = (aspect, 1.0) if aspect <= 1 else (1.0, 1 / aspect)
    s = fill * min(W / pw, H / ph)
    base = np.array([[-pw, -ph], [pw, -ph], [pw, ph], [-pw, ph]]) * s / 2
    ang = np.deg2rad(rng.choice([rng.uniform(-12, 12), rng.uniform(-45, 45), rng.uniform(-180, 180)], p=[0.6, 0.3, 0.1]))
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    q = base @ R.T
    # camera tilt: far edge narrower (perspective), random direction
    if rng.random() < 0.6:
        k = rng.uniform(0.03, 0.35); d = rng.uniform(0, 2 * math.pi); u = np.array([math.cos(d), math.sin(d)])
        proj = q @ u; t = (proj - proj.min()) / (np.ptp(proj) + 1e-6)
        q = q * (1 - k * t)[:, None] + q.mean(0) * (k * t)[:, None]
    q += rng.normal(0, 0.025, (4, 2)) * s
    q += np.array([W / 2, H / 2]) + rng.normal(0, 0.06, 2) * [W, H]
    # keep all corners in the frame (a few px margin): shrink if too big, then shift
    ext = q.max(0) - q.min(0); k = min(1.0, (W - 8) / ext[0], (H - 8) / ext[1])
    q = (q - q.mean(0)) * k + q.mean(0)
    mn, mx = q.min(0), q.max(0)
    q += np.clip(-mn + 3, 0, None) - np.clip(mx - [W - 3, H - 3], 0, None)
    return q.astype(np.float32)


def canonical(c):
    m = c.mean(0); ang = np.arctan2(c[:, 1] - m[1], c[:, 0] - m[0])
    order = sorted(range(4), key=lambda i: (float(ang[i]), i))
    sums = [float(c[i, 0] + c[i, 1]) for i in order]
    tl = min(range(4), key=lambda k: (sums[k], k))
    return c[order[tl:] + order[:tl]]


def paste(img, layer, alpha):
    a = alpha[..., None]
    return img * (1 - a) + layer * a


def warp_layer(src, q, W, H, bend=None):
    h, w = src.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(q))
    lay = cv2.warpPerspective(src, M, (W, H), flags=cv2.INTER_LINEAR)
    al = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    if bend is not None:
        lay = cv2.remap(lay, bend[0], bend[1], cv2.INTER_LINEAR); al = cv2.remap(al, bend[0], bend[1], cv2.INTER_LINEAR)
    return lay, al


def bend_field(rng, W, H, amp):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = cv2.resize(rng.normal(0, 1, (3, 3, 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
    return xx + f[..., 0], yy + f[..., 1]


def scene(seed, W=384, H=512):
    """Returns (photo uint8 HxWx3, corners 4x2 px canonical)."""
    rng = np.random.default_rng(seed)
    if rng.random() < 0.2: W, H = H, W  # landscape photo
    img = background(rng, H, W)

    # clutter on the surface
    for _ in range(int(rng.integers(0, 5))):
        cw, chh = rng.uniform(30, 260), rng.uniform(15, 200)
        box = cv2.boxPoints(((rng.uniform(0, W), rng.uniform(0, H)), (cw, chh), rng.uniform(0, 180))).astype(np.int32)
        r = rng.random()
        if r < 0.25 and photos():  # book / box with a printed cover
            ph = photos()[rng.integers(len(photos()))]; lay, al = warp_layer(ph.astype(np.float32), box.astype(np.float32), W, H); img = paste(img, lay, al)
        elif r < 0.45:  # phone / keyboard / dark object
            cv2.fillConvexPoly(img, box, rc(rng, 0, 50).tolist())
        elif r < 0.6:  # pen
            p0 = np.array([rng.uniform(0, W), rng.uniform(0, H)]); d = rng.normal(0, 1, 2); d /= np.linalg.norm(d) + 1e-6
            cv2.line(img, tuple(np.int32(p0)), tuple(np.int32(p0 + d * rng.uniform(60, 200))), rc(rng, 0, 255).tolist(), int(rng.integers(3, 8)))
        else:
            cv2.fillConvexPoly(img, box, rc(rng).tolist())

    kind = rng.choice(["printed", "scan", "notebook", "receipt", "card"], p=[0.32, 0.28, 0.22, 0.1, 0.08])
    aspect = {"receipt": rng.uniform(0.25, 0.45), "card": 1.585}.get(kind, rng.choice([0.707, 0.773, rng.uniform(0.6, 0.85)]))
    if kind not in ("receipt", "card") and rng.random() < 0.12: aspect = 1 / aspect  # landscape page
    fill = rng.choice([rng.uniform(0.3, 0.6), rng.uniform(0.6, 0.85), rng.uniform(0.85, 0.99)], p=[0.3, 0.45, 0.25])
    if kind == "card": fill = min(fill, rng.uniform(0.4, 0.8))
    q = page_quad(rng, W, H, aspect, fill)

    paper = np.array([rng.uniform(215, 255)] * 3, np.float32) * np.array([rng.uniform(0.92, 1), rng.uniform(0.96, 1), 1], np.float32)
    if rng.random() < 0.1: paper = rc(rng, 170, 250)  # coloured paper
    if kind == "receipt": paper = np.array([rng.uniform(225, 250)] * 3, np.float32)

    # what lies under the page: other sheets, a notebook body, or nothing
    under = rng.random()
    c0 = q.mean(0)
    if kind == "notebook" and rng.random() < 0.7:
        thick = rng.uniform(2, 9)
        d = rng.normal(0, 1, 2); d /= np.linalg.norm(d) + 1e-6
        cover = (q - c0) * rng.uniform(1.0, 1.05) + c0 + d * thick
        cv2.fillConvexPoly(img, np.int32(cover), rc(rng, 20, 220).tolist())
        for k in range(1, int(thick)):  # visible stack of pages
            cv2.polylines(img, [np.int32((q - c0) * 1.0 + c0 + d * k)], True, (paper * rng.uniform(0.75, 0.95)).tolist(), 1)
    elif under < 0.25:
        for _ in range(int(rng.integers(1, 4))):
            q2 = (q - c0) @ np.array([[1, 0], [0, 1]]) + c0
            a = np.deg2rad(rng.uniform(-15, 15)); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            q2 = (q - c0) @ R.T * rng.uniform(0.95, 1.1) + c0 + rng.normal(0, 0.06, 2) * [W, H]
            cv2.fillConvexPoly(img, np.int32(q2), (paper * rng.uniform(0.9, 1.04)).tolist())

    ph, pw = (700, int(700 * aspect)) if aspect <= 1 else (int(700 / aspect), 700)
    content = page_image(rng, kind, pw, ph, paper)
    bend = bend_field(rng, W, H, rng.uniform(1, 6)) if rng.random() < 0.3 else None
    if rng.random() < 0.15:  # soft drop shadow under the page
        _, sal = warp_layer(np.zeros((ph, pw, 3), np.float32), q + rng.normal(0, 3, 2), W, H)
        img = img * (1 - cv2.GaussianBlur(sal, (0, 0), 4)[..., None] * rng.uniform(0.2, 0.5))
    lay, al = warp_layer(content, q, W, H, bend)
    if rng.random() < 0.5:  # paper edge slightly darker (thickness / light falloff)
        e = cv2.morphologyEx((al > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        lay = lay * (1 - e[..., None] * rng.uniform(0.05, 0.25))
    img = paste(img, lay, al)
    if kind == "notebook" and rng.random() < 0.5:  # spiral binding along one side
        i = rng.integers(4); a, b = q[i], q[(i + 1) % 4]; n = int(np.linalg.norm(b - a) / rng.uniform(8, 14))
        inward = (c0 - (a + b) / 2); inward /= np.linalg.norm(inward) + 1e-6
        for k in range(1, n):
            p = a + (b - a) * k / n
            cv2.ellipse(img, tuple(np.int32(p - inward * 2)), (4, 2), math.degrees(math.atan2(*(b - a)[::-1])) + 90, 0, 360, rc(rng, 30, 200).tolist(), 2)
    if rng.random() < 0.25:  # fingers holding the page
        skin = np.array([rng.uniform(60, 160), rng.uniform(90, 180), rng.uniform(140, 230)], np.float32)
        for _ in range(int(rng.integers(1, 3))):
            i = rng.integers(4); a, b = q[i], q[(i + 1) % 4]; mid = a + (b - a) * rng.uniform(0.15, 0.85)
            out = mid - c0; out /= np.linalg.norm(out) + 1e-6
            cen = mid + out * rng.uniform(5, 40)
            cv2.ellipse(img, tuple(np.int32(cen)), (int(rng.uniform(12, 30)), int(rng.uniform(30, 70))), math.degrees(math.atan2(out[1], out[0])) + 90, 0, 360, skin.tolist(), -1)

    # lighting
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1 + rng.uniform(-0.4, 0.4) * (xx / W - 0.5) + rng.uniform(-0.4, 0.4) * (yy / H - 0.5)
    light *= rng.choice([rng.uniform(0.35, 0.7), rng.uniform(0.7, 1.15)], p=[0.3, 0.7])
    for _ in range(int(rng.integers(0, 3))):  # shadows (hand, phone, head)
        poly = np.int32(rng.uniform(-0.3, 1.3, (int(rng.integers(3, 7)), 2)) * [W, H]); sh = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(sh, cv2.convexHull(poly), 1.0); sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(2, 30))
        light *= 1 - sh * rng.uniform(0.2, 0.6)
    cast = rng.choice([np.array([1, 1, 1]), np.array([0.75, 0.92, 1.1]), np.array([1.1, 1, 0.9]), rng.uniform(0.8, 1.15, 3)], p=[0.35, 0.35, 0.1, 0.2]) if False else [np.array([1, 1, 1]), np.array([0.72, 0.9, 1.1]), np.array([1.1, 1, 0.9]), rng.uniform(0.8, 1.15, 3)][rng.choice(4, p=[0.35, 0.35, 0.1, 0.2])]
    img = img * light[..., None] * np.asarray(cast, np.float32)
    if rng.random() < 0.2:  # glare
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 90)
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(60, 200))[..., None]
    if rng.random() < 0.4:  # vignetting
        v = 1 - rng.uniform(0.1, 0.4) * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2)
        img *= v[..., None]
    # camera
    if rng.random() < 0.15:
        k = int(rng.integers(3, 9)); ker = np.zeros((k, k), np.float32); ker[k // 2] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1); ker = cv2.warpAffine(ker, M, (k, k)); ker /= ker.sum() + 1e-6
        img = cv2.filter2D(img, -1, ker)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    img = img + rng.normal(0, rng.uniform(1, 9), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 95))]); img = cv2.imdecode(enc, 1)
    return img, canonical(q)


def letterbox256(img, corners):
    h, w = img.shape[:2]; s = min(256 / w, 256 / h); nw, nh = round(w * s), round(h * s)
    ox, oy = (256 - nw) // 2, (256 - nh) // 2
    out = np.zeros((256, 256, 3), np.uint8)
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, corners * [nw / w, nh / h] + [ox, oy]


if __name__ == "__main__":
    import sys, time
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t = time.time(); tiles = []
    for i in range(n):
        im, c = scene(i)
        lb, c2 = letterbox256(im, c)
        v = lb.copy(); cv2.polylines(v, [np.int32(c2)], True, (0, 255, 0), 1)
        for j, p in enumerate(c2): cv2.circle(v, tuple(np.int32(p)), 3, [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)][j], -1)
        tiles.append(np.hstack([lb, v]))
    print("ms/scene", (time.time() - t) / n * 1000)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles) - len(tiles) % 4, 4)]
    cv2.imwrite(os.path.join(HERE, "peek.jpg"), np.vstack(rows))
