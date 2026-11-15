"""Realistic phone photos of already-cropped pages, with the perfect clean result and region masks."""
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
def _font(linux, windows):
    """The DejaVu font on Linux, a similar standard font on Windows."""
    import os
    for f in (linux, os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", windows)):
        if os.path.exists(f): return f
    return linux
F = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"); FB = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf")
FS = _font("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "times.ttf"); FM = _font("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "consola.ttf")
WORDS = ("account electricity bill customer meter reading units amount payable due date the of and to in for on with "
         "total charge vat service period previous present tariff student marks result exam roll class signature").split()

def txt(rng, n): return " ".join(rng.choice(WORDS, n)).capitalize()

def page(rng, kind):
    """Returns clean RGB page (float), paper colour, and masks dict."""
    W, H = (900, 2200) if kind == "receipt" else (1240, 1754)
    paper = {"colored": np.array(rng.choice([[245, 235, 180], [250, 215, 220], [215, 230, 245], [225, 240, 210]])),
             "receipt": np.array([248, 248, 244])}.get(kind, np.array([rng.integers(238, 252)] * 3) - [0, rng.integers(0, 5), rng.integers(0, 10)])
    img = Image.new("RGB", (W, H), tuple(int(v) for v in paper)); d = ImageDraw.Draw(img)
    big = Image.new("L", (W, H), 0); db = ImageDraw.Draw(big)        # large dark/colour areas that must survive
    col = Image.new("L", (W, H), 0); dc = ImageDraw.Draw(col)        # coloured marks
    faint = Image.new("L", (W, H), 0); dfn = ImageDraw.Draw(faint)   # faint marks (pencil, thermal)
    ink = (20, 20, 28)
    if kind == "receipt":
        fnt = ImageFont.truetype(FM, 30); y = 60; gray = int(rng.integers(95, 140))
        for i in range(48):
            t = f"{txt(rng, 2)[:18]:<18} {rng.integers(10, 999):>5}.{rng.integers(0, 99):02d}"
            d.text((50, y), t, font=fnt, fill=(gray, gray, gray)); dfn.text((50, y), t, font=fnt, fill=255); y += 42
        return np.array(img, float), paper.astype(float), big, col, faint
    if kind == "book":
        fnt = ImageFont.truetype(FS, 22); y = 80
        for i in range(62):
            d.text((90, y), txt(rng, 12), font=fnt, fill=ink); y += 26
    else:
        d.text((90, 70), "RESULT SHEET" if kind == "form" else "ELECTRICITY BILL", font=ImageFont.truetype(FB, 60), fill=ink)
        fnt = ImageFont.truetype(F, 24); y = 170
        if kind in ("form", "photo"):   # coloured header bar with white text (must stay coloured)
            c = tuple(int(v) for v in rng.choice([[30, 70, 160], [150, 30, 40], [20, 110, 70]]))
            d.rectangle([90, y, W - 90, y + 60], fill=c); db.rectangle([90, y, W - 90, y + 60], fill=255); dc.rectangle([90, y, W - 90, y + 60], fill=255)
            d.text((110, y + 15), "Student information", font=fnt, fill=(255, 255, 255)); y += 90
        for i in range(10):
            d.text((90, y), txt(rng, 9), font=fnt, fill=ink); y += 36
        # table, some cells light-grey shaded
        for r in range(6):
            for c in range(4):
                box = [90 + c * 265, y + r * 46, 90 + (c + 1) * 265, y + (r + 1) * 46]
                if r == 0 and kind == "form": d.rectangle(box, fill=(205, 205, 205))
                d.rectangle(box, outline=(50, 50, 55), width=2)
                d.text((box[0] + 10, box[1] + 9), str(rng.integers(10, 9999)), font=fnt, fill=ink)
        y += 300
        if kind == "photo":   # a photo and a solid black box
            ph = rng.integers(0, 255, (8, 10, 3)).astype(np.uint8)
            ph = cv2.resize(ph, (420, 320), interpolation=cv2.INTER_CUBIC)
            ph = cv2.GaussianBlur(ph, (0, 0), 6) * 0.8
            img.paste(Image.fromarray(ph.astype(np.uint8)), (90, y)); db.rectangle([90, y, 510, y + 320], fill=255)
            d.rectangle([600, y, 1100, y + 140], fill=(15, 15, 18)); db.rectangle([600, y, 1100, y + 140], fill=255)
            d.text((620, y + 50), "IMPORTANT", font=ImageFont.truetype(FB, 40), fill=(255, 255, 255))
            y += 360
        for i in range(8):
            d.text((90, y), txt(rng, 9), font=fnt, fill=ink); y += 36
        # highlighter over a line (multiply)
        hl = np.array(img, float); yy = y - 36 * 5
        hl[yy - 4:yy + 30, 85:800] *= np.array([1.0, 0.96, 0.35]); img = Image.fromarray(hl.astype(np.uint8)); d = ImageDraw.Draw(img)
        dc.rectangle([85, yy - 4, 800, yy + 30], fill=255)
        # blue pen handwriting + pencil note
        pts = [(100 + i * 9, y + 60 + 18 * np.sin(i / 2.0) + rng.normal(0, 2)) for i in range(70)]
        d.line(pts, fill=(25, 45, 160), width=3); dc.line(pts, fill=255, width=3)
        pts = [(100 + i * 9, y + 140 + 14 * np.sin(i / 1.7)) for i in range(60)]
        d.line(pts, fill=(150, 150, 150), width=2); dfn.line(pts, fill=255, width=2)
        # round stamp
        d.ellipse([850, y + 40, 1110, y + 190], outline=(170, 30, 40), width=6); dc.ellipse([850, y + 40, 1110, y + 190], outline=255, width=6)
        d.text((900, y + 95), "PAID", font=ImageFont.truetype(FB, 44), fill=(170, 30, 40)); dc.text((900, y + 95), "PAID", font=ImageFont.truetype(FB, 44), fill=255)
    return np.array(img, float), paper.astype(float), big, col, faint

def photograph(clean, rng, scale):
    """Simulate a phone photo of the page: lighting, colour cast, shadows, glare, noise, blur, JPEG."""
    H, W = clean.shape[:2]
    img = cv2.resize(clean, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    # paper texture
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 0.02 + 1
    img = img * tex[..., None]
    cast = np.array(rng.choice([[1.0, 0.86, 0.66], [0.93, 1.0, 0.9], [0.95, 0.97, 1.0], [1.0, 0.95, 0.85], [0.9, 0.95, 1.0]]))
    expo = rng.uniform(0.5, 1.05)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi); gx, gy = np.cos(ang), np.sin(ang)
    grad = 1 + rng.uniform(0, 0.35) * ((xx / w - 0.5) * gx + (yy / h - 0.5) * gy)
    vig = 1 - rng.uniform(0, 0.3) * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    light = grad * vig
    sh = np.zeros((h, w), np.float32)
    nsh = rng.integers(0, 3)
    for k in range(nsh):  # hand / phone shadows: hard or soft
        poly = np.array([[rng.uniform(-0.2, 1.2) * w, rng.uniform(-0.2, 1.2) * h] for _ in range(5)], np.int32)
        m = np.zeros((h, w), np.float32); cv2.fillConvexPoly(m, cv2.convexHull(poly), 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.choice([rng.uniform(3, 10), rng.uniform(25, 90)]))
        sh = np.maximum(sh, m * rng.uniform(0.3, 0.65))
    shadow_col = np.array([0.92, 0.97, 1.08])  # shadows lit by blue-ish sky light
    L = expo * light[..., None] * cast * (1 - sh[..., None] * (2 - shadow_col))
    out = img * L
    if rng.random() < 0.3:  # glare hotspot
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h; r = rng.uniform(0.08, 0.2) * w
        out += (np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))) * rng.uniform(30, 80))[..., None]
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), rng.uniform(0.6, 1.3))
    noise_sd = 2 + (1.05 - expo) * 14  # darker photo = noisier
    out = out + rng.normal(0, 1, out.shape) * np.sqrt(np.maximum(out, 1)) * noise_sd / 12 + rng.normal(0, noise_sd / 3, (h, w, 1))
    out = np.clip(out, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(75, 93))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), nsh

KINDS = ["bill", "form", "photo", "book", "receipt", "colored"]
def sample(seed, scale=None):
    rng = np.random.default_rng(seed); kind = KINDS[seed % len(KINDS)]
    clean, paper, big, col, faint = page(rng, kind)
    scale = scale or rng.uniform(1.4, 2.2)
    photo, nsh = photograph(clean, rng, scale)
    h, w = photo.shape[:2]
    ideal = np.clip(clean / paper * 255, 0, 255)                                 # paper -> pure white
    ideal = cv2.resize(ideal, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8)
    rs = lambda m: cv2.resize(np.array(m), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return dict(photo=photo, ideal=cv2.cvtColor(ideal, cv2.COLOR_RGB2BGR), big=rs(big), col=rs(col), faint=rs(faint), kind=kind, shadows=nsh)
