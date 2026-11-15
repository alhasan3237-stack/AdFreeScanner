"""Synthetic phone photos of documents with known page corners."""
import cv2, numpy as np, random
from PIL import Image, ImageDraw, ImageFont

def _font(linux, windows):
    """The DejaVu font on Linux, a similar standard font on Windows."""
    import os
    for f in (linux, os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", windows)):
        if os.path.exists(f): return f
    return linux
FONT = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf")
FONTB = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf")
WORDS = ("account electricity bill customer meter reading units consumed amount payable due date "
         "the of and to in for on with total charge vat service period previous present tariff").split()

def make_page(rng, W=1240, H=1754):
    tint = rng.integers(225, 250)
    paper = np.full((H, W, 3), (tint, tint - rng.integers(0, 6), tint - rng.integers(0, 12)), np.uint8)
    img = Image.fromarray(paper); d = ImageDraw.Draw(img)
    d.text((90, 80), "ELECTRICITY BILL", font=ImageFont.truetype(FONTB, 64), fill=(20, 20, 30))
    d.text((90, 170), "Dhaka Power Distribution Co.  Bill date: 12/10/2026", font=ImageFont.truetype(FONT, 26), fill=(40, 40, 50))
    f = ImageFont.truetype(FONT, 24); y = 260
    for i in range(32):
        line = " ".join(rng.choice(WORDS, rng.integers(6, 11)))
        d.text((90, y), line.capitalize(), font=f, fill=(25, 25, 35)); y += 38
        if i == 12:  # a table
            for r in range(5):
                for c in range(4):
                    d.rectangle([90 + c * 260, y + r * 44, 90 + (c + 1) * 260, y + (r + 1) * 44], outline=(60, 60, 70), width=2)
                    d.text((100 + c * 260, y + r * 44 + 8), f"{rng.integers(10, 9999)}", font=f, fill=(25, 25, 35))
            y += 240
    # blue stamp + red signature
    d.ellipse([850, 1450, 1110, 1600], outline=(30, 60, 170), width=6)
    d.text((880, 1505), "PAID", font=ImageFont.truetype(FONTB, 48), fill=(30, 60, 170))
    d.line([(150, 1560), (260, 1500), (330, 1580), (420, 1510)], fill=(170, 30, 40), width=5)
    return np.array(img)

def background(rng, kind, W, H):
    if kind == "wood":
        x = np.linspace(0, 1, W)[None, :]; yy = np.linspace(0, 1, H)[:, None]
        grain = np.sin((x * 40 + np.sin(yy * 6) * 2) * np.pi) * 0.5 + 0.5
        base = np.stack([90 + 50 * grain, 60 + 35 * grain, 35 + 20 * grain], -1)
    elif kind == "white_desk":  # low contrast: light grey-white table
        base = np.full((H, W, 3), (205, 208, 210), float)
    elif kind == "dark":
        base = np.full((H, W, 3), (45, 45, 50), float)
    elif kind == "pattern":  # checkered cloth
        c = ((np.arange(W)[None, :] // 60 + np.arange(H)[:, None] // 60) % 2)[..., None]
        base = np.where(c, np.array([200, 80, 70.]), np.array([230, 220, 200.]))
    else:  # beige bedsheet, low contrast with warm paper
        base = np.full((H, W, 3), (200, 195, 180), float)
    base = base + rng.normal(0, 6, (H, W, 1)) + rng.normal(0, 3, (H, W, 3))
    return np.clip(base, 0, 255)

def render(seed, kind=None, shadow=True, clutter=False, finger=False):
    rng = np.random.default_rng(seed)
    W, H = 1944, 2592
    kind = kind or rng.choice(["wood", "white_desk", "dark", "pattern", "beige"])
    bg = background(rng, kind, W, H)
    page = make_page(rng)
    ph, pw = page.shape[:2]
    # random perspective: page fills 45-80% of frame, tilted up to ~12°
    cx, cy = W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-200, 200)
    s = rng.uniform(0.55, 0.85) * H / ph
    ang = np.deg2rad(rng.uniform(-12, 12))
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) * s
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = base @ R.T
    pts += rng.uniform(-0.06, 0.06, (4, 2)) * np.array([pw * s, ph * s])  # perspective
    # make top narrower sometimes (phone tilted)
    k = rng.uniform(0, 0.12); pts[0, 0] += k * pw * s / 2; pts[1, 0] -= k * pw * s / 2
    pts += [cx, cy]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), np.float32(pts))
    warped = cv2.warpPerspective(page, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H))
    m = (cv2.GaussianBlur(mask, (3, 3), 0) / 255.0)[..., None]
    img = bg * (1 - m) + warped * m
    if clutter:  # a phone and a pen lying near the page
        p0 = pts.mean(0)
        cv2.rectangle(img, (int(p0[0] + 600), int(p0[1] - 900)), (int(p0[0] + 900), int(p0[1] - 300)), (20, 20, 25), -1)
        cv2.line(img, (100, int(H * 0.85)), (int(W * 0.6), int(H * 0.95)), (30, 30, 160), 25)
    if finger:  # thumb over the bottom-left corner
        bl = pts[3]
        cv2.ellipse(img, (int(bl[0] + 30), int(bl[1] + 20)), (110, 200), 30, 0, 360, (215, 170, 140), -1)
    # lighting: vignette + gradient
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.82 + 0.25 * (xx / W) * rng.uniform(0.3, 1) - 0.15 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W)
    img = img * light[..., None]
    if shadow:  # hard-ish shadow of a hand/phone from one side
        sh = np.zeros((H, W), np.float32)
        poly = np.array([[0, H * rng.uniform(0.2, 0.5)], [W * rng.uniform(0.45, 0.7), H * rng.uniform(0.35, 0.6)],
                         [W * rng.uniform(0.5, 0.8), H], [0, H]], np.int32)
        cv2.fillPoly(sh, [poly], 1.0)
        sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(8, 30))
        img = img * (1 - rng.uniform(0.35, 0.55) * sh[..., None])
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.1)
    img = np.clip(img + rng.normal(0, 3.5, img.shape), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    gt = pts / [W, H]
    return img, gt.astype(np.float32), kind

if __name__ == "__main__":
    import os; os.makedirs("/root/bench/samples", exist_ok=True)
    for i, (k, c, f) in enumerate([("wood", False, False), ("white_desk", False, False), ("pattern", True, False), ("beige", False, True), ("dark", True, True)]):
        img, gt, kind = render(100 + i, k, True, c, f)
        cv2.imwrite(f"/root/bench/samples/s{i}_{kind}.jpg", cv2.resize(img, None, fx=0.3, fy=0.3))
    print("ok")
