"""Hard scenes for edge detection, with known corners."""
import cv2, numpy as np
from synth import make_page, background

SCENES = ["hands", "clutter", "paper_on_paper", "fills_frame", "steep", "dim"]

def render_hard(seed):
    rng = np.random.default_rng(seed); scene = SCENES[seed % len(SCENES)]
    W, H = 1944, 2592
    kind = rng.choice(["wood", "dark", "beige", "white_desk"]) if scene != "paper_on_paper" else rng.choice(["wood", "dark"])
    img = background(rng, kind, W, H)
    page = make_page(rng); ph, pw = page.shape[:2]
    if scene == "clutter":  # books, keyboard, sticky notes, a phone
        for _ in range(int(rng.integers(5, 9))):
            x, y = rng.uniform(-200, W), rng.uniform(-200, H); w_, h_ = rng.uniform(200, 900), rng.uniform(150, 700)
            ang = rng.uniform(0, 180); col = rng.integers(20, 240, 3).astype(float)
            box = cv2.boxPoints(((x, y), (w_, h_), ang)).astype(np.int32)
            cv2.fillConvexPoly(img, box, col.tolist())
            if rng.random() < 0.4:  # keyboard-like texture
                for k in range(30):
                    cx, cy = x + rng.uniform(-w_ / 2, w_ / 2), y + rng.uniform(-h_ / 2, h_ / 2)
                    cv2.rectangle(img, (int(cx), int(cy)), (int(cx + 40), int(cy + 40)), (col * 0.6).tolist(), 2)
    fill = {"fills_frame": rng.uniform(0.9, 0.97), "steep": rng.uniform(0.55, 0.75)}.get(scene, rng.uniform(0.5, 0.8))
    s = fill * H / ph
    cx, cy = W / 2 + rng.uniform(-80, 80), H / 2 + rng.uniform(-100, 100)
    ang = np.deg2rad(rng.uniform(-10, 10))
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) * s
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]); pts = base @ R.T
    pts += rng.uniform(-0.04, 0.04, (4, 2)) * np.array([pw * s, ph * s])
    k = rng.uniform(0.25, 0.4) if scene == "steep" else rng.uniform(0, 0.1)
    pts[0, 0] += k * pw * s / 2; pts[1, 0] -= k * pw * s / 2
    pts += [cx, cy]
    if scene == "fills_frame":  # keep corners just inside the frame
        pts[:, 0] = np.clip(pts[:, 0], 15, W - 15); pts[:, 1] = np.clip(pts[:, 1], 15, H - 15)
    if scene == "paper_on_paper":  # a second white sheet underneath, offset and rotated
        other = make_page(np.random.default_rng(seed + 999)) * 0 + np.array([238, 236, 232])
        off = rng.uniform(-300, 300, 2); a2 = np.deg2rad(rng.uniform(-25, 25))
        R2 = np.array([[np.cos(a2), -np.sin(a2)], [np.sin(a2), np.cos(a2)]])
        p2 = (base @ R2.T) + [cx, cy] + off
        M2 = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), np.float32(p2))
        w2 = cv2.warpPerspective(other.astype(np.uint8), M2, (W, H)); m2 = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M2, (W, H))
        img = np.where(m2[..., None] > 0, w2, img)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), np.float32(pts))
    warped = cv2.warpPerspective(page, M, (W, H)); mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H))
    m = (cv2.GaussianBlur(mask, (3, 3), 0) / 255.0)[..., None]
    img = img * (1 - m) + warped * m
    if scene == "hands":  # two hands holding the sides, thumbs over the page edges
        skin = np.array([215, 175, 150.]) * rng.uniform(0.7, 1.05)
        for side in (0, 1):
            a, b = (pts[0], pts[3]) if side == 0 else (pts[1], pts[2])
            mid = a + (b - a) * rng.uniform(0.35, 0.65)
            out = (mid - pts.mean(0)); out /= np.linalg.norm(out)
            cv2.ellipse(img, tuple(np.int32(mid + out * 220)), (330, 230), 0, 0, 360, skin.tolist(), -1)
            cv2.ellipse(img, tuple(np.int32(mid - out * 40)), (70, 130), 0, 0, 360, skin.tolist(), -1)
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.85 + 0.2 * (xx / W) - 0.12 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W)
    img = img * light[..., None]
    if scene == "dim": img = img * rng.uniform(0.3, 0.45)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.1)
    noise = 12 if scene == "dim" else 3.5
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), (pts / [W, H]).astype(np.float32), scene

if __name__ == "__main__":
    import sys, importlib, time
    mod = importlib.import_module(sys.argv[1]); N = int(sys.argv[2]); base = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res = {s: [] for s in SCENES}; t = 0
    for i in range(N):
        img, gt, scene = render_hard(base + i)
        small = cv2.resize(img, None, fx=640 / 2592, fy=640 / 2592, interpolation=cv2.INTER_AREA)
        t0 = time.time(); q = mod.detect_color(small); t += time.time() - t0
        diag = np.linalg.norm((gt[2] - gt[0]) * [1944, 2592])
        err = 1.0 if q is None else max(np.linalg.norm((q[j] - gt[j]) * [1944, 2592]) for j in range(4)) / diag
        res[scene].append(err)
    allr = sum(res.values(), [])
    print(f"{sys.argv[1]} hard: {sum(e < 0.02 for e in allr)}/{len(allr)}  avg {t / N * 1000:.0f} ms")
    for s in SCENES: print(f"  {s:15s} {sum(e < 0.02 for e in res[s])}/{len(res[s])}  errs {[round(e, 3) for e in res[s]]}")
