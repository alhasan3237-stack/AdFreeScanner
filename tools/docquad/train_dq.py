"""Fine-tune DocQuadNet-256 (converted from the MakeACopy app) on the realistic scenes.
Targets exactly as in its original training: corner heatmaps (64x64, sigma 2, BCE on logits) and
page mask (64x64 polygon, BCE on logits), RGB 0..1 input, letterboxed with black bars."""
import numpy as np, torch, torch.nn.functional as F, cv2, time, sys, os, pickle, math
torch.set_num_threads(2); torch.manual_seed(0)
sys.path.insert(0, "/root/bench")
EPOCHS = int(os.environ.get("EPOCHS", "3")); LR = float(os.environ.get("LR", "3e-4")); BS = 16
X = np.concatenate([np.load(f"/root/gen/data_{p}.npy", mmap_mode="r") for p in (0, 1)]) if False else None
parts = [np.load(f"/root/gen/data_{p}.npy", mmap_mode="r") for p in (0, 1)]
Cs = [np.load(f"/root/gen/corners_{p}.npy") for p in (0, 1)]
index = [(p, i) for p in (0, 1) for i in range(len(Cs[p]))]
print("scenes", len(index), flush=True)
xs = np.arange(64) + 0.5; GX, GY = np.meshgrid(xs, xs)

def canonical(c):
    m = c.mean(0); ang = np.arctan2(c[:, 1] - m[1], c[:, 0] - m[0])
    order = sorted(range(4), key=lambda i: (float(ang[i]), i)); sums = [float(c[i, 0] + c[i, 1]) for i in order]
    tl = min(range(4), key=lambda k: (sums[k], k)); return c[order[tl:] + order[:tl]]

def targets(c):
    c = np.clip(c, 0, 255.999); cc = c / 4.0
    hm = np.exp(-((GX[None] - cc[:, 0, None, None]) ** 2 + (GY[None] - cc[:, 1, None, None]) ** 2) / 8.0).astype(np.float32)
    m = np.zeros((64, 64), np.uint8); cv2.fillConvexPoly(m, np.int32(np.round((cc - 0.5) * 16)), 1, shift=4)
    return hm, m.astype(np.float32)

def sample(p, i, rng):
    img = np.array(parts[p][i]); c = Cs[p][i].copy()
    if rng.random() < 0.5: img = img[:, ::-1]; c[:, 0] = 256 - c[:, 0]
    if rng.random() < 0.25: img = img[::-1]; c[:, 1] = 256 - c[:, 1]
    if rng.random() < 0.3:  # rotate 90 (portrait <-> landscape layout)
        img = np.rot90(img); c = np.stack([c[:, 1], 256 - c[:, 0]], 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    if rng.random() < 0.5: img = np.clip((img - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-25, 25), 0, 255)
    c = canonical(c); hm, m = targets(c)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
    return x, hm, m

model = torch.load("/root/dq/docquad_torch.pt", weights_only=False)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = len(index) // BS; total = steps_per_epoch * EPOCHS
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

def predict(img_bgr):
    """Same as the app: letterbox to 256 with black bars, returns corners (px) and peak probs."""
    h, w = img_bgr.shape[:2]; s = min(256 / w, 256 / h); nw, nh = round(w * s), round(h * s); ox, oy = (256 - nw) // 2, (256 - nh) // 2
    lb = np.zeros((256, 256, 3), np.uint8); lb[oy:oy + nh, ox:ox + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255)
    with torch.no_grad(): hm, mk = model(x)
    hm = hm[0].numpy(); pts = []; pk = []
    for k in range(4):
        a = hm[k]; iy, ix = np.unravel_index(np.argmax(a), a.shape); pk.append(1 / (1 + np.exp(-a[iy, ix])))
        y0, y1, x0, x1 = max(0, iy - 1), min(64, iy + 2), max(0, ix - 1), min(64, ix + 2)
        wt = 1 / (1 + np.exp(-a[y0:y1, x0:x1])); yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = (wt * (xx + 0.5)).sum() / wt.sum() * 4; cy = (wt * (yy + 0.5)).sum() / wt.sum() * 4
        pts.append(((cx - ox) * w / nw, (cy - oy) * h / nh))
    return np.array(pts), np.array(pk)

DATA = pickle.load(open("/root/sets/all.pkl", "rb"))
def evaluate(tag):
    model.eval(); res = {}
    for name, sm, gt, fn in DATA:
        h, w = sm.shape[:2]; pts, pk = predict(sm); q = pts / [w, h]
        diag = np.linalg.norm((gt[2] - gt[0]) * [w, h]); e = max(np.linalg.norm((q[j] - gt[j]) * [w, h]) for j in range(4)) / diag
        r = res.setdefault(name, [0, 0, 0.0]); r[0] += e < (0.02 if name in ("standard", "unseen", "hard") else 0.03); r[1] += 1; r[2] += min(e, 0.5)
    print(tag, {k: f"{a}/{b} err{c / b:.3f}" for k, (a, b, c) in res.items()}, flush=True)
    model.train(); return res

evaluate("before")
rng = np.random.default_rng(1); step = 0
for ep in range(EPOCHS):
    perm = rng.permutation(len(index)); t0 = time.time(); run = 0.0
    for b in range(steps_per_epoch):
        batch = [sample(*index[j], rng) for j in perm[b * BS:(b + 1) * BS]]
        x = torch.from_numpy(np.stack([t[0] for t in batch]).astype(np.float32))
        hm = torch.from_numpy(np.stack([t[1] for t in batch])); m = torch.from_numpy(np.stack([t[2] for t in batch]))[:, None]
        ph, pm = model(x)
        loss = F.binary_cross_entropy_with_logits(ph, hm) + F.binary_cross_entropy_with_logits(pm, m)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        run = 0.98 * run + 0.02 * loss.item() if b else loss.item()
        if b % 100 == 0: print(f"ep{ep} {b}/{steps_per_epoch} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    torch.save(model, f"/root/ft/dq_ep{ep}.pt")
    evaluate(f"epoch {ep}")
