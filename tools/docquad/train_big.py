"""Large-scale fine-tuning of the app's page-corner network (docquad.onnx) on a GPU.

Fresh scenes are generated on the fly by gen2.py (never the same scene twice), so the sample can
be as large as you like without filling the disk. The model is checked before training and after
every epoch on two held-out sets, and the best epoch is exported:
  A. 1,500 gen2 scenes from a seed range that training never uses,
  B. 300 full-size photos from synth.py, a separate generator (different backgrounds, shadows,
     fingers), so it measures generalisation, not memorisation of gen2.

Usage (Windows or Linux):
  set DQ_PHOTOS=<folder with OpenCV's samples/data images>
  python train_big.py <docquad.onnx in> <work folder> [scenes per epoch] [epochs]
Writes <work>/docquad_best.onnx and prints the scores. Needs: torch (CUDA), opencv-python-headless
4.10, numpy, pillow, onnx.
"""
import math, os, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import gen2
from onnxtorch import OnnxNet

TRAIN_BASE = 1_000_000      # training seeds start here (old training used 100,000+)
HOLDOUT_BASE = 90_000_000   # held-out seeds, never used for training
xs = np.arange(64) + 0.5; GX, GY = np.meshgrid(xs, xs)


def targets(c):
    """Corner heatmaps (64x64, sigma 2) and page mask, exactly as the original training."""
    c = np.clip(c, 0, 255.999); cc = c / 4.0
    hm = np.exp(-((GX[None] - cc[:, 0, None, None]) ** 2 + (GY[None] - cc[:, 1, None, None]) ** 2) / 8.0).astype(np.float32)
    m = np.zeros((64, 64), np.uint8); cv2.fillConvexPoly(m, np.int32(np.round((cc - 0.5) * 16)), 1, shift=4)
    return hm, m.astype(np.float32)


class Scenes(Dataset):
    def __init__(self, n): self.n = n; self.offset = 0
    def __len__(self): return self.n

    def __getitem__(self, i):
        seed = TRAIN_BASE + self.offset + i
        for retry in range(5):  # a rare font/drawing error must not stop a long run
            try:
                im, c = gen2.scene(seed); break
            except Exception:
                seed += 50_000_000
        img, c = gen2.letterbox256(im, c)
        rng = np.random.default_rng(seed + 7)
        if rng.random() < 0.5: img = img[:, ::-1]; c[:, 0] = 256 - c[:, 0]
        if rng.random() < 0.25: img = img[::-1]; c[:, 1] = 256 - c[:, 1]
        if rng.random() < 0.3: img = np.rot90(img); c = np.stack([c[:, 1], 256 - c[:, 0]], 1)
        img = np.ascontiguousarray(img).astype(np.float32)
        if rng.random() < 0.5: img = np.clip((img - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-25, 25), 0, 255)
        c = gen2.canonical(c); hm, m = targets(c)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(hm), torch.from_numpy(m[None])


def decode(hm, ox, oy, nw, nh, w, h):
    """Same corner decoding as the app (peak + 3x3 weighted centre)."""
    pts = []
    for k in range(4):
        a = hm[k]; iy, ix = np.unravel_index(np.argmax(a), a.shape)
        y0, y1, x0, x1 = max(0, iy - 1), min(64, iy + 2), max(0, ix - 1), min(64, ix + 2)
        wt = 1 / (1 + np.exp(-a[y0:y1, x0:x1])); yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = (wt * (xx + 0.5)).sum() / wt.sum() * 4; cy = (wt * (yy + 0.5)).sum() / wt.sum() * 4
        pts.append(((cx - ox) * w / nw, (cy - oy) * h / nh))
    return np.array(pts)


def quad_error(p, gt):
    """Largest corner distance, whichever corner the network calls "top left" (the crop is the same)."""
    best = 1e9
    for order in (p, p[::-1]):
        for k in range(4):
            q = np.roll(order, k, 0); best = min(best, np.linalg.norm(q - gt, axis=1).max())
    return best


def letterbox(img):
    h, w = img.shape[:2]; s = min(256 / w, 256 / h); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    ox, oy = (256 - nw) // 2, (256 - nh) // 2
    lb = np.zeros((256, 256, 3), np.uint8); lb[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return lb, (ox, oy, nw, nh, w, h)


def holdout_sets(work):
    path = os.path.join(work, "holdout.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True); return [(str(n), x, g, m) for n, x, g, m in d["sets"]]
    print("building held-out sets...", flush=True)
    import synth
    A_x, A_g, A_m = [], [], []
    for i in range(1500):
        im, c = gen2.scene(HOLDOUT_BASE + i)
        lb, meta = letterbox(im); A_x.append(lb); A_g.append(c); A_m.append(meta)
    B_x, B_g, B_m = [], [], []
    kinds = ["wood", "white_desk", "dark", "pattern", "beige"]
    for i in range(300):
        img, gt, _ = synth.render(5000 + i, kinds[i % 5], shadow=(i % 3 != 2), clutter=(i % 4 == 1), finger=(i % 7 == 3))
        lb, meta = letterbox(img); B_x.append(lb); B_g.append(np.asarray(gt) * [img.shape[1], img.shape[0]]); B_m.append(meta)
    sets = [("A gen2 held-out", np.stack(A_x), np.stack(A_g), np.array(A_m)),
            ("B synth.py", np.stack(B_x), np.stack(B_g), np.array(B_m))]
    np.savez(path, sets=np.array(sets, dtype=object)); return sets


@torch.no_grad()
def evaluate(model, sets, dev, tag):
    model.eval(); out = {}
    for name, X, G, M in sets:
        errs = []
        for b in range(0, len(X), 128):
            x = torch.from_numpy(cv2.cvtColor(X[b:b + 128].reshape(-1, 256, 3), cv2.COLOR_BGR2RGB)
                                 .reshape(-1, 256, 256, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            hm = model(x)[0].float().cpu().numpy()
            for j in range(len(hm)):
                ox, oy, nw, nh, w, h = M[b + j]; gt = G[b + j]
                p = decode(hm[j], ox, oy, nw, nh, w, h)
                diag = np.linalg.norm(gt[2] - gt[0])
                errs.append(quad_error(p, gt) / diag)
        e = np.array(errs)
        out[name] = (float((e < 0.02).mean()), float((e < 0.05).mean()), float(np.median(e)), float(np.minimum(e, 0.5).mean()))
    print(tag, "  ".join(f"{n}: <2% {a*100:.1f}%  <5% {b*100:.1f}%  median {c*100:.2f}%  mean {d*100:.2f}%"
                          for n, (a, b, c, d) in out.items()), flush=True)
    model.train(); return out


def score(r):  # higher is better: fraction of near-exact corners on both sets
    return sum(v[0] + 0.5 * v[1] for v in r.values())


def main():
    src, work = sys.argv[1], sys.argv[2]
    per_epoch = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    lr = float(os.environ.get("LR", "1e-4")); bs = 64
    os.makedirs(work, exist_ok=True); torch.manual_seed(0)
    dev = "cuda"
    model = OnnxNet(src).to(dev).train()
    sets = holdout_sets(work)
    best = evaluate(model, sets, dev, "before"); best_score = score(best); best_tag = "original"
    model.save(os.path.join(work, "docquad_best.onnx"))

    data = Scenes(per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * (per_epoch // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    workers = max(2, (os.cpu_count() or 4) - 2)
    for ep in range(epochs):
        data.offset = ep * per_epoch
        loader = DataLoader(data, batch_size=bs, shuffle=True, num_workers=workers, drop_last=True, prefetch_factor=4)
        t0 = time.time(); run = None
        for b, (x, hm, m) in enumerate(loader):
            x, hm, m = x.to(dev, non_blocking=True), hm.to(dev), m.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ph, pm = model(x)
            loss = F.binary_cross_entropy_with_logits(ph.float(), hm) + F.binary_cross_entropy_with_logits(pm.float(), m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            if b % 100 == 0:
                print(f"epoch {ep} batch {b}/{len(loader)} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(b + 1) * bs / (time.time() - t0):.0f} scenes/s", flush=True)
        model.save(os.path.join(work, f"docquad_ep{ep}.onnx"))
        r = evaluate(model, sets, dev, f"epoch {ep}")
        if score(r) > best_score:
            best_score, best_tag = score(r), f"epoch {ep}"
            model.save(os.path.join(work, "docquad_best.onnx"))
        print(f"best so far: {best_tag}", flush=True)
    print("DONE best:", best_tag, flush=True)


if __name__ == "__main__":
    main()
