import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, cv2
from multiprocessing import Pool
from fastgen import scene, OH, OW
torch.set_num_threads(2)

def gen(seeds):
    with Pool(2) as p: out = p.map(scene, seeds, chunksize=64)
    out = [(o[0], o[1], o[2]) for o in out]
    X = np.stack([o[0] for o in out]); Y = np.stack([o[1] for o in out]); Q = np.stack([o[2] for o in out])
    return X, Y, Q

class Block(nn.Module):
    def __init__(s, i, o, d=1):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                            nn.Conv2d(o, o, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(s, x): return s.c(x)

class PageNet(nn.Module):
    """Tiny U-Net: BGR 0-255 in, page probability out (same size)."""
    def __init__(s, c=(8, 16, 24, 32, 48)):
        super().__init__()
        s.e0, s.e1, s.e2, s.e3 = Block(3, c[0]), Block(c[0], c[1]), Block(c[1], c[2]), Block(c[2], c[3])
        s.b = nn.Sequential(Block(c[3], c[4]), Block(c[4], c[4], d=2))
        s.d3, s.d2, s.d1, s.d0 = Block(c[4] + c[3], c[3]), Block(c[3] + c[2], c[2]), Block(c[2] + c[1], c[1]), Block(c[1] + c[0], c[0])
        s.out = nn.Conv2d(c[0], 1, 1)
    def forward(s, x):
        x = x / 255.0 - 0.5
        e0 = s.e0(x); e1 = s.e1(F.max_pool2d(e0, 2)); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2))
        b = s.b(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)
        d3 = s.d3(torch.cat([up(b), e3], 1)); d2 = s.d2(torch.cat([up(d3), e2], 1))
        d1 = s.d1(torch.cat([up(d2), e1], 1)); d0 = s.d0(torch.cat([up(d1), e0], 1))
        return torch.sigmoid(s.out(d0))

if __name__ == "__main__":
    N = int(sys.argv[1]); EPOCHS = int(sys.argv[2])
    t = time.time()
    X, Y, _ = gen(range(100000, 100000 + N)); Xv, Yv, _ = gen(range(900000, 900800))
    print(f"data {X.shape} in {time.time() - t:.0f}s", flush=True)
    # keep images as bytes (1 GB); convert one batch at a time
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    Xvt = torch.from_numpy(Xv); Yvt = torch.from_numpy(Yv)
    fx = lambda t: t.permute(0, 3, 1, 2).float(); fy = lambda t: t[:, None].float()
    net = PageNet(); print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    steps = EPOCHS * (N // 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    best = 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(N); tl = 0
        for i in range(0, N - 31, 32):
            idx = perm[i:i + 32]; x = fx(Xt[idx]); y = fy(Yt[idx])
            if np.random.rand() < 0.5: x, y = x.flip(3), y.flip(3)
            x = x * (0.8 + 0.4 * torch.rand(x.shape[0], 3, 1, 1)) + 20 * torch.randn(x.shape[0], 1, 1, 1)  # colour/brightness jitter
            x = x.clamp(0, 255)
            p = net(x)
            inter = (p * y).sum((1, 2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy(p, y) + dice.mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
            if ep == 0 and i == 32 * 20: print(f"  20 batches in {time.time() - t:.0f}s total so far", flush=True)
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(fx(Xvt[i:i + 100])) for i in range(0, len(Xvt), 100)]) > 0.5
            yv = fy(Yvt) > 0.5
            inter = (pv & yv).sum((1, 2, 3)).float(); union = (pv | yv).sum((1, 2, 3)).float()
            iou = (inter / union.clamp(min=1)).mean().item()
        print(f"epoch {ep + 1}/{EPOCHS} loss {tl / (N // 32):.3f} val IoU {iou:.4f}  {time.time() - t:.0f}s", flush=True)
        if iou > best: best = iou; torch.save(net.state_dict(), "/root/net/pagenet.pt")
    print("best IoU", best)
