import cv2, numpy as np, glob, os, json, sys, time, v3, dq
v3.NET_CAND = False
pagenet = cv2.dnn.readNetFromONNX(os.path.expanduser("~/net/pagenet.onnx"))
def pagenet_prob(sm):
    pagenet.setInput(cv2.dnn.blobFromImage(sm, 1.0, (96, 128), (0, 0, 0), False, False)); return pagenet.forward()[0, 0]
def dq_image_mask(mask64, lb, w, h):
    s, ox, oy = lb
    big = cv2.resize(mask64, (256, 256), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(round(ox)), int(round(oy)); x1, y1 = int(round(ox + w * s)), int(round(oy + h * s))
    crop = big[y0:y1, x0:x1]
    return cv2.resize(crop, (96, 128), interpolation=cv2.INTER_LINEAR)
def load_sets():
    data = []
    for name in ["standard", "unseen", "hard"]:
        for f in sorted(glob.glob(os.path.expanduser(f"~/sets/{name}/*.png"))):
            data.append((name, cv2.imread(f), np.load(f[:-4] + ".npy"), "syn"))
    gt = json.load(open(os.path.expanduser("~/real/gt.json")))
    for fn, q in gt.items():
        im = cv2.imread(os.path.expanduser("~/real/set/" + fn)); h, w = im.shape[:2]; s = 640 / max(h, w)
        data.append(("real", cv2.resize(im, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA), np.array(q), fn))
    A = os.path.expanduser("~/real/dwin-gharibi_AmazingScanner/data/")
    for sub, js, tag in [("real/own/", "real/own/annotations.json", "amz"), ("test_pack/", "test_pack/ground_truth/corners.json", "pack")]:
        for it in json.load(open(A + js))["items"]:
            im = cv2.imread(A + sub + it["file"]); h, w = im.shape[:2]; s = 640 / max(h, w)
            sm = cv2.resize(im, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)
            data.append((tag, sm, np.array(it["corners"]) / [w, h], it["file"]))
    return data
def err_of(q, gt, w, h):
    if q is None: return 1.0
    q = np.array(q); diag = np.linalg.norm((gt[2] - gt[0]) * [w, h])
    best = 9.0
    for g in (gt, gt[::-1]):  # corner order may start anywhere / go either way in the labels
        for k in range(4):
            gg = np.roll(g, k, axis=0)
            best = min(best, max(np.linalg.norm((q[j] - gg[j]) * [w, h]) for j in range(4)) / diag)
    return best
def run(cfg, data, verbose=False):
    res = {}; fails = []
    for name, sm, gt, tag in data:
        h, w = sm.shape[:2]
        pts, pk, mk, lb = dq.run(sm)
        dqq = pts / [w, h]
        mode = cfg["mode"]
        v3.EXTRA = []; v3.EXTRA_W = cfg.get("extra_w", 1.0); v3.NET_W = cfg.get("net_w", 0.3)
        if mode == "dq":
            q = dqq
        else:
            prob = pagenet_prob(sm) if cfg.get("prob", "pagenet") == "pagenet" else dq_image_mask(mk, lb, w, h)
            if cfg.get("prob") == "both": prob = np.sqrt(pagenet_prob(sm) * dq_image_mask(mk, lb, w, h))
            if mode == "fuse" and pk.min() >= cfg.get("pk_min", 0.0): v3.EXTRA = [pts]
            v3.EXTRA_CONF = float(pk.min()); v3.TRUST = cfg.get("trust", 9.0); v3.AGREE = cfg.get("agree", 0.85)
            q = v3.detect_color(sm, prob=prob)
            if mode == "pick":  # trust DQ when it's confident, else v3
                if pk.min() >= cfg.get("pk_min", 0.6): q = dqq
        e = err_of(q, gt, w, h); good = e < (0.02 if name in ('standard','unseen','hard') else 0.03)
        r = res.setdefault(name, [0, 0, 0, 0.0]); r[0] += good; r[1] += 1; r[2] += e < 0.05; r[3] += min(e, 0.5)
        if not good: fails.append((name, tag if tag != "syn" else "", round(e, 3), round(float(pk.min()), 2)))
    tot = sum(v[0] for v in res.values()), sum(v[1] for v in res.values())
    print(cfg, " ".join(f"{k}:{a}/{b}|5%:{c}|err{d / b:.3f}" for k, (a, b, c, d) in res.items()), f"TOTAL {tot[0]}/{tot[1]}", flush=True)
    if verbose: print("  fails:", fails)
    return res
if __name__ == "__main__":
    data = load_sets()
    for cfg in [dict(mode="v3"), dict(mode="dq"), dict(mode="v3", prob="dq"), dict(mode="v3", prob="both"),
                dict(mode="fuse"), dict(mode="fuse", prob="dq"), dict(mode="fuse", prob="both"), dict(mode="fuse", extra_w=1.3)]:
        run(cfg, data)
