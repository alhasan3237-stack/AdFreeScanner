import cv2, numpy as np, onnxruntime as ort, os
S = ort.InferenceSession(os.path.expanduser(os.environ.get("DQ_MODEL", "~/dq/docquadnet256_trained_opset17.ort")), providers=["CPUExecutionProvider"])
def sig(x): return 1 / (1 + np.exp(-x))
def run(img):
    """img: BGR uint8 any size. Returns (corners Nx2 in img px TL,TR,BR,BL, peak probs, mask prob 64x64 in letterbox, lb)."""
    h, w = img.shape[:2]; s = min(256 / w, 256 / h); nw, nh = w * s, h * s; ox, oy = (256 - nw) / 2, (256 - nh) / 2
    M = np.float32([[s, 0, ox], [0, s, oy]])
    inter = cv2.INTER_AREA
    lb = cv2.warpAffine(cv2.resize(img, (max(1, round(nw)), max(1, round(nh))), interpolation=inter), np.float32([[1, 0, ox], [0, 1, oy]]), (256, 256), borderValue=(0, 0, 0))
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
    outs = dict(zip([o.name for o in S.get_outputs()], S.run(None, {S.get_inputs()[0].name: x})))
    hm = outs["corner_heatmaps"][0]; mk = outs["mask_logits"][0, 0]
    pts = []; peaks = []
    for c in range(4):
        a = hm[c]; iy, ix = np.unravel_index(np.argmax(a), a.shape); peaks.append(sig(a[iy, ix]))
        y0, y1, x0, x1 = max(0, iy - 1), min(64, iy + 2), max(0, ix - 1), min(64, ix + 2)
        wts = sig(a[y0:y1, x0:x1]); yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = (wts * (xx + 0.5)).sum() / wts.sum() * 4; cy = (wts * (yy + 0.5)).sum() / wts.sum() * 4
        pts.append(((cx - ox) / s, (cy - oy) / s))
    return np.array(pts), np.array(peaks), sig(mk), (s, ox, oy)
