import cv2, numpy as np, sys, glob, v3, os
v3.NET_CAND = os.environ.get("CAND","0")=="1"; v3.NET_W = float(os.environ.get("NETW","0.3"))
net = cv2.dnn.readNetFromONNX(os.path.expanduser("~/net/pagenet.onnx"))
def prob_for(img):
    blob = cv2.dnn.blobFromImage(img, 1.0, (96, 128), (0, 0, 0), False, False)
    net.setInput(blob); return net.forward()[0, 0]
fs = sorted(glob.glob(os.path.expanduser("~/real/set/r*")))
skip = {"r08", "r23", "r24"}
tiles = []
for f in fs:
    nm = os.path.basename(f)[:3]
    if nm in skip: continue
    im = cv2.imread(f); h, w = im.shape[:2]; s = 640 / max(h, w)
    sm = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    p = prob_for(sm); v3.LAST_CONF = None
    q = v3.detect_color(sm, prob=p); c = v3.LAST_CONF or (0, 1, None)
    unsure = q is None or c[0] < 0.2 or c[1] > 0.6 or (c[2] is not None and c[2] < 0.7)
    g = cv2.cvtColor(cv2.resize(im, (round(w * 400 / max(h, w)), round(h * 400 / max(h, w)))), cv2.COLOR_BGR2GRAY)
    ql = v3.detect(g)
    t = cv2.resize(im, (round(w * 360 / max(h, w)), round(h * 360 / max(h, w))))
    H, W = t.shape[:2]
    pm = cv2.resize((p * 255).astype(np.uint8), (W, H)); t[..., 2] = np.maximum(t[..., 2], (pm > 127) * 0)  # no tint
    if ql is not None: cv2.polylines(t, [np.int32(np.array(ql) * [W, H])], True, (255, 0, 0), 1)
    if q is not None: cv2.polylines(t, [np.int32(np.array(q) * [W, H])], True, (0, 0, 255) if unsure else (0, 200, 0), 2)
    lab = f"{nm} b{c[0]:.2f} r{c[1]:.2f} n{(c[2] or 0):.2f}"
    cv2.putText(t, lab, (3, 16), 0, 0.45, (0, 0, 0), 3); cv2.putText(t, lab, (3, 16), 0, 0.45, (255, 255, 255), 1)
    T = np.zeros((360, 360, 3), np.uint8); T[:H, :W] = t; tiles.append(T)
    print(lab, "UNSURE" if unsure else "sure")
while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "/root/real/out.jpg", np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
