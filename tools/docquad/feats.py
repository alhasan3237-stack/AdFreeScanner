import sys, pickle, numpy as np, cv2
sys.path.insert(0, '/root/bench'); import fuse, dq, v3
v3.NET_CAND = False; v3.NET_W = 0.3; v3.TRUST = 9; v3.AGREE = 0.75
data = pickle.load(open('/root/sets/all.pkl', 'rb'))
rows = []
for name, sm, gt, fn in data:
    h, w = sm.shape[:2]; pts, pk, mk, lb = dq.run(sm)
    dmask = fuse.dq_image_mask(mk, lb, w, h)
    v3.EXTRA = [pts]; v3.EXTRA_CONF = float(pk.min())
    v3.detect_color(sm, prob=fuse.pagenet_prob(sm)); A = v3.ALT
    # agreement of each option with the corner network's own mask
    def miou(q):
        m = np.zeros((128, 96), np.uint8); cv2.fillConvexPoly(m, np.int32(np.array(q) * [96, 128]), 1)
        b = dmask > 0.5; i = np.logical_and(m, b).sum(); u = np.logical_or(m, b).sum(); return i / u if u else 0
    rows.append(dict(set=name, fn=fn, pk=float(pk.min()), pkmean=float(pk.mean()), e_v3=fuse.err_of(A['q_v3'], gt, w, h), e_dq=fuse.err_of(A['q_dq'], gt, w, h),
                     e_raw=fuse.err_of(pts / [w, h], gt, w, h), mask_v3=miou(A['q_v3']), mask_dq=miou(A['q_dq']),
                     **{k: v for k, v in A.items() if not k.startswith('q_')}))
pickle.dump(rows, open('/root/sets/feats_%s.pkl' % sys.argv[1], 'wb')); print(len(rows))
