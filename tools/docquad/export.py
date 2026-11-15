"""Writes fine-tuned weights into the OpenCV-compatible graph (docquad.onnx) and verifies it."""
import onnx, torch, numpy as np, cv2, sys
from onnx import numpy_helper
src = sys.argv[1]; out = sys.argv[2]
model = torch.load(src, weights_only=False); model.eval()
g = onnx.load('/root/dq/docquad.onnx')
mods = dict(model.named_modules())
inits = {t.name: i for i, t in enumerate(g.graph.initializer)}
n = 0
for nd in g.graph.node:
    if nd.op_type != 'Conv': continue
    m = mods.get(nd.name.lstrip('/').replace('.', '/')) or mods.get(nd.name)
    if m is None: raise SystemExit("no module for " + nd.name)
    for idx, attr in ((1, 'weight'), (2, 'bias')):
        if len(nd.input) > idx and nd.input[idx]:
            t = getattr(m, attr).detach().numpy().astype(np.float32)
            old = g.graph.initializer[inits[nd.input[idx]]]
            assert tuple(old.dims) == t.shape, (nd.name, old.dims, t.shape)
            g.graph.initializer[inits[nd.input[idx]]].CopyFrom(numpy_helper.from_array(t, nd.input[idx]))
    n += 1
onnx.save(g, out); print("convs updated", n)
x = np.random.default_rng(3).random((2, 3, 256, 256)).astype(np.float32)
with torch.no_grad(): ref = [o.numpy() for o in model(torch.from_numpy(x))]
net = cv2.dnn.readNetFromONNX(out)
for b in range(2):
    net.setInput(x[b:b + 1]); got = net.forward(['corner_heatmaps', 'mask_logits'])
    print("opencv vs torch:", [float(np.abs(p - q[b:b + 1]).max()) for p, q in zip(got, ref)])
