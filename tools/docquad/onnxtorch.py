"""Runs the app's docquad.onnx graph as a trainable PyTorch module, and writes trained weights back.

The graph is executed node by node exactly as stored (Conv, Relu, Add, Mul, Clip, HardSigmoid,
GlobalAveragePool, Resize), with every Conv weight and bias as a trainable parameter keyed by its
ONNX initializer name. Saving just replaces those initializers, so the exported file is the same
OpenCV-compatible graph the app already loads, only with new numbers.
"""
import numpy as np, onnx, torch, torch.nn as nn, torch.nn.functional as F
from onnx import numpy_helper


def _attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class OnnxNet(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = onnx.load(path)
        g = self.model.graph
        conv_params = {i for n in g.node if n.op_type == "Conv" for i in n.input[1:] if i}
        self.key = {}  # onnx initializer name -> safe module attribute name
        self.params = nn.ParameterDict()
        self.consts = {}
        for t in g.initializer:
            arr = numpy_helper.to_array(t).copy()
            if t.name in conv_params:
                k = "p%d" % len(self.key); self.key[t.name] = k
                self.params[k] = nn.Parameter(torch.from_numpy(arr.astype(np.float32)))
            else:
                self.consts[t.name] = arr
        for k, v in list(self.consts.items()):
            if v.dtype == np.float32:
                self.register_buffer("c_" + str(len(self._buffers)), torch.from_numpy(v))
        self.cbuf = {}
        i = 0
        for k, v in self.consts.items():
            if v.dtype == np.float32: self.cbuf[k] = "c_%d" % i; i += 1
        self.nodes = [(n.op_type, list(n.input), list(n.output), _attrs(n)) for n in g.node]
        self.inp = g.input[0].name
        self.outs = [o.name for o in g.output]

    def _get(self, env, name):
        if name in env: return env[name]
        if name in self.key: return self.params[self.key[name]]
        if name in self.cbuf: return getattr(self, self.cbuf[name])
        return self.consts[name]

    def forward(self, x):
        env = {self.inp: x}
        for op, ins, outs, a in self.nodes:
            v = [self._get(env, i) if i else None for i in ins]
            if op == "Conv":
                p = a.get("pads", [0, 0, 0, 0]); assert p[0] == p[2] and p[1] == p[3]
                y = F.conv2d(v[0], v[1], v[2] if len(v) > 2 else None, a.get("strides", [1, 1]),
                             (p[0], p[1]), a.get("dilations", [1, 1]), a.get("group", 1))
            elif op == "Relu": y = F.relu(v[0])
            elif op == "Add": y = v[0] + v[1]
            elif op == "Mul": y = v[0] * v[1]
            elif op == "Clip":
                lo = float(v[1]) if len(v) > 1 and v[1] is not None else None
                hi = float(v[2]) if len(v) > 2 and v[2] is not None else None
                y = torch.clamp(v[0], lo, hi)
            elif op == "HardSigmoid": y = torch.clamp(v[0] * a.get("alpha", 0.2) + a.get("beta", 0.5), 0, 1)
            elif op == "GlobalAveragePool": y = v[0].mean((2, 3), keepdim=True)
            elif op == "Resize":
                assert a["mode"] in (b"linear", "linear") and a["coordinate_transformation_mode"] in (b"half_pixel", "half_pixel")
                sizes = v[3] if len(v) > 3 and v[3] is not None and np.size(v[3]) else None
                if sizes is not None:
                    s = [int(t) for t in (sizes.tolist() if torch.is_tensor(sizes) else sizes)]
                    y = F.interpolate(v[0], size=s[2:], mode="bilinear", align_corners=False)
                else:
                    sc = v[2].tolist() if torch.is_tensor(v[2]) else list(v[2])
                    y = F.interpolate(v[0], scale_factor=sc[2:], mode="bilinear", align_corners=False)
            else:
                raise NotImplementedError(op)
            env[outs[0]] = y
        return tuple(env[o] for o in self.outs)

    def save(self, path):
        """Writes the current weights into a copy of the original graph."""
        m = onnx.ModelProto(); m.CopyFrom(self.model)
        for t in m.graph.initializer:
            if t.name in self.key:
                arr = self.params[self.key[t.name]].detach().float().cpu().numpy()
                assert tuple(t.dims) == arr.shape
                t.CopyFrom(numpy_helper.from_array(arr, t.name))
        onnx.save(m, path)


if __name__ == "__main__":
    # Self-check: PyTorch execution must match OpenCV (what the app uses) on the original file.
    import sys, cv2
    path = sys.argv[1]
    net = OnnxNet(path).eval()
    x = np.random.default_rng(0).random((1, 3, 256, 256)).astype(np.float32)
    with torch.no_grad(): ref = [o.numpy() for o in net(torch.from_numpy(x))]
    cvn = cv2.dnn.readNetFromONNX(path); cvn.setInput(x)
    got = cvn.forward(["corner_heatmaps", "mask_logits"])
    print("max |torch - opencv|:", [float(np.abs(a - b).max()) for a, b in zip(ref, got)])
    print("trainable parameters:", sum(p.numel() for p in net.parameters()))
