import numpy as np, onnx, onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs as fbs
from onnx import helper, numpy_helper, TensorProto
buf = open('docquadnet256_trained_opset17.ort', 'rb').read()
g = fbs.InferenceSession.InferenceSession.GetRootAsInferenceSession(buf, 0).Model().Graph()
DT = {1: np.float32, 7: np.int64, 6: np.int32}
inits = []
for i in range(g.InitializersLength()):
    t = g.Initializers(i); dims = [t.Dims(j) for j in range(t.DimsLength())]
    raw = t.RawDataAsNumpy().tobytes() if t.RawDataLength() else b''
    arr = np.frombuffer(raw, DT[t.DataType()]).reshape(dims) if raw else np.zeros(dims, DT[t.DataType()])
    inits.append(numpy_helper.from_array(arr.copy(), t.Name().decode()))
def attrs(n):
    out = {}
    for j in range(n.AttributesLength()):
        a = n.Attributes(j); nm = a.Name().decode(); ty = a.Type()
        if ty == 1: out[nm] = a.F()
        elif ty == 2: out[nm] = a.I()
        elif ty == 3: out[nm] = a.S().decode()
        elif ty == 6: out[nm] = [a.Floats(k) for k in range(a.FloatsLength())]
        elif ty == 7: out[nm] = [a.Ints(k) for k in range(a.IntsLength())]
        elif ty == 8: out[nm] = [a.Strings(k).decode() for k in range(a.StringsLength())]
        else: raise Exception(f"attr type {ty} {nm}")
    return out
nodes = []; seen = {}
# node order: ORT stores in topological order by index
order = sorted(range(g.NodesLength()), key=lambda i: g.Nodes(i).Index())
for i in order:
    n = g.Nodes(i); op = n.OpType().decode(); dom = n.Domain().decode()
    ins = [n.Inputs(k).decode() for k in range(n.InputsLength())]; outs = [n.Outputs(k).decode() for k in range(n.OutputsLength())]
    a = attrs(n); name = n.Name().decode() or f"n{i}"
    if op == 'FusedConv':
        act = a.pop('activation', None); ap = a.pop('activation_params', None)
        z = ins[3] if len(ins) > 3 and ins[3] else None
        ins = [x for x in ins[:3] if x]
        conv_out = outs[0] + '_conv' if (z or act) else outs[0]
        nodes.append(helper.make_node('Conv', ins, [conv_out], name=name, **a))
        mid = conv_out
        if z:  # Conv + Add (+ activation): the sum comes before the activation
            mid = outs[0] + '_sum' if act else outs[0]
            nodes.append(helper.make_node('Add', [conv_out, z], [mid], name=name + '_add'))
        if act is None: continue
        if act == 'Relu': nodes.append(helper.make_node('Relu', [mid], outs, name=name + '_act'))
        elif act == 'HardSigmoid': nodes.append(helper.make_node('HardSigmoid', [mid], outs, name=name + '_act', alpha=ap[0], beta=ap[1]))
        elif act == 'Clip': nodes.append(helper.make_node('Clip', [mid, name+'_lo', name+'_hi'], outs, name=name + '_act')); inits += [numpy_helper.from_array(np.float32(ap[0]), name+'_lo'), numpy_helper.from_array(np.float32(ap[1]), name+'_hi')]
        else: raise Exception("activation " + act)
    else:
        nodes.append(helper.make_node(op, ins, outs, name=name, **a))
# topological order
avail = {'input'} | {t.name for t in inits}; done = []; todo = list(nodes)
while todo:
    progress = False
    for nd in list(todo):
        if all((x == '' or x in avail) for x in nd.input):
            done.append(nd); avail.update(nd.output); todo.remove(nd); progress = True
    if not progress: raise Exception("cycle / missing input: " + str([list(n.input) for n in todo[:3]]))
nodes = done
inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 256, 256])
outs = [helper.make_tensor_value_info('corner_heatmaps', TensorProto.FLOAT, [1, 4, 64, 64]), helper.make_tensor_value_info('mask_logits', TensorProto.FLOAT, [1, 1, 64, 64])]
m = helper.make_model(helper.make_graph(nodes, 'docquad', [inp], outs, inits), opset_imports=[helper.make_opsetid('', 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, 'docquad.onnx'); print("saved", len(nodes), "nodes")
import collections; print(collections.Counter(nd.op_type for nd in nodes))
# OpenCV 4.10 only broadcasts a Mul when the SMALL input comes second: swap where needed.
m = onnx.load('docquad.onnx')
inf = onnx.shape_inference.infer_shapes(m)
shp = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(inf.graph.value_info) + list(inf.graph.input)}
sw = 0
for nd in m.graph.node:
    if nd.op_type in ('Mul', 'Add') and len(nd.input) == 2:
        a, b = shp.get(nd.input[0]), shp.get(nd.input[1])
        if a and b and np.prod(a) < np.prod(b):
            x0, x1 = nd.input[0], nd.input[1]; nd.input[0] = x1; nd.input[1] = x0; sw += 1
onnx.save(m, 'docquad.onnx'); print("swapped", sw)
# OpenCV fuses "HardSigmoid -> Mul" into HardSwish even when the Mul's other input is a different
# tensor (squeeze-excitation), which is wrong. Rewrite those HardSigmoids as x*a+b clipped to [0,1].
m = onnx.load('docquad.onnx')
prod = {o: nd for nd in m.graph.node for o in nd.output}
users = {}
for nd in m.graph.node:
    for i in nd.input: users.setdefault(i, []).append(nd)
new_nodes = []; extra = []; rw = 0
for nd in m.graph.node:
    if nd.op_type == 'HardSigmoid':
        x = nd.input[0]; y = nd.output[0]
        se = any(u.op_type == 'Mul' and x not in u.input for u in users.get(y, []))
        if se:
            al = [a.f for a in nd.attribute if a.name == 'alpha'][0]; be = [a.f for a in nd.attribute if a.name == 'beta'][0]
            k = nd.name.replace('/', '_')
            extra += [numpy_helper.from_array(np.full((1,), al, np.float32), k + '_a'), numpy_helper.from_array(np.full((1,), be, np.float32), k + '_b'),
                      numpy_helper.from_array(np.float32(0), k + '_lo'), numpy_helper.from_array(np.float32(1), k + '_hi')]
            new_nodes += [helper.make_node('Mul', [x, k + '_a'], [k + '_m'], name=k + '_mul'),
                          helper.make_node('Add', [k + '_m', k + '_b'], [k + '_s'], name=k + '_add'),
                          helper.make_node('Clip', [k + '_s', k + '_lo', k + '_hi'], [y], name=k + '_clip')]
            rw += 1; continue
    new_nodes.append(nd)
del m.graph.node[:]; m.graph.node.extend(new_nodes); m.graph.initializer.extend(extra)
onnx.checker.check_model(m); onnx.save(m, 'docquad.onnx'); print("rewrote SE hardsigmoids", rw)
